# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from tensordict import TensorDict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

from torch.utils.data import Subset
import torch
WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    #  
    # Initialize base return dict
    return_dict = dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

    # Add correctness-based calculations if available
    if 'correctness' in batch.batch:
        correctness = batch.batch['correctness'].bool()  # Convert to boolean mask
        
        # Calculate lengths for correct and incorrect responses
        response_length_correct = response_length[correctness]
        response_length_incorrect = response_length[~correctness]
        
        # Add to return dict
        return_dict.update({
            'response_length_correct': response_length_correct,
            'response_length_incorrect': response_length_incorrect,
        })

    return return_dict
    
    
def compute_response_metrics(batch):
    max_response_length = batch.batch['responses'].shape[-1]
    response_info = _compute_response_info(batch)
    response_length = response_info['response_length']
    
    metrics = {
        # response length metrics (always available)
        'response_length/overall/mean': torch.mean(response_length).detach().item(),
        'response_length/overall/max': torch.max(response_length).detach().item(),
        'response_length/overall/min': torch.min(response_length).detach().item(),
        'response_length/overall/clip_ratio': torch.mean(
            torch.eq(response_length, max_response_length).float()
        ).detach().item(),
    }
    
    # Handle correct responses
    if 'response_length_correct' in response_info:
        response_length_correct = response_info['response_length_correct']
        if len(response_length_correct) > 0:
            metrics.update({
                'response_length/correct/mean': torch.mean(response_length_correct).detach().item(),
                'response_length/correct/max': torch.max(response_length_correct).detach().item(),
                'response_length/correct/min': torch.min(response_length_correct).detach().item(),
                'response_length/correct/clip_ratio': torch.mean(
                    torch.eq(response_length_correct, max_response_length).float()
                ).detach().item(),
            })
        else:
            metrics.update({
                'response_length/correct/mean': 0.0,
                'response_length/correct/max': 0.0,
                'response_length/correct/min': 0.0,
                'response_length/correct/clip_ratio': 0.0,
            })
    
    # Handle incorrect responses
    if 'response_length_incorrect' in response_info:
        response_length_incorrect = response_info['response_length_incorrect']
        if len(response_length_incorrect) > 0:
            metrics.update({
                'response_length/incorrect/mean': torch.mean(response_length_incorrect).detach().item(),
                'response_length/incorrect/max': torch.max(response_length_incorrect).detach().item(),
                'response_length/incorrect/min': torch.min(response_length_incorrect).detach().item(),
                'response_length/incorrect/clip_ratio': torch.mean(
                    torch.eq(response_length_incorrect, max_response_length).float()
                ).detach().item(),
            })
        else:
            metrics.update({
                'response_length/incorrect/mean': 0.0,
                'response_length/incorrect/max': 0.0,
                'response_length/incorrect/min': 0.0,
                'response_length/incorrect/clip_ratio': 0.0,
            })
    
    return metrics


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score_0 = batch.batch['token_level_scores_0'].sum(-1)
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score 0
        'critic/score_0/mean':
            torch.mean(sequence_score_0).detach().item(),
        'critic/score_0/max':
            torch.max(sequence_score_0).detach().item(),
        'critic/score_0/min':
            torch.min(sequence_score_0).detach().item(),
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'filtered_response_length/mean':
            torch.mean(response_length).detach().item(),
        'filtered_response_length/max':
            torch.max(response_length).detach().item(),
        'filtered_response_length/min':
            torch.min(response_length).detach().item(),
        'filtered_response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }



def compute_calculator_metrics(results, correctness_tensor, mid_vs=None):
    stats_dict = {}

    # 确保 correctness_tensor 是一个 tensor
    if not isinstance(correctness_tensor, torch.Tensor):
        correctness_tensor = torch.tensor(correctness_tensor)

    # 将 correctness_tensor 转换为 boolean 索引
    correct_indices = correctness_tensor == 1 # 假设 1 表示正确
    incorrect_indices = correctness_tensor != 1 # 假设其他值表示错误

    #
    if mid_vs:
        for key, value in mid_vs.items():
            stats_dict[f"cal/base_v/{key}"] = value

    # 计算正确和错误的样本数量
    stats_dict["cal/num_correct"] = correct_indices.sum().item()
    stats_dict["cal/num_incorrect"] = incorrect_indices.sum().item()

    # 可选：计算正确率
    total_samples = len(correctness_tensor)
    if total_samples > 0:
        stats_dict["cal/accuracy"] = stats_dict["cal/num_correct"] / total_samples


    for layer_key, indicators in results.items():
        for indicator_name, values in indicators.items():
            # 确保 values 是 torch.Tensor
            if not isinstance(values, torch.Tensor):
                values = torch.tensor(values)

            # 总体统计量
            stats_dict[f"cal/overall/layer_{layer_key}/{indicator_name}/max"] = torch.max(values).item()
            stats_dict[f"cal/overall/layer_{layer_key}/{indicator_name}/min"] = torch.min(values).item()
            stats_dict[f"cal/overall/layer_{layer_key}/{indicator_name}/mean"] = torch.mean(values).item()

            # 正确样本统计量
            correct_values = values[correct_indices]
            if correct_values.numel() > 0: # 检查是否有正确样本
                stats_dict[f"cal/correct/layer_{layer_key}/{indicator_name}/max"] = torch.max(correct_values).item()
                stats_dict[f"cal/correct/layer_{layer_key}/{indicator_name}/min"] = torch.min(correct_values).item()
                stats_dict[f"cal/correct/layer_{layer_key}/{indicator_name}/mean"] = torch.mean(correct_values).item()
            else:
                # 如果没有正确样本，可以设置为 NaN 或其他标记
                stats_dict[f"cal/correct/layer_{layer_key}/{indicator_name}/max"] = 0.0
                stats_dict[f"cal/correct/layer_{layer_key}/{indicator_name}/min"] = 0.0
                stats_dict[f"cal/correct/layer_{layer_key}/{indicator_name}/mean"] = 0.0

            # 错误样本统计量
            incorrect_values = values[incorrect_indices]
            if incorrect_values.numel() > 0: # 检查是否有错误样本
                stats_dict[f"cal/incorrect/layer_{layer_key}/{indicator_name}/max"] = torch.max(incorrect_values).item()
                stats_dict[f"cal/incorrect/layer_{layer_key}/{indicator_name}/min"] = torch.min(incorrect_values).item()
                stats_dict[f"cal/incorrect/layer_{layer_key}/{indicator_name}/mean"] = torch.mean(incorrect_values).item()
            else:
                # 如果没有错误样本，可以设置为 NaN 或其他标记
                stats_dict[f"cal/incorrect/layer_{layer_key}/{indicator_name}/max"] = 0.0
                stats_dict[f"cal/incorrect/layer_{layer_key}/{indicator_name}/min"] = 0.0
                stats_dict[f"cal/incorrect/layer_{layer_key}/{indicator_name}/mean"] = 0.0

    

    return stats_dict
def concatenate_results(calculater_lst):
    """
    Concatenate multiple results dictionaries along the batch dimension.
    
    Args:
        calculater_lst: List of results dictionaries, each with structure:
            {
                "1": {
                    "Response Entropy": tensor([...]),  # shape [batch_size]
                    "Effective Rank": tensor([...]),
                    "Curvature": tensor([...])
                },
                "2": { ... }
            }
            
    Returns:
        A single concatenated dictionary with the same structure but concatenated tensors
    """
    if not calculater_lst:
        return {}
    
    # Get all layer keys and metric keys from the first result
    layer_keys = calculater_lst[0].keys()
    metric_keys = calculater_lst[0][next(iter(layer_keys))].keys()
    
    concatenated = {}
    
    for layer in layer_keys:
        concatenated[layer] = {}
        for metric in metric_keys:
            # Concatenate all tensors for this layer and metric across batches
            tensors = [res[layer][metric] for res in calculater_lst]
            concatenated[layer][metric] = torch.cat(tensors, dim=0)
    
    return concatenated

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 calculator=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.calculator = calculator

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        # self.val_dataloader = DataLoader(dataset=self.val_dataset,
        #                                  batch_size=len(self.val_dataset),
        #                                  shuffle=True,
        #                                  drop_last=True,
        #                                  collate_fn=collate_fn)

        # sampled_indices = np.random.choice(len(self.val_dataset), size=10, replace=False)
        # sampled_dataset = Subset(self.val_dataset, sampled_indices)
        # self.val_dataloader = DataLoader(
        #     dataset=sampled_dataset,
        #     batch_size=len(sampled_dataset),  # 或更小的 batch_size
        #     shuffle=False,  # 已随机采样，无需再 shuffle
        #     drop_last=False,
        #     collate_fn=collate_fn
        # )
        # <<< INICIO DE MODIFICACIONES >>>
        # Lógica para submuestrear el dataloader de validación si se especifica
        val_sample_size = self.config.trainer.get('val_sample_size', -1)
        if val_sample_size > 0:
            print(f"Sampling {val_sample_size} examples from the validation set.")
            # Asegurarse de que el tamaño de la muestra no exceda el tamaño del dataset
            sample_size = min(val_sample_size, len(self.val_dataset))
            sampled_indices = np.random.choice(len(self.val_dataset), size=sample_size, replace=False)
            sampled_dataset = Subset(self.val_dataset, sampled_indices)
            self.val_dataloader = DataLoader(
                dataset=sampled_dataset,
                batch_size=len(sampled_dataset),  # Cargar todas las muestras en un solo batch
                shuffle=False,  # Ya se muestreó aleatoriamente
                drop_last=False,
                collate_fn=collate_fn
            )
        else:
            print("Using the full validation set.")
            self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                             batch_size=len(self.val_dataset),
                                             shuffle=True,
                                             drop_last=True,
                                             collate_fn=collate_fn)
        # <<< FIN DE MODIFICACIONES >>>

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        correctness_lst = []
        reward_tensor_lst = []
        reward_tensor_0_lst = []
        data_source_lst = []
        calculater_lst = []


        use_calculator = self.config.calculator.get('enable', True)

        for test_data in self.val_dataloader:
            
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            
            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            #  
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            prompt_len = test_batch.batch['prompts'].shape[1]  # 例如 512
            response_attention_mask = test_batch.batch['attention_mask'][:, prompt_len:]
            lens_tensor = response_attention_mask.sum(dim=-1)  # [10] - sum of non-padding tokens for each sample

             
            # <<< INICIO DE MODIFICACIONES EN _validate >>>
            # Obtener diff_stride desde la configuración de Hydra
            if use_calculator:
                diff_stride_val = self.config.calculator.get('diff_stride', 20)
                
                test_batch.batch['calculator_results'] = self.calculator(
                    hidden_states=test_batch.batch['hidden_states_decode'],  # [10, 2048, 2, 896]
                    attention_mask=response_attention_mask,  # [10, 2048]
                    compute_diff=True, 
                    diff_stride=diff_stride_val  # Usar el valor de la configuración
                )
                # <<< FIN DE MODIFICACIONES EN _validate >>>

                calculater_lst.append(test_batch.batch['calculator_results'])
            
            # Add these near where other lists are initialized
            length_lst = []

            # Inside the loop, after calculating lens_tensor:
            length_lst.append(lens_tensor.cpu())
            

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor_dict = self.val_reward_fn(test_batch, is_val=True)
            
            reward_tensor_0 = reward_tensor_dict['reward_tensor_0'] 
            reward_tensor = reward_tensor_dict['reward_tensor'] 
            correctness = reward_tensor_dict['correctness_tensor']
            
            reward_tensor_0_lst.append(reward_tensor_0)
            reward_tensor_lst.append(reward_tensor)
            correctness_lst.append(correctness)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

            

            

            
        reward_tensor_0 = torch.cat(reward_tensor_0_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        correctness_tensor = torch.cat(correctness_lst, dim=0).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        
        if use_calculator:
            calculator_cat = concatenate_results(calculater_lst)
        else:
            calculator_cat = {}
        length_tensor = torch.cat(length_lst, dim=0).cpu()  # (total_batch_size,)
        # evaluate test_score based on data source
        data_source_reward_0 = {}
        data_source_reward = {}
        data_source_correctness = {}

        data_source_calculator_overall = {}
        # 新增：用于存储正确和错误回答的 calculator 指标
        data_source_calculator_correct = {}
        data_source_calculator_incorrect = {}


        # Add these near where other data_source dictionaries are initialized
        data_source_length = {}


        if calculator_cat:
            for i in range(reward_tensor.shape[0]):
                data_source = data_sources[i]
                is_correct = correctness_tensor[i].item() # 获取当前样本正确性标签


                if data_source not in data_source_reward:
                    data_source_reward_0[data_source] = []
                    data_source_reward[data_source] = []
                    data_source_correctness[data_source] = []
                    # 初始化总体、正确、错误回答的 calculator 字典
                    data_source_calculator_overall[data_source] = {}
                    data_source_calculator_correct[data_source] = {}
                    data_source_calculator_incorrect[data_source] = {}

                    data_source_length[data_source] = []
                
                data_source_reward_0[data_source].append(reward_tensor_0[i].item())
                data_source_reward[data_source].append(reward_tensor[i].item())
                data_source_correctness[data_source].append(is_correct)
                data_source_length[data_source].append(length_tensor[i].item())


                for layer, layer_calculator in calculator_cat.items():
                    # 恢复：将总体指标添加到列表中
                    if layer not in data_source_calculator_overall[data_source]:
                        data_source_calculator_overall[data_source][layer] = {}
                    if layer not in data_source_calculator_correct[data_source]:
                        data_source_calculator_correct[data_source][layer] = {}
                        data_source_calculator_incorrect[data_source][layer] = {}

                    for indicator, value_tensor in layer_calculator.items():
                        if indicator not in data_source_calculator_overall[data_source][layer]:
                             data_source_calculator_overall[data_source][layer][indicator] = []
                        if indicator not in data_source_calculator_correct[data_source][layer]:
                            data_source_calculator_correct[data_source][layer][indicator] = []
                            data_source_calculator_incorrect[data_source][layer][indicator] = []
                        
                        data_source_calculator_overall[data_source][layer][indicator].append(value_tensor[i].item())

                        if is_correct == 1:
                            data_source_calculator_correct[data_source][layer][indicator].append(value_tensor[i].item())
                        else:
                            data_source_calculator_incorrect[data_source][layer][indicator].append(value_tensor[i].item())


        metric_dict = {}


        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        
        for data_source, rewards in data_source_reward_0.items():
            metric_dict[f'val/test_score_0/{data_source}'] = np.mean(rewards)

        for data_source, correctnesses in data_source_correctness.items():
            metric_dict[f'val/test_correctness/{data_source}'] = np.mean(correctnesses)

        # === 新增代码开始 ===
        def fill_metrics(prefix, calc_dict):
            """通用指标填充函数"""
            for ds_name, ds_data in calc_dict.items():
                for layer, layer_data in ds_data.items():
                    for metric, values in layer_data.items():
                        key = f"val/{prefix}/{ds_name}/layer_{layer}/{metric}"
                        if len(values) > 0:
                            metric_dict[key] = np.nanmean(values) if np.isnan(values).any() else np.mean(values)
                        else:
                            metric_dict[key] = 0.0

        # <<< INICIO DE MODIFICACIONES EN _validate >>>
        if calculator_cat:
            fill_metrics("cal_correct", data_source_calculator_correct)
            fill_metrics("cal_incorrect", data_source_calculator_incorrect)
            fill_metrics("cal_overall", data_source_calculator_overall)
        # <<< FIN DE MODIFICACIONES EN _validate >>>


        # Add these with the other metric calculations
        for data_source, lengths in data_source_length.items():
            metric_dict[f'val/test_overall_len/{data_source}'] = np.mean(lengths)
            # Calculate correct/incorrect lengths by filtering with correctness
            correct_lengths = [length for length, correct in zip(lengths, data_source_correctness[data_source]) if correct == 1]
            incorrect_lengths = [length for length, correct in zip(lengths, data_source_correctness[data_source]) if correct == 0]
            
            metric_dict[f'val/test_correct_len/{data_source}'] = np.mean(correct_lengths) if correct_lengths else 0.0
            metric_dict[f'val/test_incorrect_len/{data_source}'] = np.mean(incorrect_lengths) if incorrect_lengths else 0.0

        
        if 'hidden_states_decode' in test_batch.batch:
            del test_batch.batch['hidden_states_decode']
        if 'calculator_results' in test_batch.batch:
            del test_batch.batch['calculator_results']

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, remove_previous_ckpt=self.config.trainer.remove_previous_ckpt)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, remove_previous_ckpt=self.config.trainer.remove_previous_ckpt)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))
            
    # # 新增过滤方法（需在类中定义）
    # def _filter_batch(self, batch, mask: np.ndarray) -> DataProto:
    #     """根据布尔掩码过滤批次数据"""
    #     mask_tensor = torch.from_numpy(mask).to(batch.batch.device)
    
    #     # 过滤张量数据
    #     filtered_tensors = {
    #         k: v[mask_tensor] for k, v in batch.batch.items()
    #     }
    
    #     # 过滤非张量数据（如果有）
    #     filtered_non_tensors = {
    #         k: [x for x, m in zip(v, mask) if m]
    #         for k, v in batch.non_tensor_batch.items()
    #     }
    
    #     return DataProto(
    #         batch=TensorDict(filtered_tensors, batch_size=mask.sum()),
    #         non_tensor_batch=filtered_non_tensors,
    #         meta_info=batch.meta_info
    #     )
    def _filter_batch(self, batch, mask: np.ndarray) -> DataProto:
        """根据布尔掩码过滤批次数据"""
        mask_tensor = torch.from_numpy(mask).to(batch.batch.device)
    
        # 过滤张量数据
        filtered_tensors = {
            k: v[mask_tensor] for k, v in batch.batch.items()
        }
    
        # ==== 修复点：保持 non_tensor_batch 为 NumPy 数组 ====
        filtered_non_tensors = {
            k: v[mask]  # 直接使用 NumPy 布尔索引（保持数组类型）
            for k, v in batch.non_tensor_batch.items()
        }
    
        return DataProto(
            batch=TensorDict(filtered_tensors, batch_size=mask.sum()),
            non_tensor_batch=filtered_non_tensors,
            meta_info=batch.meta_info
        )


    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        # logger = Tracking(project_name=self.config.trainer.project_name,
        #                   experiment_name=self.config.trainer.experiment_name,
        #                   default_backend=self.config.trainer.logger,
        #                   config=OmegaConf.to_container(self.config, resolve=True))

        import os

        # 获取 checkpoint 路径
        checkpoint_dir = self.config.trainer.default_local_dir
        experiment_name = self.config.trainer.experiment_name

        # 构造 wandb 日志路径：只到 wandb 一级目录
        wandb_log_dir = os.path.join(checkpoint_dir, 'wandb')

        # 创建目录
        os.makedirs(wandb_log_dir, exist_ok=True)

        # 设置环境变量，告诉 wandb 把日志写入这里
        os.environ["WANDB_DIR"] = wandb_log_dir

        # 初始化 logger
        logger = Tracking(project_name=self.config.trainer.project_name,
                        experiment_name=self.config.trainer.experiment_name,
                        default_backend=self.config.trainer.logger,
                        config=OmegaConf.to_container(self.config, resolve=True),
                        default_local_dir=self.config.trainer.default_local_dir)  # 👈 增加参数

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training 这儿记住打开 ****************************************
        # currently, we only support validation using the reward_function.

        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return


        # <<< INICIO DE MODIFICACIONES EN fit >>>
        # Leer el flag para habilitar/deshabilitar el calculator
        use_calculator = self.config.calculator.get('enable', True)
        # <<< FIN DE MODIFICACIONES EN fit >>>

        # we start from step 1
        self.global_steps += 1
        metrics_old = {}
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        #  
                    
                    # 为批次中的每个样本生成一个唯一的 UUID，用于在后续处理中追踪和识别每个样本，特别是在计算 GRPO (Group-based Reward Policy Optimization) 优势时很重要
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                     
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # print("------batch--------", batch)
                    # print("------response-----", batch.batch["responses"])
                    # print("------batch.batch['attention_mask']------", batch.batch['attention_mask'])
                    # ===== 新增响应长度统计 =====
                    # response_info = _compute_response_info(batch)
                    # response_lengths = response_info['response_length']
                    # max_len = batch.batch['responses'].shape[-1]  # 获取当前批次的最大响应长度
                    # # ===== 核心修改：按样本整体屏蔽 =====
                    # # 生成样本级掩码（True表示需要屏蔽）
                    # sample_mask = (response_lengths >= max_len)  # [batch_size]
                    # # 将样本级掩码扩展到与attention_mask相同形状
                    # # 假设attention_mask形状为 [batch_size, seq_len]
                    # mask_2d = sample_mask.unsqueeze(-1).expand_as(batch.batch['attention_mask'])
                    # # 执行批量屏蔽（将被屏蔽样本的整个attention置零）
                    # adjusted_attention_mask = torch.where(
                    # mask_2d,
                    # torch.zeros_like(batch.batch['attention_mask']),
                    # batch.batch['attention_mask']
                    # )

                    response_info = _compute_response_info(batch)
                    response_lengths = response_info['response_length']
                    max_len = batch.batch['responses'].shape[-1]  # 获取当前批次的最大响应长度
                    # ===== 核心修改：按样本整体屏蔽 =====
                    # 生成样本级掩码（True表示需要屏蔽）
                    sample_mask = (response_lengths >= max_len)  # [batch_size]
                    
                    
                    
                    adjusted_attention_mask = batch.batch['attention_mask'].clone()
                    for i, mask in enumerate(sample_mask):
                        if mask:
                            adjusted_attention_mask[i, -max_len:] = 0  # 将响应部分掩码置零
                    
                    # metrics.update(compute_response_metrics(batch=batch))

                    # 更新指标

                    # <<< INICIO DE MODIFICACIONES EN fit >>>
                    if use_calculator:
                        prompt_len = batch.batch['prompts'].shape[1] # 例如 512
                        response_attention_mask = batch.batch['attention_mask'][:, prompt_len:]
                        diff_stride_train = self.config.calculator.get('diff_stride', 20)
                        
                        batch.batch['calculator_results'] = self.calculator(
                            hidden_states=batch.batch['hidden_states_decode'], # [10, 2048, 2, 896]
                            attention_mask=response_attention_mask, # [10, 2048]
                            compute_diff=True, 
                            diff_stride=diff_stride_train
                        )
                        del batch.batch['hidden_states_decode']

                    
                    #  
                    if self.config.trainer.remove_clip:
                        batch.batch['attention_mask'] = adjusted_attention_mask

                    # ===== 指标统计 =====
                    truncated_ratio = torch.sum(sample_mask.float()).item()  # 计算被屏蔽样本比例
                    
                    # print('adaptive_mask/truncated_ratio', truncated_ratio)
                    # print('adaptive_mask/total_ratio', adjusted_attention_mask.shape[0])
                    # print("------response length-----", response_lengths.cpu().numpy())
                    # print("batch.batch['attention_mask']", adjusted_attention_mask.cpu().numpy())
                    #print("lenght_ma")
                    
                    
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor_dict: dict = self.reward_fn(batch, is_val=False, metrics_old=metrics_old)  # 这儿加了一个
                        # batch.batch['token_level_scores'] = reward_tensor
                        batch.batch['token_level_scores_0'] = reward_tensor_dict['reward_tensor_0']
                        batch.batch['token_level_scores'] = reward_tensor_dict['reward_tensor']
                        batch.batch['correctness'] = reward_tensor_dict['correctness_tensor']

                         
                        # ### 新增: 接收并处理来自 RewardManager 的内部指标 ###
                        internal_reward_metrics = reward_tensor_dict.get('internal_metrics', {})
                        if internal_reward_metrics:
                            for key, value_list in internal_reward_metrics.items():
                                if value_list: # 确保列表不为空
                                    # 将列表中的值求平均，并添加到主 metrics 字典中
                                    metrics[f'reward_manager/{key}_mean'] = np.mean(value_list)
                        # ### 新增结束 ###

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                    
                    metrics.update(compute_response_metrics(batch=batch))
                    # metrics.update(compute_calculator_metrics(batch.batch['calculator_results'], batch.batch['correctness'], self.reward_fn.mids))


                    # <<< INICIO DE MODIFICACIONES EN fit >>>
                    if use_calculator:
                        metrics.update(compute_calculator_metrics(batch.batch['calculator_results'], batch.batch['correctness'], self.reward_fn.mids))
                        del batch.batch['calculator_results']
                    # <<< FIN DE MODIFICACIONES EN fit >>>



                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        #  
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                #  

                metrics_old = metrics.copy()

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
