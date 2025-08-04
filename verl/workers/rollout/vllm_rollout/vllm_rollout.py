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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length, pad_4d_tensor, pad_to_packed
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            swap_space=config.swap_space,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        if "mistral" in config.model_path.lower():
            if "24b" in config.model_path.lower():
                kwargs['stop_token_ids'] = [23836, 19464, 3263, 18993] # _end, istant, user, _start
            elif "7b-v0.1" in config.model_path.lower():
                kwargs['stop_token_ids'] = [22478, 24994, 26307, 9977, 933, 2820] # Question, Answer, \nStep
        elif "llama" in config.model_path.lower():
            kwargs['stop_token_ids'] = [14924, 16533] #Question, Answer
        elif "deepseek-math" in  config.model_path.lower():
            kwargs['stop_token_ids'] = [3631, 81038, 5726, 77398, 6713] # user, assistant, system
        elif "qwen2.5" in config.model_path.lower():
            if "7b" in config.model_path.lower():
                kwargs['stop_token_ids'] = [151645, 151643, 872,77091, 1474, 71703, 151644, 8948]
                if "math" in config.model_path.lower():
                    kwargs['stop_token_ids'] = [151645, 151643, 872, 77091, 1474, 71703, 151644, 8948, 73594]
            elif "3b" in config.model_path.lower():
                kwargs['stop_token_ids'] = [14582, 16141, 31198] # Question, Answer, Problem      
            elif "1.5b" in config.model_path.lower():
                kwargs['stop_token_ids'] = [14582, 16141, 31198] # Question, Answer, Problem
            elif "0.5b" in config.model_path.lower():
                kwargs['stop_token_ids'] = [14582, 16141, 31198] # Question, Answer, Problem
                
        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        MICRO_ROLLOUT_BATCH_SIZE = self.config.micro_rollout_batch_size
        # outputs = []
        #  
        
        responses_list = []
        log_probs_list = []
        hidden_states_decode_list = []
        hidden_states_prefill_list = []
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            for i in range(0, batch_size, MICRO_ROLLOUT_BATCH_SIZE):
                batch_end = min(i + MICRO_ROLLOUT_BATCH_SIZE, batch_size)
                idx_chunk = idx_list[i:batch_end]
                output = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    prompt_token_ids=idx_chunk,
                    use_tqdm=False)
                # outputs.append(output)
                responses_list.append(output[0])
                log_probs_list.append(output[1])
                hidden_states_decode_list.append(output[2])
                hidden_states_prefill_list.append(output[3])
        
        # hidden_states_prefill_list[0].shape
        # torch.Size([66, 896])
        # hidden_states_list[0].shape
        # torch.Size([4, 236, 1, 896])
                
        #  
        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        # response = torch.cat(responses, dim=0).to(idx.device)
        # log_probs = torch.cat(log_probs, dim=0).to(idx.device)
        new_responses_list = []
        new_log_probs_list = []
        new_hidden_states_decode_list = []
        new_hidden_states_prefill_list = []
        for i in range(len(responses_list)):
            if responses_list[i].shape[1] < self.config.response_length:
                response_ = pad_sequence_to_length(responses_list[i], self.config.response_length, self.pad_token_id)
                new_responses_list.append(response_)
                # log_probs = pad_sequence_to_length(log_probs_list[i], self.config.response_length, self.pad_token_id)
                # new_log_probs_list.append(log_probs)
                
                hidden_states_decode_ = pad_4d_tensor(hidden_states_decode_list[i], self.config.response_length, self.pad_token_id, left_pad=False) # right pad
                new_hidden_states_decode_list.append(hidden_states_decode_)

            else:
                new_responses_list.append(responses_list[i])
                # new_log_probs_list.append(log_probs_list[i])
                new_hidden_states_decode_list.append(hidden_states_decode_list[i])

            
            if hidden_states_prefill_list[i].shape[1] < self.config.prompt_length:
                
                hidden_states_prefill_ = pad_4d_tensor(hidden_states_prefill_list[i], self.config.prompt_length, self.pad_token_id, left_pad=True) # left pad
                new_hidden_states_prefill_list.append(hidden_states_prefill_)
            else:
                new_hidden_states_decode_list.append(hidden_states_decode_list[i])
    
        response = torch.cat(new_responses_list, dim=0).to(idx.device)
        # log_probs = torch.cat(new_log_probs_list, dim=0).to(idx.device)
        hidden_states_decode = torch.cat(new_hidden_states_decode_list, dim=0).to(idx.device)
        hidden_states_prefill = torch.cat(new_hidden_states_prefill_list, dim=0).to(idx.device)
 
        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        prompt_attention_mask = attention_mask
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        if hidden_states_decode.shape[0] == hidden_states_prefill.shape[0]:
             # all the tp ranks should contain the same data here. data in all ranks are valid
            merged_batch = TensorDict(
                {
                    'prompts': idx,
                    'responses': response,
                    'input_ids': seq,  # here input_ids become the whole sentences
                    # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                    'attention_mask': attention_mask,
                    'position_ids': position_ids,
                    'hidden_states_decode': hidden_states_decode,
                    'hidden_states_prefill': hidden_states_prefill
                },
            batch_size=batch_size)

        else:

            base_batch = TensorDict(
                {
                    'prompts': idx,
                    'responses': response,
                    'input_ids': seq,  # here input_ids become the whole sentences
                    # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                    'attention_mask': attention_mask,
                    'position_ids': position_ids,
                    'hidden_states_decode': hidden_states_decode
                },
                batch_size=batch_size)
        
            
            # 为 hidden_states_prefill 创建一个新的 TensorDict,使用原始的 batch_size
            prefill_batch = TensorDict(
                {
                    'hidden_states_prefill': hidden_states_prefill
                },
                batch_size=batch_size // self.config.n  # 使用未扩展的 batch size
            )

            # 在 generate_sequences 方法的返回语句之前，合并两个 batch
            # 创建一个索引映射来重复使用 prefill 数据
            repeat_indices = torch.arange(batch_size // self.config.n).repeat_interleave(self.config.n)

            # 将两个 TensorDict 组合
            merged_batch = TensorDict(
                {
                    **base_batch.to_dict(),
                    'hidden_states_prefill': prefill_batch['hidden_states_prefill'][repeat_indices]
                },
                batch_size=batch_size
            )
                
        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        #  这儿改成返合并
        return DataProto(batch=merged_batch)
