#! /bin/bash

USER_ENV=`whoami`
set -x
# export NCCL_DEBUG=DEBUG
export RAY_BACKEND_LOG_LEVEL=debug
export RAY_DEDUP_LOGS=1


export PROJECT_NAME=verl_train
export WANDB_API_KEY=22af3c074c3d3de0b406284e18bb302225ede044
export WANDB_OFFICIAL=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export HDFS_DATA_PATH=/home/hfd24/simpleRL-reason/custom/data
export HDFS_MODEL_PATH=/home/hfd24/model
export HDFS_CHECKPOINT_PATH=/home/hfd24/simpleRL-reason/custom/checkpoint
export HDFS_LOG_PATH=/home/hfd24/simpleRL-reason/custom/log
export RUN_NAME=verl-grpo
export ARNOLD_WORKER_NUM=1 # number of nodes you want to use 

# ========== 新增配置 ==========
export NCCL_PROTO=simple
export NCCL_ALGO=ring
export NCCL_SOCKET_IFNAME=eno1  # 确认是eno1而非docker虚拟网卡
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 禁用InfiniBand
export NCCL_P2P_DISABLE=1  # 禁用P2P直连（跨NUMA节点时需要）

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=/home/.cache/huggingface
# 禁用FlashAttention的核心变量
export FLASH_ATTENTION_SKIP_INIT=1
export DISABLE_FLASH_ATTENTION=1
export USE_FLASH_ATTN=0
export USE_FLASH_ATTN_2=0

# 强制使用兼容的注意力后端
export TRANSFORMERS_ATTENTION_BACKEND=eager

# 针对PyTorch的配置
export TORCH_CUDA_ARCH_LIST="7.0"  # 指定V100的计算能力
# =============================

WORKING_DIR="."

HEAD_IP="10.14.4.13"  
HEAD_PORT="6379"      

# Default values
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=32
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=1024
LEARNING_RATE=5e-7
PPO_MINI_BATCH_SIZE=16
# per GPU
PPO_MICRO_BATCH_SIZE=4
CLIP_RATIO=0.2
KL_LOSS_COEF=0.001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=4
ROLLOUT_N=4
KL_COEF=0.001
TOTAL_EPOCHS=2
DATASET_NAME=simplelr_qwen_level3to5
ROLLOUT_GPU_MEMORY_UTIL=0.5
MODEL_NAME=Qwen2.5-Math-7B
SAVE_FREQ=-1
TEST_FREQ=-1
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
MICRO_ROLLOUT_BATCH_SIZE=8
REMOVE_PREVIOUS_CKPT=True
EXP_NAME=""


# Default values for new parameters
REWARD_EMA_ALPHA=""
REWARD_INDICATOR_NAMES=""
REWARD_WEIGHTS=""
REWARD_WEIGHTS_EXPLOIT=""
METRIC_INDICES="[0,1,2]" # 默认指标索引
# <<< 修改点 1: 将 HYDRA_OVERRIDES 初始化为数组 >>>
HYDRA_OVERRIDES=()
VAL_BEFORE_TRAIN=True
VAL_SAMPLE_SIZE=-1
ENABLE_CALCULATOR=True
DIFF_STRIDE=20
ADD_REWARD=True
COMPUTE_LOG_EFFECTIVE_RANK=False
MODULATION_GAIN=1.5
OUTPUT_TOKEN_LEVEL_METRICS=False
ADV_ESTIMATOR="grpo"  # grpo or gae
CRITIC_MODEL_PATH=""
AUX_REWARD_GLOBAL_WEIGHT=1.0 # 建议也加上这个
TOKEN_LEVEL_BASELINE_TYPE="internal_mean" # <--- 在这里添加
generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local suffix_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --val_batch_size) suffix+="_valbatch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_max_prompt$2"; shift 2 ;;
      --max_response_length) suffix+="_max_response$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --ppo_micro_batch_size) shift 2 ;;
      --kl_loss_coef) suffix+="_klcoef$2"; shift 2 ;;
      --entropy_coeffient) suffix+="_entcoef$2"; shift 2 ;;
      --clip_ratio) suffix+="_clipratio$2"; shift 2 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --log_prob_micro_batch_size) suffix+="_logprobbatch$2"; shift 2 ;;
      --rollout_n) suffix+="_rollout$2"; shift 2 ;;
      --kl_coef) suffix+="_klcontrol$2"; shift 2 ;;
      --total_epochs) suffix+="_epochs$2"; shift 2 ;;
      --rollout_gpu_memory_util) shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --remove_clip) suffix+="_remove_clip$2"; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      --logger_config) LOGGER_CONFIG="$2"; shift 2 ;;
      --exp_name) EXP_NAME="$2"; shift 2 ;;
      --diff_stride) suffix+="_stride$2"; shift 2 ;;
      --reward_ema_alpha) suffix+="_ema$2"; shift 2 ;;
      --metric_indices) METRIC_INDICES="$2"; shift 2 ;;
      --modulation_gain) suffix+="_mgain$2"; shift 2 ;;
      --adv_estimator) suffix+="_$2"; shift 2 ;;
      --critic_model_path) shift 2 ;;
      *) shift ;;
    esac
  done

  # 如果命令行中没有提供 --dataset_name，则使用默认值
  # 因为上面设置了标志位，所以这里不会重复添加
  # if [ "$dataset_provided" = false ]; then
  #   suffix+="_$DATASET_NAME"
  # fi

  if [ "$suffix_provided" = true ]; then
    suffix+="_$input_suffix"
  fi
  
  echo "$suffix"
}

echo "Arguments received: $@"

# Generate a unique suffix based on the input arguments
SUFFIX=$(generate_suffix "$@")
RUN_NAME="$RUN_NAME$SUFFIX"
LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"
EXP_NAME=${exp_name}

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --val_batch_size) VAL_BATCH_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --ppo_micro_batch_size) PPO_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy_coeffient) ENTROPY_COEFFIENT="$2"; shift 2 ;;
    --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
    --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --log_prob_micro_batch_size) LOG_PROB_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --micro_rollout_batch_size) MICRO_ROLLOUT_BATCH_SIZE="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;
    --remove_clip) REMOVE_CLIP="$2"; shift 2 ;;
    --remove_previous_ckpt) REMOVE_PREVIOUS_CKPT="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    --logger_config) LOGGER_CONFIG="$2"; shift 2 ;;
    --exp_name) EXP_NAME="$2"; shift 2 ;;
    # <<< 新增参数解析 >>>
    --reward_ema_alpha) REWARD_EMA_ALPHA="$2"; shift 2 ;;
    --reward_indicator_names) REWARD_INDICATOR_NAMES="$2"; shift 2 ;;
    --reward_weights) REWARD_WEIGHTS="$2"; shift 2 ;;
    --reward_weights_exploit) REWARD_WEIGHTS_EXPLOIT="$2"; shift 2 ;;
    --val_before_train) VAL_BEFORE_TRAIN="$2"; shift 2 ;;
    --val_sample_size) VAL_SAMPLE_SIZE="$2"; shift 2 ;;
    --diff_stride) DIFF_STRIDE="$2"; shift 2 ;;
    --enable_calculator) ENABLE_CALCULATOR="$2"; shift 2 ;;
    --add_reward) ADD_REWARD="$2"; shift 2 ;;
    --compute_log_effective_rank) COMPUTE_LOG_EFFECTIVE_RANK="$2"; shift 2 ;;
    --metric_indices) METRIC_INDICES="$2"; shift 2 ;;
    --modulation_gain) MODULATION_GAIN="$2"; shift 2 ;;
    --output_token_level_metrics) OUTPUT_TOKEN_LEVEL_METRICS="$2"; shift 2 ;;
    --adv_estimator) ADV_ESTIMATOR="$2"; shift 2 ;;
    --critic_model_path) CRITIC_MODEL_PATH="$2"; shift 2 ;;
    --aux_reward_global_weight) AUX_REWARD_GLOBAL_WEIGHT="$2"; shift 2 ;;
    --token_level_baseline_type) TOKEN_LEVEL_BASELINE_TYPE="$2"; shift 2 ;; 
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "$ADV_ESTIMATOR" == "grpo" ]]; then
  if [[ "$ROLLOUT_N" -le 1 ]]; then
    echo "错误：当 adv_estimator 为 grpo 时, --rollout_n 必须大于 1."
    exit 1
  fi
  if [[ -n "$CRITIC_MODEL_PATH" ]]; then
    echo "警告：当 adv_estimator 为 grpo 时, --critic_model_path 将被忽略."
    CRITIC_MODEL_PATH="" # 确保不传递
  fi
elif [[ "$ADV_ESTIMATOR" == "gae" ]]; then
  if [[ "$ROLLOUT_N" -ne 1 ]]; then
    echo "错误：当 adv_estimator 为 gae (PPO模式) 时, --rollout_n 必须等于 1."
    exit 1
  fi
  if [[ -z "$CRITIC_MODEL_PATH" ]]; then
    echo "错误：当 adv_estimator 为 gae (PPO模式) 时, 必须通过 --critic_model_path 提供 Critic 模型路径."
    exit 1
  fi
else
  echo "错误：无效的 adv_estimator: $ADV_ESTIMATOR. 请选择 'grpo' 或 'gae'."
  exit 1
fi

# ... (End of argument parsing while loop)

# Generate a unique suffix based on the input arguments (now without model name)
SUFFIX=$(generate_suffix "$@")

# Construct the FINAL_RUN_NAME in the desired order: {model}_{exp}_{base}{suffix}
# For example: Qwen2.5-3B_origin_verl-grpo_max_response1280...
if [[ "$ADV_ESTIMATOR" == "gae" ]]; then
  SUFFIX+="_critic-$(basename $CRITIC_MODEL_PATH)"
fi
FINAL_RUN_NAME="${MODEL_NAME}_${EXP_NAME}_${RUN_NAME}${SUFFIX}"

# Update the log file path to use the new name
LOG_FILE_PATH="$HDFS_LOG_PATH/$FINAL_RUN_NAME.log"
# The EXP_NAME variable is now part of the FINAL_RUN_NAME

# ... (echo statements)

echo "Training with the following parameters:"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Val Batch Size: $VAL_BATCH_SIZE" 
echo "Max Prompt Length: $MAX_PROMPT_LENGTH" 
echo "Max Response Length: $MAX_RESPONSE_LENGTH" 
echo "Learning Rate: $LEARNING_RATE" 
echo "PPO Mini Batch Size: $PPO_MINI_BATCH_SIZE" 
echo "PPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE" 
echo "Micro Rollout Batch Size: $MICRO_ROLLOUT_BATCH_SIZE"
echo "KL Loss Coefficient: $KL_LOSS_COEF" 
echo "KL Loss Type: $KL_LOSS_TYPE" 
echo "Temperature: $TEMPERATURE" 
echo "Rollout N: $ROLLOUT_N" 
echo "KL Coefficient: $KL_COEF" 
echo "Total Epochs: $TOTAL_EPOCHS"
echo "Dataset Name: $DATASET_NAME"
echo "Model Name: $MODEL_NAME"
echo "Remove Clip: $REMOVE_CLIP"
echo "Remove Previous Ckpt: $REMOVE_PREVIOUS_CKPT"
echo "LOG FILE PATH: $LOG_FILE_PATH"

max_num_batched_tokens=$(expr $MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH + 50)
echo -e "Training with the following parameters:\nTrain Batch Size: $TRAIN_BATCH_SIZE\nVal Batch Size: $VAL_BATCH_SIZE\nMax Prompt Length: $MAX_PROMPT_LENGTH\nMax Response Length: $MAX_RESPONSE_LENGTH\nLearning Rate: $LEARNING_RATE\nPPO Mini Batch Size: $PPO_MINI_BATCH_SIZE\nPPO Micro Batch Size: $PPO_MICRO_BATCH_SIZE\nKL Loss Coefficient: $KL_LOSS_COEF\nKL Loss Type: $KL_LOSS_TYPE\nTemperature: $TEMPERATURE\nRollout N: $ROLLOUT_N\nKL Coefficient: $KL_COEF\nTotal Epochs: $TOTAL_EPOCHS\nDataset Name: $DATASET_NAME\nModel Name: $MODEL_NAME"

echo "Validate Before Train: $VAL_BEFORE_TRAIN"
echo "Validation Sample Size: $VAL_SAMPLE_SIZE"
echo "Calculator Diff Stride: $DIFF_STRIDE"
echo "Enable Calculator Metrics: $ENABLE_CALCULATOR"
echo "Add Reward enabled: $ADD_REWARD"
echo "Compute Log Effective Rank: $COMPUTE_LOG_EFFECTIVE_RANK"
echo "LOG FILE PATH: $LOG_FILE_PATH"

# <<< 修改点 2: 构建 hydra 参数数组 >>>
# 将覆盖参数作为独立元素添加到数组中
if [ -n "$REWARD_EMA_ALPHA" ]; then
  HYDRA_OVERRIDES+=("reward_manager.ema_alpha=$REWARD_EMA_ALPHA")
fi
if [ -n "$REWARD_INDICATOR_NAMES" ]; then
  # 这里直接使用变量，不添加额外的引号
  HYDRA_OVERRIDES+=("reward_manager.indicator_names=$REWARD_INDICATOR_NAMES")
fi
if [ -n "$REWARD_WEIGHTS" ]; then
  HYDRA_OVERRIDES+=("reward_manager.weights=$REWARD_WEIGHTS")
fi
if [ -n "$REWARD_WEIGHTS_EXPLOIT" ]; then
  HYDRA_OVERRIDES+=("reward_manager.weights_exploit=$REWARD_WEIGHTS_EXPLOIT")
fi

if [ -n "$ADD_REWARD" ]; then
  HYDRA_OVERRIDES+=("reward_manager.add_reward=$ADD_REWARD")
fi
if [ -n "$COMPUTE_LOG_EFFECTIVE_RANK" ]; then
  HYDRA_OVERRIDES+=("calculator.compute_log_effective_rank=$COMPUTE_LOG_EFFECTIVE_RANK")
fi
if [ -n "$METRIC_INDICES" ]; then
  HYDRA_OVERRIDES+=("calculator.metric_indices=$METRIC_INDICES")
fi
if [ -n "$MODULATION_GAIN" ]; then
  HYDRA_OVERRIDES+=("reward_manager.modulation_gain=$MODULATION_GAIN")
fi
if [ -n "$OUTPUT_TOKEN_LEVEL_METRICS" ]; then
  HYDRA_OVERRIDES+=("calculator.output_token_level_metrics=$OUTPUT_TOKEN_LEVEL_METRICS")
fi
if [ -n "$CRITIC_MODEL_PATH" ]; then
  HYDRA_OVERRIDES+=("critic.model.path=$CRITIC_MODEL_PATH")
fi
# v-- 在这里添加下面的代码块 --v
if [ -n "$TOKEN_LEVEL_BASELINE_TYPE" ]; then
  HYDRA_OVERRIDES+=("reward_manager.token_level_baseline_type=$TOKEN_LEVEL_BASELINE_TYPE")
fi
ray job submit --address=${HEAD_IP}:${HEAD_PORT} \
  --entrypoint-num-cpus=1 \
  --runtime-env-json='{
    "working_dir": "'${WORKING_DIR}'",
    "excludes": [
    "/.git/",                    
    "/checkpoint/",
    "/custom/checkpoint/",
    "/custom/log/",
    "/examples/simplelr_math_eval/data/tabmwp/test.jsonl"
  ],
    "env_vars": {
      "RAY_DEBUG": "1",
      "GLOO_SOCKET_IFNAME": "eno1",
      "NCCL_SOCKET_IFNAME": "eno1",
      "NCCL_PROTO": "simple",
      "NCCL_ALGO": "ring",
      "VLLM_DTYPE": "float16",
      "CUDA_VISIBLE_DEVICES": "0,1", 
      "OMP_NUM_THREADS": "8" ,
      "TORCH_CUDA_ARCH_LIST": "7.0",
      "FLASH_ATTENTION_SKIP_INIT": "1",
      "VLLM_ATTENTION_BACKEND": "XFORMERS",
      "TRANSFORMERS_ATTENTION_BACKEND": "eager",
      "USE_FLASH_ATTN": "0",
      "USE_FLASH_ATTN_2": "0",
      "DISABLE_FLASH_ATTENTION": "1",
      "REWORD_FUNCTION_TYPE": "independent",
      "WANDB_API_KEY": "22af3c074c3d3de0b406284e18bb302225ede044"
    }
  }' \
  -- python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=$ADV_ESTIMATOR \
  data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
  data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/test.parquet \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  actor_rollout_ref.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
  actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
  actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
  actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
  actor_rollout_ref.rollout.micro_rollout_batch_size=$MICRO_ROLLOUT_BATCH_SIZE \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=$KL_COEF \
  critic.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
  trainer.critic_warmup=0 \
  trainer.logger=$LOGGER_CONFIG \
  trainer.project_name=$PROJECT_NAME \
  trainer.remove_previous_ckpt=$REMOVE_PREVIOUS_CKPT \
  trainer.experiment_name=$RUN_NAME \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=$ARNOLD_WORKER_NUM \
  trainer.remove_clip=$REMOVE_CLIP \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$FINAL_RUN_NAME \
  "${HYDRA_OVERRIDES[@]}" \
  trainer.val_before_train=$VAL_BEFORE_TRAIN \
  trainer.val_sample_size=$VAL_SAMPLE_SIZE \
  calculator.diff_stride=$DIFF_STRIDE \
  calculator.enable=$ENABLE_CALCULATOR \
  trainer.total_epochs=$TOTAL_EPOCHS 2>&1 | tee -a $LOG_FILE_PATH \
  # trainer.total_training_steps=2 \
  


