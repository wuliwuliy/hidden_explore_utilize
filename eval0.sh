#!/bin/bash

# =======================================================
#               参数配置区域
# =======================================================
DTYPE="torch.float16"
#    在这里集中管理你的路径
HDFS_PATH="/home/hfd24/simpleRL-reason/custom"
MODEL_BASE_PATH="/home/hfd24/model/qwen"

# 1. 定义要遍历的 Run Names (模型/检查点)
#    在这里添加你所有想要测试的 --run_name
RUN_NAMES=(
    "DeepSeek-R1-Distill-Qwen-1.5B_origin_verl-grpo_max_response2048_batch48_rollout6_valbatch6_ppomini24_logprobbatch1_klcoef0.001_entcoef0.001_epochs2_simplelr_qwen_level3to5"
)

# 2. 定义要遍历的温度值
TEMPERATURES=(0.0)

# 3. 其他固定参数
USE_WANDB="false"
CALCULATE_METRICS="true"
METRICS_TO_CALC="Effective Rank"
METRIC_ORDERS="0,1,2"
METRIC_STRIDE=20
SPECIFIC_STEPS="0,120"
BENCHMARKS="aime24,amc23,math500,olympiadbench,gsm8k,minerva_math" # aime24,amc23,math500,olympiadbench,gsm8k,minerva_math"

# =======================================================
#               嵌套循环主体
# =======================================================

# 外层循环: 遍历每一个 run_name
for run_name in "${RUN_NAMES[@]}"
do
    # 内层循环: 遍历每一个温度值
    for temp in "${TEMPERATURES[@]}"
    do
        echo "========================================================================"
        echo ">>>>>  RUNNING EVALUATION  <<<<<"
        echo ">>>>>  Run Name: ${run_name}"
        echo ">>>>>  Temperature: ${temp}"
        echo "========================================================================"

        # 核心修改：定义输出目录为 "${run_name}/eval_results_temp_${temp}"
        FINAL_OUTPUT_DIR="eval_results_temp_${temp}"
        
        # 确保目录存在
        mkdir -p "${FINAL_OUTPUT_DIR}"

        bash eval_math_nodes.sh \
            --run_name "${run_name}" \
            --template qwen-boxed  \
            --init_model DeepSeek-R1-Distill-Qwen-1.5B \
            --tp_size 1 \
            --add_step_0 true  \
            --temperature ${temp} \
            --top_p 0.95 \
            --max_tokens 2048 \
            --benchmarks ${BENCHMARKS} \
            --n_sampling 1 \
            --visible_gpus 0,1 \
            --output_dir "${FINAL_OUTPUT_DIR}" \
            --use_wandb_arg ${USE_WANDB} \
            --calculate_metrics ${CALCULATE_METRICS} \
            --metrics_to_calc "${METRICS_TO_CALC}" \
            --metric_orders "${METRIC_ORDERS}" \
            --metric_stride ${METRIC_STRIDE} \
            --specific_steps "${SPECIFIC_STEPS}" \
            --num_test_sample_per_dataset -1 \
            --hdfs_home "${HDFS_PATH}" \
            --init_model_base_path "${MODEL_BASE_PATH}" \
            --dtype "${DTYPE}"
    done
done

echo "========================================================================"
echo "All evaluations are complete."
echo "========================================================================"