# False / True 
bash train_grpo_math_tune_ray.sh \
    --model_name qwen/Qwen2.5-0.5B --max_response_length 800 \
    --critic_model_path "" \
    --train_batch_size 8 \
    --rollout_n 3 \
    --val_batch_size 12 \
    --ppo_mini_batch_size 8 \
    --ppo_micro_batch_size 2 \
    --log_prob_micro_batch_size 2 \
    --micro_rollout_batch_size 2 \
    --kl_loss_coef 0.001 \
    --entropy_coeffient 0.001 \
    --rollout_gpu_memory_util 0.68 \
		--logger_config "['console','wandb']" \
    --rollout_tp 1 --save_freq 20 --test_freq 5 --total_epochs 2 \
    --exp_name "test" --dataset_name "simplelr_abel_gsm8k_level1" \
    --reward_ema_alpha 0.5 \
    --reward_weights "[0.0, 0.0, 1.0]" \
    --reward_weights_exploit "[0.0, 1.0, 0.0]" \
    --reward_indicator_names "['Effective Rank diff 2', 'Effective Rank diff', 'Effective Rank']" \
    --val_before_train True \
    --val_sample_size 13 \
    --enable_calculator True --metric_indices "[0,1]" \
    --output_token_level_metrics False --compute_log_effective_rank False \
    --add_reward False --diff_stride 20 --modulation_gain 1.5  --adv_estimator "grpo" 

    # python monitor_gpu.py -H 10 -S 10 -g 0 1 2 3 -o ./custom/log_gpu 
    # llama/Llama-3.2-1B 可以
    # google/gemma-2b 可以 
    # google/gemma-2-2b