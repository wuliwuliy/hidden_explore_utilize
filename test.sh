ray job submit \
  --working-dir . \
  --runtime-env-json='{
    "excludes": [
      "./checkpoint/verl-grpo_Qwen2.5-0.5B-Instruct_max_response512_batch4_rollout4_klcoef0.0001_entcoef0.001_simplelr_math_35/global_step_40/actor/huggingface/*.json",
      "./custom/data/",
      "/examples/simplelr_math_eval/data/tabmwp/test.jsonl"
    ]
  }' \
  -- python -c "print('Dry run')"