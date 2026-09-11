# Retired early-stage entrypoints

User explicitly requests sole B training implementation, no retained old code. Removed old semantic Stage1 trainers, fixed-machine submitters, and one-off step79 training pipeline with its implementation-specific tests. Historical outputs/task evidence/archive remain unchanged. Generic collection tools and Stage3/RL are not included.

- experiments/navigation_baseline/submit_sft1_alltrain_8gpu_embedlr_with_eval.sh
- experiments/navigation_baseline/submit_sft1_alltrain_8gpu_lora_with_eval.sh
- experiments/navigation_baseline/submit_sft1_train_vagen79.sh
- experiments/navigation_baseline/submit_sft1_train_vagen79_alltrain.sh
- experiments/navigation_baseline/submit_sft1_train_vagen79_maskfix.sh
- experiments/navigation_baseline/train_sft1_qwen25vl.py
- experiments/navigation_baseline/train_sft1_vagen48.slurm
- experiments/navigation_baseline/train_sft1_vagen79_1node4gpu.slurm
- experiments/navigation_baseline/train_sft1_vagen79_1node4gpu_alltrain.slurm
- experiments/navigation_baseline/train_sft1_vagen79_1node8gpu_alltrain_embedlr.slurm
- experiments/navigation_baseline/train_sft1_vagen79_1node8gpu_alltrain_lora.slurm
- experiments/navigation_baseline/train_sft1_vagen79_2node8gpu.slurm
- experiments/navigation_baseline/train_sft1_vagen79_maskfix_1node4gpu.slurm
- experiments/training/sft/evaluation/run_step79_stage1_stage2_eval.sh
- experiments/training/sft/evaluation/pipeline_contract.py
- tests/training/sft/evaluation/test_pipeline_contract.py

## Early-stage evaluation replacement

- experiments/navigation_baseline/submit_sft1_lora_ckpt_eval_watcher.sh
- experiments/navigation_baseline/sft1_eval_one_model_valtest.slurm
- experiments/navigation_baseline/sft1_lora_ckpt_eval_watcher.slurm
- experiments/navigation_baseline/submit_sft1_eval_vagen48.sh
- experiments/navigation_baseline/submit_sft1_eval_epoch17_nimloth.sh
- experiments/navigation_baseline/summarize_sft1_eval_rollouts.py
- experiments/navigation_baseline/sft1_eval_vagen79_greedy_valtest.slurm
- experiments/navigation_baseline/submit_sft1_eval_epoch17_vs_step79.sh
- experiments/navigation_baseline/compare_sft1_eval_summaries.py
- experiments/navigation_baseline/resubmit_sft1_eval_vagen48.sh
- experiments/navigation_baseline/run_sft1_lora_ckpt_eval_watcher.sh
- experiments/navigation_baseline/submit_sft1_eval_vagen79_after_train.sh
- experiments/training/sft1/eval_greedy_valtest.slurm
- experiments/training/sft1/run_parent_checkpoint_eval_arm.sh
- experiments/training/sft1/eval_parent_checkpoints_test300.slurm
- experiments/training/sft1/ckpt_eval_watcher.slurm
- experiments/training/sft1/submit_ckpt_eval_watcher.sh
- experiments/training/sft1/finalize_parent_checkpoint_eval.py
- experiments/training/sft1/compare_eval_summaries.py
- tests/training/sft1/test_finalize_parent_checkpoint_eval.py

Retain experiments/training/sft1/summarize_eval_rollouts.py only because deferred historical Stage3 (experiments/training/sft2/eval_greedy_valtest.slurm) calls it; early eval will use the unified new summary. No Stage3 caller change in this scope.
