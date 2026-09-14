# Stage1 protocol-weight16 existing-resume contract

User-approved scope: same original run, model parameter scope, data/cache, seed42, FSDP8, original epoch7 optimizer/scheduler/RNG/convergence. Only action-number/start/end/EOS weighting changes to16; other targets1. Weighted validation metric added for comparison; unweighted LM continues to drive original min2/patience2/1% stop rule. No hard epoch cap or automatic switch of objective.

Source commit pending final check/commit. Local source /workspace/remote2/nimloth/.worktree/fix-sft1-termination-retrain; remote new /mnt/nimloth/.worktree/sft1-protocol-weight16. Python /mnt/nimloth/venv/bin/python3. Original config retained under /mnt/nimloth/.worktree/fix-sft1-termination-retrain/configs/training/sft1/format.yaml. Use existing experiments/training/sft1/train.py and --resume.

Command arguments:
--source-model /mnt/nimloth/checkpoint/hf_actor
--train-jsonl /mnt/nimloth/outputs/experiments/sft1-rollout2000/20260910T173759Z_B_semantic_prompt/data/sft1_train_all.jsonl
--val-jsonl /mnt/nimloth/outputs/experiments/sft1-rollout2000/20260910T173759Z_B_semantic_prompt/data/sft1_heldout_all.jsonl
--format-eval-jsonl /mnt/nimloth/outputs/experiments/sft1-rollout2000/20260910T173759Z_B_semantic_prompt/data/sft1_heldout_all.jsonl
--output-dir /mnt/nimloth/outputs/experiments/sft1-rollout2000/20260911T115012Z_success_only_termination_w8
--config /mnt/nimloth/.worktree/fix-sft1-termination-retrain/configs/training/sft1/format.yaml
--nproc-per-node 8 --success-only --resume
--action-token-loss-weight 16 --boundary-token-loss-weight 16 --format-eval-batch-size 4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; PYTHONUNBUFFERED=1; OMP_NUM_THREADS=8; TOKENIZERS_PARALLELISM=false. CPATH project Python310 headers; no per-server paths added to source. Step126 ->127 and epoch7 ->8 must be seen in launch logs; checkpoint every10 steps; existing epoch-end intermediate pruning retained. Epoch7 immutable. Original prepared caches reused with --require-prebuilt-cache; no semantic-token reinitialization.

Monitoring: prior user authorizes recurring status checks. Every5minutes check exact owned workflow/distributed pids and logs, validation/checkpoint artifacts. Quiet routine progress; notify completed epochs with both validation losses, completion or actionable errors. Three consecutive SSH failure rounds pause monitor and notify once, do not cancel training merely from connectivity. Training to convergence; time elapsed alone is not convergence. Record final GPU/runtime outcome and cleanup exact owned resources.
