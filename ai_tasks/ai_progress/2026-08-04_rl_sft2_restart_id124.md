# 2026-08-04: RL ID124 single-node restart

- Human requested using an actually idle node. A formal one-node/eight-GPU
  topology was added without changing the objective: two TP4 rollout workers,
  four two-GPU training ranks, 16 episodes/update, and eval10x120.
- Commit `f272d7d5` also fixed all formal batches to allow the intentionally
  empty fresh-init `INITIAL_RESUME_CHECKPOINT`; remote shell/config/full-runner
  regression passed 65 tests.
- W&B maximum numeric ID was 122; exact ID124 name and formal output were
  absent. Actual VAGEN assets are train 1200+1200 and held-out 60+60 with no
  train/eval scene overlap. SFT2 epoch1 model/WM/ValueHead/training files were
  non-empty.
- Job `505936` acquired idle `dgx-39:8` at 22:19:53+08, but failed before Ray
  and environment startup. Submission set `ENV_REPO` to the VAGEN submodule;
  the controller appended `external/VAGEN`, producing a duplicated path and
  exit128 at iteration1 startup.
- Formal output was never created. Only the adjacent iteration-progress log
  exists; there is no W&B run, rollout, optimizer, consumption, metric, or
  checkpoint. ID124 cannot resume. Retry requires new ID/output/W&B identity
  and `ENV_REPO` equal to the parent Nimloth runtime worktree.

## ID125 corrected single-node launch

- ID125 uses the same verified runtime code `f272d7d5`, objective, SFT2 epoch1
  source, 1x8 topology, dataset split, token caps, and eval10x120 schedule. Its
  new output and W&B identity were absent. Exact preflight passed with
  `ENV_REPO` set to the parent runtime worktree and the resolved pinned
  `external/VAGEN` assets checked.
- Job `505944` acquired all eight GPUs on idle normal node `dgx-39` at
  22:26:26+08. The formal README confirms two TP4 rollout workers and four
  synchronized two-GPU training ranks; Slurm stderr remains empty.
- Both isolated navigation environments completed real AI2-THOR prewarm in
  about 11.1 seconds. Two independent vLLM world4 groups connected all ranks,
  loaded both safetensor shards on all eight workers in 57--60 seconds, created
  KV caches, and completed engine profile/warmup at 22:31:22+08 without
  CUDA/NCCL/OOM errors.
- Iterations 1 and 2 subsequently completed strict 16-rollout merges, finite
  synchronized updates, and durable `train/latest` checkpoints. Iteration 1
  used 320 transitions with train-rollout success 0/16 and total loss 2.87047;
  iteration 2 used 305 transitions with train-rollout success 1/16 and total
  loss 3.62398. The corresponding WM/DINO/value losses were
  0.10872/0.92707/2.29822 and 0.23518/0.91012/2.93374.
- At 22:48:49+08, job `505944` remained healthy on `dgx-39:8` with empty Slurm
  stderr and iteration 3 running in both rollout shards. One early iteration-3
  episode had succeeded, but the merge and update were not yet complete. No
  held-out evaluation exists yet, as the first 120-episode evaluation is
  scheduled after iteration 10; per-iteration success above is training-rollout
  success and is not `val_success_rate`.
