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
- This is verified healthy model-service startup, not yet a completed RL
  update. At the last readable log boundary, no episode manifest, strict
  16-rollout merge, optimizer step, or checkpoint had been confirmed.
