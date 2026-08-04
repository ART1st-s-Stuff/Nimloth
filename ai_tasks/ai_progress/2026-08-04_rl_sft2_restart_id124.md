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
