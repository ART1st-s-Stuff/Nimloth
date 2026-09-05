# 2026-06-23 VAGEN legacy retry14

## Goal
Continue VAGEN legacy navigation PPO with `prompt_format=wm`, `bi_level_gae`, `max_actions_per_step=1`, and the 2048 per-turn response cap, using a stable external env + train split.

## Current plan
1. Run external 4-GPU env on normal `dgx-16`.
2. Run single-node 7-GPU train on normal `dgx-54`.
3. Monitor env health, Ray startup, validation step 0, and the first PPO training step.

## Current status
- Env job submitted and running:
  - `458286 vagen-env-ext4gpu RUNNING dgx-16`
  - env URLs:
    - `http://10.23.0.133:8400`
    - `http://10.23.0.133:8401`
    - `http://10.23.0.133:8402`
    - `http://10.23.0.133:8403`
- Train job submitted but still pending:
  - `458296 vagen-train-resume PENDING (Priority) ReqNodeList=dgx-54`

## Output directory
`/project/peilab/atst/nimloth/outputs/experiments/training/baseline/2026-06-23/vagen_legacy_wm_bilevel_gamma1_100ep_4env8train_dgx54_16_retry14`

## Notes
- A run README was written with the current git hash:
  - `a5f7f9f3d3b53c85e505eeade431254c4d376576`
- The train job is currently blocked by Slurm priority, not by env health.
- Temporary train script was reduced to 1 node / 7 GPUs and Ray head was trimmed to `CUDA_VISIBLE_DEVICES=0..6`.
