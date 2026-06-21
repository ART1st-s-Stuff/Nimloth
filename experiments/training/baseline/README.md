# VAGEN navigation baseline (canonical)

Canonical scripts for VAGEN navigation RL baseline per `ai_rules/03_experiments_and_data.md`.

**Do not add** node-specific or retry-numbered Slurm files here. Put one-off run details under `outputs/experiments/training/baseline/<date>/`.

## Layout

| File | Purpose |
|------|---------|
| `install_vagen_env.slurm` | One-time uv/VAGEN/AI2-THOR install |
| `setup_ai2thor_env.sh` | Vulkan + AI2-THOR runtime |
| `launch_env_servers.sh` + `env_server.slurm` | Preempt env servers (2×4 GPU) for fresh training |
| `hold_preempt.slurm` + `launch_preempt_training.sh` + `run_preempt_training.sh` | Co-located env+train on 2 preempt nodes (when normal queue unavailable) |
| `launch_val_wandb_watcher.sh` + `val_wandb_watcher.slurm` | Poll training checkpoints, run val_only, upload val curves to wandb |
| `env_external_4gpu.slurm` | Normal-partition external env (4 processes) |
| `train.slurm` | Fresh 2-node×8 GPU PPO training |
| `train_resume.slurm` | Resume multinode training (`trainer.resume_mode=auto`) |
| `vagen_paper_ppo_cli.inc.sh` | Shared Table-23-aligned Hydra CLI overrides |
| `vagen_rollout_vllm_cli.inc.sh` | Shared vLLM rollout Hydra overrides (default backend; avoid sglang) |
| `vagen_env_repro_cli.inc.sh` | Reproducible env sampling + val composition assert |
| `convert_checkpoint_world_size.slurm` | HF → FSDP shard conversion |
| `prune_checkpoints.py` + `prune_checkpoints_policy.sh` | Keep latest + every 10th step + best val checkpoint |
| `convert_vagen_*_to_world_size.py` | Conversion entrypoints |
| `slurm_gpu_resources.py` | Cluster GPU inventory helper |
| `submit_env_external_4gpu.sh` | Thin sbatch wrapper for external env |
| `submit_train_resume.sh` | Thin sbatch wrapper for resume train |

Config: `configs/training/baseline/` (`train.yaml`, `val.yaml`, `defaults.yaml`).

Outputs: `outputs/experiments/training/baseline/` (Slurm logs, per-run dirs, `progress.md`).

## Latest valid reference run (2026-06)

Remote legacy run (still authoritative for SFT1 init / rollouts):

`experiments/navigation_baseline/runs/vagen_nav_dgx31_49train_dgx36env_3node_16train8env_original_base_common_resp20k_single_action_promptfix_retry2`

- Latest checkpoint: `global_step_93` (`checkpoints/latest_checkpointed_iteration.txt`)
- Resume pattern: external 4-GPU env + 2-node×8 GPU train (`train_resume.slurm`)
- Fused kernels **off** on resume; `use_kl_in_reward=True`, `kl_coef=0.001`

New runs should write to `outputs/experiments/training/baseline/<YYYY-MM-DD>/<experiment_name>/`.
To resume a legacy run, set `RUN_DIR` to the legacy path.

## Quick start (fresh training)

```bash
cd /project/peilab/atst/nimloth
sbatch experiments/training/baseline/install_vagen_env.slurm   # once
bash experiments/training/baseline/launch_env_servers.sh
sbatch experiments/training/baseline/train.slurm
```

## Preempt co-located training (env + train on held nodes)

Use when `normal` partition has no whole nodes. **Do not** add node- or run-named scripts; pass parameters via env and record the exact command in `outputs/.../README.md`.

```bash
cd /project/peilab/atst/nimloth
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

# Optional: pick idle nodes — document choice in outputs README, not in repo filenames
# NODELIST=dgx-47,dgx-55 \
EXPERIMENT_NAME=vagen_nav_wm_fresh \
RUN_DATE=$(date +%Y-%m-%d) \
bash experiments/training/baseline/launch_preempt_training.sh
```

Defaults match `configs/training/baseline/defaults.yaml` (`prompt_format=wm` in train/val yaml, 50 steps, batch 128/32, `test_freq=10`). Override with `TOTAL_STEPS`, `TRAIN_BATCH_SIZE`, `NODELIST`, etc.

Outputs: `outputs/experiments/training/baseline/<date>/<EXPERIMENT_NAME>/` plus group-level `progress.md`.

## Resume (external env + train)

```bash
# 1) env on a 4-GPU node (optional NODELIST=...)
RUN_DIR=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/2026-06-18/my_run \
  bash experiments/training/baseline/submit_env_external_4gpu.sh

# 2) train after env ready
RUN_DIR=... EXPERIMENT_NAME=my_run NODELIST=dgx-32,dgx-37 \
  bash experiments/training/baseline/submit_train_resume.sh
```

## Legacy

`experiments/navigation_baseline/` is frozen legacy. Do not add scripts there.
