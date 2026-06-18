# VAGEN navigation baseline (canonical)

Canonical scripts for VAGEN navigation RL baseline per `ai_rules/03_experiments_and_data.md`.

**Do not add** node-specific or retry-numbered Slurm files here. Put one-off run details under `outputs/experiments/training/baseline/<date>/`.

## Layout

| File | Purpose |
|------|---------|
| `install_vagen_env.slurm` | One-time uv/VAGEN/AI2-THOR install |
| `setup_ai2thor_env.sh` | Vulkan + AI2-THOR runtime |
| `launch_env_servers.sh` + `env_server.slurm` | Preempt env servers (2×4 GPU) for fresh training |
| `env_external_4gpu.slurm` | Normal-partition external env (4 processes) |
| `train.slurm` | Fresh 2-node×8 GPU PPO training |
| `train_resume.slurm` | Resume multinode training (`trainer.resume_mode=auto`) |
| `convert_checkpoint_world_size.slurm` | HF → FSDP shard conversion |
| `prune_checkpoints.py` | Keep last N + best validation step |
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
