# Phase 1 — format SFT (SFT1)

Migrate from `experiments/navigation_baseline/`:
- `train_sft1_qwen25vl.py` → `train.py`
- `train_sft1_*.slurm`, `submit_sft1_*.sh`
- `convert_sft1_rollouts_to_nimloth.py` → `convert_rollouts.py`
- eval / watcher scripts

Library: `src/nimloth/training/phase1_sft/`
Configs: `configs/training/phase1_sft/`
