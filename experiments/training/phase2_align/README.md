# Phase 2 — WM predictor + Value head (SFT2)

Migrate from `experiments/navigation_baseline/`:
- `train_sft2_qwen25vl.py` → `train.py`
- `train_sft2_*.slurm`, `submit_sft2_*.sh`, smoke slurm
- `pretrain_lewm_navigation.py` → `pretrain_predictor.py` (optional init only)
- `upload_sft2_wandb_from_csv.py`, `pick_sft1_ckpt_for_sft2.py`

Library: `src/nimloth/training/phase2_align/` (from legacy `src/nimloth/sft2/`)
Configs: `configs/training/phase2_align/latent_wm_value.yaml`

Human spec: `ai_tasks/sft2_exp.md`
Plan: `ai_tasks/sft2_phase2_plan.md`
