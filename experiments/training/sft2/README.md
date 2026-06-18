# SFT2 training (Phase 2)

Canonical location for SFT2 per `ai_tasks/sft2_exp.md`.

| File | Purpose |
|------|---------|
| `train.py` | Thin experiment entry → `nimloth.training.sft2.trainer` |
| `train_vagen79_default.slurm` | 8-GPU Slurm job (reads yaml config) |
| `submit_default_8gpu.sh` | Default: LLM freeze + vision full |
| `submit_llmvis_lora_8gpu.sh` | LLM+Vision LoRA variant |
| `submit_llmvis_lora_preempt6g_dgx11.sh` | Preempt 6-GPU variant |
| `eval_val_rollout_success.py` | Offline val trajectory success rate |
| `upload_sft2_wandb_from_csv.py` | Retroactive wandb upload from CSV |
| `pick_sft1_ckpt_for_sft2.py` | SFT1 init checkpoint picker |
| `resolve_sft1_init_for_sft2.sh` | Merge SFT1 LoRA → hf_merged for SFT2 |

Config: `configs/training/sft2/latent_wm_value.yaml`

Library: `src/nimloth/training/sft2/`；WM 在 `wm/`；Qwen 调参在 `backbone/`；离线 eval 在 `eval/`。

SFT1 checkpoints and rollout records stay under `experiments/navigation_baseline/runs/` (legacy); SFT1 scripts are in `experiments/training/sft1/`.

**Outputs:** train checkpoints/logs → `outputs/experiments/training/sft2/<YYYY-MM-DD>/<experiment_name>/`; Slurm logs → `outputs/experiments/training/sft2/slurm/`. Legacy SFT2 runs under `experiments/navigation_baseline/runs/` are not overwritten; resume via `TRAIN_OUT_OVERRIDE`.
