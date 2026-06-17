# SFT2 training (Phase 2)

Canonical location for SFT2 per `ai_tasks/sft2_exp.md`.

| File | Purpose |
|------|---------|
| `train.py` | Main training entry |
| `train_vagen79_default.slurm` | 8-GPU Slurm job (reads yaml config) |
| `submit_default_8gpu.sh` | Default: LLM freeze + vision full |
| `submit_llmvis_lora_8gpu.sh` | LLM+Vision LoRA variant |
| `submit_llmvis_lora_preempt6g_dgx11.sh` | Preempt 6-GPU variant |
| `eval_val_rollout_success.py` | Offline val trajectory success rate |
| `upload_sft2_wandb_from_csv.py` | Retroactive wandb upload from CSV |
| `pick_sft1_ckpt_for_sft2.py` | SFT1 init checkpoint picker |
| `resolve_sft1_init_for_sft2.sh` | Merge SFT1 LoRA → hf_merged for SFT2 |

Config: `configs/training/sft2/latent_wm_value.yaml`

Library: `src/nimloth/training/sft2/`

Legacy wrappers remain under `experiments/navigation_baseline/` (see `SFT2_DEPRECATED.md`).
