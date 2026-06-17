# Training configs (by phase)

| Path | Phase | Loaded by |
|------|-------|-----------|
| `phase0_vagen/defaults.yaml` | VAGEN navigation defaults | planned |
| `phase1_sft/qwen25vl_lora.yaml` | SFT1 LoRA | planned |
| `sft2/latent_wm_value.yaml` | **SFT2** WM + Value | `experiments/training/sft2/train.py --config` |

SFT2 defaults per `ai_tasks/sft2_exp.md`: LLM freeze, vision full + EMA, include failed rollouts.

Legacy `phase2_align/` → see `phase2_align/README.md` (merged into `sft2/`).
