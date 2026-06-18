# Training configs (by phase)

| Path | Phase | Loaded by |
|------|-------|-----------|
| `phase0_vagen/defaults.yaml` | VAGEN navigation defaults | planned |
| `phase1_sft/qwen25vl_lora.yaml` | SFT1 LoRA | planned |
| `sft2/latent_wm_value.yaml` | **SFT2** WM + Value | `nimloth.training.sft2.trainer`（经 `experiments/training/sft2/train.py --config`） |

SFT2 defaults per `ai_tasks/sft2_exp.md`: LLM freeze, vision full + EMA, include failed rollouts.
