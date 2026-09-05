# Training configs (by phase)

| Path | Phase | Loaded by |
|------|-------|-----------|
| `baseline/train.yaml`, `baseline/val.yaml` | **VAGEN navigation baseline** | `experiments/training/baseline/*.slurm` |
| `baseline/defaults.yaml` | Baseline hyperparam reference | docs / submit env |
| `phase0_vagen/defaults.yaml` | VAGEN navigation defaults | planned |
| `sft1/qwen25vl_lora.yaml` | **SFT1** LoRA | `experiments/training/sft1/train_8gpu.slurm` |
| `sft1/defaults.yaml` | SFT1 hyperparam reference | docs |
| `sft2/latent_wm_value.yaml` | **SFT2** WM + Value | `nimloth.training.sft2.trainer`（经 `experiments/training/sft2/train.py --config`） |
| `reconstruction/rcdm_sft2.yaml` | SFT2 latent → RCDM visualization reference | `python -m nimloth.training.reconstruction.rcdm_sft2` |

The executable SFT2 defaults are owned by `sft2/latent_wm_value.yaml`; train/freeze and objective interpretation follows [the World Model and Training contract](../../.trellis/spec/domains/world-model-and-training.md).
