# Training configs (by phase)

| Path | Phase | Loaded by |
|------|-------|-----------|
| `baseline/train.yaml`, `baseline/val.yaml` | **VAGEN navigation baseline** | `experiments/training/baseline/*.slurm` |
| `baseline/defaults.yaml` | Baseline hyperparam reference | docs / submit env |
| `phase0_vagen/defaults.yaml` | VAGEN navigation defaults | planned |
| `sft1/qwen25vl_lora.yaml` | **SFT1** LoRA | `experiments/training/sft1/train_8gpu.slurm` |
| `sft1/defaults.yaml` | SFT1 hyperparam reference | docs |
| `sft2/latent_wm_value.yaml` | **SFT2** WM + Value | `nimloth.training.sft2.trainer`（经 `experiments/training/sft2/train.py --config`） |
| `sft2/latent_wm_value_k8_dinov2.yaml` | SFT2 k=8 + 当前 RGB 的 DINOv2-L/14 CLS 直接对齐 | 同上；DINO teacher 冻结且不保存进 checkpoint |
| `sft2/latent_wm_value_k8_dinov2_cached.yaml` | 同一目标的预计算 float32 CLS 版本 | 必须显式提供已校验的 `dino_cache_dir`，禁止fallback到在线teacher |
| `sft2/latent_wm_value_k8_dinov3.yaml` | SFT2 k=8 + 当前 RGB 的 DINOv3-L/16 CLS 直接对齐 | 需要支持 DINOv3 的 Transformers 与 gated 权重 |
| `reconstruction/rcdm_sft2.yaml` | SFT2 latent → RCDM visualization reference | `python -m nimloth.training.reconstruction.rcdm_sft2` |

SFT2 defaults per `ai_tasks/sft2_exp.md`: LLM freeze, vision full + EMA, include failed rollouts.
