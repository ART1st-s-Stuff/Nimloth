# LeWM Cube latent → Nimloth decoder reproduction

目标：验证 Nimloth 的 `WMImageDecoder` 在适配 LeWM 原文 latent 规模后，能否从官方 LeWM OGBench-Cube encoder 的 192-dim `[CLS]` 表示重建原图。

这不是 LeWM paper Appendix D cross-attention decoder 的复现；decoder 使用 Nimloth 当前实现：state vector 线性展开到 patch tokens，再经过 self-attention 和 RGB patch head。

## 数据与 checkpoint

- 官方 LeWM checkpoint: HF model `quentinll/lewm-cube`
- 官方 Cube dataset: HF dataset `quentinll/lewm-cube`
- LeWM upstream code: `external/le-wm`

`cube_single_expert.tar.zst` 约 46GB。解压后默认放到 `$STABLEWM_HOME/ogbench/cube_single_expert.h5`。

## Full train with W&B

Uses the full train split (`--train-limit 0`), evaluates every 1000 optimizer steps on a bounded validation subset by default (`VAL_LIMIT=4096` sequences = 16,384 images), and uploads metrics/previews to W&B.

```bash
sbatch experiments/lewm_decoder_repro/run_full_wandb.slurm
```

## Smoke run

```bash
export PYTHONPATH=src:external/le-wm:$PYTHONPATH
export STABLEWM_HOME=/project/peilab/atst/nimloth/outputs/experiments/lewm_repro/stablewm_home

python experiments/lewm_decoder_repro/train_nimloth_decoder_on_lewm_cube.py \
  --output-dir /project/peilab/atst/nimloth/outputs/experiments/lewm_repro/2026-07-02/nimloth_decoder_cube_smoke \
  --stablewm-home "$STABLEWM_HOME" \
  --download-dataset \
  --epochs 5 \
  --train-limit 2048 \
  --val-limit 256 \
  --batch-size 16 \
  --decoder-hidden-dim 192 \
  --decoder-depth 4 \
  --decoder-heads 3
```

## 输出

- `metadata.json`: 命令参数和 git commit
- `train_log.csv`: epoch-level train/val loss
- `previews/*.png`: target/reconstruction 对比图
- `best/`, `final/`, `epoch_*`: Nimloth decoder checkpoints

## 实验边界

- LeWM encoder 冻结。
- 训练模块只有 `WMImageDecoder`。
- Decoder 输入是官方 LeWM projected CLS embedding，维度 192。
- Decoder 输出是 224×224 RGB，patch size 16。
