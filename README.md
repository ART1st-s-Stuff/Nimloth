# Flower

面向世界模型与多阶段智能体训练的实验仓库。

当前主线聚焦：

- AI2THOR 轨迹采集
- Phase2 世界模型训练（含 LeWM / SIGReg）
- Qwen visual encoder 联合训练（支持 full / LoRA / 可选 KL / 可选 EMA）

## 快速开始

### 1) 安装环境

```bash
uv sync
cp .env.example .env
```

建议在 `.env` 配置：

- `WANDB_MODE=online|offline`
- `WANDB_API_KEY`（在线模式）
- `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_RUN_PREFIX`

### 2) 数据采集（Phase1）

```bash
uv run ./scripts/phase1/wm_data_collection.sh
```

### 3) 世界模型训练（Phase2）

```bash
uv run ./scripts/phase2/wm_training.sh
```

### 4) 阈值校准

```bash
uv run ./scripts/phase2/wm_calibration.sh
```

## Qwen 联合训练（重点）

训练入口：

- `src/train/train_wm_joint.py`

两阶段开关：

- `pipeline.train.stage=stage1_wm_vision`：训练 WM + Qwen visual encoder
- `pipeline.train.stage=stage2_value_head`：占位，不执行训练

示例（小规模冒烟）：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.train_wm_joint \
  wm=lewm_qwen_llm_joint \
  pipeline.train.stage=stage1_wm_vision \
  pipeline.train.epochs=1 \
  pipeline.train.batch_size=2 \
  pipeline.train.max_samples=128 \
  pipeline.train.temporal_stride=1 \
  pipeline.train.device=cuda
```

## Qwen visual encoder 训练策略配置

配置路径：`configs/pipeline/train/default.yaml` 下的 `qwen_encoder`。

### 训练模式

- `qwen_encoder.train_mode=full`：visual 全参数训练
- `qwen_encoder.train_mode=lora`：仅训练 visual LoRA 参数

LoRA 参数：

- `qwen_encoder.lora.r`
- `qwen_encoder.lora.alpha`
- `qwen_encoder.lora.dropout`
- `qwen_encoder.lora.target_modules`

### KL 蒸馏（可选）

- `qwen_encoder.kl.enabled=true|false`
- `qwen_encoder.kl.weight`
- `qwen_encoder.kl.temperature`
- `qwen_encoder.kl.max_images_per_batch`

说明：当前 KL 为 vision token 级 teacher-student KL（teacher 为冻结原始 Qwen visual encoder）。

### Qwen visual EMA（可选）

- `qwen_encoder.ema.enabled=true|false`
- `qwen_encoder.ema.decay`
- `qwen_encoder.ema.use_ema_for_eval`

## 训练后可视化

`train_wm_joint.py` 训练后可在 test split 生成 rollout 图并上传 wandb。

可视化配置：

- `pipeline.train.post_visualization_enabled`
- `pipeline.train.post_visualization_rollouts`
- `pipeline.train.post_visualization_steps`
- `pipeline.train.post_visualization_include_sigreg_encoder_space`

其中最后一项用于额外输出 LeWM 内部 encoder（SIGReg 前）空间轨迹图。

## 关键输出目录

- 采集数据：`datasets/`
- 训练模型：`models/wm/`
- 可视化产物：`outputs/dev/visualization/joint_rollout/`
- Hydra 运行产物：`outputs/hydra/`
- W&B 本地缓存：`wandb/`

## 常用脚本

- `scripts/phase1/wm_data_collection.sh`
- `scripts/phase2/wm_training.sh`
- `scripts/phase2/wm_calibration.sh`
- `scripts/storage.sh`（清理 models/datasets/outputs）

## 备注

- 默认训练设备为 `cuda`，如需 CPU 请显式覆盖 `pipeline.train.device=cpu`。
- AI2THOR 不可用时会直接报错，不自动回退。

