# Flower (Phase 1 + Phase 2 最小闭环)

本版本实现以下流程：

1. 采集轨迹（支持 `mock` 与 `ai2thor` 后端）并生成 `manifest.jsonl`
2. 训练 CFM 世界模型（最小实现）
3. 计算散度阈值 `theta_div`（95% 分位）

## 环境依赖（uv）

- Python 3.10+
- 使用 `uv` 管理虚拟环境与依赖

初始化环境：

```bash
uv sync
```

配置环境变量（建议）：

```bash
cp .env.example .env
```

其中：

- `WANDB_MODE=online` 默认在线上传；若未提供 `WANDB_API_KEY`，程序会明确告警并切到 offline。
- 可在 `.env` 中配置 `WANDB_PROJECT/WANDB_ENTITY/WANDB_RUN_PREFIX`。

## 运行步骤

在项目根目录执行：

```bash
uv run ./scripts/phase1/wm_data_collection.sh
uv run ./scripts/phase2/wm_training.sh
uv run ./scripts/phase2/wm_calibration.sh
```

组件可替换运行（示例）：

```bash
# 选择替代数据集配置
uv run ./scripts/phase2/wm_training.sh dataset=wm_default

# 显式指定 WM/PM/VLM 组件
uv run ./scripts/phase2/wm_training.sh wm=cfm pm=rule_based vlm=qwen_vl
```

所有入口都会在命令行显示 Rich 进度和关键指标，并自动记录 W&B 实验数据。

默认已使用 AI2THOR 无头后端（`CloudRendering`）。如需切换为 mock：

```bash
uv run python -m src.train.collect_data pipeline.collect.env.backend=mock
```

默认采集规模：

- scenes: `FloorPlan1-10` 与 `FloorPlan201-210`
- 每个 scene: `num_episodes_per_scene=50`
- 每个 episode: `max_steps_per_episode=50`
- 并行采集: `num_workers=4`（按 scene 多进程并行）

采集支持断点续跑（默认 `pipeline.collect.collect.operation.resume=true`），再次执行会复用最近一次 `wm_data_collection` 目录并从已有样本后继续。

如需清空所有 phase1 采集结果后重跑：

```bash
uv run ./scripts/phase1/wm_data_collection.sh clean
```

## 关键输出

- 数据收集：`datasets/phase1/wm_data_collection/<datetime>/manifest.jsonl`
- 模型训练：`models/phase2/wm_training/<datetime>/wm.pt`
- 训练指标：`models/phase2/wm_training/<datetime>/train_metrics.json`
- 阈值校准：`models/phase2/wm_calibration/<datetime>/theta_div.json`
- Hydra 默认：`outputs/hydra/...`

## 配置结构（Hydra 组件组）

- `configs/dataset/`: 训练/校准数据集组件（如 `wm_default`）
- `configs/wm/`: 世界模型组件（如 `cfm`）
- `configs/pm/`: 规划器组件（如 `none`、`rule_based`）
- `configs/vlm/`: 视觉语言模型组件（如 `none`、`qwen_vl`）
- `configs/pipeline/`: 任务流程参数（`collect/train/calib/rollout`）

启动进度可视化服务：

```bash
uv run ./scripts/start_visualization_server.sh
```

可视化服务器是单服务整合页，包含：

- 数据集进度：采集样本统计与样本预览
- 训练进度：`wm_training` 运行列表、`last_loss`、checkpoint 状态
- 校准与 Rollout：当前状态探测与后续扩展占位

## 说明

- 采集层采用 adapter 设计：`src/data/mock_env.py` 与 `src/data/ai2thor_env.py` 可插拔。
- 当 `backend=ai2thor` 但运行环境不可用时，程序会直接报错并终止，不会自动回退。
- 未来 VLM rollout 默认保存并上传观测图片、Prompt、CoT（见 `configs/pipeline/rollout/default.yaml`）。

