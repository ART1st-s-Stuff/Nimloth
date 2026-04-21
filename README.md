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

## 运行步骤

在项目根目录执行：

```bash
uv run ./scripts/collect_data.sh
uv run ./scripts/train_wm.sh
uv run ./scripts/calibrate_wm.sh
```

切换 AI2THOR 后端：

```bash
uv run python -m src.train.collect_data data.env.backend=ai2thor
```

## 关键输出

- 数据清单：`datasets/phase1/raw/manifest.jsonl`
- 模型参数：`models/phase2/wm/wm.pt`
- 训练指标：`models/phase2/wm/train_metrics.json`
- 阈值结果：`models/phase2/wm/theta_div.json`

## 说明

- 采集层采用 adapter 设计：`src/data/mock_env.py` 与 `src/data/ai2thor_env.py` 可插拔。
- 当 `backend=ai2thor` 但运行环境缺少可用构建或图形能力时，可通过 `data.env.fallback_to_mock_on_error=true` 自动回退到 mock，避免阻塞流程。

