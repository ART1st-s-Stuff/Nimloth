# 输出与可视化规范（2026-04-21）

- 重要性: high
- L: 1
- D: 1

## 目录规范

- 所有训练、数据收集、校准等操作统一写入 `outputs/<phase>/<task>/<datetime>/`。
- 当前任务名约定：
  - phase1 数据收集：`wm_data_collection`
  - phase2 训练：`wm_training`
  - phase2 校准：`wm_calibration`
- Hydra 默认输出目录统一放在 `outputs/hydra/...`。

## 脚本规范

- 脚本按阶段分层到 `scripts/<phase>/<task>.sh`。
- 兼容入口 `scripts/*.sh` 保留为转发壳，后续可逐步下线。

## 可视化规范

- W&B 跟踪逻辑迁移至 `src/visualize/wandb_tracker.py`。
- 进度服务入口为 `src/visualize/progress_server.py`，在同一服务器内整合：
  - 数据集 progress（manifest 统计与样本预览）
  - 训练 progress（`train_metrics.json` + `wm.pt` 状态）
  - 校准与 rollout 状态占位
- 统一启动脚本：`scripts/start_visualization_server.sh`。

