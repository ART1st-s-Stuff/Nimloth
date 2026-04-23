将训练链路统一为接口化实现，作为后续 WM/PM/VLM 的公共约束。

已落地接口（`src/core`）：
- `StorageProvider`
  - 统一 run 目录解析与状态管理（running/completed/failed）。
- `DataProvider`（继承 `StorageProvider`）
  - 统一 `train()/val()/test()` 数据访问契约。
- `Model`
  - 统一 `train_step` / `eval_step`。
  - 提供默认 `train(DataProvider)` / `test(DataProvider)` 循环。
- `ModelProvider`（继承 `StorageProvider`）
  - 统一 `save`、`save_checkpoint`、`load_checkpoint`。
  - 支持 checkpoint 与最终导出模型分离（如 EMA 与主模型）。

模块适配：
- `WMModelAdapter`：已在 `train_wm.py` 接入，封装 batch 训练输入。
- `PMModelAdapter` / `VLMModelAdapter`：提供同构占位接口，后续训练入口可直接扩展。

训练命令约定（Phase2 WM）：
- 默认行为：如果 latest run 未完成，则自动续训；若 latest run 已完成，则新建训练。
- 强制新训：`scripts/phase2/wm_training.sh --new ...`
  - `--new` 会透传为 `pipeline.train.operation.force_new_run=true`。

后续规范：
- 新增训练脚本时，必须优先复用 `src/core` 的接口抽象，不再直接散落实现存储和 checkpoint 逻辑。