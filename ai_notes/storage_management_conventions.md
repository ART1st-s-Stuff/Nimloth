importance: high
L: 1
D: 2

# 存储结构与清理约定

- 数据集目录结构：`datasets/<dataset-name>/<datetime>/...`，并在 `<dataset-name>` 层维护 `metadata.json`。
- 模型目录结构：`models/<wm|pm|vlm>/<model-config-name>/<datetime>/...`，并在 `<model-config-name>` 层维护 `metadata.json`。
- `metadata.json` 至少包含：`latest`、`runs`、`updated_at`。
- 存储清理统一使用 `storage` 命令：`trim` / `discard` / `reset`。
- `trim` 与 `discard` 必须保证每个分组至少保留一个运行目录。
- VLM LoRA 运行目录可使用 `base_weight_ref.json` 指向基础权重；清理时若基础权重仍被引用则禁止删除。
