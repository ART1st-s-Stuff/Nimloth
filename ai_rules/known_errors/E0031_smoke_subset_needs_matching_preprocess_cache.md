# E0031 — Smoke subset needs a matching preprocess cache

## 错误

在 SFT2 smoke 中设置 `--max-train-records 2 --max-val-records 2`，同时把
full-data compact cache 作为 `--require-prebuilt-cache` 输入，错误地认为可直接
复用其中前两条记录。

## 实际结果

Job `476051` 在训练前被严格 gate 拒绝：cache manifest 的 transition count 与
smoke subset 期望 count 不同。没有 optimizer step、checkpoint 或数据污染。

## 原因

SFT2 cache gate 同时验证 fingerprint 和展开后的样本数。即使 processor、模型、
query mode 与 JSONL 相同，`max_records` 改变了期望 transition count，full cache
也不能冒充 subset cache。

## 正确做法

- GPU smoke 使用 `--require-prebuilt-cache` 时，先在 CPU partition 按相同
  `max_train_records/max_val_records` 建立独立 cache；或
- 不设置 subset limit，严格按 full cache 的 manifest 运行。

不得关闭 count gate，也不得在 GPU allocation 内临时重建 cache。
