# E0010 — epoch 中途 checkpoint 必须保存并恢复 DataLoader 位置

## 错误

SFT2 的 `latest` / `step_*` 在 epoch 中途保存时只记录 `epoch`。旧 resume 一律从 `epoch + 1` 开始，导致当前 epoch 尚未处理的 micro-batches 被静默跳过。

## 正确做法

- 中途 checkpoint 记录 `epoch_complete=false` 和 `micro_step_in_epoch`。
- 只在 optimizer boundary 保存，避免丢失尚未 step 的累计梯度。
- Resume 使用同一 epoch/seed/sampler，并跳过已经消费的 micro-batches；检查位置不超过当前 DataLoader 长度且符合 grad-accum boundary。
- epoch/best/final checkpoint 记录 `epoch_complete=true`，从下一 epoch 恢复。
- 还必须复现后续 stochastic micro-step 的 RNG；协议见 E0011。
- 旧 checkpoint 没有该字段时，为兼容历史行为按 epoch-complete 处理；不能把旧中途 checkpoint 宣称为精确 resume。
