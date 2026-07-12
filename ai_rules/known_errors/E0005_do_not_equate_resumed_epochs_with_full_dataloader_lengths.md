# E0005 — 不要未经确认把 resume 后的 epoch 直接换算成固定 step 数

## 错误
根据日志中的 `Size of train dataloader: 18`，直接把用户要求的“继续30个 epoch”解释成 `30 × 18 = 540` 个 PPO updates，并设置绝对 target step。

## 问题
18 确实是当前 dataloader 每个完整 epoch 的 batch 数，但 checkpoint 恢复了 stateful dataloader state。恢复后的第一个 outer-loop epoch 可能只消费当前 epoch 的剩余部分，因此“设置 `trainer.total_epochs=30`”与“强制执行540次更新”不是同一个语义。

## 正确做法
先向用户明确说明18的证据，再确认用户需要：
1. trainer 原生语义：从 checkpoint 恢复后设置 `trainer.total_epochs=30`；或
2. 固定更新量语义：无论 sampler 当前位置如何都执行540次 PPO updates。

在得到确认前，不启动对应长训练任务。
