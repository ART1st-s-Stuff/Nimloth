# E0002 — 不要在训练未结束前把当前 step 误当成总 step

## 错误
曾在训练尚未结束时，看到日志中的当前 step（如 `1520` / `1539`），就过早声称本轮 1 epoch 总 step “大约 1540”。

## 问题
训练未结束时，当前 step 只是当前进度，不是总步数。

## 正确做法
如果用户问“到底有多少 step”，优先用以下方法之一：
1. 直接根据数据规模和训练配置计算：
   - `num_train_samples`
   - `world`
   - `batch_size`
   - `grad_accum`
2. 或等待训练结束后，读取：
   - `train_step_log.csv` 最后一行 `global_step`
   - `training_state.pt`
   - `epoch_* / final / sft2_done.flag`

## 本项目这次的正确结果
在双卡-per-shard配置下：
- `num_train_samples = 54702`
- `world = 4`
- `batch_size_per_rank = 2`
- `grad_accum = 4`

因此：
- `micro_batches_per_epoch_per_rank = ceil(54702 / (2 * 4)) = 6838`
- `optimizer_steps_per_epoch = ceil(6838 / 4) = 1710`

所以这轮 1 epoch 的总 step 是：
```text
1710
```
