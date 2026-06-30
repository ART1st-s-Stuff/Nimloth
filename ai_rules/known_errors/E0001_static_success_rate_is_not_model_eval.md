# E0001 — 不要把静态 success 比例当成模型评估结果

## 错误
曾错误地把 SFT2 训练结束日志中的：
- `success_rate`
- `val_rollout_success_rate`

解释为模型性能指标，并试图用它们比较 `Vision Full` 和 `Vision LoRA`。

## 实际定义
### `success_rate`
来自：
- `src/nimloth/training/sft2/evaluate.py`
- `src/nimloth/training/sft2/metrics.py`

其定义是当前验证 batch 中：
```python
sum(1.0 for item in items if item.get("success")) / len(items)
```
即 **step/item 级 success 标签比例**。

### `val_rollout_success_rate`
来自：
- `src/nimloth/eval/rollout.py`

其定义是整个 `val_jsonl` 中：
```python
sum(1 for record in records if bool(record.get("success", False))) / len(records)
```
即 **trajectory/record 级 success 标签比例**。

## 正确认识
- 这两个字段都不是模型 rollout 结果；
- 它们只是验证数据集静态标签统计；
- 不能用来比较不同训练设置的模型效果。

## 正确做法
比较模型效果时，应优先看：
- 真正的 rollout / greedy eval 结果；
- 或训练内有模型区分度的数值，如 `wm_mse`、`value_*`。
