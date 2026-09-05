# E0001 — 不要把静态 success 比例当成模型评估结果

## 错误
曾错误地把 SFT2 训练结束日志中的：
- `success_rate`
- `val_rollout_success_rate`

解释为模型性能指标，并试图用它们比较 `Vision Full` 和 `Vision LoRA`。

## 实际定义

历史上的 `success_rate` 与 `val_rollout_success_rate` 都只读取现有 JSONL
里的 `success` 标签，没有执行模型。当前静态统计入口位于：

- `src/nimloth/wm/statistics.py`
- `experiments/training/sft2/diagnosis/report_dataset_success.py`

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

SFT2 生产训练不再记录这些静态 success 字段，也不允许用它们选择
checkpoint；当前 checkpoint 指标为 `val_wm_mse`。
