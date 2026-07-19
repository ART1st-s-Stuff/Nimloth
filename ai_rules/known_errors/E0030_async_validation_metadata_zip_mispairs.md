# E0030：异步 validation 结果不能按列表位置与输入 metadata 配对

## 已确认现象

VAGEN validation 在 `ray_trainer.py::_validate()` 中执行：

```python
micro_validation_rst = self.test_rollout_manager.recording_to_log()
for item, env_config, uid in zip(micro_validation_rst, env_configs, uids):
    item.update(...)
```

`recording_to_log()` 按 recorder/environment 的运行时插入顺序返回结果；这个顺序会受异步环境响应和 environment reuse 影响，并不保证等于 `env_configs` 输入顺序。按位置 `zip` 会把一个任务的 `data_source`、`env_seed`、`eval_set`、`uid` 写到另一个任务的 trajectory/metrics 上。

2026-07-20 resolution A/B 的原始 dump 已确认：

- A 有 16/120 行的 metadata `eval_set` 与实际 runtime `config_id` 冲突；
- B 有 14/120 行冲突；
- 同一 eval_set 内还可能发生无法靠 `config_id` 发现的 seed 置换。

## 禁止做法

- 禁止直接用受影响 JSONL 的 `(data_source, env_seed)` 做 paired comparison。
- 禁止把 W&B 按 runtime `config_id` 聚合结果与 JSONL 按错误 metadata 聚合结果混为一谈。
- 禁止在没有一致性 gate 的情况下让比较工具把缺失/不一致 metadata 静默当作失败。

## 当前处理

- `compare_rollout_resolution_probe.py` 现在读取 `metrics.success`，并在 `config_id` 与 `eval_set` 可见冲突时 fail fast。
- 既有 A/B 的总成功数和按 runtime `config_id` 的聚合仍可信。
- 既有 A/B 使用 control-batch、runtime config、instruction 和初始帧 RMSE 做了非破坏性 task identity 恢复；可用于诊断性 paired test，但无法恢复精确 seed 标签。

## 正式修复要求

rollout manager 在 reset 时必须保存 `env_id -> input index/uid/metadata` 映射；trainer 必须按稳定 identity 合并结果，不能按返回位置 `zip`。修复后需增加跨 config、异步乱序和 environment reuse 的测试。需要精确 `(data_source, env_seed)` 证据时，必须用修复后的代码重跑。
