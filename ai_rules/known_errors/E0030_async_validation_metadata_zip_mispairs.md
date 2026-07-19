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

## 正式修复

rollout manager 在 reset 时保存 `env_id -> input index` 映射；trainer 通过`attach_validation_input_metadata()`按稳定 identity 合并结果，并对缺失、重复、越界和不完整映射fail fast，不再按返回位置`zip`。

- 255路径VAGEN：`192c35a fix(validation): preserve rollout input identity`
- 旧504路径VAGEN：`ef851af fix(validation): preserve rollout input identity`
- 测试覆盖乱序返回、environment reuse、映射错误、local/service manager contract和trainer禁止旧zip；255 lineage为7 passed，旧504 lineage连同source eval contract为8 passed。

需要精确 `(data_source, env_seed)` 证据时，仍必须用修复后的代码重跑；旧dump不能被修复代码追溯恢复seed标签。
