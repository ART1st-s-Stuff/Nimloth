# E0064：训练续跑不能用日志行数代替已提交状态

## 已发生的错误

RL outer runner 用 CSV 最后一行推断已完成 iteration，并在启动下一轮前移动 `train/latest`。进程在移动后或当前轮提交前中断时，重启既找不到 `latest`，也不会复用已存在的 policy snapshot；若 CSV 已提前写入，还会把未提交更新误认为完成。

## 原因

把报告日志当成事务提交记录，并假设 checkpoint 搬移、CSV、fresh consumption 和 controller 生命周期是一个原子操作。

## 正确做法

以连续的 committed fresh-consumption 记录及其完整 checkpoint 为恢复依据。复用已搬移的 policy snapshot；把当前未提交 checkpoint、rollout/reference/Ray 输出和多出的 CSV 行归档后，从上一个 committed checkpoint 重试。

## 证据

- `src/nimloth/training/rl/continuation.py`
- `experiments/training/rl/run_vllm_online_ppo_full.sh`
- `tests/training/rl/test_full_runner.py`

