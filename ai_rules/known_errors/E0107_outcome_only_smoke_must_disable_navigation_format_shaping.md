# E0107: Outcome-only smoke must disable Navigation format shaping

## 已发生的错误

ID168完成真实8-trajectory rollout后，batch compiler在optimizer前拒绝数据：
`NavigationEnvConfig.per_turn_format_reward`默认是`0.01`，而integration dataset config
未覆盖它，导致intermediate reward非零。

## 正确做法

纯结果joint training的实际Navigation config必须显式设置：

- `per_turn_format_reward: 0.0`
- `format_reward: 0.0`
- `success_reward: 1.0`

compiler应继续拒绝任何nonzero intermediate reward，禁止静默丢弃shaping reward。
启动前测试必须从实际dataset YAML构造/核验这些字段，不能只检查算法config。

## Evidence

- `external/VAGEN/vagen/envs/navigation/navigation_env.py`：默认per-turn reward。
- `external/VAGEN/vagen/joint_policy/training_contract.py`：outcome-only fail-closed。
- 服务器ID168 `failure_analysis.md`：真实rollout/compiler边界。
