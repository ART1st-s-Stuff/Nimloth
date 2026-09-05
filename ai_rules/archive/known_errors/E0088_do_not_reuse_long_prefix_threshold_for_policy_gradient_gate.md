# E0088: Policy 梯度门禁不得沿用无关的长前缀阈值

## 现象

PlannerPolicyHead 的新鲜 rollout 已完整生成，但梯度门禁在 forward 之前失败：真实最长
final prefix 为 4,136 tokens，而启动器沿用了旧显存诊断的 14,000-token 硬阈值。

## 根因

旧阈值用于筛选特定长上下文显存压力样本，不是证明 PlannerPolicyHead PPO 梯度回传的
必要条件。把它复制到真实新鲜 batch 的策略梯度 gate，会在数据本身不产生 14k prefix
时错误拒绝有效输入。

## 正确做法

- 策略梯度 gate 从行为 checkpoint 匹配的新鲜 batch 中选择真实最长 final prefix，并只
  要求 prefix 非空。
- 长上下文显存压力测试若仍需要，必须作为单独目标和单独数据合同，不能冒充 PPO 梯度
  正确性测试。
- 失败输出不得恢复。只有未消费且通过完整 fingerprint 校验的 rollout 可以作为新 ID 的
  不可变输入；新的 gate 证据写入新的输出目录。

对应失败：ID144。对应启动器：
`experiments/training/rl/run_planner_policy_gpu_gate_4x4_on_hold.sh`。
