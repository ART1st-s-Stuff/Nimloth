# E0087: PlannerPolicyHead PPO 必须分别验证 rollout state 与重算 state

## 错误结论

只在完整 prefix 重算得到的 current state 上构造相同 logits 和非零输入梯度，就断言
PlannerPolicyHead 测试同时满足 behavior freshness 与 PPO 梯度门禁。

## 原因

`RLAlgorithm.planner_old_policy_statistics()` 使用持久化的
`ExecutedTransition.rollout_decision_state()` 重算并校验 behavior log-probs；
`actor_transition_step()` 则使用完整 prefix 重算的 current state 计算新策略概率。
测试 fixture 中两者不一定相同。只验证后者会遗漏 old-policy freshness 失败。

## 正确做法

- 在 rollout decision state 上验证 PolicyHead logits 与保存的 behavior log-probs 一致。
- 在完整 prefix 重算 state 上另行验证 PPO ratio 位于预期裁剪区间且输入梯度非零。
- 独立数学 probe 必须使用测试中的两种真实 state，不能用一个假定 state 代替。

对应回归测试：`tests/training/rl/test_episode_training.py::test_planner_policy_ppo_reaches_full_prefix_qwen_and_policy_head`。
