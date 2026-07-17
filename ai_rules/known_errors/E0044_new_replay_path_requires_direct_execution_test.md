# E0044: 新 replay helper 必须有直接执行测试

## 已发生错误

ID21成功收集2条schema-v3 trajectories后，在`attach_policy_and_reference_token_log_probs()`首次执行时触发`NameError: validate_rollout_trajectory is not defined`，因为trainer调用了rollout validator却未导入。此前全套测试未直接执行该helper，因此漏检。

## 正确做法

- 新增训练阶段helper时，测试必须直接执行或至少解析其全部全局依赖，不能只测下游数学函数。
- helper引用跨模块validator/utility时必须显式导入，并用测试断言绑定到预期实现。
- 真实rollout smoke通过后，进入replay/backward前的错误仍属于0-step terminal failure；不得复用identity或声称PPO/KL已验证。
