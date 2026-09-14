# 预处理缓存

本包负责 Qwen transition 编码缓存的通用存储和构建流程。它消费 rollout
transition 与 Qwen 编码适配器，不包含 SFT2 优化语义。

Transition expansion v4 保存可选 `action_success`；加载时核对 source transition
的 outcome 和 action-value target。新增监督字段需要新 cache fingerprint。
