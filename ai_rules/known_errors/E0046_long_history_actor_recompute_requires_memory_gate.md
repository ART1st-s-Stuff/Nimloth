# E0046: 长历史 actor recompute 必须先过显存 gate

## 已发生错误

ID24在iteration1收集8条/157个最长20-turn的多模态transition并完成PPO-old/reference replay后，使用transition microbatch2执行带梯度的actor recompute。79.19GiB GPU几乎全部占满，所有trainer rank在Qwen LoRA MLP forward中OOM；global/optimizer step仍为0。此前ID22只验证了最多2-turn的microbatch2，不能证明20-turn历史安全。

## 正确做法

- pilot前必须用接近20-turn、完整图片历史的transition直接执行actor recompute与backward显存测试。
- rollout `batch_size`是transition microbatch；长历史默认先测试1，不能沿用短smoke的2。
- 显存gate必须覆盖PPO-new有梯度forward，而不只覆盖无梯度generation、PPO-old或reference replay。
- OOM发生在首个optimizer step前且已有真实rollout artifact时，该identity仍须terminal，禁止resume/reuse。
