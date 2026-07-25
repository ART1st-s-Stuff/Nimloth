# E0053：reasoning必须屏蔽多模态与聊天控制token

## 已确认错误

turn-response logits processor只屏蔽Nimloth latent/action protocol token时，Qwen仍可能在CoT
中采样`<|image_pad|>`等tokenizer special token。该CoT进入后续历史后会增加没有对应真实图片
的placeholder：前端processor可能直接越界，若进入EngineCore则会触发embedding
`masked_scatter`数量断言。

## 正确做法

- reasoning阶段屏蔽tokenizer全部special token以及Nimloth protocol token；`</think>`普通
  token序列保持可生成。
- behavior输出落盘前验证reasoning token IDs不含被禁控制token。
- 每次vLLM请求前验证文本中的Qwen image placeholder数量严格等于绑定图片数量。
- 禁止丢弃失败trajectory后继续用较小、选择性batch训练；指定数量不完整时整体rollout失败。
