# E0056：禁止用重编码文本替代behavior token trace

## 已确认错误

ID103的四个length-truncated CoT保存了真实vLLM token IDs，且这些IDs解码后与环境使用的
assistant response完全一致。但截断落在byte-level token序列中间时，tokenizer decode不是
可逆映射；把文本重新encode会把512个behavior token变成1362--1418个token。reference
replay错误要求重编码结果等于原IDs，因此拒绝了有效trajectory。

## 正确做法

- behavior生成的token IDs是PPO/reference replay的权威continuation，必须原样追加。
- 文本绑定检查使用`decode(saved token IDs) == assistant response continuation`；不得要求
  `encode(decode(ids)) == ids`。
- length truncation必须继续持久化finish reason和truncated状态，不能静默丢弃或重新采样。
- 回归测试必须包含decode不可逆但保存IDs仍能精确replay的tokenizer。
