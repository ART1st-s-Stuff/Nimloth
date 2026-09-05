# E0095：实验后审计必须先检查结果schema

## 已确认错误

ID161 TP8 capture gate成功后，只读摘要脚本直接读取`one_turn_result.json['response_ids']`，
但该结果文件没有这个顶层展示字段，导致摘要脚本`KeyError`。实验主体、validator和cleanup均已完成，
错误只发生在后置审计。

## 正确做法

- 后置审计先打印或校验实际JSON keys，再提取字段。
- 区分“运行时已校验的原始response IDs/masks/log-probs”和“结果JSON实际持久化的字段”；
  没有持久化时不得在摘要中声称可以从结果文件重建。
- 后置摘要失败不改变实验主体结论，但必须保留hold直到用正确schema完成cleanup审计。
