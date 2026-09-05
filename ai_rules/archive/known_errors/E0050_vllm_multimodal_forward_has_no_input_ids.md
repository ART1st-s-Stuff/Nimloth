# E0050：vLLM V1多模态forward参数没有input IDs

## 已确认错误

vLLM 0.11 V1对多模态模型统一先构造`inputs_embeds`，随后以`input_ids=None`调用model
forward，即使当前decode位置只是文本token也一样。只从forward参数读取token ID的hidden
capture hook在真实图片rollout中会得到空序列；文本fake test无法覆盖该路径。

## 正确做法

- hidden仍取当前同一次model forward输出；对齐token ID从V1 runner为该forward准备的
  `input_ids.gpu` buffer读取，行数必须与hidden严格一致。
- 测试必须显式覆盖`input_ids=None, inputs_embeds=...`的多模态调用约定。
- 禁止用HF二次forward替代vLLM rollout hidden state；behavior与planner state必须来自同一次
  vLLM生成。
