# E0062：RL rollout、replay 和 checkpoint 必须使用同一 processor 分辨率

## 错误

Python 侧给 processor 覆盖了较小的 `max_pixels`，但 vLLM 从 checkpoint artifact
重新创建自己的多模态 processor。结果 behavior rollout 和 HF replay 看到了不同的图像
token；保存 checkpoint 时还会把 Python 侧覆盖写入下一轮 artifact，令相邻 iteration 的
policy 行为语义发生变化。

## 规则

- RL 默认保留初始化 checkpoint 自带的 processor 像素上下界，不得在 launcher 中静默
  写死另一个值。
- 若人类明确批准覆盖，必须把同一像素上下界传给 vLLM multimodal processor、HF rollout、
  reference replay 和训练 replay，并在 fresh manifest 中记录实际生效值。
- 训练加载 processor 后必须和 fresh manifest 校验；不一致时在 optimizer step 前失败。
- checkpoint 保存不得无意改变下一 iteration 的 processor 配置。

## 证据

ID110 的 ID46 初始化 artifact 为 `max_pixels=100352`，update-1 vLLM 使用该 artifact；
launcher 同时给 HF replay 强制传入 `3136`，而 update-1 保存的
`preprocessor_config.json` 变为 `max_pixels=3136`。因此该 gate 被取消且不可续训。
