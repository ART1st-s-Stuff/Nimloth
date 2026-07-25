# E0052：增长的多模态history不能复用vLLM V1 prefix cache

## 已确认错误

逐turn增加图片的agent prompt在vLLM 0.11 V1默认prefix cache下可能复用旧文本token前缀，
但新请求的多模态placeholder与embedding调度切片不再一致。真实六图请求触发CUDA
`masked_scatter_size_check`并杀死EngineCore。

## 正确做法

- Nimloth Qwen多模态vLLM rollout显式设置`enable_prefix_caching=False`。
- 多图smoke必须跑到history增长后的后续turn；单图首请求成功不能证明该路径健康。
- 禁止把prefix cache崩溃后的部分图片或空trajectory当作有效rollout。
