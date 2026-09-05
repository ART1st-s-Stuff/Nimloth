# E0052：禁止用单次配置相关性断言多模态cache根因

## 已确认错误

ID98在六图请求同时观察到prefix复用和CUDA `masked_scatter_size_check`，因此把崩溃归因于
prefix cache。ID99显式关闭prefix cache后，`num_computed_tokens=0`且
`num_common_prefix_blocks=[0]`，相同六图请求仍原样崩溃，证伪了该归因。

## 正确做法

- cache根因必须通过对照重跑验证；配置与失败同时出现不能单独证明因果。
- 记录失败请求的placeholder范围、feature identifier、`scheduled_encoder_inputs`和实际cache
  开关；区分token prefix cache、processor cache和GPU encoder cache。
- 多图smoke必须跑到history增长后的后续turn；单图首请求成功不能证明该路径健康。
- 禁止把崩溃后的部分图片或空trajectory当作有效rollout。
