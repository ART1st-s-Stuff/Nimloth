# E0051：vLLM V1 UtilityResult默认不保留嵌套tensor类型

## 已确认错误

vLLM 0.11 V1的utility RPC在默认安全模式下不为任意嵌套result保存Python类型信息。worker
返回`dict[str, torch.Tensor]`时，字段可保留，但tensor在前端成为原生tuple/list；直接要求
`torch.Tensor`会失败。

## 正确做法

- worker extension只跨utility RPC返回普通数值容器，前端显式重建指定dtype的tensor。
- 重建后继续检查维度、finite和所有TP rank一致性，不能因序列化妥协数值门禁。
- 禁止仅为自定义结果开启`VLLM_ALLOW_INSECURE_SERIALIZATION`。
