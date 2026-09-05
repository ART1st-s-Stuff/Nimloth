# E0007 — PEFT adapter resume 不能直接调用 model.load_state_dict

## 错误

SFT1 resume 对 `adapter_model.safetensors` 直接执行 `model.load_state_dict(..., strict=False)`。PEFT 保存时会移除 adapter name；raw load 不会恢复它，实际出现 700 unexpected keys，却继续训练并保存了无效 epoch2。

## 正确做法

已经注入同配置 adapter 的模型应使用 `peft.set_peft_model_state_dict`。加载后再用 `get_peft_model_state_dict(..., save_embedding_layers=True)` 与保存文件逐 key 检查：

- 所有保存 key 都存在；
- shape 和 tensor value 完全一致；
- unexpected keys 为空。

不能因为 `strict=False` 没有抛异常就声称 resume 成功。无效 checkpoint 必须明确隔离，不能继续作为 latest 使用。

当前服务器 PEFT 还会从 Transformers 4.55.4 导入不存在的 `EmbeddingParallel`。对明确没有 tensor-parallel plan 的模型，可在调用 PEFT loader 前添加缺失类 sentinel；不得假装真正的 TP 模型也兼容。

此版本对 `modules_to_save.weight` source keys 可能同时返回 unexpected，但已把它们映射到 adapter-name wrapper。只有当这些 keys 来自 saved state 且完整 post-load tensor exact check 通过时才可过滤；其他 unexpected 必须失败。
