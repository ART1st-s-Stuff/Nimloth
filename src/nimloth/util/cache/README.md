# 预处理缓存

本包负责 Qwen transition 编码缓存的通用存储和构建流程。它消费 rollout
transition 与 Qwen 编码适配器，不包含 SFT2 优化语义。

Transition expansion v4 保存可选 `action_success`；加载时核对 source transition
的 outcome 和 action-value target。新增监督字段需要新 cache fingerprint。

`image_reuse.py` 校验旧缓存的图像来源指纹、顺序、索引、网格、dtype 和
分片结构，然后仅将图像分片 hardlink 到新缓存。来源指纹基于路径、大小和
mtime；复用记录另保存分片 SHA256。文本和 labels 由当前编码流程重新生成，
旧 manifest 和 transition 分片不复用。新旧目录不得相同或互相包含。
