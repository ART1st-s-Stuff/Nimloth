# Design

## Boundary

修改 `nimloth.training.sft.stage1` 的格式生成批处理、配置入口和相邻测试。训练 loss、数据缓存、checkpoint 与正式环境评估保持原合同。

## Data flow

1. 按原顺序构造前 32 条 prompt、图片和记录元数据。
2. 以 `format_eval_batch_size` 切片。每批将文本一次性交给 processor，并按 prompt 中图片占位符的原顺序提供图片。
3. 在局部上下文中将 tokenizer padding 切到左侧，使用 `padding=True` 产生批输入；生成后无条件恢复原设置。
4. FSDP 继续由所有 rank 处理完全相同的批次，并保留 `synced_gpus=True`。
5. 从统一的 padded prompt 宽度切出每条新 token；在首个 EOS 后裁掉生成器补入的 padding，再调用现有严格 validator。
6. 主 rank 发布批次进度与耗时；epoch 指标增加 LM 验证和格式生成耗时字段。

## Compatibility

- `format_eval_batch_size` 是正式可覆盖参数，标准 YAML 为 4。
- 保存的逐样本 JSONL schema 和严格 parser 语义不变。
- Stage 2 复用 `evaluate_format`，允许尾批并保持原有 `generate`/`inject` 判定路径。
- 不把性能代码写入实验启动脚本，也不改动正在运行进程的文件视图。

## Risk controls

- 批量多模态 processor 的图片展平顺序必须与每条 prompt 中 image token 顺序一致；测试覆盖多条和不同长度。
- 右侧 padding 会让 decoder-only 模型从 padding 位置继续生成，因此只在生成临界区使用左侧 padding并恢复。
- 批大小过大会抬高显存峰值，标准值 4 作为保守起点；可用 CLI 覆盖，无 OOM 自动降级。
- 远程恢复只使用完整 `COMMITTED` epoch checkpoint；若首轮真实生成 OOM 或语义不符，停止新进程并从同一 checkpoint 回到已提交版本。

