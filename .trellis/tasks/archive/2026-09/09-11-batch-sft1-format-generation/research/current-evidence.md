# Current evidence

- `stage1/trainer.py:evaluate_format` 对每条样本单独构造 processor 输入并调用一次 `generate(max_new_tokens=128, do_sample=False)`。
- FSDP 分支通过 `generation_model` materialize root 参数，所有 rank 使用相同 prompt 和 `synced_gpus=True`。
- tokenizer 在训练器初始化中使用右侧 padding；批量 decoder-only 生成必须局部切换为左侧 padding。
- a100-1 第 3 个 epoch 最后训练 step 到 validation metrics 写出间隔约 1606 秒；日志没有 LM 验证与格式生成的独立时间点。
- 第 3 个 epoch 格式通过率为 27/32，证明当前判定合同已有可对照产物。
