# E0086: Qwen 梯度检查点必须同时处于 train mode

## 现象

planner RL 启动参数和加载日志显示已调用 `gradient_checkpointing_enable()`，但长决策
前缀的 Qwen 重算仍在单层 language MLP forward 耗尽整卡显存。仅检查配置开关会误判
激活检查点已经生效。

## 根因

Transformers 4.55.4 的 `PreTrainedModel.from_pretrained()` 在加载结束时执行
`model.eval()`。同版本 Qwen2.5-VL 文本模型只在
`self.gradient_checkpointing and self.training` 同时成立时调用 checkpoint forward。
原 RL trainer 开启了前一个条件，却从未把底层 Qwen 切回 train mode。ID137 因此在
首个 optimizer step 之前保留完整长前缀激活，并在 GPU3/GPU5 的 language MLP forward
分别 OOM。

## 防复发

- planner 的独立 vLLM rollout 模型与可微训练 Qwen 必须分别管理 mode；训练 Qwen 在
  DDP/FSDP 包装前显式进入 train mode。
- 不得用 CLI 的 `gradient_checkpointing=true` 代表运行时已生效。启动时必须检查至少
  一个模型子模块的 `gradient_checkpointing` 为真且该子模块 `training` 为真；不满足时
  fail closed。
- CPU/接口测试只能证明 mode 契约。恢复训练前仍需 production-shaped GPU 门禁，使用
  真实长前缀完成 forward、backward 和 optimizer step，并记录峰值显存；它不能复用
  ID137 的未提交 rollout。

## 证据边界

远端正式环境 Transformers 4.55.4 源码中，`modeling_utils.py` 明确在
`from_pretrained()` 结尾执行 `model.eval()`；Qwen2.5-VL 的文本 forward 明确以
`if self.gradient_checkpointing and self.training` 选择检查点路径。该代码证据解释
ID137 的激活保留，但在新的真实 GPU backward 门禁完成前，不能宣称显存问题已被实验
证明解决。
