# E0051: Qwen checkpoint/FSDP修复必须覆盖真实LoRA参数层级

## 已发生错误

ID26的真实20-turn/4548-token actor forward通过，但root-only FSDP在首个backward触发`CheckpointError: recomputed values ... different metadata`。

ID27改为Qwen decoder/vision block的layer-wise FSDP auto-wrap。Nested FSDP gate通过，真实actor仍在首个backward触发同类错误，证明layer-wise auto-wrap本身不充分。

Transformers4.55的`GradientCheckpointingLayer.__call__`在Qwen block内部checkpoint `partial(super().__call__)`。VAGEN/VERL固定Transformers4.49，在外层model loop checkpoint已由FSDP替换的`decoder_layer.__call__`；同名配置的执行边界不同。

ID28禁用HF内部checkpoint，并在raw Qwen block外应用PyTorch external non-reentrant `CheckpointWrapper`后再auto-wrap FSDP。Tiny全参数Qwen多模态gate在8rank完成forward/backward/Adam，证明基本wrapper顺序可工作；但生产actor的冻结base+LoRA r64+`use_orig_params=True`仍在真实首个backward报同一错误。错位列表含`[2048,64]`、`[64,11008]`等LoRA形状。0 actor optimizer、0 critic、无显存结果。这证明全参数tiny gate不能代表PEFT/FSDP层级。

## 正确做法

- 保持checkpoint determinism check开启；禁止隐藏错误。
- 下一direct gate必须复现生产actor：PEFT LoRA r64、冻结base/vision、完整decoder宽度、相同`use_orig_params`和auto-wrap policy；不能再用全参数tiny模型代替。
- 先检查实际module hierarchy及每个FSDP unit内参数的`requires_grad`组成，再测试LoRA-aware leaf auto-wrap。`use_orig_params=False`只有在每个flatten unit的参数都具有一致`requires_grad`时才可尝试；这是候选方案，尚未验证。
- 只有exact LoRA distributed train-mode forward/backward/optimizer gate通过后，才允许再次运行真实20-turn显存gate。
- Activation-checkpoint、FSDP、PEFT wrap和`use_orig_params`都属于checkpoint/resume协议；改变后旧identity不可resume。
