# E0051: HF内部checkpoint不能只靠layer-wise FSDP auto-wrap修复

## 已发生错误

ID26首次让actor recompute在train mode真正启用gradient checkpointing。真实20-turn/4548-token forward通过，但root-only FSDP在首个backward触发`CheckpointError: recomputed values ... different metadata`。

ID27将actor和critic改为Qwen decoder layer及vision block的layer-wise FSDP auto-wrap。Nested FSDP runtime gate通过，真实20-turn actor forward再次完成，但8个rank仍在首个actor backward触发同类`CheckpointError`；saved/recomputed tensor从position 8起错位。没有actor optimizer step、critic阶段或显存结果。这证明layer-wise auto-wrap本身不充分。

Transformers 4.55的`GradientCheckpointingLayer.__call__`在Qwen block内部建立`partial(super().__call__)`的non-reentrant checkpoint。该checkpoint位于已经进入的FSDP unit内部，重算没有重放同一个wrapper-level执行边界。

VAGEN确实为actor和critic都设置了`enable_gradient_checkpointing=True`及`use_reentrant=False`，但其VERL固定`transformers==4.49.0`。4.49的Qwen实现由外层model loop checkpoint `decoder_layer.__call__`；FSDP auto-wrap替换layer后，重算会重新进入FSDP wrapper。Nimloth服务器环境是Transformers4.55.4/PyTorch2.8，不能把VAGEN配置名相同视为执行边界相同。

## 正确做法

- 禁用Qwen/Hugging Face block内部的gradient checkpointing。
- 在与FSDP相同的decoder/vision block边界，使用PyTorch `checkpoint_wrapper(..., CheckpointImpl.NO_REENTRANT)` / `apply_activation_checkpointing`建立外部activation-checkpoint wrapper，再验证FSDP与checkpoint wrapper的嵌套顺序。
- 必须先做直接distributed train-mode forward/backward测试，再运行真实20-turn显存gate。
- 不能关闭checkpoint determinism check来掩盖错误，也不能用forward或eval-mode成功声称兼容。
- Activation-checkpoint与FSDP wrapper协议均须进入checkpoint/resume metadata；改变协议后旧identity不可resume。
