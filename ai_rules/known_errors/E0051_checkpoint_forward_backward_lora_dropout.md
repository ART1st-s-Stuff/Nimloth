# E0051: Checkpoint forward与backward重算期间不能恢复LoRA dropout

## 已发生错误

ID26–ID28的真实20-turn/4548-token actor都在首个backward触发`torch.utils.checkpoint.CheckpointError`。曾依次归因为root-only FSDP、HF内部checkpoint边界、PEFT/FSDP参数层级；这些解释都不完整。

最终CPU最小复现确认直接原因：`_temporary_deterministic_train()`在forward期间把LoRA dropout从0.05改为0，但context在`loss.backward()`之前退出并恢复0.05。Non-reentrant checkpoint在backward重算时因此多执行了dropout mask操作。ID28错误的第一个recomputed tensor正是`[1,4548,2048] bool`，对应LoRA dropout输入mask；saved列表此处原本是`[2048,64]` LoRA矩阵，后续tensor整体错位。

ID28的全参数tiny gate没有LoRA dropout，所以通过；它不能覆盖这个错误。CPU exact-LoRA测试已稳定复现：forward后退出context再backward会报同类`CheckpointError`，把backward放在context内则成功。

## 正确做法

- PPO old/new/reference必须确定性；RL actor的LoRA dropout固定为0，不能沿用SFT默认0.05。
- CLI、launcher、memory probe和严格resume protocol都必须显式记录/校验`lora_dropout=0`；非零值fail-fast。
- 如果临时修改任何影响forward算子序列的状态，checkpoint backward必须在状态恢复前完成。
- 保持checkpoint determinism check开启；禁止隐藏错误。
- Tiny gate必须复现实际可训练模块和dropout配置，不能只用全参数无dropout模型替代。
- 这次根因不证明FSDP或PEFT本身不兼容；修复后的真实distributed backward仍需单独验证。
