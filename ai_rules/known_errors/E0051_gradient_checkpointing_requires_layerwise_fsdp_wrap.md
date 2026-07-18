# E0051: Gradient checkpointing需要layer-wise FSDP wrap

## 已发生错误

ID26首次让actor recompute在train mode真正启用gradient checkpointing。真实20-turn/4548-tokenforward通过，但root-only FSDP在forward后reshard整模参数；non-reentrant checkpoint backward逐层重算时拿到不同参数/tensor布局，8个rank均触发`CheckpointError: recomputed values ... different metadata`。此前eval mode静默关闭checkpointing，所以短smoke没有暴露该问题。

## 正确做法

- Qwen FSDP与gradient checkpointing组合必须按decoder layer和vision block auto-wrap，不能只wrap整个PEFT root。
- 显存/训练gate必须包含真实train-mode checkpointed backward；仅forward或eval-mode replay不能证明兼容。
- FSDP wrap policy属于checkpoint/resume协议，改变后旧identity和旧checkpoint不得无审计复用。
