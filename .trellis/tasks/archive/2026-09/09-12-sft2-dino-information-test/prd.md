# Stage2 DINO 信息恢复测试

目标：在 a100-2 使用最新完整 Stage1 checkpoint 进行有界测试训练，衡量 state 恢复 DINO 特征的信息量，不以启动成功或训练 loss 下降代替质量结论。

输入：20260911T115012Z_success_only_termination_w8/train/epoch_014，已远程核实 COMMITTED epoch14 step252；位于 a100-1。a100-2 n30196 当前8卡空闲，仅有评估数据，没有此完整权重。

保留既有 Stage2 合同：K64、8x8，全部轨迹 DINO，成功轨迹 LM，完整轨迹一次因果前向。原始输入与 checkpoint 不修改。

验收：独立验证轨迹报告样本数、逐轨迹聚合 MSE、余弦相似度、相对训练集均值基线的恢复改善；比较训练前后并报告分布。配对打乱 state 评估验证是否依赖对应观察。不能据此宣称完整 DINO 或 RGB 重建能力。

待审核预算：一次8卡测试，最多2 epochs或6小时，先到即停止并评估；训练前和每个epoch评估。每10步保存，保留所有测试ckpt。预算耗尽不称作收敛，不自动扩大预算。

## Authorized rerun 2026-09-12
Use latest Stage1 run 20260912T130307Z_token_fp32_lr5e5_action2_boundary8_epoch5, epoch002 (step36). Keep embedding and lm_head FP32 masters exactly as Stage1, BF16 forward; fresh LoRA LR5e-5, embedding LR5e-5; preserve BF16 projector with independent LR1e-6. Keep existing two-epoch DINO information test scope, K64/grid8, eight GPUs on a100-2, six-hour total controller budget including one-hour evaluation reserve, ten-step checkpoints. Preserve previous run. Initial checkpoint and standalone evaluations remain required. Current user explicitly authorized continuation and these changes.

## Authorized fresh training and optimizer override 2026-09-12

Start a fresh Stage2 run from the latest Stage1 epoch002 model on a100-1. Do not load the
existing Stage2 epoch001/epoch002 model, optimizer, scheduler, RNG, global step, or
convergence history. Train until convergence, monitoring full-validation
`validation_total_loss`; stop after two consecutive epochs with relative improvement below
1%. The new run starts at epoch 1 and establishes its own validation history.

The Stage2 default optimizer contract is LoRA LR `5e-5`, projector LR `5e-5`, the current
K query-token rows in both input embedding and independent LM head at LR `5e-5`, and the
eight action-number rows plus action-start, action-end, and EOS format rows at LR `1e-5`.
All other vocabulary rows are frozen exactly, including against Adam momentum and AdamW
weight decay. Selected token rows use FP32 masters and BF16 forward. Checkpoint identity
must record the selected-row schema, exact token IDs, both row learning rates, FP32 master
precision, and input-embedding plus independent-LM-head scope, and resume must fail closed
on a mismatch. The existing epoch002 evaluation on a100-2 remains in scope and must not be
cancelled or overwritten.
