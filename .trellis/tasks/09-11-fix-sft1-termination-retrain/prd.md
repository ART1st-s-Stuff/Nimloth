# 修复 SFT1 终止格式并重新训练

## Goal

让 Stage 1 模型在生成一个完整的 CoT 和动作块后正确结束回答，使训练期格式指标与正式 rollout 的严格解析一致；从原始 `hf_actor` 的既定语义初始化重新训练到验证集 LM loss 收敛，并用标准 held-out 120 episodes 验证 success rate。

## Background

- 已停止的 epoch 5 评估完成 12/120 个 Base episode，成功 0。全部已完成 episode 都运行满 20 步。
- 220/220 个已检查 step 的生成包含动作块，但在 `<|action_end|>` 后没有生成真实 EOS `<|im_end|>`，而是继续生成新的 `<|action_start|>`、padding 和 `<|im_start|>`，最终达到 512 token 上限。
- 正式 Stage 1 parser 对完整回答执行严格匹配，因此这 220 个输出均为 `invalid_response_envelope`，实际发送空动作并每步得到 -0.2；这直接解释了当前成功数为 0。
- 实际缓存标签以 `<|action_end|><|im_end|>` 结束，EOS 监督没有缺失。当前 loss 将动作起始、动作结束和八个动作编号 token 统一赋权 8，EOS 权重为 1。
- 训练期 `format_correct_rate` 当前通过正则搜索中间动作块，允许尾随内容；epoch 2 至 epoch 5 的 32/32 因此没有证明生成会在动作块后结束。
- 动作边界已从 EOS 初始化，八个动作编号已从各自语义词初始化。当前问题不通过 FP32 参数、动作约束生成、在 `action_end` 强制截断或评估侧修复来掩盖。
- 当前 B 训练输入是 1709 条 `sft1_train_all.jsonl`，其中同时包含成功与失败 rollout。转换流程已经生成并严格核验 `sft1_train_success.jsonl`，但本轮没有使用它。
- Stage 1 名称是格式训练，但实际 LM loss 监督完整 CoT 和动作，而不只监督格式 token；因此失败 rollout 也会把未完成任务的推理和动作序列作为模仿目标。

## Requirements

- R1：训练期 Stage 1 格式评估必须要求完整 CoT、恰好一个合法动作块和模型生成的终止 token；动作块后的非终止内容必须失败。其解析口径应与正式 Stage 1 rollout 一致。
- R2：Stage 1 loss 必须显式区分动作选择 token、动作边界 token 和 EOS 的权重语义。改变权重范围后，checkpoint/objective 身份必须阻止旧优化器状态静默恢复；token/label 未变化时不得无意义重建预处理缓存。
- R2a：仅八个 `<|action_(i)|>` 动作编号 token 使用权重 8；`<|action_start|>`、`<|action_end|>`、EOS 和其余有效回答 token 的权重均为 1。
- R3：保持已经审核的 B prompt、语义 token 初始化、BF16/FSDP/LoRA、数据划分、学习率、有效 batch、保存清理和收敛规则；从原始 `/mnt/nimloth/checkpoint/hf_actor` 派生新的初始化模型并使用全新运行目录。
- R4：训练位于 a100-1，从现有 1709 条 train 与 193 条内部 heldout 源分区分别筛选成功 rollout；实际训练和验证条数以启动前审计为准。至少训练 2 个 epoch，以完整成功-heldout 子集的未加权回答 LM loss 为准；连续 2 轮相邻改善不足 1% 时停止。运行时限只触发完整 checkpoint 后续训，不算收敛。
- R5：每 10 个 optimizer step 保存；完整 epoch checkpoint 核验后只清理被该 epoch 覆盖的本次中间 step checkpoint，保留 epoch、best、final 和失败证据。
- R6：收敛后先以固定 32 条完整 heldout prompt 执行严格格式门禁；通过后使用统一入口在 Base 60 + Common Sense 60、seed 1..60、每 episode 最多 20 步的 test split 上执行 Stage 1 success-rate 评估。保存原始模型输出、严格格式结果、实际环境输入、奖励、终止原因和分组/总体汇总。
- R7：训练、导出和正式评估都绑定已提交且远程一致的 commit。不得覆盖或删除原始 `hf_actor`、旧 B 训练、已停止评估、已暂停 Stage 2 及其 checkpoint/输出。
- R8：正式启动前统计成功子集的 trajectory/assistant-turn 数量、八类动作实际计数（允许零计数）和 train/heldout 隔离。训练 LM 只使用成功 train rollout，收敛 LM loss 使用成功 heldout rollout；严格格式生成仍覆盖完整 heldout prompt，最终质量只由真实环境 held-out 评估判定。不得为补齐成功子集未覆盖的动作而混入失败 rollout。

## Acceptance Criteria

- AC1：回归测试证明合法回答只有在动作块后生成 EOS 时通过；尾随文本、重复动作块、达到长度上限和缺少 EOS 均失败。训练期与正式 Stage 1 parser 对同一组样例给出一致结果。
- AC2：loss 测试逐 token 验证最终批准的三类权重、shift/mask/归一化、BF16 梯度和 checkpoint 身份；Stage 2 不继承 Stage 1 专用权重。
- AC3：远程 preflight 核验代码 commit、Python、数据与 split、缓存标签尾部、初始化 lineage、可训练/冻结模块、FSDP 保存恢复和空输出身份；真实多卡门禁产生有限 loss 并能精确恢复。
- AC4：正式训练产生逐步 loss、每轮完整验证、严格格式样本和可恢复 checkpoint；只在既定收敛条件满足后发布 `CONVERGED`/final。
- AC5：严格原文门禁通过后，标准 held-out 评估完整完成 120/120，并报告 Base、Common Sense 和总体 success rate；若运行中仍有格式失败，结果按实际失败保留，不进行生成约束或修复。
- AC6：正式 120-episode 环境评估前，使用同一导出 checkpoint、同一生成实现和固定 32 条完整 heldout prompt 执行严格原文门禁；至少 31/32 输出必须生成 EOS 且正文完整匹配。未达标则保存逐条原文和终止原因并停止，不启动数小时的环境评估。

## Out of Scope

- 不恢复或继续当前暂停的 Stage 2 K64 训练。
- 不改变 prompt、动作语义初始化、训练/验证划分、模型精度、LoRA 范围、学习率或 Stage 2/Stage 3/RL 目标。
- 不用强制在 `action_end` 停止生成作为模型通过格式验收的证据。
