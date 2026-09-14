# 执行计划
- [x] 用户授权纠正spec漂移后重启，以验证LM loss收敛为目标及每10步保存/成功后清理。
- [x] 实现stage1无query语义与相关回归测试；明确数据/缓存/checkpoint阶段身份。
- [x] 更新本次launch/cleanup与必要调用方参数，旧K配置不再进入stage1。
- [x] 接入已确认的收敛状态、checkpoint恢复与时限暂停；停止条件不是固定epoch。
- [x] 独立审查实际代码路径与stage2不回归，完成相关测试。
- [ ] 提交本次专用branch修正，同步远程专用worktree；CPU数据/模型/真实CLI检查。
- [ ] 构建并全量核验新format-only缓存，刷新资源后启动一次8GPU训练。
- [ ] 核验optimizer step/loss，记录进程、commit、run/监控与交接。
- [ ] epoch成功及final核验后清理当前run step checkpoints；记录最终结果。

## FSDP追加执行
- [x] 用户批准FSDP替代DDP。
- [x] 实现阶段配置、FSDP训练和完整保存恢复，保留非FSDP路径。
- [x] 独立审查及CPU回归。
- [x] 远程8卡有限保存/恢复验证，15分钟总截止。
- [x] 新run复用cache全量验证并启动真实收敛训练，确认有限optimizer步及运行状态。

## 动作加权实现边界
- [x] 用户审核通过仅修改sft1_step伪代码的spec。
- [x] 实现目标token加权CE及CLI/YAML校验，保持其他阶段和现有累积方式；缓存仍是原token/label。
- [x] 把权重纳入恢复身份；验证仍使用未加权LM loss。
- [x] 分析式loss/梯度/mask/shift、非法配置和恢复兼容回归；独立review。
- [x] 用户确认权重8及原始hf_actor初始化；真实8GPU长序列/保存恢复检查通过，新run已完成首个优化步。
本批source职责为stage1 loss、CLI/config、trainer/checkpoint及模块README；research launch/capacitygate传入明确权重。维持BF16参数和现有LoRA精度，不改token初始化、学习率、数据、其他stage或convergence。新加权目标从何checkpoint初始化尚需明确，不能复用旧optimizer/convergence冒充同目标恢复。

B execution boundary: approved prompt-only derived data, research semantic initializer, exact optimizer-step budget stop reusing safe checkpoint; optional research format evaluation/launcher. No human spec changes, no C learning-rate experiment, no original data/checkpoint mutation. Source entrypoints pinned before remote launch; default trainer semantics unchanged absent budgetflag.

## 已批准：Stage 2/3 按完整轨迹成功标记选择 LM
行为边界：全部真实轨迹保留状态监督；LM 按成功回答（Stage2）或成功起点窗口（Stage3）求均值。修改数据元信息、模型损失、累积/多卡归约及恢复身份，保留单次 Qwen 前向和已有 WM/value/SIGReg 语义。缺失 success 拒绝，不从动作/窗口回报猜测。测试覆盖混合/全失败、不等 token 数、累积与多 rank 分母。Stage2 初始化使用新 Stage1 epoch7，正式启动前核验其 termination/prompt 修正和全部轨迹数据。
