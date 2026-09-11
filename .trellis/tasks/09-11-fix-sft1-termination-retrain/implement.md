# 执行计划

- [ ] 建立独立 child-task worktree/branch，核验基点包含已提交的 Stage 1 统一评估入口，保留并发 `external/le-wm` 状态。
- [ ] 修改 Stage 1 加权 CE：八个动作编号权重 8，边界/EOS/其余为 1；新增明确 objective/checkpoint 身份并拒绝旧目标恢复。
- [ ] 将训练期 Stage 1 格式评估改为 token 终止检查加共享严格正文解析；保留 Stage 2 query 专用行为。
- [ ] 更新 `src/nimloth/training/sft/spec.md` 的 Stage 1 伪代码、Stage 1 README 和标准配置说明，使权重范围及严格终止验收与实现一致。
- [ ] 增加逐 token loss、shift/mask/归一化、BF16 梯度、checkpoint 兼容、EOS/length/tail/repeated-action 及训练/正式 parser 一致性测试。
- [ ] 运行聚焦测试、受影响 SFT/agent 测试、Ruff/compile 和 Trellis check；修复审查发现的问题后提交代码。
- [ ] 在 a100-1 只读核验并发布成功 train/heldout 派生视图：记录数、turn 数、动作分布、hash、split overlap 和图片/标签完整性；覆盖不足则停止。
- [ ] 从原始 `hf_actor` 建立并核验新的语义初始化产物；按成功子集建立或严格复用逐记录 cache，发布不可覆盖 manifest。
- [ ] 刷新 a100-1 GPU/进程状态，记录完整 launch contract，运行 8-GPU FSDP 有限更新、保存、恢复及严格生成门禁。
- [ ] 门禁通过后，以全新运行身份从 fresh optimizer 训练；每 10 步保存，按 epoch 核验后清理，跨 6 小时运行段从完整 checkpoint 续训，直至验证 LM loss 满足既定收敛规则。
- [ ] 导出收敛 final，先执行严格原文门禁；通过后运行标准 held-out 120 success-rate 评估，否则保存失败并停止。
- [ ] 按实验结束规范更新 task research/progress，记录实际 commit、命令、数据、模型、资源、checkpoint、指标、限制和恢复边界；完成最终 Trellis check、spec 同步和提交。

## 验证范围

- 聚焦：Stage 1 loss/config/checkpoint/format evaluation 与共享 Stage 1 parser。
- 回归：Stage 2 query alignment、统一 early-stage evaluation、checkpoint export/FSDP roundtrip。
- 真实门禁：a100-1 8-rank BF16/FSDP 下一步更新、保存恢复、导出重载和无约束生成。
- 质量：成功-heldout 完整 LM loss 收敛、完整 heldout prompt 严格格式、标准 test 120 环境 success rate。CPU 和有限门禁不替代质量结论。

## 停止与回滚点

- 成功子集规模或动作覆盖不满足输入门禁时，不启动 GPU 训练。
- 保存恢复、dtype、初始化 lineage 或严格生成机制门禁失败时，不启动正式训练。
- 正式运行出现 OOM、NaN、非计划退出或不完整 checkpoint 时停止诊断，不自动盲目重试。
- 收敛 checkpoint 的严格原文门禁仍呈系统性终止失败时，不启动 120-episode 环境评估。
