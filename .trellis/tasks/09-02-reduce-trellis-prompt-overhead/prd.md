# 降低 Trellis prompt 开销并简化低风险流程

## Goal

把典型Agent在读取源码前的项目/Trellis上下文开销从约36–46k token降到不超过10k，并让已批准范围内的低风险小修改只执行最小充分验证，不重复请求授权、启动subagent或写progress commit。

## Requirements

- Main SessionStart只注入compact overview、当前task/work-item摘要、artifact hash/path；不得内联完整JSONL context与三份task artifact。
- `trellis_subagent`只提供一份task context：完整task artifacts至多一次；JSONL默认提供去重后的path/reason索引，Agent按实际修改需要读取相关文件。
- canonical path去重；task artifact若出现在JSONL中不得重复；绝对路径必须正确处理，missing/truncated必须可见。
- task/artifact更新只发送变化摘要或新hash/path；禁止因progress/checklist/git status变化重复追加完整context snapshot。
- Child dispatch必须明确已提供的内容和已满足的加载步骤，禁止机械重读已内联文件；相邻源码仍按需读取。不得为此fork此前pristine的Trellis生成agent/prompt文件。
- 增加low-risk fast path：现有approved scope内、无protected/remote/destructive/schema/public-contract风险、紧密相关小改不新建task、不启动implement/check agent、不重复完整验证或审批。
- 验证分层并复用：小步只跑focused RED/GREEN；受影响层检查在work-item边界运行；完整lint/typecheck/build在最终批次至多一次。相关输入未变时复用成功证据。
- 审批只在权限边界变化时触发；不得重复请求已授予权限。实验launch、远程/破坏性操作、protected data、push/merge仍保持精确审批。
- `on-progress`改为work-item完成、风险/设计变化、实验状态变化或跨session交接时触发；同一item连续小修合并记录。
- 保持Trellis唯一任务权威、CoT/state、Git/worktree、实验和诚实性红线。

## Acceptance Criteria

- [ ] 自动化fixture证明Main初始Trellis context不超过16KB，且不含完整PRD/design/implement正文。
- [ ] implement/check child初始Trellis context各不超过32KB、每个task artifact最多出现一次、JSONL文件正文默认不内联。
- [ ] absolute/relative/canonical duplicate context entry测试通过；遗漏和截断可诊断。
- [ ] artifact/progress变化只产生bounded delta，不追加完整task context。
- [ ] Pi child dispatch明确task artifact已提供且不得重读；此前pristine的Pi/channel agent与start/continue/finish生成文件保持原版。
- [ ] low-risk fast path和approval reuse规则在`AGENTS.md`、workflow、spec及相关skills/prompts中一致。
- [ ] 高风险实验、远程破坏、protected file和push/merge门禁回归测试/文本检查通过。
- [ ] 给出修改前后同一fixture的bytes和估算token对比。

## Authorization

- 人类已明确要求立即进行prompt修正，并授权创建本治理task与规划。
- `AGENTS.md`为人类编写文件；只在最终scope经人类批准后按本task精确修改。
- 当前仅规划；不修改memory JSONL，不启动实验，不push/merge。
