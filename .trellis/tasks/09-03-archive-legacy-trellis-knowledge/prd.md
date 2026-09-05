# 需求 — 归档legacy Trellis知识和进度系统

## Goal

逐文件复核156份known-error记录，把仍有效的通用合同提炼到scoped specs，并按人类选择的现有命名空间byte-preserving归档known errors、旧`ai_tasks/`、`AI_branch_progress.md`和`AI_issues.md`，使active prompt/routing只依赖Trellis与spec。

## Requirements

- 以完整路径而非数字前缀识别156个`E*.md`；生成source SHA-256、topic、evidence、disposition、target spec、destination manifest。
- 只有经当前源码/test/spec重新核验的通用模式进入spec；`retain-contract`必须记录owning spec路径、唯一section anchor和逐字excerpt并由机器验证。实现局部、版本/机器特定或不值得稳定化的行为归为`archive-only`，不得为保留分类而膨胀spec；不复制事故叙述，不把archive作为现行规则源。
- Known errors归档到`ai_rules/archive/known_errors/`，保留文件名和原index作为traceability历史。
- `ai_tasks/`现有内容归档到`ai_tasks/archive/pre-trellis/<原相对路径>`；根部`AI_branch_progress.md`和`AI_issues.md`归档到`ai_tasks/archive/pre-trellis/root/`。
- Active specs/skills/docs/code/tests/Slurm引用迁移到现行spec、Trellis task或明确的archive provenance路径；不得改变实验语义。五个SFT1 Slurm purpose在post-move指向`ai_tasks/archive/pre-trellis/sft1_exp.md`并保留`no actor/critic update`。
- 现有Trellis task artifacts是历史/并发状态：只为所有当前`in_progress` task准备会阻断active context validation的exact manifest rewrite，不重写其叙述或状态；冲突/归属不明时停止。
- 移动前展示完整source→destination manifest、hash、hash-bound rewrite patch和命令并取得独立破坏性批准；未批准不得`git mv`、应用post-move rewrite或删除。
- Move preflight拒绝symlink/非目录destination ancestor和repo-root escape，要求全部destination absent，并在首个move前创建/复核全部destination目录；每次mutation前再次检查。
- Execute完成后验证destination SHA/blob、source absence、active legacy refs、current active task context、Markdown links和focused tests；提供但不执行deterministic partial rollback。
- 不修改memory JSONL。`local:M0013`保持非权威，除非人类另行批准memory纠正。

## Acceptance Criteria

- [ ] Manifest恰好覆盖156个E文件、完整ai_tasks tracked tree及两个根部历史文件，duplicate prefixes不丢失。
- [ ] 156项均有可审计disposition和证据；所有retain mapping逐字解析到owning spec，四个needs-human-decision不编码规则；spec diff只含重新核验的通用合同。
- [ ] 人类批准后移动前后blob/SHA-256一致，目标无collision，source无原历史正文。
- [ ] Active prompt/spec/skill/source不再把known errors、AI_branch_progress或ai_tasks当作当前authority。
- [ ] 必需的active task context和launcher provenance引用解析到有效替代路径。
- [ ] Markdown links、Trellis context validation、focused launcher/static tests及archive contract tests通过。
- [ ] 独立review无P0–P2；未经批准不commit/merge/push。

## Out of Scope

- 不改实验算法/配置，不运行实验、GPU、Slurm或remote job。
- 不重写archive正文，不审批memory，不删除live approval runtime。
- 不归档当前`.trellis/tasks/`。
