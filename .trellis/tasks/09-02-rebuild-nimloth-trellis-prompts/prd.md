# 交互式重建 Nimloth Trellis prompts

## Goal

以发布版Trellis `0.6.16`为不修改的通用基线，重建一个低重复、可升级的Nimloth项目覆盖层：常驻prompt只保留必要安全不变量，完整合同进入spec，操作流程进入project-local skill，独立extension承担可见性和通用跨session问答传输。

## Background

- 项目记录版本为Trellis `0.6.15`，本机发布版CLI/templates为`0.6.16`。
- 当前规则重复分布在`AGENTS.md`、`.trellis/workflow.md`、governance specs、skills、Trellis Pi extension和pi-app中，并存在审批、context和生命周期冲突。
- 上游Trellis使用对话式阶段门禁，没有本地五分类typed-approval runtime。
- Pi原生扫描skill并只把name/description常驻注入；普通能力由description匹配，强制生命周期路由由上游workflow breadcrumb负责，完整skill正文按需读取。
- Nimloth没有统一的root lint/typecheck/test命令；现有172个Python test文件中，静态CPU检查与真实GPU/Slurm实验必须分开管理。

## Requirements

### R1 — Ownership and baseline

- 发布版Trellis `0.6.16`的workflow、scripts、generated prompts/agents、context injection和bundled skills保持上游语义，不维护Nimloth Trellis fork。
- `trellis update`管理的通用文件不得承载Nimloth私有政策；项目能力使用独立local extension/skill，工程合同使用`.trellis/spec/`。
- Pi是主要验收平台；`AGENTS.md`、spec和共享`.agents/skills`尽量平台中立，其他平台的生成文件保持上游模板。

### R2 — Thin always-on safety kernel

`AGENTS.md`仅包含：

1. 项目内权威顺序：当前人类指令 → `AGENTS.md`安全内核 → 上游Trellis workflow → project specs → reviewed task artifacts → 当前源码证据 → 经核验memory；task不得静默放宽spec。
2. 诚实与实质不确定性边界：低风险、可逆实现细节自主决定；目标/语义、公共合同、授权、安全、不可逆影响、无法验证或不可调和冲突必须询问。
3. 替代机制边界：proxy、stub、approximation或不完整实现必须在Trellis task中披露并批准，且不得冒充真实实现或验收证据；普通隔离型test double例外，但不能替代被验收行为。
4. 破坏性与protected-data边界。
5. Git mutation不变量与protected-branch边界。
6. 不可漏掉的project-local高风险skill路由。

`AGENTS.md`不包含CoT/state等实现规范、Trellis workflow副本、Slurm/worktree具体命令、known-error规则、work-item runtime或approval协议。它继续由人类治理；最终task计划明确列出并获批准的`AGENTS.md`修改无需逐次再审批。

### R3 — Task and approval semantics

- 无active task时严格遵循上游task-creation consent：simple/trivial代码工作也先询问是否建task；用户拒绝后方可inline处理。任何真实实验执行仍需experiment task；复现或仅部分参数变化的trivial实验使用轻量小task。
- task创建不等于实施批准；复杂task的`prd.md`、`design.md`、`implement.md`经最终摘要审查并收到后续明确回复后才进入实施。
- 移除`planning / implementation / experiment_launch / commit / push_merge` typed approval、artifact-hash receipt及其Trellis生命周期暗示。
- pi-app可保留独立、通用的跨session问题队列，只把标准人类回复可靠送回source session；它不得定义Trellis approval kind、签发receipt、判断授权、启动task或成为第二审批权威。
- 破坏性动作（删除数据、覆盖不可恢复产物、destructive SQL、force Git等）无论是否有task，都必须在执行前展示精确目标、命令和影响并取得批准。
- protected data允许边界内只读检查；修改、移动、覆盖或删除必须执行前精确批准。

### R4 — Experiment, remote and Git authorization

- 禁止在本地执行训练、评估、数据收集、rollout等真实实验；本地静态检查和CPU单元测试不属于实验执行。
- 所有真实实验在具备所需资源的远程环境运行，并由`meta.kind=experiment` task记录source、参数、预计时长、运行次数和输出范围。
- 预计在10分钟内结束的remote reproduction或部分参数重跑可使用轻量experiment task，启动无需额外询问。remote job从成功提交时刻起设置15分钟总hard deadline，排队等待与实际运行共同计时；到期仍为pending或running时必须取消，防止因长期等不到节点而阻塞。取消后的控制流由task依赖决定：若该实验暂不阻断当前代码批次，则记录为deferred并继续其他工作；到阶段边界或其结果成为后续工作的真实前置条件时，停止并报告blocker，不得把缺失结果当作通过。GPU/CPU资源数量不设固定上限；AI应先检查剩余可用资源，并在不抢占或干扰现有任务的前提下灵活选择资源。
- 预计超过10分钟的实验启动必须明确批准。真实GPU/Slurm/model-quality结果不得由静态或CPU测试替代。
- routine、非破坏性remote操作在已批准task scope内无需逐命令审批；破坏性操作、protected-data mutation、预算超限或scope实质变化仍触发各自门禁。
- 每个`meta.kind=experiment` task自动获得其声明repository、remote和source/target branches范围内的task级Git同步权限，可覆盖相关实现、测试、修复及多次正常commit/push/merge，直到task完成。
- 非实验task不继承上述Git同步例外；repository/remote/target变化、scope扩大或task完成使授权失效。force操作始终视为破坏性动作。
- 每个repository在本地config/spec声明可写开发branch；`main`等protected branch默认不可修改，只有reviewed task的明确例外可以进入。

### R5 — Worktree safety

- 每个Trellis task使用自己的Git branch，并在task metadata中记录；不同task不得共享同一实施branch。
- 每次mutation前核验目标repository、task branch和status，并保留无关或并发dirty changes。
- 单一主task在canonical directory checkout其独立task branch；出现并行时，当前主task保留在canonical，新增并行task各自在worktree checkout自己的branch。Canonical存在未提交修改时不得切换到其他task branch。
- worktree使用结束后必须cleanup。只有tracked、untracked、ignored和nested payload均核验为空且普通non-force remove足够时可自动清理；发现payload或需要force时必须精确询问。
- canonical paths、`.local`共享、创建验证及cleanup命令只存在于Git/worktree spec和skill，不常驻`AGENTS.md`。

### R6 — Work-item visibility as an independent capability

- 保持上游`implement.md`为复杂task的人类可读执行计划，不要求`[W-xxx]`或修改其默认语义。
- 独立work-item extension/skill只在复杂task进入实施时自动激活；规划和轻量工作不注入该prompt。
- extension按task、heading和checkbox文本透明派生item identity；plan文字变化使旧assignment失效并要求重新选择。
- executor/item/state/timestamp保存在extension-owned、project-local、gitignored runtime中，不使用Trellis runtime namespace，不复制task、plan或checkbox状态。
- extension/runtime失败只能降级可见性，不得阻断Trellis、改变task lifecycle或声称plan完成。

### R7 — Context, validation and progress

- 恢复上游`0.6.16` context injection和Pi implement/check agent，不保留compact-locator/delta补丁；恢复后先测量基线，再另立任务处理剩余开销。
- 不修改上游agent/workflow验证措辞。Nimloth Python quality spec定义项目检查：实施迭代运行focused RED/GREEN；独立check运行一次最终affected-scope静态/CPU suite；真实实验证据由实验预算合同负责。
- Nimloth专用中途progress只用于跨session交接、外部长job状态变化或即将中断；普通work-item和连续小修依赖上游task artifacts、journal和finish-work。
- 保留上游finish-work：work commits按Phase 3.4审查后，显式finish-work可自动创建task-archive commit和session-journal commit。

### R8 — Specs, memory and legacy knowledge

- project spec变更不静默混入实施。task结束时AI可准备未提交spec diff，但必须单独展示完整diff、理由和影响；仅获批内容进入commit plan，拒绝时只回退spec修改。
- curated memory只作按需证据：相关时通过skill查询并重新核验；创建、纠正或upvote需要人类批准；操作细节不常驻prompt。
- 审计全部156个`ai_rules/known_errors/E*.md`，以当前源码/证据复核，提炼仍有效的通用failure patterns写入对应spec，不复制事故叙述。
- 迁移后把原known-error记录及索引移动到历史archive；archive仅供追溯，不再是现行规则源。
- 归档并停止写入`AI_branch_progress.md`和旧`ai_tasks/`；它们退出active prompt，新进度只使用Trellis与本PRD规定的有限交接记录。

### R9 — Remove TaskTree globally from pi-app

- Trellis是Nimloth唯一task authority。
- 从pi-app全局删除`pi-task-tree` adapter、专用TaskTree panel/model、TaskTree-specific mutation mapping、文档和测试。
- 保留被其他adapter使用的通用`workspace-json`基础设施。

### R10 — Legacy approval cleanup

- 移除typed-approval framework前，将现有request/receipt runtime完整复制为`.local`下的只读审计快照。
- 删除live legacy runtime属于破坏性操作，实施时必须另行展示精确路径并取得批准。
- 新系统不迁移、不读取archive中的approval记录。

### R11 — Skill routing

- 普通project-local skill依赖Pi/Trellis原生description匹配；description必须具体描述触发意图。
- `AGENTS.md`只点名绝不能漏掉的高风险路由：实验、Slurm/remote GPU、worktree cleanup、memory mutation及长job/跨session交接。
- 生命周期skill继续由不修改的上游workflow breadcrumb路由；具体步骤和参数仅位于`SKILL.md`及references。

## Acceptance Criteria

- [x] 发布版manifest、direct-source及`trellis update --dry-run`差异分类证明通用Trellis workflow/scripts/Pi prompts/agents/extension恢复`0.6.16`；project-owned overlay未混入template-managed文件。
- [x] `AGENTS.md`只含R2定义的薄内核与R11高风险路由，不含实现规范或流程副本；其精确diff经过本task审查。
- [x] 对话式task/implementation/commit门禁符合上游；不存在active typed-approval tool、kind、receipt或授权投影。
- [x] pi-app标准问题在session切换、background、later、explicit decline及worker cancellation下保持source correlation，且不生成Trellis授权状态。
- [x] 实验预算、10分钟估算/从提交起15分钟总deadline（覆盖pending+running）、deadline取消后的defer/blocker路由、remote、Git同步、destructive/protected-data、per-task branch和worktree合同均可由spec/skill执行并有边界测试。
- [x] 独立work-item extension在复杂task实施时展示当前item/executor/state，重启可恢复，plan变化会过期；缺失或失败时Trellis正常工作。
- [x] 上游context恢复后的prompt体积和重复读取有测量记录；本task未以未验证的compact补丁替代基线。
- [x] focused迭代检查、独立最终affected-scope检查和真实实验验证三者在spec与task报告中明确区分。
- [x] 156个known-error文件均有核验处置记录；有效通用pattern进入spec，原文件移入可追溯archive且不再自动加载。
- [x] `AI_branch_progress.md`和旧`ai_tasks/`归档并退出active prompt。
- [x] pi-app不再注册、显示、读写或文档化TaskTree专用能力，共享`workspace-json`回归通过。
- [x] legacy approval runtime已完整归档为只读`.local`快照；live文件保持不变，只有取得执行时精确批准后才删除。
- [x] Nimloth受影响的静态/CPU测试、Trellis validation、pi-app受影响测试/typecheck/build及完整diff检查完成；Web TS仅有已独立复现的外部worktree-layout基线，未启动未经预算授权的实验。
- [x] 所有已验收Nimloth child成果已汇入`task/rebuild-nimloth-trellis-prompts`，并在最终复审APPROVED后以非force fast-forward合入Nimloth `dev`；post-merge smoke通过。

## Out of Scope

- 修改上游Trellis npm package或维护项目fork。
- 改变CoT/state、训练、评估、模型或数据接口语义；这些属于各自domain/implementation spec。
- 在本task中启动正式训练、评估、收集、rollout或Slurm job。
- 保留typed approval作为授权系统，或迁移其receipt到新问题队列。
- 为非Pi平台创建Nimloth专属generated prompt/agent分叉。
- 在恢复并测量`0.6.16`前继续优化context压缩。

## Evidence

- `research/upstream-trellis-approval-compliance-audit-2026-09-02.md`
- `research/validation-cost-and-boundary-audit-2026-09-03.md`
- Pi skills documentation: `/workspace/pi-app/node_modules/@earendil-works/pi-coding-agent/docs/skills.md`
- Published Trellis `0.6.16` workflow and `trellis-meta` architecture references under `/home/user/.local/share/npm/lib/node_modules/@mindfoldhq/trellis/dist/templates/`
