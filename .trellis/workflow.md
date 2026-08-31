# Nimloth 开发工作流

Trellis是Nimloth唯一的当前开发任务系统。本工作流保留Trellis原生的`planning` → `in_progress` → `completed/archive`状态合同，同时执行[`AGENTS.md`](../AGENTS.md)与[`.trellis/spec/`](spec/)中的项目安全、实验、进度和审查门禁。

## 核心原则

1. **授权缺失或存在不确定性时必须停止。** 若只读证据可以解决问题，先研究；否则询问人类。
2. **实施前必须规划。** 创建任务只批准规划。经审查后启动任务才批准实施。实验启动还需要单独审批。
3. **只注入经选择的证据。** 整理相关spec、研究和单条known error；禁止注入整个known-error库。
4. **按所有权持久化。** 任务保存当前工作，工作空间日志保存会话，curated memory保存经审查的可复用经验，旧任务/问题文件只保留历史。
5. **提交前必须审查。** 工作提交前展示完整范围和验证；只有结束审查完成后才能执行归档/会话日志记账。

## Trellis 系统

- 任务：`.trellis/tasks/<task>/`，包含`task.json`、`prd.md`、可选的`design.md`/`implement.md`、研究和JSONL上下文清单。
- 规范：`.trellis/spec/{governance,experiments,python,domains,guides}/`。
- 工作空间：`.trellis/workspace/`中的会话日志。
- Curated memory：`.memory/`、`.local/memory/`和`.agents/skills/memory/`；禁止直接编辑memory JSONL。
- 原始召回：`trellis mem`是未经核验的对话搜索。

常用命令：

```bash
python3 ./.trellis/scripts/task.py current --source
python3 ./.trellis/scripts/task.py create "<title>" --slug <name>
python3 ./.trellis/scripts/task.py start <task>
python3 ./.trellis/scripts/task.py validate <task>
python3 ./.trellis/scripts/get_context.py --mode packages
python3 ./.trellis/scripts/get_context.py --mode phase --step <X.Y>
```

## Phase Index

```text
Phase 1: Plan    → classify risk, obtain task consent, research, persist and review artifacts
Phase 2: Execute → implement approved scope, apply progress/experiment gates, verify repeatedly
Phase 3: Finish  → full-scope check, memory/spec review, complete-diff review, work commits, wrap-up
```

### Task threshold

多文件或歧义实施、项目规则/工作流变更、任何实验、GPU、Slurm、远程长job、收集、评估或rollout-train，以及需要持久设计、交接或跨session的工作，都必须使用Trellis任务。

创建任务前必须先征得创建同意。如果人类拒绝，广泛工作必须停止；仅可继续一轮解释、不产生持久决策的只读查询，或边界明确、低风险的小修改。

Trellis是唯一的任务权威。Pi TaskTree必须保持空，不得复制状态、优先级、层级、focus、验收标准或backlog。

[workflow-state:no_task]
当前没有活动任务。先对请求分类，并在创建Trellis任务前征得任务创建同意。
多文件/歧义工作、规则/工作流变更、实验/远程job和持久或跨session工作都必须使用任务。若人类拒绝，禁止继续广泛工作；只允许解释、不产生持久决策的只读查询，或边界明确、低风险的小修改。
Pi TaskTree必须保持空；Trellis是唯一的开发任务权威。
[/workflow-state:no_task]

### Phase 1: Plan

- 1.0 创建任务 `[required · once]`
- 1.1 探索要求与风险 `[required · repeatable]`
- 1.2 证据研究 `[optional · repeatable]`
- 1.3 配置已选上下文 `[required · once for sub-agent platforms]`
- 1.4 最终规划审查并启动任务 `[required · once]`
- 1.5 完成标准

[workflow-state:planning]
必须停留在规划阶段。创建任务不等于实施审批。
复杂工作必须完成并审查`prd.md`、`design.md`和`implement.md`；选择相关spec、源码证据、单条known error和经核验的memory。为子代理平台整理两份JSONL清单。
实验任务必须设置`meta.kind=experiment`、完成实验合同，并保留单独且明确的启动审批门禁。
运行`task.py start`前，必须请人类批准最终产物。
[/workflow-state:planning]

[workflow-state:planning-inline]
必须停留在规划阶段。创建任务不等于实施审批。
必须完成所需任务产物、已选spec/源码/known-error/memory证据，以及适用的实验合同。内联平台可以跳过JSONL整理，但编辑前必须加载同一批证据。
运行`task.py start`前，必须请人类批准最终产物。
[/workflow-state:planning-inline]

### Phase 2: Execute

- 2.1 实施 `[required · repeatable]`
- 2.2 质量检查 `[required · repeatable]`
- 2.3 回滚或重新规划 `[on demand]`

[workflow-state:in_progress]
只能实施经审查的任务范围。先读取curated JSONL条目，再读取`prd.md`、`design.md`和`implement.md`；编辑前检查相邻源码/测试。
主会话流程：`trellis-implement` -> `trellis-check` -> memory/spec审查 -> 完整diff与验证审查 -> 获批的工作提交 -> `/trellis:finish-work`。
子代理递归保护：当前`trellis-implement`或`trellis-check` agent必须直接完成自身职责，禁止再次启动这两种角色。
出现实质里程碑后必须执行`on-progress`。实验必须具备完整合同和单独、明确的启动审批；完成/失败/取消/暂停时必须触发`on-experiment-end`。
最终检查必须覆盖所有受影响的spec层，并包含相关known errors、链接/config/hooks和语义证据。
[/workflow-state:in_progress]

[workflow-state:in_progress-inline]
只能实施经审查的任务范围。编辑前必须加载任务产物，以及相关governance/domain/experiment/Python specs、源码证据、单条known error和经核验的memory。
流程：开发前审查 -> 编辑 -> 全范围检查 -> memory/spec审查 -> 完整diff与验证审查 -> 获批的工作提交 -> `/trellis:finish-work`。
出现实质里程碑后必须执行`on-progress`；实验仍需单独启动审批，并强制执行结束门禁。
[/workflow-state:in_progress-inline]

### Phase 3: Finish

- 3.2 调试复盘 `[on demand]`
- 3.3 进度、memory与spec审查 `[required · once]`
- 3.4 完整diff审查与工作提交 `[required · once]`
- 3.5 Finish-work审查与记账 `[required · once]`

[workflow-state:completed]
工作提交已经完成。必须展示finish-work的归档/会话日志影响并取得人类验收，之后`/trellis:finish-work`才能执行自动记账提交。
[/workflow-state:completed]

### Phase rules

1. 必须按顺序执行各步骤；禁止跳过标记为required的门禁。
2. 要求、范围、语义或授权发生变化时，必须返回规划阶段。
3. 任务产物已存在只表示无需重复创建，不代表可以跳过对当前内容的审查。
4. 未经人类明确批准，任务产物不得覆盖人类prompt、`AGENTS.md`或硬性spec。
5. 禁止在不匹配的门禁阶段启动实验、修改受保护文件、编辑memory JSONL或commit。

## Phase 1: Plan

目标：在实施前建立已授权、有来源证据支持且可验证的工作。

#### 1.0 Create task `[required · once]`

取得任务创建同意后运行：

```bash
python3 ./.trellis/scripts/task.py create "<title>" --slug <name>
```

此时禁止运行`start`。只有可独立验证的交付物才能使用父/子任务，并且必须在子任务产物中写明依赖顺序。如果`task.py current --source`已指向获批任务，则跳过创建。

#### 1.1 Requirements and risk exploration `[required · repeatable]`

在`prd.md`中记录要求、排除项、授权、验收标准和未决决定。复杂或高风险工作还必须包含：

- `design.md`：所有权、合同、备选方案、兼容性、回滚；
- `implement.md`：有序修改、验证命令、审查/审批门禁。

规划期间：

- 必须应用[governance](spec/governance/index.md)和[调查/不确定性](spec/guides/investigation-and-uncertainty.md)；
- 必须识别受保护文件、main/worktree风险、CoT/state语义、实验/远程操作和无法识别的dirty文件；
- 证据无法解决必要决定时，只提出一个清晰问题；
- 未经明确批准，禁止使用临时替代或近似机制。

实验任务必须设置`task.json.meta.kind = "experiment"`，并加入[实验任务合同](spec/experiments/task-contract.md)中的每个字段。实施审批仍不授权实验启动。

#### 1.2 Evidence research `[optional · repeatable]`

研究当前源码、测试、配置、数据/元数据、模块READMEs、外部参考或runtime行为。把持久发现写入任务的`research/`目录。必须区分已核验事实、假设和未决决定。

使用[known-error索引](../ai_rules/known_errors/README.md)，按修改路径/概念选择单条相关记录。只有可能节省调查时才搜索curated memory；依赖前必须执行`get`并重新阅读证据。原始`trellis mem`输出只能作为线索。

#### 1.3 Configure selected context `[required · once for sub-agent platforms]`

在`implement.jsonl`和`check.jsonl`中整理仓库相对路径的`{"file":"...","reason":"..."}`行，内容限于相关spec、任务研究和已选known errors。跳过没有`file`的seed行。禁止仅因即将修改就列入源码文件，也禁止注入全部known errors。

- 实施上下文：完成修改所需的合同/证据；
- 检查上下文：审查质量/语义所需的合同/证据。

内联平台必须在编辑前直接加载同一批证据，可以跳过JSONL整理。

#### 1.4 Final planning review and activate `[required · once]`

展示最终范围、任务产物、假设/未决决定、已选证据、验证计划，以及受保护文件/实验门禁。必须请求人类实施审批。只有获批后才能运行：

```bash
python3 ./.trellis/scripts/task.py start <task>
```

对实验而言，启动任务只批准实施/准备。精确启动合同必须在执行前再次取得单独审批。

#### 1.5 Completion criteria

- `prd.md`存在且符合当前人类要求；
- 复杂工作已有经审查的`design.md`和`implement.md`；
- 已选择相关spec/源码/known errors以及实际依赖的memory证据；
- 子代理平台已有curated implement/check JSONL行；
- 人类已批准实施，且任务状态为`in_progress`；
- 实验任务已有`meta.kind=experiment`和完整合同，但尚未启动。

## Phase 2: Execute

目标：只实施经审查的范围，并建立最新验证证据。

#### 2.1 Implement `[required · repeatable]`

主会话通常启动`trellis-implement`，其prompt首行为`Active task: <path>`。实施agent加载JSONL条目和任务产物，阅读相邻源码/测试，直接编辑并运行针对性检查。它禁止启动另一个实施或检查agent。

Pi的work-item runtime producer可用时，Agent必须维护显式cursor：

- 开始或切换实质`implement.md`计划项前，调用`trellis_work_item select`并提供精确task/item引用；禁止猜第一个未勾选项或从assistant文本/tool名推断。
- 状态进入验证、委派、等待、阻塞或失败时立即`update`/`block`；证据只保存typed ref与短summary。
- `trellis_subagent`委派必须提供显式`<taskRef>#<workItemRef>`；generic subagent只能继承已声明primary item，不能改为另一item。
- 完成只由`implement.md`checkbox表达：先更新checkbox并审查diff，再`release`或选择下一项。runtime工具不得直接标记done。
- 非Pi平台或producer不可用时不得伪造runtime；继续以task artifact为权威并明确说明没有live assignment projection。

每次编辑前：

- 核验branch/worktree和完整dirty状态；
- 阅读每个受影响层索引中的开发前检查清单及所属模块文档；
- 保留无关修改与受保护内容；
- 如果源码与规划冲突或范围扩大，必须停止并返回Phase 1。

完成可验证子任务、关键修复、重要设计决定、实验阶段、规则变更或推翻既有结论后，必须立即执行`.agents/skills/on-progress/`。当前细节留在Trellis任务中；只有确有必要时才添加简短branch里程碑。禁止创建新的`ai_tasks/ai_progress/`文件。

启动实验前必须执行`on-experiment-start`，展示精确最终合同/资源/命令并取得明确的人类批准。启动后持续监控到健康运行。任何终止状态都必须在当前对话中执行`on-experiment-end`。

#### 2.2 Quality check `[required · repeatable]`

主会话通常启动`trellis-check`；实施子代理禁止自行启动它，只能报告需要该检查。必须审查并修复：

- 任务PRD/设计/计划合规性与范围；
- 每个受影响spec索引中的质量检查；
- 已选known errors和最终diff概念搜索；
- 针对性测试以及受影响的跨模块/全范围检查；
- 适用的config/JSON/TOML/YAML解析和Python/TypeScript/shell语法；
- 工作流变更涉及的Markdown链接、generated-adapter合同、任务/上下文验证和`git diff --check`；
- 受保护文件、memory hashes、产品/实验/输出边界和无法识别的dirty路径。

必须报告精确命令与结果。缺失依赖、不可用硬件或未运行的平台reload/probe都必须明确标记为未验证项。

#### 2.3 Roll back or re-plan `[on demand]`

- 要求/设计缺陷或需要新授权 → 更新任务产物、请求审查，再重新启动实施；
- 实施缺陷 → 只回滚本任务修改，保留无关工作，然后重新实施；
- 证据缺失 → 执行只读研究并持久化发现；
- 存在不安全/歧义条件 → 停止并询问。

## Phase 3: Finish

目标：使完整结果可审查，在不重复的前提下保留有用知识，并分离工作提交与记账。

#### 3.2 Debug retrospective `[on demand]`

同一问题需要反复修复时，必须归类根因以及此前尝试失败的原因。只有已经发生且确认的失败才能新增known error。稳定且强制的预防规则应提升到所属spec；禁止把memory或known error当作任务日志。

#### 3.3 Progress, memory, and spec review `[required · once]`

- 完成`implement.md`/任务检查清单，并记录验证证据/未决风险。
- 只有本任务改变branch级状态时，才新增或更新一条简短`AI_branch_progress.md`里程碑。
- 评估每条使用过的memory：执行`get`、重新阅读证据；只有内容正确且确实有帮助时才upvote。必须通过skill纠正错误记录。禁止运行human-only审批。
- 只有紧凑、可复用且spec/文档尚未清晰表达的经验才能新增memory。
- 稳定的跨任务规则或新建立的合同必须更新spec。模块局部行为留在模块README。

#### 3.4 Complete-diff review and work commits `[required · once]`

运行最终全范围检查并查看：

```bash
git status --porcelain
git diff --stat
git diff --check
git log --oneline -5
```

一次性展示：

- 按用途分组的全部修改文件；
- 与任务验收标准的语义对应关系；
- 精确验证证据和未验证项；
- 有意的生成内容/Trellis update冲突；
- 排除在工作提交之外的未知dirty文件；
- 建议的逻辑工作commit分组/消息。

必须请求一次性的人类commit审批。获批后只能提交列出的工作组；禁止amend、push、merge，也禁止混入归档/会话日志记账。如果人类拒绝或选择手工提交，则停止commit并遵循其决定。平台/任务角色的禁止提交限制具有更高优先级。

#### 3.5 Finish-work review and bookkeeping `[required · once]`

工作提交完成且worktree处于已审查状态后，展示`/trellis:finish-work`将归档和记录的内容，包括自动记账提交（`session_auto_commit: true`）。必须在调用finish-work前取得人类验收。归档和会话日志提交只能发生在工作提交之后，禁止早于完整diff审查。

## Platform consistency and upgrade boundary

- Pi：`.pi/extensions/trellis`、prompts和agents。该extension使用callback/session `ctx.cwd`作为活动根目录，并将根目录纳入context cache key；`process.cwd()`只能作为启动回退。Adapter变更后，必须先运行`/reload`再执行实时子代理probe。
- Claude Code：`.claude/hooks`、commands、agents和`.claude/skills -> ../.agents/skills`。
- Codex：`.codex/hooks`、agents、config和共享`.agents/skills`；启用/批准全局原生hook是用户机器操作。
- 仓库自有Nimloth规则位于`.trellis/spec/`和非`trellis-*`项目skills中。禁止把私有规则放入上游随附的`trellis-*` skills。
- `.trellis/workflow.md`和Pi根目录adapter有意偏离生成的默认模板。每次`trellis update --dry-run`都必须审查冲突；禁止手工编辑`.trellis/.template-hashes.json`或runtime session state。
