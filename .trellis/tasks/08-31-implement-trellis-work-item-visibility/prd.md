# 实现 Trellis 工作项级实时可见性

## Goal

端到端实现“Pi session/run → Trellis task tree → implement.md work item → executor/runtime status → evidence”的可见性，使pi-app Trellis面板准确显示哪个主Agent或子代理正在完成哪一条计划项、状态如何，并让人类能在界面内清楚审查Agent新建task的PRD/design/plan及精确授权门禁，而不是只显示task生命周期`in_progress`。

## Source decision

经人类明确确认，第一版必须包含工作项级定位。只显示run/tool/elapsed而无法关联Trellis plan item的pi-app-only方案已被否决。

设计证据：

`../archive/2026-08/08-31-research-pi-app-trellis-visibility/research/pi-app-trellis-visibility-design-2026-08-31.md`

## Child ownership

1. `08-31-trellis-work-item-runtime-producer`
   - work-item ID与Markdown parser；
   - task tree/plan/runtime dashboard JSON合同；
   - hash-bound review package与typed approval request/receipt合同；
   - runtime assignment writer；
   - Pi extension显式cursor、heartbeat与subagent handoff；
   - workflow/spec/skill同步。
2. `08-31-pi-app-trellis-task-tree-consumer`
   - session-aware dashboard读取；
   - 合并pi-app AppEvent observed activity；
   - task tree/plan item/executor/evidence UI；
   - task review、artifact diff与typed approval UI；
   - `ask_user_question`跨session correlation缺陷调查/修复；
   - fallback、stale/conflict和cross-session验证。

## Requirements

- `task.json.parent/children/status`是task tree权威。
- `implement.md`heading、item文本、顺序和checkbox是plan权威。
- 新item使用task-local显式`[W-xxx]`；旧item使用标记为unstable的derived ID渐进兼容。
- `.trellis/.runtime/execution/<context-key>.json`只保存task/item引用、assignment与简短证据，不复制任务清单。
- runtime支持main/subagent、多assignment、working/verifying/delegated/waiting_human/waiting_external/blocked/failed、heartbeat/stale/conflict。
- pi-app必须展示task tree路径、当前work item、executor、elapsed、blocked target、evidence和next action。
- 新建task必须有独立review视图，直接呈现`prd.md`、`design.md`、`implement.md`、parent/children、改动仓库/路径、验收命令、风险、排除项和未决问题；不能要求人类从聊天摘要猜审批范围。
- 审批操作必须区分`批准规划/继续规划`、`批准实施`、`批准实验启动`、`批准commit`、`批准push/merge`等不同授权类型；禁止单一含糊的“Approve task”。
- 实施审批必须绑定taskRef、approval kind、artifact hashes、精确scope/exclusions和request/session identity；审批后artifact变化必须显示失效并重新审查。
- 人类必须能在review界面选择approve、decline或comment，并看到该响应将返回哪个Pi session/request。
- pi-app必须提供workspace-scoped待处理问题/审批队列；background request和session switch只挂起/转移可见性，不能自动伪装为user decline。
- Pending UI identity必须保留root/workspace、sessionFile、request ID和toolCallId；并发request不得由单一global slot覆盖。
- “稍后处理”、用户明确拒绝、session reset、background、worker abort必须是不同语义；系统取消保留reason，不能格式化成用户拒绝。
- observed tool/run事实与Agent-declared plan状态必须可区分。
- pi-app通过Trellis提供的versioned dashboard JSON读取task/plan/runtime，避免在两个仓库各自实现不同Markdown语义。
- `ctx.cwd`、resolved root与session/context key隔离；同session ID跨root不得串状态。
- 没有producer、无active item、malformed/stale runtime时安全降级，不猜first-unchecked、不解析assistant文本。
- 调查`ask_user_question`在其他session回答/切换/取消时误返回`User declined to answer questions`的可能性；若确认，先添加cross-session/request-correlation RED test再修复，若不能复现则保留证据缺口而不伪称修复。
- Pi TaskTree保持空；禁止读取、写入或镜像`.pi/task-tree/`。
- 第一版只要求Pi producer；runtime/dashboard schema保持平台中立，Claude/Codex producer不在本task范围。

## Authorization and exclusions

- 人类已批准创建本parent与两个child task；这只授权规划。
- 未取得最终implementation approval前，不修改Nimloth产品/adapter/spec/workflow或pi-app源码。
- 不启动实验、GPU、Slurm或远程job。
- 不commit、push、merge或清理既有dirty changes。
- pi-app当前`main`有无关dirty changes；未经批准不得在该worktree修改。实施需使用独立pi-app worktree或先由人类处理现状。
- 不批量改写现有376条plan item；迁移只覆盖新任务模板和经审查的当前active item。

## Acceptance Criteria

- [ ] Producer child交付versioned task/plan/runtime dashboard JSON及fixtures。
- [ ] Producer child交付稳定ID、legacy fallback、assignment writer、Pi cursor与subagent handoff。
- [ ] Consumer child在pi-app展示task tree、plan sections/items、current assignment、executor、state、evidence和elapsed。
- [ ] 新建task review页完整展示三份规划artifact、变更边界、validation、风险、parent/children和相对上次审查的变化。
- [ ] Typed approval request/receipt绑定artifact hashes和session/request；approve、decline、comment均准确回传且授权类型不混淆。
- [ ] Background/session-switch问题进入可恢复的workspace待处理队列；不会因未显示或切换session而自动变成user-declined。
- [ ] 并发问卷保留完整root/session/request/toolCall identity，不覆盖、不串线且可逐项恢复。
- [ ] artifact在审批后变化会使旧receipt失效，不能继续据此`task.py start`或执行更高风险操作。
- [ ] `ask_user_question`跨session/切换/取消/并发行为有源码结论和RED tests；确认的缺陷已修复，未确认则明确记录未复现与残余风险。
- [ ] `done`只来自checkbox；runtime/plan冲突显式显示且fail closed。
- [ ] working heartbeat、waiting跨turn、session shutdown、stale/orphan/conflict行为通过测试。
- [ ] 同root多session、同session多executor和同session ID跨root隔离通过测试。
- [ ] 无producer的普通Trellis项目保持静态panel fallback。
- [ ] 当前已有Trellis任务可使用legacy ID无修改读取；不发生批量migration。
- [ ] Pi TaskTree未被使用或改变。
- [ ] Nimloth adapter验证、pi-app unit/typecheck/lint/build和端到端人工probe全部通过。
- [ ] 完整双仓库diff、残余风险和提交分组经人类审查后才允许commit。

## Proposed implementation contract requiring approval

- Dashboard CLI：`python3 ./.trellis/scripts/task.py dashboard --json`；selected session通过明确`TRELLIS_CONTEXT_ID`传入。
- Heartbeat：live `working/verifying/delegated`每10秒更新；超过30秒且无live worker证据标记stale。`waiting_human/waiting_external/blocked`不依赖heartbeat，但持续显示age并在release、supersede或引用失效时结束。
- Typed evidence：`artifact | test | command | commit | job | approval | url`；只保存ref、最多200字符summary和timestamp，不保存完整日志、tool args或CoT。
- Review/approval：dashboard提供raw artifacts、结构化sections、artifact SHA-256、last-review diff和pending typed request；pi-app按钮通过精确session/request channel返回`approve | decline | comment`，Agent验证hash后才执行对应门禁。
- pi-app隔离worktree：branch `feature/trellis-work-item-visibility`，path `/workspace/pi-app/.worktree/feature-trellis-work-item-visibility`，base exact `1b2834dcb70bb593a4fcab8aa5357437c5632b0f`。该路径由人类在实施中明确指定；不包含当前dirty main的未提交内容，也不清理它们。
- Producer先锁定schema/fixtures；consumer随后实现。Parent task拥有跨仓库integration test与最终验收。
- 本批准若取得，只授权implementation与上述pi-app worktree创建；仍不包含commit、push或merge。
