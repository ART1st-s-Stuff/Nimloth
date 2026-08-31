# 实现 pi-app Trellis task-tree consumer

## Goal

在pi-app Trellis侧栏中消费versioned Trellis dashboard与runtime assignments，合并Pi AppEvent，展示任务树、计划清单、当前work item、主/子代理、实时状态、elapsed、blocker、evidence和next action；同时提供Agent新建task的完整规划审查与精确typed approval界面。

## Parent/dependency

Parent：`08-31-implement-trellis-work-item-visibility`。

依赖producer child锁定dashboard-v1 fixture。Consumer不得自行重新定义Markdown grammar、done状态或runtime state machine。

## Requirements

- selected renderer session identity必须传入side-panel provider；不得使用Electron main隐式环境猜current task。
- 以异步child process调用Trellis dashboard JSON，避免现有5秒同步`execSync`阻塞main process。
- task tree支持parent/children、archived completed child、plan sections/items和progress。
- 每个planning task提供review页：直接查看PRD、design、implement、artifact hash/变更、目标仓库/路径、scope/exclusions、validation、风险、未决问题和child边界。
- 审批按钮明确区分规划、implementation、experiment launch、commit和push/merge；当前request未包含的权限不得由UI暗示或扩大。
- approve、decline、comment必须返回request绑定的workspace/root + session + toolCall/request ID；切换session后仍能看清响应目标。
- 提供workspace-scoped pending-attention queue；background question/approval必须可见且保持pending，session switch只suspend显示，不向Worker自动返回cancel。
- 并发request按root/session/request/toolCall存储；禁止single-slot覆盖。稍后处理、明确拒绝、session reset、background和worker abort必须保留不同语义。
- artifact hash变化后旧审批显示expired并禁止继续使用；不存在可靠correlation channel时review保持只读。
- 当前assignment突出task#item、executor、declared state、elapsed、blocked target、next action和typed evidence。
- AppEvent observed run/tool/subagent activity与declared assignment合并展示并明确来源。
- 支持main/subagent、多session、多executor和同item concurrency warning。
- 支持legacy unstable ID、stale/orphan/conflict、malformed dashboard和unknown schema。
- producer/dashboard不存在时回退当前静态Trellis panel，不把`in_progress`当实际working。
- task/plan/runtime变化与session切换触发安全刷新；foreign background event不得点亮当前session。
- 不修改Trellis task/plan，不使用Pi TaskTree；approval response只通过明确的Pi request channel或producer定义的runtime receipt合同返回。
- 已确认background/session-switch自动cancel路径及Renderer correlation/liveness缺口；实施需覆盖跨session回答、selected session切换、背景run、dialog cancel/close及并发request，并RED-first修复。精确`User declined to answer questions`字符串生产者仍是证据缺口，不得夸大。
- UI与现有Run panel/tree tool card复用presentation semantics，避免重复定义working/tool/thinking。

## Authorization

- 当前只批准规划，不批准修改`/workspace/pi-app`。
- pi-app当前实际branch为`main`，ahead 3且有无关dirty changes；未经明确批准不得原地修改。
- Approved implementation worktree：`/workspace/pi-app/.worktree/feature-trellis-work-item-visibility`，branch `feature/trellis-work-item-visibility`，base `1b2834dcb70bb593a4fcab8aa5357437c5632b0f`；人类在实施中明确指定该nested路径，已创建并验证clean。
- 不commit、push、merge或清理现有pi-app changes。

## Acceptance Criteria

- [ ] 正确展示task tree、plan sections/items和checkbox progress。
- [ ] Review页无需依赖聊天摘要即可审查三份artifact、scope/exclusions、validation、风险、parent/children和相对上次request的变化。
- [ ] Typed approval操作与exact artifact hashes/session/request绑定；不同approval kinds不互相授权。
- [ ] approve、decline、comment在跨session/切换/取消/并发场景准确完成目标request。
- [ ] Workspace pending-attention queue展示所有session待处理request；切换session或后台执行不自动取消。
- [ ] 稍后处理与明确拒绝分离，system cancel reason不会显示为user decline。
- [ ] selected session正确关联current task/item；root/session切换不串状态。
- [ ] main/subagent assignment、observed tool、elapsed、queue和evidence实时更新。
- [ ] waiting/blocked/stale/orphan/conflict/concurrency状态有明确UI和测试。
- [ ] lifecycle `in_progress`与runtime working语义完全分离。
- [ ] legacy ID、missing producer、unknown schema和reader失败安全fallback。
- [ ] renderer不直接解析assistant文本或猜first-unchecked。
- [ ] 无Pi TaskTree读写或状态镜像。
- [ ] focused unit tests、typecheck、lint、build通过。
- [ ] 开发模式人工验证长bash、approval wait、external wait、subagent、session切换和reload。
- [ ] `ask_user_question`调查有源码证据和可复现结论；确认bug则RED test与修复通过，未确认则记录测试覆盖和残余证据缺口。
