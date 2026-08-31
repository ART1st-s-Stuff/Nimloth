# Implementation Plan — pi-app consumer

## Bootstrap

- [x] [W-001] 核验approved pi-app worktree path/branch/base与clean status。
- [x] [W-002] 导入并锁定producer dashboard-v1 fixtures。
- [x] [W-003] 添加consumer RED tests覆盖session/task/item/status路径。
- [x] [W-004] 添加review package、artifact invalidation与approval correlation fixtures。

## Main/IPC GREEN

- [x] [W-010] 扩展IPC contract传递selected session identity。
- [x] [W-011] 实现async dashboard reader及schema/error/fallback guard。
- [x] [W-012] 实现root/session隔离和安全refresh/watch机制。

## Renderer GREEN

- [x] [W-020] 建立dashboard + AppEvent merge presentation model。
- [x] [W-021] 实现current execution summary。
- [x] [W-022] 实现task tree、plan sections/items和progress。
- [x] [W-023] 实现executor/subagent/evidence/next-action展开。
- [x] [W-024] 实现legacy/stale/orphan/conflict/concurrency UI。
- [x] [W-025] 实现planning task review页及raw PRD/design/implement、hash/diff、scope和validation展示。
- [x] [W-026] 实现typed approval approve/decline/comment、目标session提示和artifact-change invalidation。

## Compatibility and refactor

- [x] [W-030] 复用RunPanel/tree-card status语义，消除重复状态推导。
- [x] [W-031] 保留static Trellis fallback并修正“活跃任务”误导文案。
- [x] [W-032] 验证background/foreign session event不串panel。
- [x] [W-033] 确认无TaskTree依赖。
- [x] [W-034] 追踪`ask_user_question`完整IPC/store/dialog/result correlation并形成源码结论；证据见parent `research/ask-user-question-cross-session-correlation-2026-08-31.md`。
- [x] [W-035] 添加session switch、background、cancel/close、concurrent request和submit/cancel race RED tests。
- [x] [W-036] 将single-slot store重构为workspace-scoped pending-attention registry，background/switch保持pending。
- [x] [W-037] 端到端保留root/session/request/toolCall identity并区分later/user-decline/system-cancel语义。
- [x] [W-038] Main枚举live pending request并在Renderer reload后按workspace恢复attention registry；worker exit、late response和跨root隔离fail closed。
- [x] [W-039] 接收Trellis扩展真实typed approval custom request并将review决策精确回传source tool call。

## Check

- [x] [W-040] focused unit/component tests通过。
- [x] [W-041] `npm run typecheck`、`npm run lint`、`npm run build`通过。
- [ ] [W-042] 人工验证long tool、waiting、subagent、switch/reload/fallback。
- [ ] [W-043] 人工验证新task review、approve/decline/comment、artifact变更失效和跨session questionnaire。
- [x] [W-044] 独立P0/P1 review、完整diff和残余风险通过。
- [x] [W-045] 展示pi-app commit分组并取得批准；实现已提交为`4efdce1`。

## Guardrails

- 一名writer，仅在approved isolated worktree。
- 不修改/清理dirty main。
- 不自行实现不同Trellis parser/state machine。
- 不commit/push，除非另行明确批准。
