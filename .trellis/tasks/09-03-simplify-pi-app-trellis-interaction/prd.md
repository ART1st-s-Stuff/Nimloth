# 需求 — 简化pi-app Trellis交互并删除TaskTree

## Goal

在pi-app保留通用、可寻址的跨session问题transport与generic adapter基础设施，同时彻底删除Trellis typed approval、旧dashboard和全局pi-task-tree专用能力；新的Trellis Task Browser与Work-item Activity继续只读工作。

## R1 — Generic question transport必须保留

- `select/confirm/input/editor/custom questionnaire/image review`请求保留producer workspace、session、tool-call/request identity并能回到原session。
- 切换workspace/session、进入background或稍后处理不得自动decline；明确回答、明确拒绝、timeout/abort和worker cancellation保持可区分。
- 保留pending recovery、dedupe、suspend/resume、terminal attention和`extension.respondUI`通用IPC。
- 不新增Trellis-specific method、kind、receipt、artifact hash或lifecycle mutation。

## R2 — 删除typed approval

- 删除worker bridge、main forwarding、renderer parser/store/host中所有`trellis_approval`分支、payload校验、correlation/supersede和专用UI。
- 删除shared approval types、fixtures、locale、tests及只服务typed approval的代码。
- 不读取archive/live approval runtime，不把approval退化为generic questionnaire伪装继续。

## R3 — 删除旧Trellis dashboard

- 删除`task.py dashboard` consumer、`TrellisDashboardV1` types/fixture/model、`WorkspaceTasksSidePanel`和旧state provider。
- Builtin Trellis adapter改用无状态`renderer-owned`provider marker和现有`workspace-tasks`component；不放宽generic adapter schema。
- `TrellisSidePanel`、Task Browser、Work-item Activity及其strict IPC保持唯一Trellis UI。

## R4 — 全局删除TaskTree

- 删除builtin `pi-task-tree` adapter、TaskTree panel/model、specialized registry mapping、mutation fixtures、docs和tests。
- 删除后catalog、settings、right-panel与probe中不再出现pi-task-tree。
- Trellis是Nimloth唯一task authority；pi-app不提供第二计划/status store。

## R5 — Generic adapter能力必须保留

- 保留`workspace-json` reader、generic adapter panel、adapter state/dispatch IPC和mutation command安全检查。
- 将原TaskTree专用mutation test改为neutral fixture，证明能力仍是generic。
- 保留非TaskTree adapters及其catalog顺序/override/probe语义。

## R6 — Docs与absence

- 更新中英文adapter docs、IPC/threat model/context/architecture文档，删除旧dashboard、typed approval和TaskTree陈述。
- Residual search对`trellis_approval`、`pi-task-tree`、`task-tree`、`workspace-trellis`、dashboard legacy symbols为零；历史任务artifact不在pi-app repository范围。

## Acceptance Criteria

- [ ] Stable Trellis panel无需旧dashboard state即可显示Documents与Activity。
- [ ] Generic跨session问题在切换后保持pending并回到原session；明确拒绝/timeout/abort/cancel可区分。
- [ ] Production code和active docs无typed approval kind/receipt/hash/supersede逻辑。
- [ ] pi-task-tree adapter/panel/model/mapping/docs/tests全部删除，catalog absence测试通过。
- [ ] `workspace-json`和generic adapter dispatch保持GREEN，并由neutral fixture覆盖。
- [ ] Task Browser与Work-item Activity focused tests保持GREEN。
- [ ] Affected unit、Node/Web typecheck、lint、build和独立review通过；clean-base既有失败单独归因。

## Out of Scope

- 不修改Pi core、Trellis npm/upstream、Nimloth producer或Task Browser语义。
- 不删除live approval runtime或历史Nimloth task artifacts。
- 不push、不合入默认branch、不cleanup历史worktree。
