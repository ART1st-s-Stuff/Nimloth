# 实施计划 — pi-app Trellis交互清理

## W1 — 隔离与RED

- [x] 创建并核验pi-app task worktree，base=`475c06b`，保持integration/canonical不变。
- [x] 固定Trellis direct panel、generic question transport、generic workspace-json/mutation及TaskTree/approval absence RED。

## W2 — 删除旧dashboard

- [x] 增加`renderer-owned`无状态provider并切换builtin Trellis adapter。
- [x] 删除workspace-trellis reader/provider、dashboard shared types/fixture/model和WorkspaceTasksSidePanel。
- [x] 保持Task Browser/Work-item Activity regression GREEN。

## W3 — 删除typed approval

- [x] 删除worker/main/renderer/shared中的`trellis_approval`payload、validation、correlation、presentation和tests。
- [x] 保持generic request identity、pending recovery、suspend/resume、response/decline/timeout/abort/cancel tests GREEN。

## W4 — 删除TaskTree

- [x] 将adapter mutation tests改为neutral workspace-json fixture。
- [x] 删除pi-task-tree builtin、loader entry、panel/model、registry mapping、专用docs/tests。
- [x] 保持generic adapter catalog/override/probe、workspace-json及dispatch GREEN。

## W5 — Docs与absence

- [x] 更新active中英文docs/locales；删除旧dashboard、typed approval和TaskTree陈述。
- [x] 对production/active docs运行residual symbol scan并记录精确允许项（若有）。

## W6 — 最终验证

- [x] 运行focused/affected/full unit、Node/Web typecheck、lint、build和CRLF-aware diff check。
- [x] 对clean parent复现无关baseline failure，独立review完整删除范围和generic transport数据流。
- [x] 展示完整pi-app diff和精确commit范围；未经批准不commit/merge/push。

## Guardrails

- 不修改Pi core、Trellis upstream、Nimloth producer或live runtime。
- 不删除generic extension UI、workspace-json、adapter dispatch或其他builtin adapters。
- 不在canonical/integration worktree实施；不cleanup历史worktree。
- 不commit、merge、push直到对应精确门禁。
