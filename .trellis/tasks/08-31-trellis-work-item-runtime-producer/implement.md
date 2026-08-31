# Implementation Plan — producer

## Contract RED

- [x] [W-001] 锁定plan grammar、ID uniqueness、legacy hash和dashboard fixtures。
- [x] [W-002] 锁定assignment schema、state machine、heartbeat与typed evidence。
- [x] [W-003] 添加parser/tree/runtime/dashboard失败矩阵测试。
- [x] [W-004] 锁定review-set hash、typed approval request/receipt和artifact-change invalidation fixtures。

## Parser and dashboard GREEN

- [x] [W-010] 实现task tree与implement.md parser。
- [x] [W-011] 实现dashboard-v1 projection及CLI JSON入口。
- [x] [W-012] 验证现有active/archived tasks和376条legacy items。
- [x] [W-013] 输出raw planning artifacts、hash/diff、scope/exclusions和pending approval projection。

## Runtime GREEN

- [x] [W-020] 实现atomic assignment store与root/context validation。
- [x] [W-021] 实现state transitions、heartbeat、stale、orphan和conflict。
- [x] [W-022] 实现evidence约束与无secret/无大payload guard。
- [x] [W-023] 实现approval request/receipt identity、kind和artifact-hash validation。

## Pi integration GREEN

- [x] [W-030] 注册work-item cursor工具并解析`ctx.cwd/contextKey`。
- [x] [W-031] 绑定main tool lifecycle和heartbeat。
- [x] [W-032] 绑定Trellis/generic subagent handoff与多executor。
- [x] [W-033] 验证reload、session shutdown和foreign cwd。

## Policy and refactor

- [x] [W-040] 更新workflow、tasks/progress spec及相关skills。
- [x] [W-041] 检查template hashes、`trellis platforms`和update dry-run divergence。
- [x] [W-042] 输出consumer fixture与调用合同。
- [x] [W-043] 注册真实typed approval请求工具，发布hash-bound Pi UI request并记录精确receipt。

## Check

- [x] [W-050] focused tests、TypeScript parse/bundle和Python static checks通过。
- [x] [W-051] 独立P0/P1 check通过，无TaskTree/memory/runtime-pointer越界。
- [x] [W-052] 展示完整diff并取得producer commit批准；实现已提交为`4d1c845c`。

## Guardrails

- 无批量plan migration。
- 无实验/远程job。
- 无commit/push，除非另行明确批准。
- 不手改`.trellis/.runtime/sessions/`或`.template-hashes.json`。
