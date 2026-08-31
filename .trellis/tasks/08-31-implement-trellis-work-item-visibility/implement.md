# Implementation Plan — end-to-end work-item visibility

## 1. Freeze cross-child contract

- [x] [W-001] 审查并锁定显式ID、legacy derived ID与plan grammar。
- [x] [W-002] 锁定dashboard-v1、runtime assignment-v1和typed evidence fixtures。
- [x] [W-003] 锁定heartbeat/stale/waiting/conflict状态转换。
- [x] [W-004] 取得pi-app隔离worktree exact branch/path/base批准。
- [x] [W-005] 锁定task review package、artifact hash/diff和typed approval request/receipt合同。

## 2. Producer child

- [x] [W-010] 完成Trellis plan parser、task tree和dashboard JSON。
- [x] [W-011] 完成runtime assignment atomic writer/read/validation。
- [x] [W-012] 完成Pi work-item cursor、tool heartbeat与subagent handoff。
- [x] [W-013] 同步workflow/spec/skills并验证旧task兼容。
- [x] [W-014] 发布consumer使用的versioned fixtures和producer验证证据。
- [x] [W-015] 在dashboard中提供raw planning artifacts、hashes、review diff和pending approval projection。

## 3. Consumer child

- [x] [W-020] 在隔离pi-app worktree接入session-aware async dashboard reader。
- [x] [W-021] 合并AppEvent observed activity与declared assignments。
- [x] [W-022] 实现task tree、plan item、executor、status和evidence UI。
- [x] [W-023] 实现fallback、stale/conflict/error和session/root隔离。
- [x] [W-024] 完成focused tests、可用typecheck、lint和build；canonical composite typecheck的nested依赖路径限制单独记录。
- [x] [W-025] 实现task review页，展示PRD/design/plan、边界、风险、validation、parent/children和变更。
- [x] [W-026] 实现hash-bound typed approval的approve/decline/comment及失效提示。

## 4. Integration

- [x] [W-030] 验证main Agent在plan item间select/update/release。
- [x] [W-031] 验证Trellis subagent与generic subagent assignment显示。
- [x] [W-032] 以runtime/bridge测试验证heartbeat、waiting-human/external、failure和stale；Electron人工long-tool probe受环境阻塞并单独报告。
- [x] [W-033] 验证workspace/session切换、reload和普通Trellis fallback。
- [x] [W-034] 确认实现未读取、写入或镜像Pi TaskTree；入口既有untracked payload未获清理授权，保持不动并报告。
- [x] [W-035] 调查并复现`ask_user_question`跨session/切换/取消/并发correlation，确认缺陷后先添加RED test再修复。
- [x] [W-036] 验证不同授权类型和artifact更新不会复用错误approval receipt。
- [x] [W-037] 实现并验证Trellis扩展到pi-app review页的真实typed approval request/receipt transport。

## 5. Finish

- [x] [W-040] 完成双仓库spec/check、完整diff和残余风险审查。
- [x] [W-041] 展示Nimloth与pi-app提交分组并分别取得commit批准。
- [ ] [W-042] 完成parent/children验收、memory/spec复核和finish-work。

## Approval gates

- 当前只批准任务创建与规划。
- Producer或consumer implementation均需最终计划审查和明确批准。
- pi-app worktree创建、任何commit/push/merge均需对应授权。
- 不包含实验、GPU、Slurm或远程job操作。

## Validation outline

```bash
python3 ./.trellis/scripts/task.py validate <parent-and-children>
# Producer focused parser/runtime/extension tests and adapter bundle checks
# Consumer: npm run test:unit -- <focused files>
# Consumer: npm run typecheck && npm run lint && npm run build
# End-to-end manual pi-app probe in approved isolated worktree
```
