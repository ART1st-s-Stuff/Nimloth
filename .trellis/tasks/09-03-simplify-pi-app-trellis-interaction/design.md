# 技术设计 — pi-app Trellis交互清理

## 1. 目标图

```text
Trellis adapter
  panelComponent: workspace-tasks -> TrellisSidePanel
  stateProvider: renderer-owned -> {}
  Documents -> strict task-browser IPC
  Activity  -> strict work-item projection IPC

generic extension UI queue
  producer workspace/session/request -> pending/suspended -> exact response route

generic adapters
  workspace-json + dispatch command
```

不存在dashboard/approval/TaskTree业务层。

## 2. Trellis panel registration

保持`AdapterSidePanel.stateProvider`必填，避免扩大public adapter schema。Main registry新增`renderer-owned`provider，返回空对象且不读workspace；Trellis adapter改用该marker。专用`TrellisSidePanel`不调用`adapter.sidePanel.getState`。删除`workspace-trellis`provider及其reader后，generic adapter状态API保持兼容。

## 3. 删除顺序

1. RED固定Trellis renderer-owned registration及Task Browser/Activity持续可用。
2. 删除`WorkspaceTasksSidePanel`、dashboard model/types/fixture/reader/provider。
3. RED固定generic UI切换、pending recovery、明确拒绝/timeout/abort/cancel语义；删除`trellis_approval`各层分支。
4. 用neutral `workspace-json` fixture固定generic mutation；删除pi-task-tree adapter、panel/model及registry mapping。
5. 更新active docs/locales，运行residual absence scan。

## 4. Typed approval边界

只删除method/kind=`trellis_approval`及其payload-specific validation/presentation。以下保持：generic request identity、producer root/session map、custom UI opaque payload transport、pending queue、extension response/cancel、worker termination cleanup。不得把typed approval payload自动转成questionnaire。

## 5. TaskTree边界

删除builtin catalog入口和specialized renderer；不删除`workspace-json-state.ts`、`GenericAdapterSidePanel`、`adapter-side-panel-command.ts`或IPC contracts。原mutation test使用临时neutral adapter fixture。

## 6. 测试

- Trellis adapter catalog/renderer-owned provider/direct stable panel；
- Task Browser与Work-item Activity regression；
- generic extension UI enqueue/recovery/session switch/response/decline/abort/cancel；
- generic adapter workspace-json read与neutral mutation；
- builtin catalog absence及residual symbol scan；
- affected docs links/JSON parse、unit/type/lint/build。

## 7. 分支与回滚

pi-app worktree从integration `475c06bcdcf48b28499d59550082b02342545041`创建`task/simplify-pi-app-trellis-interaction`。不修改canonical或integration worktree。回滚为单一pi-app child commit；Nimloth仅记录task artifacts。
