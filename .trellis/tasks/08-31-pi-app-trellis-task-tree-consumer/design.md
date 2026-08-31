# Design — pi-app Trellis task-tree consumer

## Input contract

唯一Trellis语义输入是producer child发布的`trellis-work-item-dashboard-v1`。pi-app不拥有Markdown/parser/runtime业务规则。

Renderer额外拥有selected session的AppEvent/UI state：run status、active tool/status、elapsed、queue和tree tool details。

## Main process

- IPC request包含workspace和selected session identity。
- async spawn项目内Trellis dashboard command，设置明确context key。
- stdout size/time/schema guard；stderr和非零status成为可见reader issue。
- workspace containment和foreign cwd隔离。
- unknown schema整体拒绝，不部分猜测字段。

## Renderer merge

以`taskRef/workItemRef/contextKey/runId/toolCallId`关联declared assignment和observed activity。

优先级：

1. plan done/pending来自dashboard plan；
2. declared semantic state来自assignment；
3. observed tool/run只补充事实；
4. 冲突显示issue，不覆盖上层权威。

## UI structure

- 顶部“当前执行”摘要；
- 下方task parent/children tree；
- task展开为plan sections/items；
- current item显示executor children、evidence与next action；
- lifecycle badge显式标“任务阶段”；
- stale/conflict/legacy warning可展开查看原因；
- planning task提供“审查与审批”tab，展示raw PRD/design/implement、artifact hash/diff、scope/exclusions、validation、风险和parent/children；
- pending request显示明确授权类型及`approve | decline | comment`，并标示目标session/request；
- artifact changed/unknown correlation时禁用approval并解释原因。

## Approval response correlation

调查已证明Main的`request.id → source Worker`路由可防止基本答案串线，但现有Renderer丢弃`toolCallId/session`、使用single-slot store，且background/session-switch自动cancel。

Consumer改为workspace-scoped pending-attention registry：

```text
key = root/workspace + sessionFile + requestId + toolCallId
```

- `pendingById`保存并发questions/approvals；当前dialog只是其中一个view；
- background request进入attention list并标source session，不自动cancel；
- session switch只suspend当前dialog，request继续pending；
- “稍后处理”只关闭view；明确“拒绝/取消请求”才向source Worker发user terminal result；
- worker abort/exit等系统终止保留reason并显示为system-cancelled；
- 点击attention item可切source session或原地review，response始终按request id回原Worker；
- unknown、duplicate或late response fail closed并留下可诊断issue。

Task approval在此identity上额外绑定taskRef、approval kind和artifact hashes。

## Refresh

- workspace/session切换：重新请求dashboard；
- task lifecycle tool结束/run settle：节流刷新dashboard；
- runtime file变化：main watch或短轮询，具体机制在实现时按Electron约束选择；
- AppEvent：renderer本地实时更新observed activity；
- 保留manual refresh。

## Fallback

- dashboard command missing/unsupported：使用现有static reader并显示“无工作项实时数据”；
- runtime assignment absent：显示plan tree但无current item；
- malformed/unknown version：可见错误，不保留无标记旧数据；
- approval transport缺失或request correlation不可靠：review保持只读，不生成虚假decline。

## Isolation

pi-app implementation必须在独立worktree。当前main dirty文件与本child scope无关，writer不得覆盖或依赖它们。
