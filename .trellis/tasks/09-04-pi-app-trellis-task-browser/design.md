# 技术设计 — pi-app Trellis任务浏览器

## 1. 架构边界

任务浏览器是pi-app的通用只读磁盘投影，分为三层：

```text
registered worktree discovery
  → Trellis task/artifact reader
  → Trellis panel task-browser UI
```

它不调用旧`task.py dashboard`，不读取approval receipt或work-item runtime，也不成为Trellis生命周期权威。后续删除旧dashboard/typed approval时，本浏览器的数据合同保持不变。

## 2. Worktree source identity

Main复用已实现并验证的`listGitReviewTargets(trustedWorkspace)`，只接受同Git common directory registered worktrees。每次读取前重新核验目标注册、realpath、`.git`入口和common-dir。

```ts
type TrellisTaskSource = {
  sourceId: string       // common-dir + worktree realpath的稳定hash
  path: string
  realPath: string
  branch: string | null
  headSha: string
  isCurrentWorkspace: boolean
}
```

当前source排序第一，其余按branch/path稳定排序。Source消失或身份变化时返回该source的issue，不读取其他同名task替代。

## 3. Task instance模型

```ts
type TrellisTaskInstance = {
  instanceId: string     // sourceId + task relative location的hash
  source: TrellisTaskSource
  taskRef: string
  location: string
  title: string
  description?: string
  status: string
  lifecycle: 'incomplete' | 'completed' | 'unknown'
  archived: boolean
  issue?: { code: string; message: string }
}
```

- `planning`、`in_progress`及其他非终态进入`incomplete`。
- `completed`进入`completed`。
- metadata缺失、JSON损坏或status未知进入`unknown`，可见显示，不静默归类。
- 同一`taskRef`位于不同source时产生不同`instanceId`，不合并、不按mtime或HEAD猜测“最新”。

## 4. Lazy archive读取

默认请求只扫描每个source的`.trellis/tasks/<task>/task.json`，跳过`archive/`，确保`未完成`首屏不被历史目录拖慢。

首次进入`已完成`tab时才请求archive：

```text
.trellis/tasks/archive/<YYYY-MM>/<task>/task.json
```

Active目录中status=`completed`的task和archive task共同进入completed结果；同一个磁盘实例只返回一次。Archive月份、task目录和文件均按稳定顺序读取，并设置数量/单文件大小上限；溢出或损坏显示issue。

## 5. Artifact allowlist与读取

只允许以下相对artifact：

```text
task.json
prd.md
design.md
implement.md
progress.md
implement.jsonl
check.jsonl
research/*.md
```

Main接收`instanceId + artifactId`，从最新source/task inventory解析真实路径，不接受Renderer提交absolute path。读取前：

1. 重新发现source并核验worktree identity；
2. 重新解析instance location；
3. 拒绝`..`、absolute path、symlink escape和不在allowlist的扩展名；
4. 核验realpath仍位于该task目录；
5. 应用单文件byte limit并返回UTF-8文本或可见错误。

Missing artifact返回`present=false`，不生成占位内容。`research/`只列出当前task目录直接/递归Markdown文件，并稳定排序。

## 6. IPC与Main实现

新增与旧dashboard分离的只读合同：

```text
trellisTaskBrowser.list
  request: { includeArchive: boolean }
  response: { sources, tasks, issues }

trellisTaskBrowser.listArtifacts
  request: { instanceId }
  response: { artifacts, issues }

trellisTaskBrowser.readArtifact
  request: { instanceId, artifactId }
  response: { present, kind, content?, issue? }
```

所有handler从`getTrustedWorkspaceRoot()`取得repository，不接收workspace root。Zod schema为strict；Main每次请求重新验证source/instance，错误fail closed。

读取逻辑放入独立`trellis-task-browser-reader.ts`，避免继续扩展将被后续cleanup的dashboard schema/parser。

## 7. Renderer UI

`WorkspaceTasksSidePanel`增加独立`任务`入口，与现有运行状态/审批区域分离：

- 生命周期tabs：`未完成`（默认）、`已完成`；unknown实例在当前tab顶部可见标记。
- Source filter默认`全部worktrees`，可选单个branch/path。
- Task行显示title、status、source branch/path、archive标记和issue。
- 选择task后显示artifact tabs；Markdown使用既有renderer并可切换只读source，JSON/JSONL只显示source。
- 缺失、损坏、过大或source消失使用明确空态/错误态。
- 快速切换source/task/artifact时，request identity包含workspace、instanceId、artifactId和generation；旧响应不得覆盖新选择。

不提供编辑、状态更新、approval、Git或workspace切换按钮。

## 8. 与后续pi-app清理的关系

本task只新增独立task browser合同。旧`TrellisDashboardV1`、ApprovalCard、execution projection和TaskTree删除由`09-03-simplify-pi-app-trellis-interaction`处理。新browser不得import这些旧类型；后续删除旧dashboard时只需保留browser reader/IPC/UI。

## 9. 测试

### Main

临时Git repository建立两个worktree及重复task refs，覆盖：

- current/other source发现与稳定排序；
- 同名task多版本保持独立；
- incomplete/completed/unknown分类；
- archive lazy读取；
- malformed task、missing artifact、oversize；
- allowlist、absolute/`..`、symlink escape；
- source删除、realpath替换、foreign common-dir；
- handler拒绝Renderer注入root/path。

### Renderer

覆盖默认未完成tab、首次completed加载archive、source标签/filter、同名多版本、artifact renderer、missing/error状态及快速切换旧响应隔离。断言所有操作均不调用workspace/session/worker/Trellis mutation或Git mutation IPC。

## 10. 修改范围与回滚

预计修改shared IPC contracts/channels、Main task browser reader/handler、Trellis side panel及中英文文案/tests。不修改Trellis task文件、旧approval删除逻辑、generic question transport、workspace/session/worker或Git mutation。

回滚只删除新增reader/IPC/browser UI并恢复panel入口；没有数据迁移或repository mutation需要回滚。
