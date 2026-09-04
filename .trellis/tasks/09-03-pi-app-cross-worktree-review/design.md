# 技术设计 — pi-app Commit Review与Merge Review

## 1. UI结构

Review面板拆成两个顶层模式，状态彼此独立：

```text
Review
├─ Commit Review
│  ├─ Worktree selector
│  └─ Staged diff
└─ Merge Review
   ├─ Base branch selector
   ├─ Compare branch selector
   └─ Committed diff since merge-base
```

默认进入Commit Review，默认worktree为当前workspace root。选择其他目标只改变Review面板状态，不调用`workspace.open`、`activateWorkspace`、worker/session API或Git checkout。

## 2. 数据合同

### 2.1 Review目标目录

```ts
type ReviewWorktree = {
  path: string
  realPath: string
  branch: string | null
  headSha: string
  isCurrentWorkspace: boolean
  lockedReason?: string
}

type ReviewBranch = {
  ref: string
  displayName: string
  sha: string
  kind: 'local' | 'remote'
}
```

Main从当前trusted workspace出发读取目标，Renderer不得提交任意repository/path/ref。

### 2.2 Commit Review响应

```ts
type CommitReviewSnapshot = {
  mode: 'commit'
  repositoryCommonDir: string
  target: ReviewWorktree
  stagedRaw: string
  statusSummary: string
  mutable: boolean
  snapshotKey: string
}
```

`stagedRaw`只来自目标worktree中的：

```bash
git diff --cached
```

`mutable`只有在目标realpath等于当前trusted workspace realpath时为`true`。其他worktree即使属于同一repository也始终为`false`。

### 2.3 Merge Review响应

```ts
type MergeReviewSnapshot = {
  mode: 'merge'
  repositoryCommonDir: string
  base: ReviewBranch
  compare: ReviewBranch
  mergeBaseSha: string
  committedRaw: string
  snapshotKey: string
}
```

Main重新解析两个ref的commit SHA，计算唯一merge-base，再读取：

```bash
git diff <merge-base-sha>..<compare-sha>
```

响应不包含worktree status、stagedRaw、unstagedRaw或untracked patch。

## 3. Main进程边界

在`src/main/git-workspace.ts`增加三个只读操作：

1. `listGitReviewTargets(workspaceRoot)`：
   - 运行`git worktree list --porcelain`读取registered worktrees；
   - 运行`git for-each-ref`读取local和remote-tracking branches；
   - 返回规范化path/ref/SHA，不返回任意shell文本作为identity。
2. `readCommitReviewSnapshot(workspaceRoot, selectedWorktree)`：
   - 将selected path与刚重新读取的worktree registry精确匹配；
   - 核验realpath、Git common dir和`.git`入口；
   - 只读取selected worktree index与HEAD。
3. `readMergeReviewSnapshot(workspaceRoot, baseRef, compareRef)`：
   - 两个ref都必须精确存在于重新读取的branch列表；
   - 用`rev-parse --verify --end-of-options <ref>^{commit}`绑定SHA；
   - 计算merge-base并读取committed diff。

所有Git调用继续使用`execFile`参数数组，不经过shell。任一步目标消失、ref变化、存在多个merge-base或Git common dir不一致时返回可见错误，不回退当前workspace。

## 4. IPC分离

新增三个只读IPC，而不是把不同含义塞入现有`review.getDiff`：

```text
review.listGitTargets
review.getCommitDiff
review.getMergeDiff
```

- `review.listGitTargets`不接收repository路径，始终使用当前trusted workspace。
- `review.getCommitDiff`只接收列表中返回的worktree identity。
- `review.getMergeDiff`只接收列表中返回的完整branch refs。
- Main每次请求都重新校验目标，不能信任Renderer缓存。

现有mutation IPC保持不变，但Renderer只有在Commit Review的`mutable=true`时才渲染unstage和commit入口。Main原有`authorizeTrustedCwd`继续作为最后一道mutation校验；新只读IPC不能调用mutation helper。

## 5. Renderer结构

将当前`ReviewPanel`拆分为：

```text
review-panel.tsx                # 顶层mode切换
commit-review-panel.tsx         # worktree selector + staged-only diff
merge-review-panel.tsx          # Base/Compare selector + committed diff
use-review-targets.ts           # 读取worktree/branch候选
use-commit-review-data.ts       # mode+worktree identity隔离
use-merge-review-data.ts        # mode+base+compare identity隔离
```

可复用现有`parseGitDiff`与diff展示组件，但需要增加显式`readOnly`/`mutationEnabled`参数：

- 当前worktree Commit Review：`mutationEnabled=true`，保留unstage及commit；
- 其他worktree Commit Review：`mutationEnabled=false`；
- Merge Review：`mutationEnabled=false`。

Commit Review不调用现有`groupReviewFiles`的unstaged路径，不合成untracked patch。Merge Review直接解析`committedRaw`。

快速切换target时，每次请求使用以下identity防止旧响应覆盖新选择：

```text
commit:<common-dir>:<worktree-realpath>:<head-sha>
merge:<common-dir>:<base-sha>:<compare-sha>:<merge-base-sha>
```

## 6. 评论与文件打开

Review评论至少绑定：

- mode；
- repository common-dir identity；
- Commit Review的worktree realpath与HEAD SHA，或Merge Review的Base/Compare/merge-base SHA；
- file path和hunk identity。

选择其他worktree时，“在编辑器打开”使用经过Main验证后的target root加repository-relative path。Merge Review中的文件可能不存在于当前workspace工作目录，第一版不提供“直接打开当前文件”；只显示diff，避免打开错误版本。

## 7. 刷新策略

- 打开面板、切换mode、切换worktree或切换branch时立即刷新；
- 当前workspace继续响应现有Git watcher；
- 非当前worktree和branch第一版使用显式刷新，不增加跨目录watcher；
- 请求进行中再次切换时丢弃旧identity响应，不串入新目标。

## 8. 测试边界

### Main/Git tests

使用临时repository建立至少两个branch和两个worktree，覆盖：

- 当前与非当前worktree的staged diff严格隔离；
- unstaged/untracked内容不出现在Commit Review；
- 不同Git common dir路径被拒绝；
- worktree在list后被删除时fail closed；
- local/remote branch解析；
- merge-base diff只包含Compare引入的committed变化；
- ref消失、Base=Compare及无可用merge-base；
- 所有读取不改变HEAD、index、workspace config或branch checkout。

### Renderer tests

覆盖：

- 默认Commit Review与当前worktree；
- 两个模式显示不同selector；
- 非当前worktree和Merge Review隐藏mutation controls；
- 当前worktree保留unstage/commit；
- 空staged不回退unstaged；
- 快速切换不会显示旧target结果；
- workspace/session/worker状态不因Review target变化而改变。

## 9. 实施文件范围

预计仅涉及：

- `packages/shared/ipc-contract.ts`与IPC channels；
- `src/main/git-workspace.ts`、review handler及其tests；
- `src/renderer/src/features/review/`及其tests；
- 中英文Review文案。

不修改workspace/session/worker实现、Trellis adapter、TaskTree、generic question transport或其他side panel。

## 10. 回滚

删除新增只读IPC与两个新panel/hook，恢复原`ReviewPanel`调用`review.getDiff`即可。数据不落盘，不创建branch/worktree，不迁移用户状态，因此不需要数据回滚。
