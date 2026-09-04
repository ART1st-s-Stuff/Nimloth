# W1隔离与clean-base证据 — 2026-09-03

## 隔离结果

- Repository：`/workspace/pi-app`
- Parent integration worktree：`/workspace/pi-app/.worktree/task-rebuild-nimloth-trellis-prompts`
- Parent branch：`task/rebuild-nimloth-trellis-prompts`
- Exact base：`5dd9adc188569b4b1b4b0b6a0049bd1fb6bd39a7`
- Child worktree：`/workspace/pi-app/.worktree/task-pi-app-cross-worktree-review`
- Child branch：`task/pi-app-cross-worktree-review`
- Git common dir：`/workspace/pi-app/.git`
- Child status：clean

创建前parent integration worktree为clean且位于exact base。pi-app canonical的既有dirty/untracked内容仅作只读核验，未被stage、覆盖、删除或提交。

## Focused baseline

命令：

```bash
../../node_modules/.bin/vitest run \
  src/main/__tests__/git-workspace.test.ts \
  src/main/__tests__/git-workspace-async-contract.test.ts
```

结果：

```text
Test Files  2 passed (2)
Tests       7 passed (7)
Duration    851ms
```

当前base没有专门的Review panel/handler tests；W2与W5将分别补Main Git语义和Renderer交互RED。已知repository-wide typecheck、rewind和CSS问题本item未重复运行，最终检查时与parent validation audit中的clean-base证据比对。

## 已确认代码边界

- Shared contract/channels：`packages/shared/ipc-contract.ts`、`packages/shared/ipc-channels.ts`
- Main Git读取：`src/main/git-workspace.ts`
- Review handler/schema：`src/main/ipc/handlers/review.ts`、`src/main/ipc/schemas.ts`
- Renderer Review：`src/renderer/src/features/review/`
- 文案：`src/renderer/src/locales/en/review.json`、`src/renderer/src/locales/zh/review.json`

W1未修改任何pi-app源码或测试。
