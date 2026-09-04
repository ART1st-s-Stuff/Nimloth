# Pre-commit review findings — 2026-09-04

独立只读reviewer检查了pi-app `task/pi-app-cross-worktree-review`完整未提交diff，对照已批准PRD/design给出`NEEDS CHANGES`，当前不能进入commit review。

## P1 findings

1. `npm run typecheck`的`fluent.tsx:107 TS2742`不能继续归为clean-base：相同依赖下的archived clean HEAD可通过，feature diff触发该失败。
2. WSL模式下Git返回Linux-native common-dir/worktree paths，新代码直接交给host `realpathSync`/`path.resolve`，Windows host无法正确校验。
3. Commit snapshot并发读取HEAD、patch和name-status，可能组合不同HEAD/index时点。
4. 非当前worktree文件打开复用通用shell IPC，没有在Main端重新校验worktree注册、realpath、common-dir与路径containment。
5. 评论只按review identity、file和`hunkIndex`绑定；HEAD不变但staged patch变化时可能串到其他hunk。

## 验证证据

- 新focused tests：27 passed / 4 files。
- Full unit：1091 passed、1 skipped、1 failed；rewind失败可在clean HEAD独立复现。
- Typecheck：TS2742失败；clean HEAD通过，因此归属于feature diff。
- ESLint `--quiet`：通过。
- 普通`git diff --check`因tracked CRLF文件报告CR字符；此前CRLF-aware检查通过，后续仍须显式报告该格式边界。
- 无新dependency、lockfile变化或staged文件。

## 经人类确认的replan

拆为三个独立批次：W8 Main正确性、W9 Renderer安全与评论identity、W10 typecheck/最终验证/独立复审。每批完成后停止，不得连续实施或commit。

完整review artifact：`/home/user/.pi/agent/sessions/--workspace-remote2-nimloth--/subagent-artifacts/outputs/3f157025-4af6-4e86-a035-e299b8778ad7/review-report.md`。
