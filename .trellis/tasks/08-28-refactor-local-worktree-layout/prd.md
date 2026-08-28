# PRD：以 `nimloth` 为唯一开发根重构本地 worktree

## Goal

降低 Pi Agent GUI 代码审阅中的多 sibling-worktree 干扰，把日常本地开发收敛到 `/workspace/remote2/nimloth`；只有确有隔离必要时，才在该根目录的 `.worktree/` 下创建 linked worktree。

## User value

- Pi Agent GUI 的主要审阅对象稳定为 `nimloth`，不再被父目录数十个 `nimloth-*` worktree 干扰。
- `nimloth` 同时拥有真实 `.git/` common dir 与 `.local/`，不再通过另一个 sibling 间接提供 Git/machine state。
- 新任务默认直接在 canonical root 工作，隔离 worktree 成为有明确理由的例外。

## Corrected intent

2026-08-28 人类更正：canonical root 应为 `/workspace/remote2/nimloth`，**不是** `nimloth-dev`。

先前设计中的两个术语因此作废：

- `dormant main` 原指保留 `nimloth` 只作为 `nimloth-dev` 的后台 Git/.local owner，但不在其中开发；这与当前要求相反。
- `最终独立化` 原指把 `nimloth-dev` 重建成 standalone repo 后删除 `nimloth`；现在无需这样做，因为 `nimloth` 已经是 standalone main worktree/common Git owner。

## Confirmed facts

1. 当前注册 39 个 worktree：1 main + 38 linked；28 clean、11 dirty。
2. `/workspace/remote2/nimloth` 已是 main worktree：`git_dir=.git`、`common_dir=.git`，并实际拥有 `.local/`。
3. `nimloth` 当前 checkout `main`，有 10 个 dirty entries；其中部分内容不在 `dev`，不能在切换 branch 时直接覆盖。
4. `nimloth-dev` 当前 checkout `dev`，在完成prompt任务后相对 `origin/dev` ahead 372，并有 6 类 dirty entries，包括三个 active Trellis task目录、`AI_branch_progress.md`、Pi TaskTree与le-wm状态。
5. 当前 active experiment `08-26-state-interface-v2-sft1-canary-exp` 硬绑定 sibling path，且对应 worktree 有 9 个未提交 smoke-preparation entries。
6. `08-27-audit-remote-experiment-storage` 的 `.worktree` 引用均为远程存储路径，不应被本地路径重构改写。
7. `nimloth` 与 `dev` 版本的 `.gitignore` 均已有 `.worktree` ignore；当前 sibling-path 规则由 `AGENTS.md`、governance spec 与 `git-worktree` skill 共同定义。
8. 初始证据见 [`research/local-worktree-audit-2026-08-28.md`](research/local-worktree-audit-2026-08-28.md)。

## Requirements

### R1 — Canonical local root

- 唯一日常开发根为 `/workspace/remote2/nimloth`。
- 新任务默认不创建 worktree。
- 只有并行修改、实验 exact-source 隔离、危险集成/回归隔离或人类明确要求等必要理由存在时，才创建：

```text
/workspace/remote2/nimloth/.worktree/<branch-name-with-slashes-replaced>
```

### R2 — Canonical branch

- 人类于2026-08-28明确选择：canonical root的日常开发branch为`dev`。
- 把`dev` checkout到`/workspace/remote2/nimloth`，保留执行时精确HEAD、`origin/dev` tracking与全部local-only commits。
- `main`仅作为普通branch保留，不删除、不把canonical root固定为长期checkout `main`。
- branch cutover前仍必须分别保存并验证`nimloth`的main dirty state与`nimloth-dev`的dev dirty state；本决定不授权覆盖或丢弃任一侧内容。

### R3 — Lossless migration

- 删除/移动前逐个核对 branch、HEAD、upstream、dirty/untracked/ignored/submodule 状态。
- 不删除 branch/tag/ref，不丢弃未提交内容，不覆盖 protected memory，不改变远端 Git/实验数据。
- `nimloth` 当前 10 个 dirty entries 与 `nimloth-dev` 当前 6 个 dirty entries必须分别保存、比较和决定归属，不能直接合并或覆盖。
- 所有其他 dirty worktree 必须有经人类批准的 exact disposition：保留并迁移、先由原任务 commit、可恢复归档，或明确丢弃。
- 禁止未经 exact-path approval 使用 `git worktree remove --force`、`rm -rf`、reset/clean 或等价破坏操作。

### R4 — Active task continuity

- 本任务当前位于 `nimloth-dev`；cutover 时必须把 task artifacts、必要 runtime/handoff 信息和未提交 task state无损迁入 `nimloth`。
- 08-26 的本地 experiment branch 在迁移前完成其自身 complete-diff/commit 决策，或采用另行批准的无损归档方案。
- 若 08-26 仍需隔离 worktree，则从 exact branch/commit 重建为：

```text
/workspace/remote2/nimloth/.worktree/exp-state-interface-v2-sft1-canary
```

- 08-27 的远程 `.worktree` 路径和历史审计证据保持不变。

### R5 — Local machine state

- `nimloth/.local/` 继续作为唯一 machine-specific state owner。
- child worktree 的 `.local` 统一指向 canonical root 的 `.local`，不再指向其他 sibling。
- submodule URL/commit/status 在迁移前后有可比较证据。

### R6 — Rule consistency

实施阶段在得到明确批准后同步：

- `AGENTS.md`；
- `.trellis/spec/governance/git-worktrees-and-protected-files.md`；
- `.agents/skills/git-worktree/SKILL.md`。

规则必须同时定义默认直接开发、child worktree 必要性标准、path、创建验证、`.local` 链接、cleanup 和禁止 `--force` 的边界。

### R7 — Reviewability and rollback

- destructive migration 前生成 exact manifest 和回滚材料；先在 disposable sandbox 验证 `dev` 从 linked worktree 回到 main worktree路径以及 nested child worktree流程。
- cutover 时暂停 Pi/其他 Git writer，迁移后从 `/workspace/remote2/nimloth` 重开会话并重新核对当前 Trellis task。
- 分批执行，每批后验证；任一 dirty/unexpected/submodule/metadata mismatch 立即停止。

## Approved reversible batch 1

2026-08-28 人类批准在不触碰live branch/worktree topology的前提下先实施：

- disposable `/tmp` sandbox中的RED/GREEN payload迁移与nested child方法证明；
- task-local、Python标准库实现的metadata-only manifest/validator；
- 39个registered worktrees、refs/tags/remotes/config、submodule、untracked/ignored概要和canonical main/dev冲突证据；
- 仅同步`AGENTS.md`、Git/worktree governance spec与`git-worktree` skill；现有`.gitignore`语义足够时不改；
- task artifacts、progress和验证证据。

本批禁止任何live branch switch、move/remove/prune、`git worktree remove`、reset/clean/stash、recursive deletion、commit/push/merge、08-26/08-27修改或protected runtime/memory修改。Batch evidence见[`research/first-reversible-batch-2026-08-28.md`](research/first-reversible-batch-2026-08-28.md)。

## Acceptance Criteria

- [ ] `/workspace/remote2/nimloth` 是唯一日常开发根，拥有真实 `.git/` 与 `.local/`。
- [ ] `git worktree list --porcelain` 中不再存在不允许的 `/workspace/remote2/nimloth-*` sibling linked worktree。
- [ ] canonical root checkout 人类批准的日常 branch；若为 `dev`，HEAD、local-only commits、origin tracking 与迁移前一致。
- [ ] 必要 child worktree 仅位于 `nimloth/.worktree/`，且当前 08-26 worktree（若保留）branch/HEAD/submodules/dirty state 与批准合同一致。
- [ ] 原 11 个 dirty worktree 均有 evidence-backed disposition，protected memory 与未提交产品/task修改无丢失。
- [ ] `nimloth` 原 main-worktree dirty 内容与 `nimloth-dev` dirty 内容均有独立 manifest、归属决定和恢复验证。
- [ ] 所有 local refs/branches/tags、origin URL/fetch config 和 detached reachable commit 均有迁移前后清单对比。
- [ ] child `.local` 指向 `nimloth/.local`，不存在旧 sibling 绝对路径依赖。
- [ ] `AGENTS.md`、governance spec、`git-worktree` skill 和 `.gitignore` 语义一致。
- [ ] 当前 Trellis task 可从 `nimloth` 解析，相关 `task.py validate` 通过；08-26 现行路径已更新而 08-27 远程历史证据未被改写。
- [ ] `git status --short --branch`、`git diff --check`、submodule status 与 Git ref/object 验证通过。
- [ ] Pi Desktop 从 `nimloth` reload/reopen 后，人工确认代码审阅目标与 worktree 列表符合预期。

## Out of scope

- 删除 local branches/tags/remotes。
- 修改或删除远程服务器 worktree、输出、checkpoint、cache 或 Slurm/W&B 状态。
- 自动 commit/push 08-26 或其他 legacy dirty worktree 的产品修改。
- 清理非 worktree 目录 `nimloth-artifacts`、`nimloth-report`。
- 擅自处理现有 `.pi/task-tree/` 内容；它作为不相关 dirty state保留。
- 修改 Pi adapter，除非迁移后实测证明现有 `ctx.cwd` 合同失效并重新规划。

## Authorization state

- Canonical path与branch均已决定：`/workspace/remote2/nimloth` + `dev`。
- 人类已要求开始重构；该授权可覆盖sandbox验证、manifest/恢复材料和经明确列出的规则修改。
- 任何branch cutover、worktree删除、`--force`、`rm -rf`、reset/clean或dirty payload disposition仍需在展示精确路径与恢复证据后单独批准。
