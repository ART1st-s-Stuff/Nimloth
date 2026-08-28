# 本地 worktree 与 Trellis 任务审查（2026-08-28）

> 第一批可逆实施已生成标准库metadata-only manifest、sandbox方法证明与下一批exact-path清单；见[`first-reversible-batch-2026-08-28.md`](first-reversible-batch-2026-08-28.md)。本文件以下内容保留为实施前的初始审查证据。

## 2026-08-28 人类更正后的解释

Canonical root 应为 `/workspace/remote2/nimloth`，不是 `nimloth-dev`。这使初始审查中的 Git topology 风险转化为迁移优势：`nimloth` 已经拥有 main/common `.git/` 与真实 `.local/`，无需把 `nimloth-dev` 重建为 standalone repo。

新的关键 cutover 是：保存 `nimloth` 当前 main dirty state与 `nimloth-dev` 当前 dev dirty/task state，解除 `dev` 在 linked worktree 的占用，然后把批准的日常 branch checkout 到 `/workspace/remote2/nimloth`。先前所谓 `dormant main` 与 `nimloth-dev 最终独立化` 两条路线均已作废。

额外只读比较确认：`nimloth` 当前 main working tree 的 `ai_tasks/vagen_baseline.md`、两个 baseline scripts等内容不在 `dev`；多个 skill 文件虽存在于 `dev` 但内容不同。因此不能用直接 checkout/reset覆盖 main dirty state。

## 审查范围

本轮仅做本地只读审查和任务规划；没有删除、移动、清理、commit、push 或修改任何既有 worktree。审查命令包括：

```bash
python3 ./.trellis/scripts/task.py current --source
git status --short --branch
git worktree list --porcelain
git -C <worktree> status --short --branch --untracked-files=all
git -C <worktree> submodule status --recursive
git rev-parse --git-dir --git-common-dir
```

Git 官方合同参考：<https://git-scm.com/docs/git-worktree>。其中明确说明：主 worktree 不能通过 `git worktree remove` 删除；linked worktree 的私有管理目录位于 common Git dir 的 `worktrees/` 下；clean worktree 才能无 `--force` 删除；包含 submodule 的 worktree 可能需要额外处理。

## 1. 当前 Git 拓扑

- 注册 worktree 共 **39** 个：1 个 main worktree + 38 个 linked worktree。
- clean 28 个，dirty 11 个；没有 locked/prunable entry；有 1 个 detached worktree。
- 当前 `/workspace/remote2/nimloth-dev` 是 linked worktree：

```text
.git -> /workspace/remote2/nimloth/.git/worktrees/dev
common Git dir = /workspace/remote2/nimloth/.git
.local -> /workspace/remote2/nimloth/.local
```

- 因此不能直接删除 `/workspace/remote2/nimloth`：这会同时破坏 `nimloth-dev` 的 Git metadata 与机器本地 `.local` 状态。
- `/workspace/remote2/nimloth-dev/.gitignore` 已包含 `.worktree`，可承载未来必要的 nested linked worktree。
- `/workspace/remote2/nimloth-artifacts` 与 `/workspace/remote2/nimloth-report` 不是 `git worktree list` 注册项，本任务不应把它们当作 worktree 删除。

## 2. Dirty worktree 风险清单

以下 11 个 worktree 当前非 clean；任何删除、`--force`、reset、stash、archive 或覆盖都需要另行精确审查。

| Path | Branch | 状态摘要 | 风险 |
|---|---|---:|---|
| `/workspace/remote2/nimloth` | `main` | 10 entries | main/common Git dir；含 tracked、untracked、submodule 与 `.local` 状态 |
| `/workspace/remote2/nimloth-dev` | `dev` | 6 entries | 当前开发根；含两个 active task、当前规划 task、`AI_branch_progress.md`、Pi TaskTree 和 le-wm 状态 |
| `/workspace/remote2/nimloth-exp-rl-k1ep1-h4-smoke` | `exp/rl-k1ep1-h4-smoke` | 2 entries | 两个 launcher 修改；upstream 已 gone |
| `/workspace/remote2/nimloth-exp-sft2-value-v3-rl-h1k1` | `exp/sft2-value-v3-rl-h1k1` | 9 entries | SFT2 launcher/validator/known-error 等未提交修改；branch behind 3 |
| `/workspace/remote2/nimloth-exp-state-interface-v2-sft1-canary` | `exp/state-interface-v2-sft1-canary` | 9 entries | 当前 08-26 active experiment 的 smoke-preparation 修改，必须保留 |
| `/workspace/remote2/nimloth-feat-planner-verl-vagen-scaffold` | `feat/planner-verl-vagen-scaffold` | 1 entry | le-wm submodule dirty |
| `/workspace/remote2/nimloth-feat-ppo-value-critic` | `feat/ppo-value-critic` | 39 entries | 大范围 PPO/WM 源码与测试未提交修改 |
| `/workspace/remote2/nimloth-feat-reconstruct` | `feat/reconstruct` | 1 entry | 修改 protected `.memory/memories.jsonl`，禁止直接丢弃或手改 |
| `/workspace/remote2/nimloth-fix-env-reproduction` | `fix/env-reproduction` | 5 entries | VAGEN submodule、skills、`.local`、临时 Slurm 文件 |
| `/workspace/remote2/nimloth-recon-compare-qwen` | `recon-compare-qwen` | 1 entry | 修改 protected `.memory/memories.jsonl` |
| `/workspace/remote2/nimloth-refine-scripts` | `refactor/refine-scripts` | 20 entries | 多个脚本修改/删除与 untracked local files |

其余 28 个 worktree 顶层状态 clean。移除 clean worktree 不会删除对应 local branch，但 upstream gone/no-upstream branch 仍应在移除前记录 ref/HEAD。唯一 detached worktree `nimloth-fix-sft2-review-bugs` 的 HEAD `950d3fcb...` 已可由 `dev` 等 refs 到达，不是当前 dangling commit。

本次尚未对所有 worktree 的 ignored outputs/cache/venv 做逐路径容量与价值审计；这是 destructive cleanup 前的必做 gate，不能用 clean Git status 代替。

## 3. 当前 Trellis 任务影响

### 08-26 `state-interface-v2-sft1-canary-exp`

- `task.json.worktree_path`、`prd.md`、`design.md`、`implement.md`、`research/exact-local-contract-v1.md` 均把本地执行路径绑定到：

```text
/workspace/remote2/nimloth-exp-state-interface-v2-sft1-canary
```

- 该 worktree 当前有 9 个未提交 smoke-preparation entries；不能直接删除或重建。
- 其远端路径 `/project/peilab/atst/nimloth/.worktree/...` 属于实验服务器合同，不是本地父目录 cleanup；不能全局替换。
- 最安全的迁移依赖是：先按 08-26 自身的 complete-diff/commit gate 处理其产品修改，使 branch 可从 exact commit 重建；随后在新根的 `.worktree/exp-state-interface-v2-sft1-canary` 重新创建并验证 exact branch/submodules。

### 08-27 `audit-remote-experiment-storage`

- 没有 local branch/worktree 绑定，`task.json.worktree_path=null`。
- 其中 `.worktree`/`.worktrees` 均指远程存储审计对象；本地规则重构不应改写其历史证据或远程路径。
- 该任务当前 acceptance 已记录完成，但 status 仍为 `in_progress`；是否 finish/archive 属于该任务自身，不纳入本次 cleanup。

### 08-28 `refactor-local-worktree-layout`

- 本任务负责本地规则、迁移清单、执行顺序和验证；不接管 08-26 的产品 commit、实验 launch 或 08-27 的远程清理。

## 4. 需要同步的规则入口

当前 sibling-worktree 规则至少存在于：

1. `AGENTS.md`（人类编写，实施时必须再次明确批准修改）；
2. `.trellis/spec/governance/git-worktrees-and-protected-files.md`；
3. `.agents/skills/git-worktree/SKILL.md`。

`.trellis/workflow.md` 只要求核验 worktree/branch 和绑定 mutation，没有固定 sibling 命名，预计无需改动。Pi adapter 已以 `ctx.cwd` 解析当前根；若不修改 adapter，则仅需在迁移后做 Pi Desktop reload/manual probe，不应为本次目标扩展 adapter 代码。

## 5. 核心技术结论（按人类更正更新）

1. `/workspace/remote2/nimloth` 已是正确的 canonical Git/.local owner；无需 clone、mirror replacement 或 Git metadata surgery。
2. 真正的 branch cutover 风险是：`nimloth` 当前 checkout dirty `main`，而现行开发历史和 task state位于 dirty linked `dev`；两边必须分别保存和审阅。
3. 推荐最终让 `/workspace/remote2/nimloth` checkout `dev` 作为日常开发根；若选择继续在 `main` 日常开发，需要明确放宽现有 main safety rule并重新规划 368 个 local-only dev commits。
4. 必要 worktree的目标路径改为 `/workspace/remote2/nimloth/.worktree/<slug>`。
5. 任何方案都必须先为 dirty worktree给出逐路径 disposition；禁止用 `--force` 一次性清掉。
