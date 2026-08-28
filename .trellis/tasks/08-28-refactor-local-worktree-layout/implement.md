# 实施计划草案：以 `nimloth` 为唯一开发根

> 状态：第一批可逆实施已完成；canonical root/branch已决定，sandbox/manifest/规则证据已落盘。所有live cutover与destructive命令仍需精确审批。

## P0 — 决策与精确清单

- [x] 创建 Trellis task 并完成初始只读拓扑/任务审查。
- [x] 记录 39 worktree、28 clean/11 dirty、main/common Git 与 `.local` 依赖。
- [x] 人类更正 canonical root 为 `/workspace/remote2/nimloth`；废弃“保留 dormant main/把 nimloth-dev 独立化”两条旧路线。
- [x] 核实 `nimloth` 已是 standalone main/common Git + `.local` owner，无需 clone/replacement。
- [x] 核实 `nimloth` 当前 main dirty 内容部分不在 `dev`，不能直接 checkout 覆盖。
- [x] D1：人类决定canonical `nimloth`的日常branch为`dev`。
- [ ] 在任何destructive batch前，为10个non-canonical dirty worktree逐个锁定disposition；未经exact approval不执行destructive action。
- [ ] 在任何删除前，审计每个目标worktree的ignored outputs/cache/venv/runs与占用。

## P1 — Sandbox RED/GREEN

在 disposable 临时仓库验证最终方案，不触碰 Nimloth live worktree：

- [x] RED：dirty/含 submodule linked worktree不能直接被安全移除或迁回 main path。
- [x] GREEN：验证 linked `dev` 工作区 staged/unstaged index+worktree及untracked/ignored/symlink payload保存、clean后detach释放 branch、main worktree checkout `dev`、payload恢复。
- [x] 验证 task artifacts与 runtime/handoff在路径切换后可恢复，并实际反向clean/detach/reattach/restore证明保留linked path可回滚。
- [x] 验证 `<root>/.worktree/<slug>` 创建、branch checkout、`.local` 链接、status 与submodule-free child普通cleanup。
- [x] 输出可审查命令；不形成自动 `--force` fallback，不手工修改 live `.git/worktrees/*`。当前Git对含submodule worktree的普通remove拒绝已作为stop gate记录。

## P2 — 规则修改（implementation approval必须明确包含`AGENTS.md`）

- [x] 修改 `AGENTS.md`：canonical root=`/workspace/remote2/nimloth`，默认直接开发，必要 child位于 `.worktree/`；明确日常 branch/main safety。
- [x] 修改 `.trellis/spec/governance/git-worktrees-and-protected-files.md`。
- [x] 修改 `.agents/skills/git-worktree/SKILL.md` 的 create/setup/verify/cleanup 命令。
- [x] 确认 `.gitignore` 的 `.worktree` 规则覆盖 target；现有规则足够，未调整。
- [x] `rg '../nimloth-|nimloth-dev|nimloth-<branch'`：现行AGENTS/workflow/spec/skills无旧sibling合同；08-26现行local绑定与08-28历史/迁移证据保留，08-27 remote路径未改。
- [x] 未修改`.trellis/workflow.md`或Pi adapter；现有`ctx.cwd`合同足够，live cutover后再执行reopen/reload验证。

## P3 — Canonical/main 与 dev dirty-state gates

- [ ] 暂停其他 Git writer，重新运行完整 worktree manifest。
- [ ] `nimloth` 当前 main：展示 10 个 entries、submodules、ignored payload与 `.local`；逐项决定 preserve/archive/integrate/discard。
- [ ] `nimloth-dev` 当前 dev：展示 6 个 parent entries、le-wm nested state、ignored Trellis runtime；生成可恢复 payload与hash。
- [ ] 证明 main payload与dev payload分别可恢复，禁止直接覆盖同名文件。
- [ ] 保留 current task、08-26、08-27 task artifacts及 session handoff证据。
- [ ] 两个 protected memory dirty legacy worktree单独处理，禁止丢弃或手改 memory JSONL。

## P4 — Canonical root cutover

若 D1 选择 `dev`：

- [ ] 按 sandbox 证明的方法精确clean后detach `nimloth-dev`以释放`dev`；保留该linked path作为rollback载体，不在cutover批remove。
- [ ] 在 `/workspace/remote2/nimloth` checkout exact `dev`，核对 HEAD=`217e5765...` 或执行时批准的新 exact HEAD、tracking 与 local refs。
- [ ] 恢复批准的 dev dirty/task state；main旧dirty payload保持独立归档或按批准处理。
- [ ] 关闭旧 Pi session，从 `/workspace/remote2/nimloth` reopen/reload；恢复并验证当前 task。

若 D1 选择 `main`：

- [ ] 先重新规划 368 个 local-only dev commits、现行 Trellis task artifacts与 main safety rule如何处理；未经新批准不执行 cutover。

## P5 — Active nested worktree

- [ ] 08-26 回到 owning task完成 smoke-preparation complete-diff与单独 commit决策；本任务不代为 commit。
- [ ] 在 source clean/exact 后移除旧 sibling registration/path。
- [ ] 从 exact branch/commit 创建：

```text
/workspace/remote2/nimloth/.worktree/exp-state-interface-v2-sft1-canary
```

- [ ] 初始化并核对 VAGEN/VERL/le-wm exact commits；`.local` 指向 `/workspace/remote2/nimloth/.local`。
- [ ] 更新 08-26 当前 path metadata/现行合同，保留历史 progress/research路径。
- [ ] 复跑 08-26 focused/static/CPU gate；不自动进入 GPU/Slurm。

## P6 — 其他 sibling worktree cleanup

- [ ] 28 个 clean worktree逐个核对 HEAD/branch/ref/submodule/ignored payload decision。
- [ ] 分批 `git worktree remove <exact-path>`；若 populated submodule导致拒绝，停止并重新规划，不自动加 `--force`。
- [ ] 其他 8 个 dirty linked worktree逐个按批准选择 migrate/archive/keep/discard。
- [ ] 每批后验证 `git worktree list`、branch/ref existence与 canonical status。
- [ ] 最终删除 `nimloth-dev` rollback copy需要单独 exact-path destructive approval。
- [ ] 不删除 local branch/tag/ref或 remote branch。

## P7 — Full verification

- [ ] `git worktree list --porcelain` 只出现 `/workspace/remote2/nimloth` 与批准的 nested entries。
- [ ] `git rev-parse --git-dir --git-common-dir` 在 canonical root均为 `.git`。
- [ ] canonical branch/HEAD/tracking与批准合同一致；all refs/tags/remotes/config与pre-migration manifest一致。
- [ ] `git status --short --branch`、`git diff --check` 与 Git ref/object验证通过。
- [ ] recursive submodule status与批准基线一致。
- [ ] `.local/SERVER.md`/local memory可读，nested child link不指向旧 sibling。
- [ ] `task.py current --source` 与相关 `task.py validate` 从 `nimloth` 通过。
- [ ] `rg` 证明现行 sibling-path规则已消除；历史/remote path仅保留在应保留位置。
- [ ] Pi Desktop 从 `nimloth` reload/reopen 后由人类确认代码审阅体验。

## P8 — Review/commit/cleanup gates

- [ ] 展示完整 changed files、迁移 manifest、验证证据、残余风险与 commit groups。
- [ ] 单独取得 project rule commit approval；不 push/merge。
- [ ] destructive filesystem cleanup使用单独 exact-path approval，不能由代码 commit approval隐含授权。
- [ ] 应用 `on-progress`、memory/spec review、finish-work流程。

## 第一批可逆实施证据

- Sandbox structured evidence：[`evidence/sandbox-proof.json`](evidence/sandbox-proof.json)。
- Metadata-only pre-migration manifest（含显式actual-work-tree绑定的initialized submodule dirty/untracked/ignored概要）：[`evidence/pre-migration-manifest.json`](evidence/pre-migration-manifest.json)。
- 结论与下一批exact-path清单：[`research/first-reversible-batch-2026-08-28.md`](research/first-reversible-batch-2026-08-28.md)。
- Validator与验证命令：[`tools/worktree_manifest.py`](tools/worktree_manifest.py)、[`tools/sandbox_worktree_proof.py`](tools/sandbox_worktree_proof.py)；最终结果见[`research/validation-evidence-2026-08-28.md`](research/validation-evidence-2026-08-28.md)。
- 本批没有live topology/branch改变，因此P3–P6仍未启动，也不更新`AI_branch_progress.md`。

## Stop conditions

- canonical branch 未决定；
- 任一 worktree出现未清单化 dirty/untracked/ignored/submodule状态；
- main 与 dev payload归属不明确或同名内容冲突；
- branch/ref/object/config/task state对比不一致；
- protected memory处理不明确；
- 08-26 active experiment尚未形成可重建的 exact source；
- 需要 `--force`、`rm -rf`、reset/clean但没有 exact human approval；
- Pi/Trellis 在 canonical root无法解析当前项目/任务。
