# 设计草案：以 `nimloth` 为 canonical root

> 状态：第一批可逆实施已完成。canonical路径=`nimloth`、日常branch=`dev`；sandbox、metadata-only manifest和规则同步已有验证证据，live cutover/删除仍需精确审批。

## 1. Correct target

```text
/workspace/remote2/
└── nimloth/
    ├── .git/                    # 已存在的 main/common Git dir
    ├── .local/                  # 已存在的 machine-specific state owner
    ├── .worktree/               # ignored；仅必要时存在
    │   └── exp-state-interface-v2-sft1-canary/
    └── <project files>
```

`nimloth-dev` 和其他 sibling linked worktree 最终移除；`nimloth-artifacts`、`nimloth-report` 不是 Git worktree，不在本任务删除范围。

## 2. 对上一版术语的澄清

### `dormant main` 原本是什么意思

原方案假设继续在 `nimloth-dev` 开发，因此建议让 `nimloth` 只保留 `.git` common dir 与 `.local`，平时不打开、不修改。它是“后台仓库管理员”，所以叫 dormant main。

这与更正后的意图相反：现在要直接在 `nimloth` 开发，因此该方案作废。

### `最终独立化` 原本是什么意思

因为 `nimloth-dev/.git` 实际指向 `nimloth/.git/worktrees/dev`，如果要删除 `nimloth` 并只留 `nimloth-dev`，就必须重新构造 `nimloth-dev/.git`、迁移 refs/objects/config 和 `.local`，使其成为 standalone main worktree。

现在选择保留并直接使用 `nimloth`；它已经是 standalone main worktree，所以无需 clone/replacement 或 Git metadata surgery。该方案也作废。

## 3. Canonical branch decision

### A. `nimloth` checkout `dev`（已选择）

迁移后：

```text
path: /workspace/remote2/nimloth
branch: dev
role: 唯一日常开发根
```

理由：

- 第一批实施前`dev`相对`origin/dev` ahead 372，包含现行Trellis/project contracts；执行cutover时以重新核验的exact HEAD/ahead为准；
- 当前任务和两个 active task 的未提交 artifacts 位于 `nimloth-dev`；
- `AGENTS.md` 禁止未明确批准在持有 `main` 的 worktree 修改；
- 路径与 branch 解耦后，`nimloth` 目录不要求必须 checkout `main`。

迁移要求：先无损保存 `nimloth` 当前 main dirty state和 `nimloth-dev` 当前 dev dirty state；解除 `dev` 的 linked-worktree checkout 后，再在 `nimloth` checkout exact `dev` 并恢复批准内容。

### B. `nimloth` 保持 `main`（已拒绝）

该路线会要求日常直接在`main`开发或频繁切换，并需显式放宽main safety rule、重新安置`dev`的local-only commits与现行task artifacts。人类已选择`dev`，本任务不再实施该路线。

## 4. Worktree creation policy

默认：直接在 canonical `nimloth` 的批准日常 branch 上开发。

只有满足至少一个条件时才允许 child worktree：

1. active experiment 需要 exact-source、clean branch 与 canonical root dirty state隔离；
2. 两个已批准任务确实需要并行修改重叠路径；
3. merge/rebase/reproduction 会显著增加覆盖当前工作区的风险；
4. 人类明确要求隔离审阅。

不充分理由：沿用旧习惯、每个 task 默认一棵 worktree、仅为命名整齐。

Child path：

```text
/workspace/remote2/nimloth/.worktree/<branch-name-with-/-replaced-by-->
```

## 5. Cutover design

### Stage 1 — Exact manifests

第一批使用task-local Python标准库工具生成schema化metadata-only manifest：文件payload仅记录path/status/kind/size/SHA-256，config value与remote endpoint仅记录fingerprint，禁止复制protected memory或secret内容。Ignored在superproject和initialized submodule中采用top-level collapsed概要；任何recursive ignored审计仍是exact-path cleanup gate。Submodule审计显式绑定每个actual work tree，避免共享Git dir的`core.worktree`误读其他sibling。

分别记录：

- canonical `nimloth` 当前 main branch/HEAD、10 个 dirty entries、submodules、ignored payload、`.local`；
- `nimloth-dev` 当前 dev branch/HEAD、6 个 dirty entries、task/runtime state、submodules；
- 其他 9 个 dirty sibling worktree；
- all refs/tags/remotes/upstreams 与 detached reachable commit。

当前只读比较已证明：main working tree 的 `ai_tasks/vagen_baseline.md`、两个 baseline scripts 等内容不在 `dev`，多个 skill 文件虽在 `dev` 但内容不同；不能简单 checkout 覆盖。

### Stage 2 — Sandbox method proof

在 disposable repo 证明：

1. 一个 branch 从 dirty/含 submodule linked worktree迁回 main worktree path；
2. dirty payload archive/restore 与 task runtime handoff 可验证；
3. nested `.worktree/` 创建、`.local` link 与 cleanup；
4. 不需要手工修改 `.git/worktrees/*`，也没有自动 `--force` fallback。

第一批实际证明补充了关键边界：当前Git对含submodule的linked worktree即使clean/deinit仍拒绝普通remove。安全GREEN方法因此是保存index/worktree与extra payload→精确clean→linked worktree detach释放branch→main worktree checkout `dev`→恢复payload；并已实际反向clean/detach/reattach/restore证明旧linked path可作为rollback载体，不要求在cutover前remove。Nested child的普通cleanup必须先精确unlink已核验`.local` symlink；若branch含submodule而remove被拒绝，立即停止并取得该exact path的新决定。

### Stage 3 — Preserve and release `dev`

- 为 `nimloth-dev` 生成 exact filesystem/diff/task manifest与可恢复副本；不依赖 parent stash 捕获 submodule内 untracked或 ignored runtime。
- 暂停 Pi/其他 writer。
- 按 sandbox 证明的方法清理或暂存 linked dev 工作区，使 `dev` 不再被该 linked worktree占用。
- 保留 rollback copy，直到 canonical root全部验证通过。

### Stage 4 — Canonical root branch cutover

若批准 `dev`：

- 先处理/保存 `nimloth` 当前 main dirty payload；
- 在 `/workspace/remote2/nimloth` checkout exact `dev`；
- 恢复经批准的 dev dirty/task state；
- 重开 Pi session，重新选择/验证本 Trellis task。

### Stage 5 — Other siblings

- 28 个 clean worktree按 exact manifest分批 `git worktree remove`；包含 populated submodule导致拒绝时停止，不自动 `--force`。
- 08-26 active dirty worktree优先由 owning task形成 exact commit，再在 `nimloth/.worktree/`重建。
- 其他 8 个 dirty linked worktree逐个选择 migrate/archive/keep/discard；protected memory不得擅自丢弃。

## 6. Task migration semantics

### 当前 08-28 task

task artifacts从 `nimloth-dev` 无损迁入 `nimloth`；cutover 后新的 Pi session必须能通过 `task.py current --source` 或明确 task selection恢复 planning/in-progress状态。

### 08-26

- 历史 `progress.md`/research 中旧 sibling path 作为 lineage 保留。
- `task.json.worktree_path` 与描述“当前执行位置”的合同更新为 nested path。
- remote `/project/.../.worktree/...` 不变。

### 08-27

不修改远程 `.worktree`/`.worktrees` 历史审计证据。

## 7. Rule ownership

- `AGENTS.md`：canonical root、批准日常 branch、必要 child worktree 与 main safety boundary；实施前需明确批准修改。
- governance spec：mutation、protected、cleanup、review 和 nested path合同。
- `git-worktree` skill：动态使用 `/workspace/remote2/nimloth` root下 `.worktree/`，统一 `.local`。
- `.trellis/workflow.md`：没有固定 sibling path，预计不改。
- Pi adapter：先用现有 `ctx.cwd` 合同从 `nimloth` reopen/reload；无失败证据不改。

## 8. Rollback

- destructive step前保存 machine-readable manifest与可恢复 payload/hash。
- canonical cutover前保留 `nimloth-dev` rollback copy；未通过 ref/status/submodule/task/Pi验证不删除。
- 每批只处理一种 disposition；任何 mismatch停止，不自动升级为 `--force`。
- branch/ref删除不属于本任务。
