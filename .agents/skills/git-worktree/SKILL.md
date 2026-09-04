---
name: git-worktree
description: >-
  在Nimloth按per-task branch和canonical单槽合同创建、验证或cleanup并行/隔离worktree，并共享机器专用.local状态。
---

# Git Worktree

## 触发条件

在本地修改仓库、判断是否需要隔离worktree、创建/清理child worktree或修复缺失的共享`.local/`状态之前使用。

## 权威合同

必须阅读：

- [`AGENTS.md`](../../../AGENTS.md)，获取直接安全内核；
- [Git/worktree/受保护文件spec](../../../.trellis/spec/governance/git-worktrees-and-protected-files.md)；
- [权限与安全](../../../.trellis/spec/governance/authority-and-safety.md)。

固定合同：

- canonical root为`/workspace/remote2/nimloth`；`dev`是integration/base branch，每个修改repository的task使用自己记录的专用branch；
- canonical只提供一个主task槽位；额外并行task、实验exact-source或危险集成使用独立child worktree；
- canonical有任何未归属tracked/untracked变化时禁止切branch、stash、reset、clean或覆盖；
- 除非人类prompt明确允许，否则禁止修改实际持有`main`的worktree；
- 每条仓库mutation必须在同一次调用中绑定精确目标cwd，并核验`pwd -P`、`git rev-parse --show-toplevel`、实际branch和status；
- `.agents/skills/`下的可移植项目skills是Git跟踪实体，禁止使用指向其他clone/worktree的符号链接；只有机器专用状态可以位于ignored `.local/`。

## 选择canonical主槽位或child

先核验canonical的实际branch和完整status，不根据目录名猜测：

```bash
ROOT=/workspace/remote2/nimloth
(
  cd "$ROOT" &&
  test "$(pwd -P)" = "$ROOT" &&
  test "$(git rev-parse --show-toplevel)" = "$ROOT" &&
  pwd -P &&
  git branch --show-current &&
  git status --short --branch --untracked-files=all
)
```

只有一个主task、canonical clean且人类已确认task branch/base时，才可在canonical创建或checkout该专用branch。已有主task或dirty state时保留现场，为新增并行task创建child worktree。任一归属、branch或base不明确都停止。

## 何时允许child worktree

只有至少满足一项并已纳入当前授权，才创建child：

1. 已批准的并行任务会修改重叠路径；
2. 实验需要exact-source/clean branch与canonical dirty state隔离；
3. merge、rebase、回归复现等高风险操作需要隔离；
4. 人类明确要求独立审阅对象。

沿用旧习惯、task命名整齐或“每个task一棵worktree”不是充分理由。

## 创建nested child

先锁定branch与已批准start point；branch存在性或起点不明确时停止，禁止猜测。

```bash
ROOT=/workspace/remote2/nimloth
CONTROLLER=<verified-canonical-or-integration-worktree>
BRANCH=feat/my-feature
START_POINT=<approved-exact-ref-or-commit>
SLUG=$(printf '%s' "$BRANCH" | tr '/' '-')
WT_DIR="$ROOT/.worktree/$SLUG"

(
  cd "$CONTROLLER" &&
  test "$(pwd -P)" = "$CONTROLLER" &&
  test "$(git rev-parse --show-toplevel)" = "$CONTROLLER" &&
  test "$(git rev-parse --path-format=absolute --git-common-dir)" = "$ROOT/.git" &&
  git status --short --branch --untracked-files=all &&
  test ! -e "$WT_DIR" &&
  test ! -L "$WT_DIR" &&
  git rev-parse --verify "$START_POINT^{commit}" >/dev/null &&
  git worktree add -b "$BRANCH" "$WT_DIR" "$START_POINT"
)
```

若branch已经存在，只能在确认应复用该exact branch后省略`-b "$BRANCH"`：

```bash
(
  cd "$CONTROLLER" &&
  test "$(pwd -P)" = "$CONTROLLER" &&
  test "$(git rev-parse --show-toplevel)" = "$CONTROLLER" &&
  test "$(git rev-parse --path-format=absolute --git-common-dir)" = "$ROOT/.git" &&
  git status --short --branch --untracked-files=all &&
  test ! -e "$WT_DIR" &&
  test ! -L "$WT_DIR" &&
  git rev-parse --verify "$BRANCH^{commit}" >/dev/null &&
  git worktree add "$WT_DIR" "$BRANCH"
)
```

禁止添加自动`--force` fallback，也禁止手改`.git/worktrees`解决占用或冲突。

## 配置并验证`.local`

Canonical root保留真实`.local/`。Child中只有确认`.local`尚不存在时才创建链接，禁止用`ln -sfn`覆盖未知内容。

```bash
(
  cd "$WT_DIR" &&
  test "$(pwd -P)" = "$WT_DIR" &&
  test "$(git rev-parse --show-toplevel)" = "$WT_DIR" &&
  test "$(git branch --show-current)" = "$BRANCH" &&
  test "$(git rev-parse --path-format=absolute --git-common-dir)" = "$ROOT/.git" &&
  git status --short --branch &&
  test ! -e .local &&
  test ! -L .local &&
  ln -s "$ROOT/.local" .local &&
  test -L .local &&
  test "$(realpath .local)" = "$ROOT/.local" &&
  git status --short --branch &&
  test -f .local/SERVER.md &&
  test -f .agents/skills/git-worktree/SKILL.md &&
  test -f .agents/skills/slurm/SKILL.md
)
```

命令结束后再次从controller核验registration，不能只相信目录名：

```bash
(
  cd "$CONTROLLER" &&
  test "$(pwd -P)" = "$CONTROLLER" &&
  test "$(git rev-parse --path-format=absolute --git-common-dir)" = "$ROOT/.git" &&
  git worktree list --porcelain &&
  test "$(git -C "$WT_DIR" rev-parse --show-toplevel)" = "$WT_DIR" &&
  test "$(git -C "$WT_DIR" branch --show-current)" = "$BRANCH"
)
```

## 远程worktree

阅读`.local/SERVER.md`和[`slurm` skill](../slurm/SKILL.md)。远程代码必须保持在已批准commit，禁止直接在服务器上修改生产代码。远程历史中的`.worktree`/`.worktrees`路径是证据，不因本地布局重构而全局替换。

## Clean cleanup

Cleanup必须有精确child path。先检查tracked、untracked、ignored和recursive submodule；任何未批准payload、路径/branch mismatch或检查代价过大都停止。

```bash
ROOT=/workspace/remote2/nimloth
CONTROLLER=<verified-canonical-or-integration-worktree>
BRANCH=feat/my-feature
SLUG=$(printf '%s' "$BRANCH" | tr '/' '-')
WT_DIR="$ROOT/.worktree/$SLUG"

(
  cd "$WT_DIR" &&
  test "$(pwd -P)" = "$WT_DIR" &&
  test "$(git rev-parse --show-toplevel)" = "$WT_DIR" &&
  test "$(git branch --show-current)" = "$BRANCH" &&
  git status --short --branch --untracked-files=all &&
  git ls-files --others --ignored --exclude-standard &&
  git submodule status --recursive &&
  git submodule foreach --recursive 'git status --short --branch --untracked-files=all' &&
  git submodule foreach --recursive 'git ls-files --others --ignored --exclude-standard' &&
  git submodule foreach --recursive 'test -z "$(git status --porcelain=v1 --untracked-files=all)"' &&
  test -z "$(git status --porcelain=v1 --untracked-files=all)" &&
  test -L .local &&
  test "$(realpath .local)" = "$ROOT/.local"
)
```

人工审阅parent和所有populated recursive submodule的ignored输出，确认除已核验`.local` symlink外没有要保留的内容后，仅unlink该symlink，再执行普通remove；失败时恢复链接并停止：

```bash
(
  cd "$WT_DIR" &&
  test "$(pwd -P)" = "$WT_DIR" &&
  test "$(git rev-parse --show-toplevel)" = "$WT_DIR" &&
  test "$(git branch --show-current)" = "$BRANCH" &&
  git status --short --branch --untracked-files=all &&
  test -L .local &&
  test "$(realpath .local)" = "$ROOT/.local" &&
  unlink .local
) &&
(
  cd "$CONTROLLER" &&
  test "$(pwd -P)" = "$CONTROLLER" &&
  test "$(git rev-parse --show-toplevel)" = "$CONTROLLER" &&
  test "$(git rev-parse --path-format=absolute --git-common-dir)" = "$ROOT/.git" &&
  git status --short --branch --untracked-files=all &&
  if ! git worktree remove "$WT_DIR"; then
    test -d "$WT_DIR" &&
      (
        cd "$WT_DIR" &&
        test "$(pwd -P)" = "$WT_DIR" &&
        test "$(git rev-parse --show-toplevel)" = "$WT_DIR" &&
        test "$(git branch --show-current)" = "$BRANCH" &&
        git status --short --branch --untracked-files=all &&
        test ! -e .local &&
        test ! -L .local &&
        ln -s "$ROOT/.local" .local
      )
    exit 1
  fi &&
  test ! -e "$WT_DIR" &&
  test ! -L "$WT_DIR" &&
  ! git worktree list --porcelain | grep -Fqx "worktree $WT_DIR"
)
```

当前Git可能对含submodule的worktree即使clean/deinit仍拒绝普通remove。此时停止并报告精确路径、HEAD、branch、status、ignored与submodule证据；不得自动添加`--force`。未经该精确路径的人类批准，也禁止`rm -rf`、reset/clean、手改`.git/worktrees`或用全局`git worktree prune`代替精确cleanup。
