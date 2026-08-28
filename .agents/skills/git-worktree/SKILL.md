---
name: git-worktree
description: >-
  创建并验证Nimloth git worktree，强制branch与路径的命名关系，并共享机器专用的.local状态。在本地修改仓库、创建branch worktree或worktree缺少.local时使用。
---

# Git Worktree

## 触发条件

在本地修改仓库、创建worktree或修复缺失的共享`.local/`状态之前使用。

## 权威合同

必须阅读：

- [`AGENTS.md`](../../../AGENTS.md)，获取直接安全内核；
- [Git/worktree/受保护文件spec](../../../.trellis/spec/governance/git-worktrees-and-protected-files.md)；
- [权限与安全](../../../.trellis/spec/governance/authority-and-safety.md)。

规则：

- 本地路径为`../nimloth-<branch-name>`，并将`/`替换为`-`；
- 除非人类prompt明确允许，否则禁止修改持有`main`的worktree；
- 必须核验实际路径和branch；禁止根据目录名推断branch；
- 每条会修改仓库的命令，都必须在同一次调用中使用明确的工具cwd，或先执行`cd "$WT_DIR" && ...`；
- `.agents/skills/`下的可移植项目skills必须是纳入版本控制的实体，禁止使用指向其他克隆/worktree的符号链接；
- 只有机器专用状态可以留在被忽略的`.local/`下。

## 创建

从持有共享本地状态的仓库/worktree运行：

```bash
BRANCH="feat/my-feature"
WT_DIR="../nimloth-$(printf '%s' "$BRANCH" | tr '/' '-')"

pwd
git status --short --branch
git worktree add -b "$BRANCH" "$WT_DIR" <approved-start-point>
```

如果branch已经存在，省略`-b`并传入现有branch。策略不明确时，禁止猜测起点或切换重要branch。

## 配置共享本地状态

```bash
MAIN="../nimloth"  # replace with the actual shared-local-state worktree
cd "$WT_DIR" && \
  ln -sfn "$MAIN/.local" .local
```

禁止为`git-worktree`、`slurm`或其他可移植仓库skills创建符号链接：它们必须通过Git获得。若未来某项skill确定只适用于单台机器，应把它放在`.local/`下，而不是添加指向`.agents/skills/`的绝对符号链接。

在同一条绑定目标的命令中验证：

```bash
cd "$WT_DIR" && \
  pwd && \
  git branch --show-current && \
  git status --short --branch && \
  test -f .local/SERVER.md && \
  test -f .agents/skills/git-worktree/SKILL.md && \
  test -f .agents/skills/slurm/SKILL.md
```

## 远程 worktree

阅读`.local/SERVER.md`和[`slurm` skill](../slurm/SKILL.md)。远程代码必须保持在已批准commit，禁止直接在服务器上修改生产代码。

## 清理

移除前，必须确认目标中没有未提交或未push的重要工作。然后从控制该目标的worktree运行：

```bash
git worktree remove "$WT_DIR"
git worktree prune
```

未经针对已核验精确路径的明确批准，禁止使用`--force`。
