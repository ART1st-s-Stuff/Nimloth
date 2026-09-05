---
name: git-worktree
description: 创建、核验或清理任务 worktree，以及配置其共享本地状态时使用。
---

# Git worktree 操作

分支、目录和清理边界以 [Git 与 worktree](../../../.trellis/spec/governance/git-worktrees-and-protected-files.md)为准。只读取当前操作需要的章节；普通文件编辑不需要重走本流程。

## 确认目标

根据当前开发者的本地约定、任务元数据和 Git 登记，确定控制目录、任务分支、基点及目标路径。旧任务记录中的绝对路径不能直接用于另一台机器。

在目标目录通过工具的工作目录参数或显式切换目录，核验：

```sh
git rev-parse --show-toplevel
git rev-parse --path-format=absolute --git-common-dir
git branch --show-current
git status --short --branch --untracked-files=all
git worktree list --porcelain
```

同时核验实际 cwd。公共 Git 元数据目录可能不在主目录的 `.git` 下；使用 Git 返回的路径核对仓库身份，不自行拼接。Git 版本不支持上述路径选项时，解析并核验其返回的相对路径。

主目录空闲、干净且分支和基点已确认时，可用作主任务目录；存在并行工作或未提交内容时保留现场，为需要隔离的任务使用独立 worktree。不要为取得干净状态而 stash、reset 或 clean。

## 创建或复用

先确认目标路径不存在，包括失效链接；任务分支没有被其他 worktree 占用，基点可解析为 commit。记录本次使用的精确基点。

以下为命令形式，变量须根据当前环境完成核验后设置，不能复制其他开发者的值：

```sh
# 新分支
git -C "$task_controller" worktree add -b "$task_branch" "$task_worktree" "$task_base"

# 已确认复用的现有分支：不要再次创建分支
git -C "$task_controller" worktree add "$task_worktree" "$task_branch"
```

每次修改前在同次调用中完成规则要求的 cwd、root、分支和 status 核验；上述两条命令是互斥选项。失败后检查原因，不添加 `--force` 或手工修改 Git 登记。

创建后从控制目录检查登记，再从新目录核验 root、分支、HEAD 和公共 Git 元数据目录，确认它属于同一仓库。把实际分支、基点、工作目录记入当前任务；不复制其他任务或整个未提交工作区。

## 共享本地状态

先确定 `.local` 等机器状态的真实拥有者及访问方式，不默认要求固定目录或符号链接。

- 使用链接时，目标位置必须不存在；核验解析后的目标确为预期共享存储，且链接被 Git 忽略。不要用覆盖式链接命令替换现有内容。
- 使用其他本地约定时，验证当前 worktree 能读取所需配置，且移除它不会删除共享状态；不自动搬迁或改造已有布局。
- 只有涉及远程操作时才要求对应服务器配置可读，本地 worktree 创建不以服务器配置为前提。
- 仓库自有 skills 保持版本控制中的实体文件，不能通过绝对链接引用另一棵 worktree。

## 清理

只检查明确归属当前任务的精确目录。确认预期分支和 HEAD，并检查普通文件、ignored 内容、嵌套仓库及已初始化的递归 submodule；不能仅看顶层 status。

可用的只读检查包括：

```sh
git -C "$task_worktree" status --short --branch --untracked-files=all
git -C "$task_worktree" ls-files --others --ignored --exclude-standard
git -C "$task_worktree" submodule status --recursive
git -C "$task_worktree" submodule foreach --recursive 'git status --short --branch --untracked-files=all'
git -C "$task_worktree" submodule foreach --recursive 'git ls-files --others --ignored --exclude-standard'
```

还须检查未登记为 submodule 的嵌套目录。存在待保留内容、未明归属或无法完成检查时，保留现场并请求决定。不要为了让检查通过而删除 ignored 文件或反初始化 submodule。

满足 spec 的自动清理条件后：记录共享状态链接的目标，只解除已核验的链接，再从仍存在的控制目录执行普通 `git worktree remove`。核验目录和登记均已消失。

若移除失败，保持目录和登记；若刚解除了链接，仅在原位置仍为空且目标未变时恢复原链接，否则报告现状。不得覆盖并发创建的内容。submodule 导致普通 remove 失败也必须停止，不自动改用 force、递归删除或全局 prune。

## 远程 worktree

远程路径和连接入口来自当前环境的 `.local/SERVER.md`。按[实验规则](../../../.trellis/spec/experiments/task-contract.md)同步已提交源码；Slurm 环境使用 [slurm skill](../slurm/SKILL.md)。本地目录调整不改变历史记录中的远程路径。
