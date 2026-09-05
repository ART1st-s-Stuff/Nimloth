# E0094: 仓库初始化命令必须显式绑定目标 worktree

## 错误

创建新 worktree 后，执行 `trellis init` 时遗漏 `cd`，使第一次初始化落在原 `dev` worktree，而不是目标 worktree。

## 原因

把“此前已创建目标 worktree”误当成后续 shell 命令会自动继承目标目录；实际每次工具调用仍使用会话默认工作目录。

## 正确做法

- 所有会修改仓库的命令必须在同一条 shell 命令中使用 `cd "$WT_DIR" && ...`，或使用工具提供的显式 cwd 参数。
- 执行前在同一命令中校验 `pwd` 和 `git branch --show-current`；不能只依赖之前一次工具调用的目录状态。
- 若误写到其他 worktree，先比较修改前基线和卸载 dry-run，再只移除本次生成的文件，保留原有未跟踪内容。

本次误初始化已用 Trellis 管理清单卸载，并手工清除卸载后残留的 `.claude/settings.json`、`.codex/` 和 `.gitattributes`；原 `dev` worktree 已恢复到操作前的 `external/le-wm` 与 `.pi/task-tree` 状态。
