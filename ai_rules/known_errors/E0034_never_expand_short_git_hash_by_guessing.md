# E0034 — Never expand a short Git hash by guessing

## 错误路径

在更新 ID19 实验 output README 时，agent 已知短 hash `1eee7c5`，却写入了一个
未经命令核实的错误完整 hash。该错误在 Slurm 提交前被发现并修正，没有污染运行
commit；正确实验 commit 是 `1eee7c5373234069724a808b452bddc783ea3f88`。

## 正确做法

- 需要完整 commit 时必须直接运行 `git rev-parse HEAD`，逐字复制输出。
- 写入实验 README 后必须重新读取，并与服务器 worktree 的
  `git rev-parse HEAD` 比较。
- 短 hash 只能原样标成短 hash；禁止自行补全。
