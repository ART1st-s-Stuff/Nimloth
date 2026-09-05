# E0100：不要用 `compileall` 做 clean-worktree 只读语法检查

## 已确认错误

在 joint PPO 训练事务实现期间，agent 以为 `PYTHONDONTWRITEBYTECODE=1 python -m compileall` 不会写文件，并在开发 worktree 运行该命令。`compileall` 的职责正是显式生成 `.pyc`，不会被该环境变量变成只读检查，因此在 parent、VAGEN 和 VERL worktree 生成了大量未跟踪 `__pycache__/`。这些新目录随后被清理；既有且不属于本任务的 `external/le-wm/__pycache__/` 保持不动。

## 正确做法

- clean-worktree 中的只读 Python 语法检查使用 `ast.parse` 逐文件解析，不使用 `compileall` 或 `py_compile`。
- `PYTHONDONTWRITEBYTECODE=1` 只用于阻止普通 import 自动写 bytecode，不能推断显式 bytecode 编译工具也不会写文件。
- 若确实需要 `.pyc` 验证，只能在隔离测试 worktree 或临时复制目录运行，并在前后检查所有 parent/submodule worktree 状态。
- 清理误生成文件时必须排除任务开始前就存在的无关未跟踪内容，禁止借机修改或删除它们。
