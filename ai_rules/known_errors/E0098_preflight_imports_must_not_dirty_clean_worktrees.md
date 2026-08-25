# E0098：CPU preflight import 不得污染待运行的 clean worktree

## 已确认错误

在 ID163 TP8 guided gate 提交前，agent 在准备作为正式运行源的服务器 clean worktree 中执行了真实 ID74 frozen-snapshot CPU preflight。该 preflight 未设置 `PYTHONDONTWRITEBYTECODE=1`，导入 `external/le-wm/module.py` 时生成了未跟踪的 `external/le-wm/__pycache__/`。正式 runner 的源码洁净门禁随后正确拒绝该 worktree，导致 ID163 在 checkpoint hash、AI2-THOR、Ray、vLLM 和模型加载前失败。

## 正确做法

- 正式 GPU 运行必须使用未被任何 import、测试或临时工具写入过的全新 Git worktree；测试 worktree 与 production run worktree 分离。
- 所有只读 CPU/import preflight 都必须显式设置 `PYTHONDONTWRITEBYTECODE=1`，并在完成后重新检查 parent、VAGEN、VERL、le-wm 和其他子模块的 `status --porcelain --untracked-files=all`。
- 任何 clean-worktree gate 失败都不得删除痕迹后复用同一实验输出；先保存失败证据，再以新数字 ID、空输出目录和新 worktree重试。
- 不得把 preflight failure 表述为模型、环境、guided policy 或 GPU runtime 的实验结论。
