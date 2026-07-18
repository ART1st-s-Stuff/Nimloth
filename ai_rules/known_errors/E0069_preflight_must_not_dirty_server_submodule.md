# E0069 — Preflight不得用pycache污染clean server worktree

## 已发生的错误

ID50的真实env semantic preflight成功；它导入`external/le-wm`时生成未跟踪`__pycache__`，使随后trainer脚本的clean-repo gate静默失败。环境已完成create/reset/close，但dataset、Ray、W&B、模型和optimizer均未启动。

## 正确做法

- 外部launcher及其所有srun child在任何Python调用前export`PYTHONDONTWRITEBYTECODE=1`。
- 运行前仍须删除历史agent生成的submodule pycache并确认递归worktree clean。
- 不要放宽正式trainer的clean-repo gate来掩盖preflight副作用。

## 证据

- `experiments/training/rl/launch_verl_online_in_hold.sh`
- `experiments/training/rl/run_verl_online_world8_smoke.sh`
- ID50 env preflight和server git status。
