# E0053 — submodule目录存在不代表已初始化

## 已发生的错误

VERL gate只用`Path.exists()`判断`external/VAGEN/verl`是否可用。未初始化submodule仍有空目录；随后`git -C <空目录>`向上找到父repo，返回Nimloth commit并触发错误的VERL commit mismatch。ID29因此在模型加载前失败。

## 正确做法

- 检查submodule的实际模块入口，例如`external/VAGEN/verl/verl/__init__.py`。
- 入口不存在时才切换到已验证的共享VERL checkout。
- fallback后再次检查模块入口，再核对VERL commit。
- 禁止仅凭目录存在或`git -C`成功断言submodule已初始化。

## 证据

- `experiments/training/rl/run_verl_exact_replay_worker_gate.py`
- ID29 README：`outputs/experiments/training/rl/2026-07-18/29_smoke_verl455_fullactor_fullcritic_exactreplay_id22traj0_maskedgae/README.md`
