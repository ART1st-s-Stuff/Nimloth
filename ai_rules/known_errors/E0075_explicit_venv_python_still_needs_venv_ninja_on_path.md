# E0075 — 显式venv Python仍须把venv ninja加入PATH

## 已发生的错误

ID57通过tied-head vLLM模型加载并进入KV-cache profiling；FlashInfer sampling JIT用subprocess查找`ninja`，但launcher只显式调用venv Python、没有把venv bin加入PATH，因此8 worker均FileNotFound。尚未W&B/rollout/update。

## 正确做法

- 继续用`.venv-vagen-main/bin/python3`显式启动，避免stale activate/shebang。
- 同时prepend该venv的`bin`到PATH，为FlashInfer等子进程提供实际ELF `ninja`。
- 不得因此改回source activate或直接调用带错误shebang的torchrun。

## 证据

- `experiments/training/rl/run_verl_online_world8_smoke.sh`
- ID57 FlashInfer JIT stack trace。
