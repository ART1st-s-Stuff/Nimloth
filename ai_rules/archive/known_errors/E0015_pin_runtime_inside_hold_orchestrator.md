# E0015：hold 内手动启动不能依赖调用者传入 Python runtime

## 已发生的错误

在 hold `471146` 中手动启动 source-parity smoke 时没有导出 `PYTHON_ENV`。`common_env.sh` 回退到 worktree `.venv`（Transformers 4.49），FSDP 在源 checkpoint 加载前因 config 类型不兼容失败。

## 原因

此前只在提交命令里约定使用 `.venv-vagen-main`，长任务 orchestrator 自身没有固定或验证 runtime。换一个启动入口后，约定没有生效。

## 正确做法

需要特定 runtime 的 Slurm/hold orchestrator 必须在脚本内部设置并验证解释器路径，同时把实际路径写入启动日志。不能依赖 activate、父 shell 或人工 launch command 隐式传递。

## 证据

- 修复：`experiments/training/sft1/rollout_full_6gpu_preempt.slurm`
- 失败输出：`.../full_2e66e97/rollout/source_parity_smoke_seed1_488dfa2/`
