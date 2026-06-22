# Port Nimloth verl changes onto VAGEN legacy verl

## 任务目标
- 在 `external/VAGEN/verl` 中从正确可复现基线 `3f55021e08b98315213e10d10b709c7f6bbe84f7` 创建/整理 `nimloth/vagen-legacy` 分支。
- 将 `origin/nimloth/main` 中与 Nimloth/VAGEN legacy 需要相关的改动适配到 legacy verl，而不是继续基于错误的 `d78186d4`。

## 当前计划
1. 修正 submodule gitdir 的 `core.worktree` 指向当前工作区。
2. 将 `external/VAGEN/verl` 重置到 `3f55021e` 并创建 `nimloth/vagen-legacy`。
3. 逐个评估并移植 Nimloth 相关 commits：latent action、per-turn token、Qwen3-VL/value-head、tracking 等。
4. 运行轻量检查（语法/导入或 git diff 审查）。
5. 汇报未移植/不适用的 newer verl async/sglang 部分。

## 已完成步骤
- 确认正确基线为 `origin/vagen-legacy = 3f55021e`。
- 修正了 `external/VAGEN/verl` 的 gitdir `core.worktree`，之前错误指向 `/workspace/remote2/nimloth-fix-env-reproduction/...`。
- 创建/切换到 `nimloth/vagen-legacy`，并 `reset --hard 3f55021e`、清理错误工作树残留。
- 已手工移植适配：
  - Nimloth latent/action token extraction，使用 `<|action_(idx)|>` token 格式。
  - VAGEN legacy vLLM SPMD rollout 的 `max_response_per_turn` 支持。
  - rollout config 中 `extract_latent_action` / `max_response_per_turn` 开关。
  - FSDP worker 的 rollout 后 latent extraction 及显式 `extract_latent_action` RPC。
  - critic `value_mask` 兼容。
  - wandb navigation validation metric flattening。

## 文件修改
- `external/VAGEN/verl/verl/workers/rollout/latent_action.py`：新增 legacy-compatible latent/action extraction helper。
- `external/VAGEN/verl/verl/workers/fsdp_workers.py`：接入 latent extraction。
- `external/VAGEN/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`：接入 per-turn response cap，并使用 `max_trajectory_length` 做 vLLM context。
- `external/VAGEN/verl/verl/trainer/config/ppo_trainer.yaml`：新增 rollout config 字段。
- `external/VAGEN/verl/verl/trainer/config/ppo_megatron_trainer.yaml`：新增 rollout config 字段。
- `external/VAGEN/verl/verl/workers/critic/dp_critic.py`：兼容 `value_mask`。
- `external/VAGEN/verl/verl/utils/tracking.py`：wandb metric flattening。

## 验证命令和结果
- 本地：`python -m py_compile external/VAGEN/verl/verl/workers/rollout/latent_action.py external/VAGEN/verl/verl/workers/fsdp_workers.py external/VAGEN/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py external/VAGEN/verl/verl/workers/critic/dp_critic.py external/VAGEN/verl/verl/utils/tracking.py`：通过。
- 服务器 dgx-40（通过既有 hold job `457147`）：同一组 `py_compile` 命令通过，输出 `PY_COMPILE_OK`。
- 服务器 dgx-40 import smoke：因服务器默认 Python 环境 pandas/numpy 不兼容失败（`pandas` 要求 `numpy>=1.22.4`，当前为 `1.21.5`），未能作为代码验证结论；不是本次改动引入的语法错误。
- `git -C external/VAGEN/verl diff --stat`：提交前显示 6 个修改文件 + 1 个新增文件。

## 待确认问题
- 未移植新版 verl 中依赖新架构的 async SGLang server / `verl/workers/config/rollout.py` / `verl/trainer/config/rollout/rollout.yaml` / agent loop concurrency 改动，因为 legacy verl 没有这些文件/架构，直接 cherry-pick 会引入大范围新版 verl 结构。
- `Qwen3-VL ValueHead` 的 `load_valuehead_model` patch 未直接适配：legacy verl 当前没有该函数；若后续确实使用 TRL value-head critic，需要单独设计 legacy 接入点。
