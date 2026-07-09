# 2026-07-08 VAGEN legacy-dev resume to 1-action / 20-turn / +30 steps

## Goal
- 将 `external/VAGEN` 切到 `nimloth/vagen-legacy-dev`。
- 从服务器 checkpoint `/project/peilab/atst/vagen_ckpt` 继续训练 VAGEN navigation。
- 训练配置要求：`max_actions_per_step=1`、每个 rollout 最多 `20` turn。
- 先续训 `30` 个 global step（当前按 checkpoint metadata 解释为从 step300 续到 step330）。

## Current plan
1. 本地分支把 VAGEN submodule 切到 `nimloth/vagen-legacy-dev`，并验证 train split / prompt 配置仍符合要求。
2. 提交并 push `exp/vagen-1action`。
3. 在 superpod 创建对应远程 worktree，同步到该 commit，初始化 submodule。
4. 新建服务器输出目录与 README；把 `/project/peilab/atst/vagen_ckpt` 以 `global_step_300` 的形式挂到新 run 的 `checkpoints/` 下。
5. 优先使用 **legacy service** 路径（`legacy_preempt_reproduction.slurm` + `run_legacy_reproduction.sh`）启动 1 env node + 1 train node 的 strict ws8 resume 训练，并监控到健康启动。
6. 若 normal 分区先出现 `2 env + 7 train` 的碎片资源，则改用 **non-strict normal fallback**：单独起 legacy env service，再把 actor/critic HF export 转成 fresh ws7 checkpoint（global_step_300），从该 ws7 checkpoint 继续跑到 step330；明确接受“模型权重续上，但 optimizer / lr scheduler 不严格续上”。

## Current status
- 已确认本地新 worktree：`/workspace/remote2/nimloth-exp-vagen-1action`
- 已确认服务器 checkpoint 目录：`/project/peilab/atst/vagen_ckpt`
- 已确认 checkpoint 不是 `global_step_*` 目录，而是裸 `actor/ critic/ data.pt` 结构。
- 已从 `actor/extra_state_world_size_8_rank_0.pt` 读到 lr scheduler `last_epoch=299` / `_step_count=300`，因此当前按 **step300 checkpoint** 处理。
- 已确认当前 `train_resume.slurm` 默认配置已经是 `max_actions_per_step=1` + `max_turns=20`；切到 `nimloth/vagen-legacy-dev` 后会补上 `base_train/common_sense_train` train split 支持。
- 已将 `.gitmodules` 中 `external/VAGEN` 的 branch 改为 `nimloth/vagen-legacy-dev`，并把 submodule pointer 切到 `2046ab16ec4d797759f1f369d696d61113486340`。
- 已 push 本地提交 `650aa6b1f6b235f8395f3964f791a06f30a05365` 到 `origin/exp/vagen-1action`。
- 已在 superpod 创建远程 worktree：`/project/peilab/atst/nimloth/.worktree/exp-vagen-1action`。
- 已确认 superpod 直接 `python3` 缺少 `gym`，不能直接跑当前 VAGEN；因此为 `env_external_4gpu.slurm` 和 `train_resume.slurm` 补上显式 `source /project/peilab/atst/nimloth/.venv/bin/activate`。
- 已在远程 worktree 用 `.venv` 成功 import `vagen.env.navigation.env.NavigationEnv`，并确认 `ValidEvalSets` 包含 `base_train`。
- 已验证 `vagen.server.server` 与 `vagen.trainer.main_ppo` 在远程 worktree + `.venv` 下可导入，说明 legacy service 路径可用。
- 已在 superpod 成功把 legacy service 路径启动到 Ray / dataset build / trainer init 阶段；第一次失败于 **critic 初始化路径错误**：`critic.model.path` 不能指向 actor 的 Qwen2.5-VL HF export，已改为 `/project/peilab/atst/vagen_ckpt/critic/huggingface`。
- 第二次失败暴露出 legacy `verl` 的 critic 初始化分支只靠 `"Qwen2.5-VL" in local_path` 判断是否走 `Qwen2_5_VLForTokenClassification`；当路径是本地 checkpoint 目录时不会命中。已在 `external/VAGEN/verl/verl/workers/fsdp_workers.py` 改为按 `critic_model_config.model_type == "qwen2_5_vl"` 判断，并把补丁 push 到 `verl` / `VAGEN` 远端，再更新 Nimloth submodule pointer。
- latest strict-resume 代码链：Nimloth `61ca730`，VAGEN `fbfd48f`，verl `1acd5b6`。
- 之后又新增 normal fallback helper，并 push Nimloth `d05c172`：
  - `experiments/training/baseline/legacy_env_service.slurm`：单独起 2-GPU legacy BatchEnvServer，并把 `base_url.txt` / `ready` / `failed` 写进 control dir。
  - `experiments/training/baseline/legacy_train_external_service.slurm`：等待 legacy env，先把 actor/critic HF export 转成 **ws7 fresh checkpoint at global_step_300**，再用 `run_legacy_reproduction.sh` 从这个 ws7 checkpoint 继续跑到 step330。
- strict ws8 preempt job `467812` 已取消，避免后续和 normal fallback 重复占资源。
- 当前 latest normal fallback jobs：
  - `468531`：`vagen-legacy-env`，normal，已在 `dgx-37` 启动并通过 health check；当前 legacy service URL：`http://10.23.1.45:5000`。
  - `468532`：`vagen-legacy-train`，normal，目标是任意 1 个 7-GPU normal 节点；当前 `PENDING (Priority)`。
- 当前 normal 侧已经成功占住 `2 env` 资源，但训练侧仍缺整块 `7 GPU` 节点；最新资源快照下 normal 只剩 `dgx-21/dgx-46/dgx-54` 各 1 张空闲 GPU，因此 train job 还没法启动。
- 当前 normal fallback run dir：`/project/peilab/atst/nimloth/outputs/experiments/training/baseline/2026-07-09/vagen_legacydev_non_strict_resume300_to330_ws7_1action_turn20_normal2env`
- 已确认之前尝试直接复用 `train_resume.slurm` / `env_external_4gpu.slurm` 不适配 legacy-dev：
  - `vagen.envs.navigation.serve` 模块不存在；
  - `vagen/gym_agent_dataset.py` 文件不存在；
  - 因此已决定改走 legacy service 方案，而不是继续补现代 external-env 路径。

## Files modified
- `.gitmodules`
- `external/VAGEN` (submodule pointer)
- `configs/training/baseline/legacy_train.yaml`
- `experiments/training/baseline/env_external_4gpu.slurm`
- `experiments/training/baseline/train_resume.slurm`
- `experiments/training/baseline/run_legacy_reproduction.sh`
- `experiments/training/baseline/legacy_env_service.slurm`
- `experiments/training/baseline/legacy_train_external_service.slurm`
- `ai_tasks/ai_progress/2026-07-08_vagen_legacydev_resume_1action.md`
- `external/VAGEN/verl/verl/workers/fsdp_workers.py` (via pushed nested submodule commit)

## Validation so far
- `git ls-remote --heads ... VAGEN.git 'nimloth/*'`
- `git show 2046ab1:vagen/env/navigation/env.py`：确认 `ValidEvalSets` 包含 `base_train/common_sense_train/long_horizon_train`
- 服务器 `torch.load('/project/peilab/atst/vagen_ckpt/data.pt')`
- 服务器 `torch.load('/project/peilab/atst/vagen_ckpt/actor/extra_state_world_size_8_rank_0.pt', weights_only=False)`
- `./.local/scripts/query-resources.sh --only-free-gpu`
- `bash -n experiments/training/baseline/train_resume.slurm experiments/training/baseline/env_external_4gpu.slurm`
- `bash -n experiments/training/baseline/run_legacy_reproduction.sh experiments/training/baseline/legacy_preempt_reproduction.slurm`
- superpod legacy service run `467798`：env service / Ray / dataset build 成功，失败点是 `AutoModelForTokenClassification.from_pretrained(actor_hf)` 不接受 `Qwen2_5_VLConfig`
- nested `verl` patch 验证：`python -m py_compile external/VAGEN/verl/verl/workers/fsdp_workers.py external/VAGEN/verl/verl/models/transformers/modeling_qwen_2_5_vl_patch.py`
- superpod `467812` 在切换到 normal fallback 方案前已取消。
- `bash -n experiments/training/baseline/legacy_env_service.slurm experiments/training/baseline/legacy_train_external_service.slurm`
- superpod 已写入 new run README：`2026-07-09/vagen_legacydev_non_strict_resume300_to330_ws7_1action_turn20_normal2env/README.md`
- superpod `468531` 已在 `dgx-37` 跑起，并记录：allocated `CUDA_VISIBLE_DEVICES=0,1`、AI2-THOR smoke `rc=0` / `rc=0`、legacy env service health OK after 6 checks。
- superpod `468532` 仍在 `PENDING (Priority)`。
- 服务器 `.venv` import smoke：能导入 `vagen.env.navigation.env.NavigationEnv`，并看到 `base_train in ValidEvalSets == True`
- 服务器 `.venv` import smoke：能导入 `vagen.server.server` 与 `vagen.trainer.main_ppo`

## Open questions / assumptions
- 当前按“再训练 30 个 global step”解释为 **300 -> 330**。若人类实际想要别的终点，需要改 `trainer.total_training_steps`。
- strict ws8 resume 仍然是语义最干净的方案；但当前已转向用户确认过的 normal fallback（接受 non-strict resume）。
- normal fallback 的关键限制：虽然 run 会从 `global_step_300` 形式继续到 `330`，但 ws7 checkpoint 是从 actor/critic HF export 新生成的，因此 **optimizer / lr scheduler 状态不会严格继承 ws8 step300 checkpoint**。
- `468531` 已证明当前这次在 `dgx-37` 分到的 physical GPU `0/1` 能通过 AI2-THOR smoke；但这仍然是本次 allocation 结果，不保证别的 allocation 也一样。
