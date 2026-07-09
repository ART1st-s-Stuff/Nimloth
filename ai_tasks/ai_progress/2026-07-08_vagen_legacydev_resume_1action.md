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
- 当前 latest env/train 状态：
  - `468531`：`vagen-legacy-env`，normal，已在 `dgx-37` 启动并通过 health check；当前 legacy service URL：`http://10.23.1.45:5000`。
  - `468532`：`vagen-legacy-train`，曾在 normal `dgx-26` 实际启动，但很快失败；失败点不是资源，而是 conversion helper 误用了现代 `vagen.main_ppo` import，报 `ModuleNotFoundError: No module named 'vagen.main_ppo'`。
- 因用户后续要求，train 一度改投到 **preempt `dgx-39`** 的 non-strict ws7 fallback；`468756` 曾在 `dgx-39` 真正启动，但随后又失败在 legacy conversion helper 的 Hydra config path 写错（它去找了不存在的 `experiments/external/VAGEN/...`）。
- 之后已回到用户更想要的 **strict ws8 dgx-39 抢占方案**：重用 `468531` 的 external legacy env，在新 run dir 下把 `/project/peilab/atst/vagen_ckpt` 挂成 `checkpoints/global_step_300` 并写入 latest marker，再提交 8-GPU strict resume train job `468767`。
- `468767` 当前状态：`PENDING (ReqNodeNotAvail,_UnavailableNodes:dgx-39)`；`sinfo -n dgx-39` 显示该节点当前是 `planned`，所以虽然表面上没有别的 running job，占用请求仍然进不去。
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
- `experiments/training/baseline/convert_legacy_vagen_hf_to_world_size.py`
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
- superpod `468532` failure log：Ray 7 GPU startup 正常，但 conversion helper 在 `convert_vagen_actor_only_to_world_size.py` 报 `ModuleNotFoundError: No module named 'vagen.main_ppo'`。
- 本地已新增 legacy 专用 conversion helper `convert_legacy_vagen_hf_to_world_size.py`，并把 `legacy_train_external_service.slurm` 的默认 `CONVERT_PY` 切到该 helper；相关 Nimloth code commit：`1947043`。
- superpod `468756` 在 `dgx-39` 真正启动后，新 helper 又失败于 Hydra main config path：`MissingConfigException: Primary config directory not found`，它错误拼成了 `experiments/external/VAGEN/vagen/trainer/config`。
- 本地随后已修复两点（当前 Nimloth commit `d658e41`）：
  - `convert_legacy_vagen_hf_to_world_size.py` 的 Hydra `config_path` 改为正确的 `../../../external/VAGEN/vagen/trainer/config`。
  - `legacy_train_external_service.slurm` 新增 `RAY_NUM_CPUS` 参数化，便于后续在碎片 4-GPU normal allocation 上把 Ray head CPU 数降到当前 allocation 能承受的范围。
- 随后转为 strict ws8 路线：superpod worktree 已 reset 到 repo head `d658e41`，新建 strict run dir `2026-07-09/vagen_legacydev_strict_resume300_to330_1action_turn20_ws8_extenv_dgx37_dgx39`，并提交 train job `468767`；但因用户随后明确要求优先走 4-GPU hold 方案，该 strict job 已取消，避免与 ws4 fallback 并发。
- 又尝试按用户要求抢 `dgx-27` 做 4-GPU bash hold：最初看到它有 4 张空闲 GPU，但在 hold job 真正运行前该节点碎片资源已被别的任务吃掉；当前服务器已没有任何 **立刻可用** 的 4-GPU 整块 normal 节点。
- 为了继续按“先占节点、再修、再 `srun`”的要求推进，已改为提交一个通用 normal 4-GPU hold：`468852 hold-1n4g`，资源请求为 `1 node / 4 GPU / 80 CPU / 160G mem`。它随后实际落在 `dgx-27`，当前仍在 `RUNNING`。
- 已为 ws4 fallback 建好 run dir：`2026-07-09/vagen_legacydev_non_strict_resume300_to330_ws4_1action_turn20_hold4g`，并在 held node `dgx-27` 上多次 `srun` 迭代修复后重新拉起。
- 迄今在 ws4 held run 上已依次修掉：
  1. legacy conversion helper 的 Hydra config path；
  2. legacy config 中缺失字段需要 `+actor_rollout_ref.rollout.agent.*` / `+actor_rollout_ref.ref.use_ref=False`；
  3. conversion 不能用假 `dummy.yaml` 冒充 parquet，已改为先用 `legacy_train.yaml` / `legacy_val.yaml` 真实生成 parquet；
  4. `/project/peilab/atst/vagen_ckpt/critic/huggingface` 没有模型权重文件，ws4 fallback 现改为用 actor HF export 初始化 critic；
  5. legacy `FSDPCheckpointManager.load_checkpoint` 在 torch 2.6+ 下需要 `weights_only=False` 才能读回 `extra_state`；
  6. legacy `vLLMRollout` 对 Qwen2.5-VL local checkpoint 也要按 `model_hf_config.model_type == "qwen2_5_vl"` 走 `limit_mm_per_prompt={"image": ...}`，不能只看路径字符串。
- 最新一次 held `srun` 已经达到：
  - ws4 actor/critic checkpoint conversion 成功，生成 `checkpoints/global_step_300/{actor,critic}/model_world_size_4_rank_{0..3}.pt`；
  - `latest_checkpointed_iteration.txt = 300`；
  - 训练主流程成功从这个 ws4 checkpoint **resume 到 global_step_300**；
  - 已进入 **validation at global step 300**，并连续完成多轮 vLLM generation / env-service rollout。
- 之后再次核实时发现：前面这次 held `srun` **并没有真的死掉**。它继续作为 Slurm job step `468852.8` 挂在 `dgx-27` 上运行，并且实际从 `global_step_300` 推进到了 `global_step_308`。
- 这次 ws4 hold-first fallback 的阶段性历史与当前状态：
  - 首次 held train step `468852.8` 已在 `21:02:48 HKT` 失败退出；它在退出前完成了 `validation@300`，并把 checkpoint 推进到 `global_step_308`；
  - 直接失败信号仍然是：Ray worker `WorkerDict`（pid `2151591`）在 `external/VAGEN/verl/verl/workers/fsdp_workers.py:492 generate_sequences` 路径里触发 **`Fatal Python error: none_dealloc: deallocating None`**，随后主任务报 `ray.exceptions.ActorDiedError`；当前没有证据足以把它断定为 OOM；
  - 按用户要求继续跑时，又发现一个 launcher 侧细节：如果仍用 `SOURCE_CHECKPOINT_STEP=300` 重启，`legacy_train_external_service.slurm` 会在“ws checkpoint already present; skip conversion”分支里把 `latest_checkpointed_iteration.txt` 重写回 `300`，导致 `resume_mode=auto` 错误地再次从 `global_step_300` 起跑；
  - 为避免浪费资源，我已主动取消那次误从 `300` 重启的新 step；
  - 之后改用 **`SOURCE_CHECKPOINT_STEP=308`** 再次拉起 held step，当前新的 train step 是 **`468852.23`**；
  - 这次已经确认：`latest_checkpointed_iteration.txt` 保持为 `308`，日志明确写出 `Found checkpoint ... global_step_308`、`Setting global step to 308`、`Resuming from .../global_step_308`；
  - 后续监控已确认它继续完成了 `validation@308`，并把训练推进到 `global_step_312`；
  - 目前已写出 `validation/308.jsonl`、`validation/310.jsonl`，并生成 `checkpoints/global_step_309`、`global_step_310`、`global_step_311`、`global_step_312`；`latest_checkpointed_iteration.txt` 当前是 `312`；
  - 日志里已经出现 `[DEBUG] step 309 rollout ends`、`[DEBUG] step 310 rollout ends`、`[DEBUG] validation at global step 310 begins/ends`、`[DEBUG] step 311 rollout ends`、`[DEBUG] step 312 rollout ends`；
  - 当前 `468852.23` 仍在 `RUNNING`，`python -m vagen.trainer.main_ppo` 进程存活；到目前为止，这次正确从 `308` 恢复后的 run 已经稳定越过旧崩溃点，并且**还没有再次出现新的 `none_dealloc` / `ActorDiedError`**。
- 我随后按 repo 里的 wandb 说明补查了 canonical 路径：
  - `experiments/training/baseline/README.md` 把 `launch_val_wandb_watcher.sh` + `val_wandb_watcher.slurm` 定义为“轮询 checkpoint、跑 val_only、把 val curve 上传到 wandb”的规范链路；
  - `experiments/training/baseline/common_env.sh` 会从 `flower/.env` / `.env` 导入 `WANDB_API_KEY`，并把 `WANDB_DIR` 设到 repo cache；
  - 当前 legacy run 本身已经通过 `trainer.logger=['console','wandb']` 自动在线同步，但每次重启都会新开一个 wandb run；
  - 为了在**不额外新占 val watcher 资源**的前提下把当前累计进度上传出去，我直接用了 repo 现成的 `experiments/navigation_baseline/upload_retry2_wandb_from_log.py --mode online`，把 `legacy_train.log` 里的累计 console metrics 追溯上传成一个 retrospective wandb run。
- retrospective 上传结果：
  - 新 run 名：`vagen_legacydev_non_strict_resume300_to330_ws4_1action_turn20_hold4g_console_retro`
  - wandb run id：`jimkqsm6`
  - 覆盖 step：`300..312`（共 13 个 parsed step）
  - 输出文件：`wandb_retro/metrics_from_console.csv`、`wandb_retro/metrics_from_console.json`
  - 用途：把多次 resume/重启切碎的自动 wandb run 重新汇总成一条更完整的进度曲线，方便人类直接看当前累计进展。
- 服务器 `.venv` import smoke：能导入 `vagen.env.navigation.env.NavigationEnv`，并看到 `base_train in ValidEvalSets == True`
- 服务器 `.venv` import smoke：能导入 `vagen.server.server` 与 `vagen.trainer.main_ppo`

## Open questions / assumptions
- 当前按“再训练 30 个 global step”解释为 **300 -> 330**。若人类实际想要别的终点，需要改 `trainer.total_training_steps`。
- strict ws8 resume 仍然是语义最干净的方案；但当前已转向用户确认过的 normal fallback（接受 non-strict resume）。
- normal fallback 的关键限制：虽然 run 会从 `global_step_300` 形式继续到 `330`，但 ws7 checkpoint 是从 actor/critic HF export 新生成的，因此 **optimizer / lr scheduler 状态不会严格继承 ws8 step300 checkpoint**。
- `468531` 已证明当前这次在 `dgx-37` 分到的 physical GPU `0/1` 能通过 AI2-THOR smoke；但这仍然是本次 allocation 结果，不保证别的 allocation 也一样。
- 当前要验证的新点：strict ws8 external-env resume 在 `dgx-39` 可用时，是否能直接跳过 conversion、从 symlinked `global_step_300` checkpoint 成功 load actor/critic shards 并进入 val-before-train。
- 当前阻塞再次变化：已经证明可以在保留当前 hold 的前提下从 `global_step_308` 正确重启，并稳定推进到 `global_step_312`；下一步重点变成：它能否继续长期稳定跑到 `330`，还是会在更后面的 rollout/validation 阶段再次命中同一个 `none_dealloc` 底层崩溃。
