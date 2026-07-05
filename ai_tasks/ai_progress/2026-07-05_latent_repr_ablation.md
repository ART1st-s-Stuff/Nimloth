# 2026-07-05 Latent Representation Ablation 执行进度

## 任务目标

在 `exp/latent-repr-ablation` 分支上按配置驱动方式实现 representation ablation 的基础设施，使后续能通过不同 YAML 配置启动不同实验，而不是为每个实验改代码。

## 当前计划

1. Phase 0：阅读并冻结接口/schema，确认可复用模块。
2. Phase 1：实现 baseline single-latent 的配置驱动离线评估链路。
3. 本地单元测试通过后，提交代码。
4. 需要服务器 smoke 时，启动指定模型 `openai-codex/gpt-5.5` 的子 agent 执行；smoke 不上传 W&B。

## 已完成步骤

- 已在 `ai_tasks/latent_repr_ablation_plan.md` 补充分 Phase 和配置驱动要求，并提交 `2e20222`。
- 已阅读现有 SFT2 dataset、WM predictor/value head、reconstruction evaluator、checkpoint helper。
- 已实现 Phase 1 single `qwen_latent` baseline 的配置驱动离线评估基础设施：严格 YAML schema、Phase-1 validator、module loader、value/predictor metrics、config-driven eval CLI。
- 已修正 eval 汇总逻辑：value/ranking/calibration 和 one-step predictor 指标按全体 encoded transitions 汇总，避免 batch size=1 时 calibration/AUC 失真。
- 已新增 `init.sft2_checkpoint` 便利配置；设置标准 SFT2 checkpoint dir 后可自动推导 qwen/state_proj/wm_predictor/value_head 路径。
- 根据 code-review 子 agent 的 blocking finding，已修复 value head checkpoint 缺失时可能静默使用随机初始化权重的问题：Phase-1 validator 和 loader 现在要求 `value_head_checkpoint/value_head.pt` 真实存在，否则报错。
- 根据 server-smoke 子 agent 结果，已修复 `test_metrics.py` 对 float32 结果精确相等的问题，改用 `pytest.approx`。
- 根据 server-smoke 子 agent 结果，已把 `LatentWMPredictor` / `StateProjector` / `ValueHead` / decoder 导入移入 loader 函数，避免 `import nimloth.representation_ablation.eval` 在未初始化 `external/le-wm` 时失败；真正评估加载 predictor 时仍会要求真实依赖存在。
- 第二次 server smoke 发现 import 仍会经由 `training.sft2.dataset` 和 `nimloth.wm.reconstruction` 触发 `nimloth.wm.__init__`；已进一步把 `TransitionQwenDataset` / `collate_transition_batch` / `extract_qwen_latents` 和 `WMImageDecoder` 导入延迟到评估函数内或 TYPE_CHECKING。
- 第三次 server smoke 确认 `import nimloth.representation_ablation.eval` 已通过，但 `import nimloth.eval.representation_ablation` 仍因 `nimloth.eval.__init__` eager import rollout 触发 le-wm 依赖；已将 `nimloth.eval.__init__` 改为 lazy `__getattr__`。
- 第四次 server smoke 在未初始化 `external/le-wm` / `external/VAGEN`、禁用 GPU/W&B 的条件下通过：representation ablation tests、py_compile、YAML load、`import nimloth.representation_ablation.eval`、`import nimloth.eval.representation_ablation` 均通过。
- 已新增 baseline A 的两个 eval config 模板：value/predictor 与 reconstruction strips。
- 子 agent 已只读溯源 low-success-source SFT2 run：step≈1000 checkpoint 选用 `.../sft2_lejepa_align_fn1_dgx56/ckpt_step1000_preserved`；该 checkpoint 是 LoRA adapter + `vision_full_state.pt`，包含 `state_proj.pt`、`wm_predictor/predictor.pt`、`value_head/value_head.pt`，可用于诊断 eval。
- 已新增诊断 eval config：`configs/eval/representation_ablation/diagnostic_low_success_sft2_step1000_value_predictor.yaml`，指向上述 step1000 checkpoint 与 `sft1_sft_records_vagen79_nimloth_format/val_all.jsonl`。
- 已新增诊断 eval smoke config：`configs/eval/representation_ablation/diagnostic_low_success_sft2_step1000_value_predictor_smoke.yaml`，限制 `max_records=2` / `max_batches=2`。
- 已修复 representation ablation eval 加载 SFT2 adapter-only checkpoint 的问题：若 `qwen_checkpoint` 是 LoRA adapter dir，则从 `adapter_config.json` 读取 base model，配置 LoRA + 加载 adapter 和 `vision_full_state.pt`。
- 已为 Plan B rollout 脚本新增 `ROLLOUT_MODEL_PATH` override，允许直接使用高成功率 step300 HF actor 路径。
- 子 agent 已溯源 SFT2/SFT1 参数：SFT2 使用 SFT1 `epoch_002/hf_merged` 初始化，数据为 vagen79 converted records，SFT2 成功 run 使用 `--no-full-trajectory-batching`、4 DDP ranks over 8 H800、`NIMLOTH_DDP_GPU_STRIDE=2`、LLM LoRA + vision full + EMA。
- 子 agent 已提出基于高成功率 VAGEN step300 的 SFT1+SFT2 重跑计划。人类确认并行执行：A 线用 low-success step1000 debug；B 线执行 Plan B（高成功率 step300 重新采集/转换数据，再跑 SFT1+SFT2）。
- A 线诊断 eval smoke 已完成：Slurm job `465669` 在 `dgx-04` `COMPLETED`，elapsed `00:02:23`，输出 `/project/peilab/atst/nimloth/outputs/experiments/representation_ablation/2026-07-05/diagnostic_low_success_sft2_step1000_smoke_a0a1446`。真实加载 checkpoint 成功，日志含 `vision_full_state_loaded=true`，产出 `summary.json` / `per_item_metrics.csv` / README / metadata。关键 smoke 指标：`num_encoded_transitions=2`，`predictor_1step_mse=0.0015321865`，`predictor_1step_cosine=0.6884475350`，`value_top1_action_acc=1.0`，`value_top2_action_acc=1.0`，`value_chosen_mse=0.0159209650`；AUC 和 depth4/8 为 NaN 是样本太小。
- B 线 Plan B 未能健康启动，已取消后续依赖任务且无 active jobs：Attempt 1 env job `465666` 在 `dgx-24` 失败（AI2-THOR smoke 仅 1/4 GPU 通过）；Attempt 2 env job `465677` 在 `dgx-56` AI2-THOR smoke 4/4 通过但 env server 失败：`No module named vagen.envs.navigation.serve`。rollout arrays `465667` / `465678` 失败，conversion/SFT1/SFT2 dependent jobs `465670-465672`、`465682-465684` 取消。
- B 线 blocking：当前服务器 `/project/peilab/atst/nimloth/external/VAGEN` commit `93c1124...` 有 `vagen.server.server`、`vagen.env.navigation.*`，但没有旧脚本需要的 `vagen.envs.navigation.serve` 和 `vagen.envs.navigation.utils.nimloth_format`。继续 Plan B 需要修改脚本/imports 或选用兼容 VAGEN checkout；本轮没有修改 repo code 或 submodule code。
- 子 agent 已只读溯源旧 step79 VAGEN 来源。直接 artifact 未记录训练时 branch/commit/dirty status；基于远程 Git reflog 重建，最可能来源为 Nimloth `main` commit `2f683f5de25c5eaf0b804f1fbadc507697c336a6`，`external/VAGEN` branch `main` commit `629c270cf069680c70282f615e7f1b83a45684ab`，嵌套 `verl` commit `6360bfe706a00b4ece9105285776b5727ee449b7`。confidence=medium；不能排除当时有未提交改动。W&B retro commit `3342f0cc...` 是事后上传脚本，不是训练 provenance。
- 人类指出 step79 记忆中应基于 `vagen_legacy`。复查服务器 VAGEN remote：`origin/nimloth/vagen-legacy` HEAD 为 `acc0e7550f73da71b66c80f0762fbfcff3905213`（2026-06-21, `Add wm alias for legacy navigation format`, `verl=869ff12...`）；`origin/vagen-legacy` HEAD 为 `787c7e2d36822ad2348255c2e838159327eb320c`；`origin/nimloth/vagen-legacy-dev` HEAD 为 `93c1124aeaa7850098f46f2b708ee224ba894861`。`629c270...` 被 `main`/`nimloth/main` 包含且 tree 中有旧 `vagen/envs/navigation/serve.py`。因此前述 reflog 推断只可作为 medium-confidence 线索，不能作为已确认 provenance。
- 人类确认其判断 step79 大概率来自 `origin/nimloth/vagen-legacy-dev`；后续 Plan B 如果需要选择 VAGEN legacy 路线，直接使用 `origin/nimloth/vagen-legacy-dev` / commit `93c1124aeaa7850098f46f2b708ee224ba894861` 作为工作假设。
- 已修复 Plan B 脚本对 `nimloth/vagen-legacy-dev` API 的兼容性：`env_external_4gpu.slurm` 现在检测旧 `vagen.envs.navigation.serve`，若不存在则使用统一 Hydra server `python -m vagen.server.server server.port=... navigation.devices='[0]' navigation.max_workers=48`；`convert_rollouts.py` 兼容从 `vagen.env.navigation.nimloth_format` 导入 Nimloth action format。
- 已开始 Phase 2 基础实现（尚未接入正式训练/eval 主路径）：新增多 latent marker 展开与 latent token-set extraction helper；新增 token-set WM predictor 和 token-set value head，支持 `(B, K, D)` 输入/输出、rollout、checkpoint save/load，并配套单元测试。
- Phase 2 token-set 服务器轻量 smoke 已通过：server worktree commit `09b1b09`，`python -m pytest tests/representation_ablation -q -p no:cacheprovider` 通过 `15 passed, 3 warnings`；`py_compile` 通过；`import nimloth.representation_ablation.qwen_tokens` 与 `token_set` 通过且不要求 LeWM 初始化。该 smoke 验证的是 token-set 模块本身；后续 `21694a3` 只改 Plan B rollout 脚本。
- Plan B retry 在 env ready 后 rollout array 仍于 Ray startup 前被 SIGTERM，未生成 records/training；子 agent 取消了 downstream jobs。已修复可疑点：移除 rollout array 中 user-wide `ray stop` / `pkill -f ray::`，改为 per-array-job `RAY_PORT` 和 `RAY_TMPDIR`，避免同节点并发 array tasks 互相杀 Ray。
- Plan B retry7 失败诊断完成：env job `465783` 在 `dgx-56` ready，rollout array `465785` 已启动 Ray 并打印 resources，但仍无 shard JSONL；downstream conversion/SFT1/SFT2 未提交。根因是 rollout command 仍调用旧 VAGEN 布局（`vagen.main_ppo` + `vagen/configs/vagen_multiturn`），而当前 legacy-dev VAGEN 使用 `vagen.trainer.main_ppo` + `vagen/trainer/config/ppo_trainer.yaml`，并要求 parquet dataset row 的 `extra_info.env_name/env_config/seed`。
- 已更新 Plan B rollout 脚本：为每个 shard 同时写 legacy YAML 与新 parquet；检测到 `vagen.trainer.main_ppo` 时使用新入口、`rollout_manager.use_service=True`、按 array task 选择一个 env server URL；旧入口保留 fallback。提交/Slurm 脚本现在支持通过 `REPO=/project/peilab/atst/nimloth/.worktree/exp-latent-repr-ablation` 使用服务器 worktree，避免硬编码主 worktree。
- Plan B retry8（Nimloth `aa09f1a`，VAGEN `93c1124`）已完成失败诊断：远程 worktree初始 clean、HEAD 正确、脚本 `bash -n` 通过；env job `465811` 在 `dgx-56`/`preempt` ports `19130-19133` ready，后已取消。rollout job `465813` 先因 retry8 长输出路径导致 Ray `AF_UNIX path length cannot exceed 107 bytes`；用 `/tmp/hs8` symlink wrapper 重交 `465818` 后确认真正进入 `vagen.trainer.main_ppo`，使用 parquet + `rollout_manager.use_service=True`，但在 `RayPPOTrainer._validate_config` 失败：缺少 `actor_rollout_ref.rollout.micro_batch_size` 或 `actor_rollout_ref.rollout.micro_batch_size_per_gpu`（配置中 rollout log-prob micro batch 字段为 `None`）。另见一个 array task 的 Ray worker grpc port `10003` 冲突。没有 env reset 证据，没有 `validation/*/shard_*/300.jsonl`，未做 conversion/SFT1/SFT2。远程 README 已更新：`/project/peilab/atst/nimloth/experiments/navigation_baseline/runs/sft1_rollouts_highsuccess_step300_greedy_parallel_retry8/README_retry8.md`。
- 已修复 retry8 暴露的 Nimloth rollout 脚本问题（未改 submodule）：新 `vagen.trainer.main_ppo` 路径加入 `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1`；Ray temp dir 默认改短到 `/tmp/ray_sft1_${USER}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}`，避免 AF_UNIX path limit；显式隔离 Ray node/object/dashboard/client/metrics/worker ports，并将 `RAY_ADDRESS` 固定为当前 array task 的 Ray head 地址，降低同节点并发串线风险。
- B 线 Plan B retry9 失败诊断完成：远程 worktree clean、HEAD=`8041689`、VAGEN=`93c1124`、脚本 `bash -n` 通过；env job `465938` 在 `dgx-56`/`preempt` ports `19130-19133` ready，后已取消。rollout array `465940_[0-3]` 全部在 Ray startup 前失败：`ray start` 因 `--ray-client-server-port` 要求 `ray[client]`，而服务器 venv 只有 minimal Ray。未进入 `vagen.trainer.main_ppo` / `_validate_config`，无 env reset 证据，无 `validation/*/shard_*/300.jsonl`，未做 conversion/SFT1/SFT2。远程 README/metadata 已更新：`/project/peilab/atst/nimloth/experiments/navigation_baseline/runs/sft1_rollouts_highsuccess_step300_greedy_parallel_retry9/README_retry9.md`、`metadata_retry9.json`。

## 文件修改

- `src/nimloth/representation_ablation/README.md`
- `src/nimloth/representation_ablation/__init__.py`
- `src/nimloth/representation_ablation/config.py`
- `src/nimloth/representation_ablation/modules.py`
- `src/nimloth/representation_ablation/metrics.py`
- `src/nimloth/representation_ablation/eval.py`
- `src/nimloth/representation_ablation/qwen_tokens.py`
- `src/nimloth/representation_ablation/token_set.py`
- `src/nimloth/eval/representation_ablation.py`
- `configs/eval/representation_ablation/a_qwen_latent_value_predictor.yaml`
- `configs/eval/representation_ablation/a_qwen_latent_reconstruction.yaml`
- `configs/eval/representation_ablation/diagnostic_low_success_sft2_step1000_value_predictor.yaml`
- `configs/eval/representation_ablation/diagnostic_low_success_sft2_step1000_value_predictor_smoke.yaml`
- `tests/representation_ablation/test_config.py`
- `tests/representation_ablation/test_metrics.py`
- `tests/representation_ablation/test_token_set.py`
- `experiments/training/sft1/rollouts_greedy_parallel.slurm`
- `experiments/training/sft1/env_external_4gpu.slurm`
- `experiments/training/sft1/submit_env_external_4gpu.sh`
- `experiments/training/sft1/submit_rollouts_greedy.sh`
- `experiments/training/sft1/convert_rollouts.py`
- 本进度文件。

## 验证命令和结果

- `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，3 passed。
- `PYTHONPATH=src ../nimloth-dev/.venv/bin/python - <<'PY' ... load_ablation_config(...)`：两个新增 YAML 模板均可解析。
- `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/representation_ablation/*.py tests/representation_ablation/*.py`：通过。
- `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/eval/representation_ablation.py`：通过。
- 修正 eval 汇总后再次运行 `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/representation_ablation/eval.py`：通过。
- 修正 eval 汇总后再次运行 `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，3 passed。
- 新增 `init.sft2_checkpoint` 后运行 `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，4 passed。
- 新增 `init.sft2_checkpoint` 后两个 eval YAML 模板仍可解析。
- 修复 value head checkpoint 缺失风险后运行 `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，5 passed。
- 修复 value head checkpoint 缺失风险后运行 `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/representation_ablation/config.py src/nimloth/representation_ablation/modules.py tests/representation_ablation/test_config.py`：通过。
- 修复 server smoke 问题后运行 `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，5 passed。
- 修复 server smoke 问题后运行 `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/representation_ablation/*.py tests/representation_ablation/*.py src/nimloth/eval/representation_ablation.py`：通过。
- 第二次修复 import 问题后运行 `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，5 passed。
- 第二次修复 import 问题后运行 `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/representation_ablation/*.py tests/representation_ablation/*.py src/nimloth/eval/representation_ablation.py`：通过。
- 修复 `nimloth.eval.__init__` eager import 后运行 `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/eval/__init__.py src/nimloth/eval/representation_ablation.py src/nimloth/representation_ablation/*.py tests/representation_ablation/*.py`：通过。
- 修复 `nimloth.eval.__init__` eager import 后运行 `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，5 passed。
- 服务器轻量 smoke（commit `7e1aa45`，未初始化 submodules，`CUDA_VISIBLE_DEVICES=""`，W&B disabled）：`python -m pytest tests/representation_ablation -q -p no:cacheprovider` 通过，9 passed；`py_compile` 通过；两个 YAML load 通过；`import nimloth.representation_ablation.eval` 和 `import nimloth.eval.representation_ablation` 均通过。
- `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation -q`：本地失败在 torch import，原因是当前本地环境缺少 `libstdc++.so.6`；服务器可用 torch 环境 smoke 已通过。
- 新增 SFT2 adapter-only eval 加载与 Plan B rollout override 后运行 `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/representation_ablation/modules.py src/nimloth/representation_ablation/eval.py tests/representation_ablation/*.py`：通过。
- 新增诊断 smoke config 后运行 YAML load：`diagnostic_low_success_sft2_step1000_value_predictor*.yaml` 均可解析。
- `bash -n experiments/training/sft1/rollouts_greedy_parallel.slurm`：通过。
- 修复 Plan B VAGEN API 兼容后运行 `bash -n experiments/training/sft1/env_external_4gpu.slurm experiments/training/sft1/convert_rollouts.py experiments/training/sft1/rollouts_greedy_parallel.slurm`：通过。
- 修复 Plan B VAGEN API 兼容后运行 `../nimloth-dev/.venv/bin/python -m py_compile experiments/training/sft1/convert_rollouts.py`：通过。
- 修复 rollout Ray 并发 cleanup 后运行 `bash -n experiments/training/sft1/rollouts_greedy_parallel.slurm`：通过。
- Phase 2 token-set 基础模块实现后运行 `../nimloth-dev/.venv/bin/python -m py_compile src/nimloth/representation_ablation/token_set.py src/nimloth/representation_ablation/qwen_tokens.py tests/representation_ablation/test_token_set.py`：通过。
- Phase 2 token-set 基础模块实现后运行 `PYTHONPATH=src ../nimloth-dev/.venv/bin/python -m pytest tests/representation_ablation/test_config.py -q`：通过，5 passed。
- Phase 2 token-set 服务器轻量 smoke：commit `09b1b09`，`python -m pytest tests/representation_ablation -q -p no:cacheprovider` 通过，15 passed；`py_compile` 通过；qwen_tokens/token_set import 通过。
- A 线服务器诊断 eval smoke：job `465669` completed，exit 0；关键 summary 见“已完成步骤”。
- B 线 Plan B jobs：env/rollout failed，dependent conversion/SFT1/SFT2 cancelled；blocking 为 VAGEN module path/API mismatch，未产生 records/training。
- 旧 step79 VAGEN provenance 只读溯源完成：artifact 无直接 provenance；reflog 重建结论见“已完成步骤”。
- 新 VAGEN rollout command 验证：本地 `bash -n experiments/training/sft1/{submit_env_external_4gpu.sh,submit_rollouts_greedy.sh,env_external_4gpu.slurm,rollouts_greedy_parallel.slurm}` 通过；远程 `/project/peilab/atst/nimloth/.worktree/exp-latent-repr-ablation/external/VAGEN` 执行 `python -m vagen.trainer.main_ppo --cfg job ...`，确认新增 Hydra overrides（parquet files、`rollout_manager.base_url/use_service`、`trainer.val_only`、`max_response_per_turn` 等）可解析；远程用 `datasets.Dataset.from_list(...).to_parquet` 写入并用 pandas 读回 parquet smoke 通过。
- Plan B retry8 服务器验证/诊断：`git status --short --branch` clean、HEAD=`aa09f1a`、`git submodule status external/VAGEN`=`93c1124...`；`bash -n experiments/training/sft1/{submit_env_external_4gpu.sh,env_external_4gpu.slurm,submit_rollouts_greedy.sh,rollouts_greedy_parallel.slurm}` 通过。Slurm env `465811` ready；rollout `465813` failed at Ray socket path length; wrapper rollout `465818` reached `vagen.trainer.main_ppo` then failed config validation (`actor_rollout_ref.rollout` missing micro batch override). `find .../validation -name '*.jsonl'` returned no shard JSONL。
- Retry8 follow-up script 验证：本地 `bash -n experiments/training/sft1/rollouts_greedy_parallel.slurm` 通过；远程 `/project/peilab/atst/nimloth/.worktree/exp-latent-repr-ablation/external/VAGEN` 下执行 `python -m vagen.trainer.main_ppo --cfg job ... actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 ...`，确认新增 micro-batch override 可被 Hydra 解析。
- Plan B retry9：远程 worktree check clean、HEAD=`8041689ca40987a982abc879879bfa790b05e1c8`、VAGEN submodule=`93c1124aeaa7850098f46f2b708ee224ba894861`、SFT1 Slurm scripts `bash -n` 通过。Env job `465938` ready；从 env allocation 内 curl 4 个 `/health` 均 OK；rollout array `465940_[0-3]` 全部失败，traceback 为 `ValueError: Ray Client requires pip package ray[client]`，失败发生在 `ray start --head`，没有 shard JSONL。

## 待确认问题

- A 线下一步：是否运行 full low-success step1000 diagnostic eval（使用非 smoke config、完整 val split），或先根据 smoke 结果调整指标/输出格式。
- B 线下一步：修复/规避 `ray start --ray-client-server-port` 对 `ray[client]` 的依赖（优先移除未使用的 client server port flag），再开新输出目录重试 Plan B；重点确认是否越过 Ray startup、是否进入 `vagen.trainer.main_ppo` 和 config validation、是否发生 env reset、是否写出真实 `validation/*/shard_*/300.jsonl`。若后续发现必须修改 submodule，必须先停止并请求人类确认。
- B 线 SFT2 初始化 checkpoint策略仍需确认：继续为可比性强制使用 SFT1 `epoch_002/hf_merged`，还是改为 SFT1 `best/hf_merged`。
- 若后续需要启动超过 3 分钟的训练/评估，会按实验规则再次向人类确认。
