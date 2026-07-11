# 2026-07-11 RL 管线修复合并与验证

## 任务目标

- 当前环境约定：如果新开 subagent，必须显式指定模型 `openai-codex/gpt-5.6-sol`。
- 将 `d6e1c1f`、`e05bb53` 中已经完成的 RL/FSDP 安全修复合并到 `feat/rl`。
- 保留当前工作区已有的 encoding `max_length=999999` 未提交修改。
- 先完成本地静态检查和单元测试；在满足实验规则后，再进行真实 GPU/环境管线 smoke 验证。

## 当前计划

1. 保存并恢复当前未提交修改。
2. Cherry-pick 两个 RL 安全修复提交，解决仅与旧 merge 记录有关的冲突。
3. 审查合并结果，修复发现的直接阻塞问题。
4. 运行 Python/shell 静态检查与 RL 单元测试。
5. 明确真实 GPU smoke 的入口、数据 split、checkpoint、训练模块、输出和资源后执行或请求确认。
6. 更新进度文件并提交阶段性成果。

## 已完成步骤

- 已确认当前位于规范 worktree `/workspace/remote2/nimloth-feat-rl`，分支为 `feat/rl`。
- 已确认开始时 `feat/rl` 与 `origin/feat/rl` 的 HEAD 均为 `5ac33bc`。
- 已保存并恢复工作区原有 encoding `max_length=999999` 修改，没有覆盖该修改。
- 已审查目标提交并完成 cherry-pick：
  - `afff665`：对应原 `d6e1c1f`，提供 FSDP/JSONL/advantage 安全修复和测试。
  - `212a6a7`：对应原 `e05bb53`，补充本地 WM 模块同步与 JSONL CLI 安全检查。
- 两次 cherry-pick 都只在当前分支不存在的旧 `ai_tasks/merge_dev.md` 上产生 modify/delete 冲突；已保持该旧 merge 记录不进入本分支，其余 RL 代码按原提交合入。
- 本地静态检查通过；本地 Python 环境没有 torch/pytest/transformers/yaml，无法执行 import/pytest runtime 验证。
- 已推送 `feat/rl` 到 `origin/feat/rl`，当前远程分支包含 `afff665`、`212a6a7`、`2542e29`。
- 已在 superpod 使用显式 `.venv-vagen-main/bin/python3` 完成 RL/WM runtime 测试。
- 已建立服务器 detached validation worktree：`/project/peilab/atst/nimloth/.worktree/feat-rl-validation`，commit `bb029e4`；其 VAGEN/le-wm submodule 已按 root gitlink 初始化。
- 已核实当前 `feat/rl` 所指 VAGEN `93c1124` 只包含 `base/common_sense/...` eval datasets，没有 `*_train` datasets；因此不能直接用当前 env collector 生成训练数据做 GPU train smoke。
- 已核实服务器已有 SFT2 warm-start checkpoint 完整，但现有 SFT1 train JSONL 是多动作转换格式，缺少 RL collector 所需的 `action_names/action_log_probs/nav_instruction`，且 action 数与观测图片数不一，不能直接冒充 RL trajectory。
- 进一步审查发现 `experiments/training/rl/rollout_env.py` 也不能作为修复后两阶段管线的数据生产端：它没有保存 final observation、`action_log_probs` 或 `nav_instruction`，输出会出现 `len(image_paths) == num_steps`，而 trainer 要求 `num_steps + 1`；其动作选择仍取普通 action-name token 的末位置 logits，没有使用当前 Nimloth `<|action_start|>`/`<|action_(idx)|>` policy 语义。因此不能用该脚本的输出声称 JSONL→FSDP 管线已验证。

## 文件修改

- Cherry-pick 修改：`src/nimloth/training/rl/`、`experiments/training/rl/README.md`、`tests/training/rl/`。
- `src/nimloth/training/rl/trainer.py`：保留任务开始前已有的 encoding `max_length=999999` 修改。
- `src/nimloth/training/rl/rollout.py`：Env collector 显式接收 dataset/split，拒绝 eval→train 冒充；使用确定 seed；保存每步 action log-probs、final observation 和完整 `trajectories.jsonl`；丢弃结构不完整的轨迹。
- `src/nimloth/training/rl/cli.py`：直接环境训练必须通过 `rollout.eval_sets` 显式提供 `*_train` datasets。
- `experiments/training/rl/rollout_env.py`：移除旧 action-name-token placeholder，复用 `EnvRolloutCollector` 的 Nimloth action policy，并在输出后强校验 RL schema。
- `configs/training/rl/e2e_smoke.yaml`、`experiments/training/rl/run_e2e_smoke.sh`：真实训练 split rollout → 2-rank FSDP step → resume step 验证入口。
- `tests/training/rl/test_rollout_schema.py`：dataset split 与 trajectory schema 测试。
- `ai_tasks/ai_progress/2026-07-11_rl_pipeline_integration_validation.md`：本任务实时进度。

## 验证命令和结果

- `python -m py_compile src/nimloth/training/rl/*.py tests/training/rl/*.py experiments/training/rl/*.py`：通过。
- `bash -n experiments/training/rl/*.slurm experiments/training/rl/*.sh`：通过。
- `PYTHONPATH=src python -m pytest -q tests/training/rl/test_rollout_jsonl.py`：本地未执行（缺少 pytest/torch）；superpod `.venv-vagen-main` 通过，`12 passed`。
- superpod 组合测试：`tests/training/rl/test_rollout_jsonl.py tests/test_wm_planning.py tests/test_wm_predictor_rollout.py tests/test_latent_extraction.py`，`35 passed, 1 warning`；warning 是测试主动调用 `std(unbiased=True)` 验证其单样本 NaN 行为。
- superpod `python -m nimloth.training.rl.cli --help`：通过，确认 `--jsonl-sources` 和 `--experiment-name` 已注册。
- `tests/test_nimloth_wm_navigation_format.py`：未通过 collection；测试仍引用旧 `vagen/envs/navigation/utils/...` 路径，而当前 gitlink `93c1124` 使用 `vagen/env/navigation/...`。这是当前分支既有 test/submodule 路径不一致，和本次 cherry-pick 无关。

## 待确认问题

- 真实 GPU smoke 预计需要加载 Qwen/SFT2 checkpoint 并连接 AI2-THOR env，预计超过 3 分钟；提交前需要按实验规则向人类说明资源、数据、checkpoint、输出和 resume 策略。
- 满足 training-split 硬规则并验证真实管线需要选择路线：
  1. 修复/替换 `rollout_env.py`，使其复用 `EnvRolloutCollector` 的 Nimloth action policy，输出完整兼容 JSONL；使用已有 `exp/vagen-1action` VAGEN commit `e699e0b` 的 `base_train/common_sense_train` 启外部 env。这会新增 train split 参数并形成明确的跨 worktree env 依赖。
  2. 只做不更新参数的 eval rollout + synthetic/离线 trainer smoke；它只能分别验证组件，不能证明真实端到端 RL 管线。
- FSDP full-tune 的 checkpoint 保存/resume 仍缺少独立端到端验证；已批准的 GPU smoke 将跑 iteration 1，再从 `best/` resume 到 iteration 2。

## 2026-07-11 GPU smoke 启动前检查

- 目的：验证修复后的真实训练 split rollout、RL JSONL、2-rank FSDP PPO/WM 更新、checkpoint 和 resume。
- 代码入口：`experiments/training/rl/run_e2e_smoke.sh`。
- 配置：`configs/training/rl/e2e_smoke.yaml`；4 episodes × 最多 2 steps，batch size 2，先 iteration 1，再 resume 到 iteration 2。
- 数据：服务器 `exp/vagen-1action` worktree 的 VAGEN commit `4607097`，`base_train.json` 实际含 1200 tasks；使用 seeds 1..4。该数据文件和 env 代码明确属于 `*_train` dataset。
- 初始化：`outputs/experiments/training/sft2/2026-06-22/sft2_llmlora_visionfull_1epoch_gamma1_ckpt100_keep2_stride2` 的 `export_best_hf` 和 `best/{state_proj,wm_predictor,value_head}`。
- 训练模块：Qwen language model full parameters、WM predictor、value head；冻结 vision tower 和 state projector。
- 输出：`outputs/experiments/training/rl/2026-07-11/post_fsdp_fix_e2e_smoke_retry1`，脚本拒绝复用非空目录。
- Resume：iteration 1 的 `best/` 保存 model/WM/optimizer；第二个 torchrun 使用 `--resume --rl-iterations 2`。
- 监控：trajectory 数/schema、transition 数、wm_mse、value_loss、actor_loss、entropy、global_step，以及 final `rl_state.pt` 的 iteration/global_step。
- 资源：单个 preempt 节点 2 GPU；env+rollout 并行占 2 GPU，之后停止 env 并用 2 GPU FSDP；预计 20–45 分钟。人类已批准。

## 2026-07-11 GPU smoke 运行状态

- 提交 hold job `471933`：preempt、单节点 2 GPU、48 CPU、160G，分配到 `dgx-44` 并进入 RUNNING。
- 在 hold allocation 中启动 `run_e2e_smoke.sh`；实际 commit `5f7ca3f`，env VAGEN `4607097`，env URL `http://10.23.1.101:8500`。
- 最后一次成功监控：env server health 正常，registered env 包含 navigation；rollout 进程已加载 2 个 Qwen checkpoint shards，GPU1 memory 增长到约 4.5 GiB，处于模型加载阶段；尚未确认首条 trajectory。
- 随后连续两次 SSH 均失败并返回 `Connection closed by UNKNOWN port 65535`；人类确认可能是网络波动，之后连接恢复。
- Rollout 成功：`base_train` seeds 1..4，共 4 trajectories / 8 transitions；每条均有 3 images、2 actions、2 组 8-way log-probs，schema guard 返回 `ALL_OK`。
- 首个 2-rank FSDP iteration 成功：`global_step=1`，`wm_mse=4.96267`、`value_loss=0.63232`、`actor_loss=0.0`、`entropy=0.64507`、`elapsed=8.3s`。
- Resume 失败：`best/model.safetensors` 中 Linear weight 被保存为 shape `[0]`，重新加载期望 `[151676,2048]`。原因是旧 `save_rl_checkpoint` 只让 rank0 对 FSDP underlying module 调用 `save_pretrained`，没有让所有 rank 参加 full-state gather；`rl_state.pt` 也只保存 rank0 optimizer shard。
- retry1 状态：失败、不可从其 checkpoint resume；rollout JSONL 可复用，但按输出隔离规则下一次使用全新 `post_fsdp_fix_e2e_smoke_retry2`。
- 本地修复中：FSDP checkpoint 改为所有 rank 进入 `FULL_STATE_DICT` collective，rank0 写完整 HF model；optimizer 按 rank 保存，resume 强制相同 world size 并加载对应 rank shard；trainer 的 periodic/best/final save 改为所有 rank 调用。
