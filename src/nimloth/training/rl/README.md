# RL 训练管线

正式RL后端已决定迁移为**VERL + full actor/full critic**。本目录原有自写trainer只保留diagnostic用途，不再用于quality pilot；下面旧管线约束仍作为数据和语义审计依据。

`verl_adapter.py`负责把一个完整多轮episode转换成一个VERL `DataProto` row：完整system/user/assistant/image transcript作为response，sampled thought/action mask1，latent queries、action delimiters、chat/environment scaffold mask0，每轮reward放在对应采样action token，terminal reward加到最后action，并保留mRoPE、multimodal inputs和审计文本。禁止按turn拆row，否则masked-GAE无法把后续turn reward归因给前面turn。Transformers4.55.4 ID39 world8 exact-replay gate已完成full actor、immutable ref、4.55-native full token critic、masked-GAE、actor/critic更新及完整checkpoint。该gate只证明离线mechanics；在线rollout、StateProjector、WM predictor和WM auxiliary worker仍未接入，完成前禁止把结果描述为正式VERL RL或quality训练。

## 协议边界

### Prompt

动态 collector 必须使用与 SFT 数据相同的消息：

1. `get_system_prompts_batch()` 只提供 system prompt；不能把它当任务。
2. `reset/step` 返回的 `obs_str` 是 user message，包含首轮 `Human Instruction:`，后续轮包含动作、环境反馈、reward、done 和 `<image>`。
3. source-eval XML 格式通过 `vagen_protocol.source_eval_text_to_nimloth()` 转为 SFT 的 Nimloth 格式；SFT converter 调用同一函数。
4. 当前 assistant turn 先由 policy 生成真实 `<think>...</think>`；框架再插入 k 个 latent query 和 `<|action_start|>`，从八个 action token 的 logits 采样。
5. 历史保存 policy 真实生成的完整 assistant response。PPO 和 latent encoding 逐字重放这些字段，不重新编造 thought、instruction 或 feedback。
6. SFT teacher-forcing prefix 使用完整轨迹，因此 runtime 的 `history_window=112` 保留全部20 turns；source VAGEN rollout 生成本身使用 `window_size=5`、每轮最多512 response tokens。

`generate` query mode 尚未实现，动态 RL 只接受显式 `inject` checkpoint。

### Reward

环境配置固定为 SFT collection 的 source-eval 协议：

- `prompt_format=source_eval_mode`
- `step_length=0.3`
- `success_threshold=1.0`
- `per_turn_format_reward=0.01`
- `success_reward=1.0`
- `format_reward=0.0`

collector 原样保存每次 `env.step()` 的 reward，并在 episode 结束后调用 `compute_reward_batch()` 保存 final reward。禁止额外碰撞惩罚或按 reward 阈值推断 success；success 读取 VAGEN `task_success` / trajectory metrics。

action-level return 为：最后一步 reward 加 final reward，然后按 `G_t = r_t + gamma * G_(t+1)` 向后累计。这对应 VAGEN 把每轮 reward 放在该 assistant response 结束位置的多轮语义。旧 SFT JSONL 没有 `step_rewards` 时仍保留历史 terminal-return 解释；新的动态 RL record 不允许缺字段。

### Trajectory schema v3

每条动态轨迹必须包含：

- `system_prompt`
- `task_instruction`
- `observation_texts`，长度 `T+1`
- `image_paths`，长度 `T+1`
- `assistant_responses`，长度 `T`
- `action_indices` / `action_names` / `action_log_probs`，长度 `T`
- `thought_token_ids` / `thought_token_log_probs`，长度 `T`，保存所有采样thought tokens及inference-engine behavior值
- `ppo_old_token_log_probs`，训练rollout长度 `T`，由actor按完整response replay重算，PPO ratio使用该字段
- `reference_token_log_probs`，训练rollout长度 `T`，每轮覆盖thought+action
- `step_rewards`，长度 `T`
- `final_reward`, `reward`, `success`, `latent_token_count`

首轮 instruction 必须与 `task_instruction` 相同；每次 step 返回的 `info.instruction` 也必须相同。legacy/taskless ID11 record 会 fail-fast，不能 resume 或进入 optimizer。

## 分布式动态 rollout

`DistributedEnvRolloutCollector` 使用：

- rank0 独占 HTTP env 和文件写入；
- 所有 rank 同序执行 FSDP policy forward；
- NCCL 承载 FSDP/logits；独立 Gloo group 承载可变延迟 env control；
- rank0 同步采样 thought token 和 action，再广播；
- 任一 prompt、schema、数值或 env 错误都 fail closed，不回退默认动作或零 log-prob。

环境 server 必须来自当前 clean worktree 的 pinned VAGEN `e7cc2d0`；旧 `exp-vagen-1action` worktree 不符合此协议。

## Encode / optimization

- 每个真实 assistant response 提取一个 k-query hidden block。
- terminal observation 没有 assistant response，因此不会伪造 terminal latent query；最后 action 仍用于 actor/value，只有缺真实 next query 的 WM pair 被跳过。
- 动态 online RL 禁止 unconditional chosen-action ranking，`lambda_rank` 必须为 0。
- `rl.batch_size` 是 transition microbatch size，不是随机保留数量。20-turn多模态pilot默认使用1；短smoke的2不能证明长历史显存安全。
- 每轮对全部 transitions 确定性 shuffle，一次 PPO epoch内每条数据恰好消费一次；每个 microbatch 执行 optimizer step。
- `algorithm.adv_estimator`可选：历史`turn_mc`由WM action ValueHead计算turn advantage并广播；`masked_gae`使用独立、全参数训练的Qwen2.5-VL token critic，为每个loss-mask token计算value/return/advantage并按VAGEN公式whiten。正式pilot固定`masked_gae(gamma=1,lam=1,reward_placement=final)`。
- PPO loss覆盖全部采样thought tokens和action token，按VAGEN response loss mask求均值。inject协议中由框架确定性插入的latent queries、`action_start/end`不属于policy采样，因此保持mask0。
- LoRA关闭时的immutable merged SFT2 base作为reference policy；actor使用与VAGEN相同的`low_var_kl`和系数0.001。VAGEN实际代码在actor KL启用时不再同时施加reward KL；runtime也强制两种placement互斥，但实现了可配置的sampled-token reward KL路径。
- `turn_mc`训练原action ValueHead；`masked_gae`冻结该旧head并单独优化Qwen token critic的clipped value loss，同时WM predictor和actor继续更新；dynamic action ranking保持0。
- actor有梯度recompute保持train mode以启用Qwen gradient checkpointing，同时临时把module dropout设为0以保持PPO确定性；LM head只计算sampled thought和action所需position的logits，避免长序列全词表logits。
- 当前k8 launcher使用`flash_attention_2`。Actor和独立critic禁用HF内部GC，把每个decoder/vision block包装为PyTorch external non-reentrant checkpoint，再由FSDP auto-wrap内部raw block。RL LoRA dropout必须固定为0：forward后、checkpoint backward前改变dropout会改变重算算子序列并触发`CheckpointError`。Attention、LoRA dropout、FSDP wrap、activation-checkpoint和recompute协议均写入resume metadata。

checkpoint 的 `rollout_protocol` / `rollout_protocol.json` 还记录advantage estimator、reward placement和critic backend；masked-GAE保存schema4 critic token values、独立critic模型及8份rank-local critic optimizer。Resume必须完全一致，旧协议checkpoint会被拒绝。

## 固定 heldout baseline

`configs/training/rl/dynamic_fsdp_k8_baseline20.yaml` 设置 `training.evaluation_only=true`：

- 只运行固定 seeds 1–20 的 heldout `base`；
- greedy `temperature=0`, `top_p=1`；
- optimizer steps 必须为 0；
- 不写 `final/` checkpoint；
- 每条任务必须与 `base.json[seed % len(tasks)]` 精确一致。

该baseline通过后才允许`RUN_MODE=pilot`。Pilot仍须使用新identity，并通过长历史actor forward+backward显存gate。

## 文件

| 文件 | 职责 |
|---|---|
| `vagen_protocol.py` | source-eval/SFT text rewrite、任务提取、action/env response、success |
| `rollout.py` | schema v3、thought token behavior trace、单进程 collector、inject action distribution |
| `distributed_rollout.py` | rank0 env + all-rank FSDP synchronized collector |
| `trainer.py` | exact replay、returns、all-transition microbatches、checkpoint/resume、eval-only |
| `loss.py` | predictor/value/full-response-token PPO及VAGEN KL losses |
| `checkpoint.py` | model/WM/optimizer checkpoint |
| `cli.py` | config/collector/model wiring |

## 验证

最小定向测试：

```bash
PYTHONPATH=src:external/VAGEN python -m pytest -q \
  tests/training/rl/test_vagen_protocol.py \
  tests/training/rl/test_vagen_rewards.py \
  tests/training/rl/test_rollout_schema.py \
  tests/training/rl/test_dynamic_rollout.py
```

真实 smoke 还必须通过 `dynamic_fsdp_k8_fragmented_4plus2plus2_env1.slurm` 的 artifact gate；该入口用4+2+2 normal碎片组成8个FSDP trainer GPUs，并要求独立的第9个AI2-THOR env GPU。仅 CPU mock不能证明真实 `obs_str`、dataset task 或 reward 路径。Env与FSDP rank共享GPU严格禁止；HTTP health也不能替代真实create+reset gate。
