# RL 训练管线

动态在线模式：Qwen policy 与 VAGEN navigation 环境交互，保存可逐字重放的轨迹，再更新 action policy、WM predictor 和 value head。

## 协议边界

### Prompt

动态 collector 必须使用与 SFT 数据相同的消息：

1. `get_system_prompts_batch()` 只提供 system prompt；不能把它当任务。
2. `reset/step` 返回的 `obs_str` 是 user message，包含首轮 `Human Instruction:`，后续轮包含动作、环境反馈、reward、done 和 `<image>`。
3. source-eval XML 格式通过 `vagen_protocol.source_eval_text_to_nimloth()` 转为 SFT 的 Nimloth 格式；SFT converter 调用同一函数。
4. 当前 assistant turn 先由 policy 生成真实 `<think>...</think>`；框架再插入 k 个 latent query 和 `<|action_start|>`，从八个 action token 的 logits 采样。
5. 历史保存 policy 真实生成的完整 assistant response。PPO 和 latent encoding 逐字重放这些字段，不重新编造 thought、instruction 或 feedback。
6. source checkpoint 使用 `history_window=112`、每轮最多 512 response tokens；20-turn episode 因此保留完整历史。

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

### Trajectory schema v2

每条动态轨迹必须包含：

- `system_prompt`
- `task_instruction`
- `observation_texts`，长度 `T+1`
- `image_paths`，长度 `T+1`
- `assistant_responses`，长度 `T`
- `action_indices` / `action_names` / `action_log_probs`，长度 `T`
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
- `rl.batch_size` 是 transition microbatch size，不是随机保留数量。
- 每轮对全部 transitions 确定性 shuffle，一次 PPO epoch内每条数据恰好消费一次；每个 microbatch 执行 optimizer step。
- advantage 在本轮全部 transitions 上、首个 optimizer step 前统一计算和标准化。
- PPO actor 当前只更新 framework 选择的 action token；生成 thought token 会保存和重放，但尚未纳入 token-level PPO loss。不能把该实现描述成完整复刻 VAGEN 的全 response-token actor loss。

checkpoint 的 `rollout_protocol` / `rollout_protocol.json` 记录 prompt、reward、schema、environment、history、sampling、microbatch、k/query mode 和 validation 协议；resume 必须完全一致。旧协议 checkpoint 会被拒绝。

## 固定 heldout baseline

`configs/training/rl/dynamic_fsdp_k8_baseline20.yaml` 设置 `training.evaluation_only=true`：

- 只运行固定 seeds 1–20 的 heldout `base`；
- greedy `temperature=0`, `top_p=1`；
- optimizer steps 必须为 0；
- 不写 `final/` checkpoint；
- 每条任务必须与 `base.json[seed % len(tasks)]` 精确一致。

在该 baseline 通过前，launcher 只允许 `RUN_MODE=smoke|baseline`，拒绝 quality pilot。

## 文件

| 文件 | 职责 |
|---|---|
| `vagen_protocol.py` | source-eval/SFT text rewrite、任务提取、action/env response、success |
| `rollout.py` | schema v2、单进程 collector、thought generation、inject action distribution |
| `distributed_rollout.py` | rank0 env + all-rank FSDP synchronized collector |
| `trainer.py` | exact replay、returns、all-transition microbatches、checkpoint/resume、eval-only |
| `loss.py` | predictor/value/action-token PPO losses |
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
