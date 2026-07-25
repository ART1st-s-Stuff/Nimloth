# RL training

The RL path uses `AgentRuntime`/`EpisodeRunner` to collect navigation
trajectories, then trains the neural `Agent` world model and optional actor.
RL 在线 planning 使用 SFT2 产出的 WM、StateProjector 和 ValueHead 作为
warm start；SFT2 本身不运行这条 rollout 路径。

## Ownership boundaries

| Owner | Responsibility |
|-------|----------------|
| `nimloth.agent` | Transcript、prompt、Qwen→WM planning policy 与 episode runner |
| `nimloth.environment` | Action vocabulary, environment session, and VAGEN navigation adapter |
| `nimloth.backbone.qwen25vl` | Qwen Backbone、prompt 输入、policy 与 PPO replay 适配 |
| `nimloth.rollout` | Trajectory schema、JSONL、连续窗口与 behavior provenance |
| `nimloth.config.agent`, `nimloth.config.rollout` | Stage-independent Agent and rollout configuration |
| `nimloth.config.rl` | Strict typed RL-phase configuration |
| `nimloth.training.rl` | RL 算法、组件装配、optimizer、checkpoint 和训练循环 |
| `experiments/training/rl/rollout_env.py` | Thin standalone rollout entry point and pre-write validation |

RL code must not construct an independent navigation prompt. Online action
selection, PPO replay, and WM state encoding all use the policy query produced
by the shared Agent template.

## Run modes

| Mode | `--env-url` | `--use-jsonl-rollout` | Intended use |
|------|-------------|-----------------------|--------------|
| Single-GPU online | required | no | local integration and online training |
| Static JSONL | not required | yes | offline WM/value training from older trajectories |
| Fresh vLLM JSONL | not required | yes | one multi-rank PPO update after exact-policy vLLM rollout |

Direct `VAGENNavigationRolloutCollector` use is rejected when `world > 1`: different
episode lengths and failures would make FSDP ranks execute different Qwen
forwards. Generate JSONL separately for distributed training. A training
collector must use an environment dataset whose name ends in `_train`; eval
assets cannot be labeled as training data.

The CLI requires one rollout mode explicitly. `actor.enabled` controls PPO and
is independent of `gradient.representation_to_backbone`. Static JSONL is
rejected when PPO is enabled. A `FreshRolloutManifest` binds one vLLM rollout to
the exact policy artifact by content fingerprint and permits exactly one
multi-rank PPO consumption. Train and validation use separate collector sources.

## Data flow

```text
environment system_prompt + obs_str + images
                    |
                    v
       AgentRuntime / AgentPromptTemplate
                    |
          Qwen 编码当前真实 state（一次）
                    |
       WM 搜索候选 action sequence（不执行 env）
                    |
       选择首动作 → EpisodeRunner.session.step（一次）
                    |
                    v
 RolloutTrajectory (structured transcript + audit prompt)
             |                       |
             v                       v
 raw trajectory window sample   exact PPO prompt replay
             |                       |
             v                       |
 shared Backbone input builder       |
             |                       |
             +---- predictor/value/actor training
```

For every step `t`, a complete trajectory stores:

- `system_prompt` and versioned `prompt_template` spec;
- `observation_texts[0:t+2]` and `image_paths[0:t+2]`;
- `action_indices[0:t+1]` and action names;
- the exact unbound `policy_messages[t]` used during rollout;
- the normalized eight-way behavior `action_log_probs[t]`;
- `sampling_temperature`、`sampling_top_p`、逐步`rewards`、聚合`reward`、
  `terminated`/`truncated`和success metadata。

There is always one final observation/image after the last action. Before
training, each saved policy prompt is rebuilt from the structured transcript
and compared with the audit copy. Missing history, stale prompt versions,
non-normalized behavior probabilities, or mismatched prompts are rejected.

Top-p and greedy policies can contain zero-probability actions. They are `-inf`
in memory, serialized as JSON `null`, and restored as `-inf` when read.

## Training semantics

With `H = predictor.history_size`, sampling first selects `H` consecutive
actions and `H + 1` prompts from the same raw trajectory. Only the sampled
windows enter the Backbone, so joint mode retains a real autograd graph and
windows never cross episode boundaries. The final prompt includes the real
next observation and all earlier turns.

```text
hidden    = backbone(policy_prompt_0..H)
states    = state_proj(hidden)
context   = states[:, :H]
targets   = stop_gradient(states[:, 1:H+1])
predicted = wm_predictor(context, actions[:, :H])
L_wm      = mse(predicted, targets)
L_sigreg  = SIGReg(states.transpose(0, 1))

Q         = value_head(context)
L_value   = regression(Q[action], discounted_returns) + ranking_loss
```

梯度模式是显式配置：

- `gradient.representation_to_backbone: true`：WM、value 和 SIGReg 均可训练
  Backbone；下一状态仍只在 WM target 分支 stop-gradient。
- `gradient.representation_to_backbone: false`：Backbone forward 在 no-grad 下
  执行；StateProjector 是否训练仍只由 `freeze.state_proj` 决定。
- `actor.enabled`：单独控制 PPO。Backbone 的可训练参数范围继续由
  `--llm-tune/--vision-tune` 决定，学习率由 `gradient.backbone_lr` 统一管理。

When actor training is enabled, PPO recomputes `new_log_prob` from the exact
same prompt and the same temperature/top-p transformation as the recorded
behavior policy. The entropy term is calculated from that transformed
distribution, including masked zero-probability actions.

`actor.credit_assignment: token`启用真正的turn内token GAE。Qwen同一次replay
forward通过`logits_to_keep`只保留loss-mask位置；TokenValueHead读取这些位置进入
`lm_head`前的hidden state，对每个sampled reasoning/action token分别预测value。
environment-step Monte Carlo return放在该turn最后一个sampled token上，前面的token
immediate reward为0，再用显式`token_credit.gamma`和`gae_lambda`反向计算：

```text
delta_i = r_i + gamma_token * V_token(i+1) - V_token(i)
A_i     = delta_i + gamma_token * lambda_token * A_(i+1)
```

policy advantage会在所有loss-mask token上whiten；critic return不whiten。注入token
仍不参与policy或critic loss。该实现是“真实environment return + turn内token GAE”，
没有把它称为完整VAGEN bi-level GAE：高层目前使用逐步reward的Monte Carlo return，
没有另做跨turn high-level GAE。

## Modules

| Module | Responsibility |
|--------|----------------|
| `nimloth.rollout.windows` | 原始 trajectory 的连续窗口计数与采样 |
| `nimloth.rollout.fresh` | policy artifact 指纹、fresh manifest 和一次性消费契约 |
| `algorithm.py` | multi-step WM/SIGReg、value/PPO 和梯度边界；不持有模型或 optimizer |
| `runtime.py` | prompt→Backbone hidden 的 joint/frozen 模式与可选 policy replay |
| `loop.py` | collect→sample→forward/backward→validate→save 生命周期 |
| `evaluation.py` | Held-out rollout collection and checkpoint metric selection |
| `rollout_runtime.py` | Collector startup constraints and online policy binding |
| `reporting.py` | RL-specific CSV/W&B metric shape over shared util helpers |
| `checkpoint_manager.py` | Runtime component state to checkpoint artifact mapping |
| `trainer.py` | 按执行顺序加载 Agent、设置 FSDP/EMA/optimizer/resume 和 adapters |
| `checkpoint.py` | 完整模型与 optimizer checkpoint helpers |
| `cli.py` | CLI adapter and independent train/eval collector selection |

## Important restrictions

- The runtime requires a complete Nimloth positive-k,
  `latent_query_mode=inject` HF checkpoint. The latent token count is read from
  checkpoint metadata and shared by behavior rollout, PPO replay, and batch
  construction. A plain base Qwen checkpoint and a standalone PEFT adapter
  directory do not satisfy this artifact contract.
- New RL JSONL must use the structured Agent schema. Old records that contain
  only `messages` cannot provide exact policy-state/PPO replay and are rejected
  by the trainer.
- JSONL cycling is suitable for offline WM/value training only. PPO accepts
  only a fresh manifest produced from the exact current policy artifact; that
  manifest cannot be reused for a second optimizer update or process.
- `latest/` records resumable progress. `best/` is updated only by the explicit
  held-out `validation.checkpoint_metric` (`success_rate` or `avg_reward`).
- LoRA plus full Vision saves `vision_full_state.pt` next to the adapter and
  restores both through the shared Qwen checkpoint helper.
- WM target 的下一状态保持 stop-gradient。ValueHead 不再擅自 detach state；
  StateProjector 与 Backbone 的梯度 ownership 分别由 `freeze.state_proj` 和
  `gradient.representation_to_backbone` 控制，并有梯度测试保护。
- `actor.credit_assignment: action` 只对 sampled action token 做 PPO；`turn` 让
  vLLM 采样 CoT，并把同一 environment step 的 Monte Carlo advantage 分配给该轮
  loss-mask reasoning/action token；`token`使用独立TokenValueHead计算逐token GAE。
  token模式要求显式配置`token_credit.gamma`、`gae_lambda`、`value_lr`、
  `value_loss_weight`、`hidden_dim`，并显式选择`rl.truncated_bootstrap`。当前只实现
  `zero`；未确认时配置解析直接失败，不猜测实验参数。
- behavior old log-prob 与 replay 都使用同一 temperature/top-p 分布；注入的 latent
  query、action boundary 和补全 delimiter 不进入 PPO loss。
- `agent.planning.enabled: true` 且actor开启时，独立vLLM rollout先让Qwen生成真实
  CoT；worker extension从同一次多模态forward截取latent hidden和action boundary
  hidden，不加载第二份HF Qwen。多步候选搜索随后全部发生在WM latent空间。
- planner 当前用叶节点最大 action-value 作为搜索启发式；模型尚无 reward/done
  head，因此不会把中间 Q-value 相加并伪装成 model-predicted return。
- planner当前只允许`search_mode=greedy`。每个预测state都选择ValueHead最高动作，
  因此`planning.horizon=2`仍只产生一条两动作候选；trajectory额外保存两个深度各自的
  完整action-value行，验证器据此重建并核对greedy选择。
- planner behavior和teacher都是首动作上的确定性分布，不需要teacher temperature。
  action token不参加PPO或reference KL；Qwen action head以显式
  `actor.planner_distillation_weight`拟合该动作。
- reference KL是actor loss，不改变environment reward、return、advantage或value
  target。冻结reference在独立重放阶段只为采样CoT token写入log-prob；manifest绑定
  reference checkpoint指纹。reward KL尚未实现，任何对应配置会被严格schema拒绝。
- `actor.max_response_tokens`限制完整CoT+协议+action response；实现先扣除协议开销再得到
  reasoning预算。当前VAGEN对齐实验使用512，而不是把512误写成reasoning-only上限。
- 在线 planning 必须从完整 RL resume 或显式 WM、StateProjector、ValueHead
  checkpoint 启动；随机初始化的 planner 不允许控制真实 environment。
- SFT2 和 RL 对 `history_size` 使用相同的 LeWM 语义，warm-start checkpoint
  必须严格匹配。RL 额外的多步预测由 `agent.planning.horizon` 控制，
  不得通过改变 `history_size` 来表达。

Example standalone rollout:

```bash
python experiments/training/rl/rollout_env.py \
  --model /path/to/nimloth-k1-inject-checkpoint \
  --env-url http://127.0.0.1:8000 \
  --output-dir outputs/rollouts/run_001 \
  --eval-set base_train \
  --split train
```

Example JSONL training:

```bash
python -m nimloth.training.rl.cli \
  --config configs/training/rl/defaults.yaml \
  --model /path/to/nimloth-k1-inject-checkpoint \
  --output-dir outputs/experiments/training/rl/run_001 \
  --use-jsonl-rollout \
  --jsonl-sources outputs/rollouts/run_001/trajectories.jsonl
```

The config-sized online PPO smoke uses
`experiments/training/rl/run_vllm_online_ppo_smoke.sh`: vLLM first consumes all
configured GPUs for behavior rollout, exits, and the same allocation then runs
one distributed update. `distributed.world_size` is the number of training
processes; `distributed.gpus_per_rank` is 1 for FSDP or 2 for balanced Qwen
model parallel. Paired replicas finish local backward first, then synchronize
all trainable gradients in deterministic optimizer-parameter order; whole-model
multi-device DDP is not used. Physical GPU count is their product. A model-parallel
launch validates that every rank's Qwen placement actually covers both local
GPUs; CPU/disk offload and single-GPU placement are rejected. This process
boundary keeps inference ownership out of the trainer and makes the policy
freshness handoff auditable.
