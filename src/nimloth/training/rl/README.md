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
- `sampling_temperature`, `sampling_top_p`, reward, and success metadata.

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
- `agent.planning.enabled: true` 时，在线 rollout 使用 `PlanningPolicy`。每个真实
  environment step 只执行一次 Qwen forward 和一次 `session.step()`，两者之间的
  多步候选搜索全部发生在 WM latent 空间。
- planner 当前用叶节点最大 action-value 作为搜索启发式；模型尚无 reward/done
  head，因此不会把中间 Q-value 相加并伪装成 model-predicted return。
- planner behavior 尚未实现可微且可重放的 PPO 概率，所以
  `agent.planning.enabled` 与 `actor.enabled` 同时开启会在加载模型前报错。
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

The 8-GPU online PPO smoke uses
`experiments/training/rl/run_vllm_online_ppo_smoke.sh`: vLLM first consumes all
eight GPUs for behavior rollout, exits, and the same eight GPUs then run one
FSDP update. This process boundary keeps inference ownership out of the trainer
and makes the policy freshness handoff auditable.
