# RL training

The RL path uses `AgentRuntime`/`EpisodeRunner` to collect navigation
trajectories, then trains the neural `Agent` world model and optional actor.

## Ownership boundaries

| Owner | Responsibility |
|-------|----------------|
| `nimloth.agent` | Transcript state, prompt version, policy/supervised message construction, episode runner |
| `nimloth.environment` | Action vocabulary, environment session, and VAGEN navigation adapter |
| `nimloth.backbone.qwen25vl` | Qwen policy, VAGEN online adapter, and latent transition encoding |
| `nimloth.rollout` | Model-independent trajectory schema, JSONL storage, sources, and transition expansion |
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
| JSONL | not required | yes | offline WM/value training from separately collected trajectories |

Direct `VAGENNavigationRolloutCollector` use is rejected when `world > 1`: different
episode lengths and failures would make FSDP ranks execute different Qwen
forwards. Generate JSONL separately for distributed training. A training
collector must use an environment dataset whose name ends in `_train`; eval
assets cannot be labeled as training data.

The CLI requires one rollout mode explicitly. A static JSONL collector is
rejected when either Qwen tune mode enables the PPO actor, because its behavior
probabilities were produced by an older policy. Train and validation always
use separate collector instances and sources.

## Data flow

```text
environment system_prompt + obs_str + images
                    |
                    v
       Agent / EpisodeRunner / AgentPromptTemplate
                    |
               QwenAgentPolicy
                    |
     action + exact behavior log probabilities
                    |
                    v
 RolloutTrajectory (structured transcript + audit prompt)
             |                       |
             v                       v
 shared policy-state encoding   exact PPO prompt replay
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

For each trajectory step, the current and next Qwen hidden states are extracted
from the same shared policy query used for action selection. The next-state
query includes the real next observation and all earlier observation/action
turns.

```text
s_t       = state_proj(qwen(policy_prompt_t)[latent_state])
s_{t+1}   = state_proj(qwen(policy_prompt_{t+1})[latent_state])
s_hat     = wm_predictor(s_t, action_t)
L_wm      = mse(s_hat, stop_gradient(s_{t+1}))

Q_t       = value_head(s_t)
L_value   = regression(Q_t[action_t], discounted_return_t) + ranking_loss
```

When actor training is enabled, PPO recomputes `new_log_prob` from the exact
same prompt and the same temperature/top-p transformation as the recorded
behavior policy. The entropy term is calculated from that transformed
distribution, including masked zero-probability actions.

## Modules

| Module | Responsibility |
|--------|----------------|
| `nimloth.rollout` | Model-independent trajectory schema, JSONL, and transition expansion |
| `nimloth.backbone.qwen25vl.rollout` | Qwen latent transition encoding |
| `components.py` | 完整 `Agent`、独立 backbone adapters、placement 和 resume |
| `batch.py` | transition 子采样与张量化 |
| `algorithm.py` | WM/value 前向、stop-gradient 与可选 PPO replay 编排 |
| `objective.py` | WM、value、PPO loss 和指标 |
| `update.py` | backward、optimizer、梯度裁剪与 EMA |
| `loop.py` | collect→encode→update→validate→save iteration 生命周期 |
| `evaluation.py` | Held-out rollout collection and checkpoint metric selection |
| `rollout_runtime.py` | Collector startup constraints and online policy binding |
| `reporting.py` | RL-specific CSV/W&B metric shape over shared util helpers |
| `checkpoint_manager.py` | Runtime component state to checkpoint artifact mapping |
| `trainer.py` | 运行模式校验与依赖装配入口 |
| `checkpoint.py` | 完整模型与 optimizer checkpoint helpers |
| `cli.py` | CLI adapter and independent train/eval collector selection |

## Important restrictions

- The current runtime requires a complete Nimloth `k=1`,
  `latent_query_mode=inject` HF checkpoint. A plain base Qwen checkpoint and a
  standalone PEFT adapter directory do not satisfy this artifact contract.
- New RL JSONL must use the structured Agent schema. Old records that contain
  only `messages` cannot provide exact policy-state/PPO replay and are rejected
  by the trainer.
- JSONL cycling is suitable for offline WM/value training only. The trainer
  fails at startup if a static JSONL source is combined with an enabled actor.
- `latest/` records resumable progress. `best/` is updated only by the explicit
  held-out `validation.checkpoint_metric` (`success_rate` or `avg_reward`).
- LoRA plus full Vision saves `vision_full_state.pt` next to the adapter and
  restores both through the shared Qwen checkpoint helper.
- 当前 WM loss 只通过当前状态更新 StateProjector；下一状态是 stop-gradient
  target。ValueHead 输入也显式 detach，因此 value loss 不更新 StateProjector。
  梯度 ownership 写在 `algorithm.py`，数学目标位于 `objective.py`，并由梯度
  测试保护；后续若改变必须作为单独算法决策处理。

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
