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
| `nimloth.backbone.qwen25vl` | Qwen Backbone、prompt 输入与 policy replay 适配 |
| `nimloth.rollout` | Trajectory schema、JSONL、锚点/预测 state 与 behavior provenance |
| `nimloth.config.agent`, `nimloth.config.rollout` | Stage-independent Agent and rollout configuration |
| `nimloth.config.rl` | Strict typed RL-phase configuration |
| `nimloth.training.rl` | RL 算法、组件装配、optimizer、checkpoint 和训练循环；`joint_critic.py`与`joint_scoring.py`保存VAGEN joint-policy尚未接线的critic snapshot及capture→Q纯scoring合同 |
| `experiments/training/rl/rollout_env.py` | Thin standalone rollout entry point and pre-write validation |

`joint_critic.py`严格复用`SharedSlotProjector -> slot mean -> ValueHead`，并可从同一
SFT2/RL checkpoint root加载projector/head、创建显式
`source_step + contract_id + score_dtype`绑定的内存frozen snapshot。
`joint_scoring.py`严格消费same-generation capture v2：episode级`request_id`仅负责sticky
routing，每次forward另有唯一`generation_id`；两者与token table、raw prior logits及
snapshot身份共同进入immutable scoring record。scorer按snapshot参数dtype构造输入，输出
dtype只来自snapshot中已哈希的contract `score_dtype`，调用者不能临时覆盖。当前没有
`joint_behavior.py`另定义identity-bearing response trace与纯assembly helper：严格绑定
request/generation/expected generation-spec、完整response IDs/mask/log-probs和真实decode文本，
再把外部已选guided action组装成behavior/execution envelope；helper没有RNG或current-Q输入。
当前仍没有optimizer、refresh schedule、Ray owner、agent-loop sampler或checkpoint lifecycle；
不得据此移除VAGEN trainer的joint-policy fail-closed门禁。

RL code must not construct an independent navigation prompt. Online action
selection, PPO replay, and WM state encoding all use the policy query produced
by the shared Agent template.

## Run modes

| Mode | `--env-url` | `--use-jsonl-rollout` | Intended use |
|------|-------------|-----------------------|--------------|
| Single-GPU online | required | no | local integration and online training |
| Static JSONL | not required | yes | non-planner offline WM/value training from migrated current-format trajectories |
| Fresh vLLM JSONL | not required | yes | one multi-rank RL update after exact-policy vLLM rollout |

Direct `VAGENNavigationRolloutCollector` use is rejected when `world > 1`: different
episode lengths and failures would make FSDP ranks execute different Qwen
forwards. Generate JSONL separately for distributed training. A training
collector must use an environment dataset whose name ends in `_train`; eval
assets cannot be labeled as training data.

The CLI requires one rollout mode explicitly. `actor.enabled` controls Qwen
action training and is independent of `gradient.representation_to_backbone`.
Static JSONL is rejected when online actor provenance is required. A
`FreshRolloutManifest` binds one vLLM rollout to
the exact policy, planner, trajectory, and optional reference artifacts by content
fingerprint. Consumption is marked in-progress before optimization and committed
only after a resumable post-update checkpoint exists. Train and validation use
separate collector sources. Planner PPO critic rejects ordinary static JSONL because
its saved decision state cannot prove a match with the current Qwen、StateProjector
and ValueHead checkpoints; planner JSONL must use the fresh manifest contract.

## Data flow

```text
environment system_prompt + obs_str + images
                    |
                    v
       AgentRuntime / AgentPromptTemplate
                    |
       Qwen 编码当前真实step的CoT state
                    |
          WM+ValueHead搜索action sequence
                    |
          只执行最佳候选的首动作
                    |
                    v
 RolloutTrajectory (每步真实Qwen state + 独立search trace)
                         |
                         v
      完整prefix Qwen -> PlannerPolicyHead + ValueHead + WM/DINO objective
                         |
              one optimizer epoch per fresh batch
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

Planner在每个真实environment step重新运行Qwen和搜索。设`P = planning.horizon`，
`P`只表示每次搜索向前模拟的候选长度；environment只执行最佳候选的首动作。trajectory
保存每一步真实CoT、Qwen hidden、投影state和独立search trace，不把上一轮候选尾部
当作下一步的真实state或实际动作。

训练使用完整episode，不采样window。每个fresh rollout batch先用update前的
ValueHead在trajectory保存的真实decision state上计算并冻结
`old_Q_t = Q_old(s_t, executed_action_t)`；PlannerPolicyHead同时冻结实际执行动作的
behavior log-prob。`old_Q`不是MCTS root score，也不是Qwen action-token log-prob。
每个fresh rollout batch只执行一个包含全部objective的optimizer epoch：

```text
old_Q = frozen(value_head(saved_rollout_decision_state)[executed_action])
old_logp = persisted PlannerPolicyHead behavior log-prob

for each real transition t:
    prefix_t = persisted prompt + all previous real history + current real CoT
    hidden_t = qwen(prefix_t)                  # current differentiable graph
    state_t = state_proj(hidden_t)
    Q_t = value_head(state_t)[executed_action_t]
    Q_clip_t = old_Q_t + clamp(Q_t - old_Q_t,
                               -ppo_clip_range, +ppo_clip_range)
    L_value = max((Q_t - return_t)^2, (Q_clip_t - return_t)^2)

    context = detached_real_states[-history_size:]
    context[-1] = state_t
    predicted_next = wm(context, previous_actions + executed_action_t)[-1]
    L_state = mse(predicted_next, saved_real_state[t+1])
    L_dino = mse(predicted_next, frozen_dino(next_image_t))

    if PlannerPolicyHead is enabled:
        logp = planner_policy(state_t)[executed_action_t]
        ratio = exp(logp - old_logp)
        L_policy = clipped_action_policy_loss(ratio, frozen_MC_advantage)
    else:
        L_policy = 0

    backward((lambda_wm * L_state + lambda_dino * L_dino
              + L_value + L_policy) / total_real_transitions)
optimizer.step()       # exactly once
global_step += 1       # only after checkpoint + fresh-consumption commit
```

`history_size`只限制WM predictor在一个预测位置最多读取多少个真实过去state；它不决定
Qwen调用次数或搜索长度。每个transition重新构建一次完整prefix的Qwen graph：历史
token/CoT是固定输入，但这次forward中处理历史的激活参与反传。backward结束后释放该
step的graph，不连接以前step已经释放的graph。ValueHead输入是可微的当前`state_t`，
所以clipped critic梯度经过ValueHead、StateProjector和本次完整Qwen prefix；它不经过
Qwen的`lm_head`，也不要求执行动作是action-token logit最大的动作。WM predictor
由独立的state/DINO loss以及多步rollout中后续decision-state value训练。planner配置固定
`value_head.lambda_rank=0`，critic只直接监督执行action的slot。所有planner配置必须显式
`ppo_epochs=1`；schema和runtime都拒绝其他值。由于同一fresh batch不再做参数更新后的重复
pass，严格同checkpoint rollout下的首轮policy ratio应为1、clip fraction应为0；clipped
公式仍用于验证behavior/current分布一致性，但不得再据此引入额外epoch。
SFT2 与 RL 的 state MSE 和 predicted-state DINO MSE 由同一个公共 objective 计算；
SFT2 从离线 cache 读取 DINO target，RL 按轨迹中的真实 next-image 路径使用固定
revision 的 frozen DINOv2 teacher，并缓存已计算的 target。RL loop 在 objective
开始前批量装配本轮实际需要的 target；algorithm 只接收已经与真实transition或
sequence next observation 对齐的 tensor，不读取其他 step 的图像。

planner不包含Qwen action objective：不重放Qwen action logits，也不蒸馏或PPO训练
Qwen action prior。这里的PPO只指ValueHead critic的frozen-old clipped regression；
`actor.enabled`只表示未启用planner时的直接Qwen actor PPO路线。

以下连续window路径只服务于未启用planner的离线/直接policy训练。With
`H = predictor.history_size`, it selects `H` consecutive actions and `H + 1`
states from the same raw trajectory; windows never cross episode boundaries.

```text
hidden    = rollout_qwen_hidden[:, window_start:window_start+H+1]
states    = state_proj(hidden)
context   = states[:, :H]
targets   = stop_gradient(states[:, 1:H+1])
predicted = wm_predictor(context, actions[:, :H])
L_wm      = mse(predicted, targets)
L_sigreg  = SIGReg(states.transpose(0, 1))

Q         = value_head(context)
L_value   = regression(Q[action], discounted_returns) + ranking_loss
```

state来源与梯度模式是显式配置：

- `gradient.state_source: recompute`：按时间位置执行Qwen state forward；可配合
  `gradient.representation_to_backbone: true`让WM、value和SIGReg训练Backbone，
  WM的下一状态监督值整体detach；同一个state在它作为current state时训练
  StateProjector。
- `gradient.state_source: rollout`：只读取trajectory明确保存的Qwen hidden，不执行
  state Qwen forward，且必须设置`gradient.representation_to_backbone: false`。
  该模式只用于不需要把representation loss传回Qwen的非planner路线。
- planner固定要求`gradient.state_source: recompute`和
  `gradient.representation_to_backbone: true`，并在每个真实transition重算完整prefix。
- 两种来源不会在batch内自动切换，也不会根据字段是否为空推断。StateProjector是否
  训练仍只由`freeze.state_proj`决定。
- `actor.enabled`：单独控制 Qwen action objective。Backbone 的可训练参数范围继续由
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
| `nimloth.rollout.fresh` | policy/planner/trajectory指纹、fresh manifest和事务式消费契约 |
| `algorithm.py` | planner transition或连续sequence的WM/value/PPO计算图；不持有模型或optimizer |
| `value.py` | planner执行动作的frozen-old PPO clipped critic objective |
| `runtime.py` | prompt→Backbone hidden 的 joint/frozen 模式与可选 policy replay |
| `planner_verl_adapter.py` | action-level Planner DataProto schema和等长rank batch packing输入 |
| `planner_verl_worker.py` | 单一complete-objective FSDP root、Ray worker和不可逆optimizer生命周期 |
| `planner_verl_factory.py` | 从显式weights-only artifacts装配未包装Agent与分组optimizer factory |
| `planner_verl_gate_factory.py` | 仅供真实Ray/FSDP mechanics gate使用的tiny可导入模型 |
| `planner_verl_checkpoint.py` | exact-world-size FSDP model/optimizer/RNG sharded checkpoint |
| `planner_verl_driver.py` | checkpoint atomic publish→fresh consumption commit事务边界 |
| `loop.py` | collect→sample→forward/backward→validate→save 生命周期 |
| `evaluation.py` | Held-out rollout collection and checkpoint metric selection |
| `rollout_runtime.py` | Collector startup constraints and online policy binding |
| `reporting.py` | RL-specific CSV/W&B metric shape over shared util helpers |
| `checkpoint_manager.py` | Runtime component state to checkpoint artifact mapping |
| `trainer.py` | 按执行顺序加载 Agent、设置 FSDP/视觉 Backbone EMA/optimizer/resume 和 adapters |
| `checkpoint.py` | 完整模型与 optimizer checkpoint helpers |
| `cli.py` | CLI adapter and independent train/eval collector selection |

## Important restrictions

- The runtime requires a complete Nimloth positive-k,
  `latent_query_mode=inject` HF checkpoint. The latent token count is read from
  checkpoint metadata and shared by behavior rollout, PPO replay, and batch
  construction. A plain base Qwen checkpoint and a standalone PEFT adapter
  directory do not satisfy this artifact contract.
- RL/SFT2 JSONL必须带`record_format: nimloth_trajectory_v1`并使用结构化Agent字段。
  未版本化记录先运行`python -m nimloth.rollout.migration`；训练读取器不做字段别名、
  prompt默认值或reward语义兼容。旧记录无法提供的token trace、planner state和真实
  terminal CoT不会由迁移器伪造；对应训练目标会拒绝这类记录。
- JSONL cycling is suitable for offline WM/value training only. PPO accepts
  only a fresh manifest whose current policy/planner and immutable trajectory
  fingerprints all match; a committed or unresolved in-progress consumption
  cannot be reused for another optimizer update.
- A multi-rank exception leaves fresh consumption `in_progress` and exits
  without an abort broadcast or cleanup barrier. This fail-closed behavior
  preserves the first rank-local traceback instead of hiding it behind a new
  mismatched collective. Single-rank failures before `optimizer.step()` may
  still release the claim synchronously.
- `latest/` records resumable progress. `best/` is updated only by the explicit
  held-out `validation.checkpoint_metric` (`success_rate` or `avg_reward`).
- LoRA plus full Vision saves `vision_full_state.pt` next to the adapter and
  restores both through the shared Qwen checkpoint helper.
- WM的下一状态监督值不更新Backbone或StateProjector；StateProjector只从
  current/start state路径训练。ValueHead不擅自detach current state；StateProjector
  与Backbone的梯度ownership分别由`freeze.state_proj`和
  `gradient.representation_to_backbone`控制，并有梯度测试保护。
- `actor.credit_assignment: action` 只对 sampled action token 做 PPO；`turn` 让
  vLLM 采样 CoT，并把同一 environment step 的 Monte Carlo advantage 分配给该轮
  loss-mask reasoning/action token；`token`使用独立TokenValueHead计算逐token GAE。
  token模式要求显式配置`token_credit.gamma`、`gae_lambda`、`value_lr`、
  `value_loss_weight`、`hidden_dim`，并显式选择`rl.truncated_bootstrap`。当前只实现
  `zero`；未确认时配置解析直接失败，不猜测实验参数。
- rollout behavior log-prob 与 replay 都使用同一 temperature/top-p 分布；注入的 latent
  query、action boundary 和补全 delimiter 不进入 PPO loss。
- `agent.planning.enabled: true`时，独立vLLM rollout在每个environment step让Qwen生成
  真实CoT；worker extension从同一次多模态forward截取latent hidden，不加载第二份
  HF Qwen。每次搜索只执行首动作，下一步用真实observation重新规划；terminal
  observation仍额外生成真实CoT/hidden作为最后一个固定state target。
- planner只用叶节点action-value作为搜索启发式；模型尚无reward/done head，
  因此不会把中间MC-return prediction相加并伪装成model-predicted return。
- planner支持`greedy`、`exhaustive`、`beam`和`mcts`。当前RL首轮方案仍固定为
  `greedy`；MCTS首先用于SFT2完成后、RL开始前的独立成功率评估。
  `exhaustive`批量模拟全部
  `action_count ** horizon`条latent动作序列；`beam`逐层批量扩展并按叶节点启发式裁剪；
  H=1 MCTS以SFT2的`prediction_horizon=K`为树深，用UCT选择节点，并以
  `Q(predicted_state_{K-1}, final_simulated_action)`做leaf evaluation。trajectory
  在每个真实step保存候选序列、叶值和root score；MCTS还保存candidate/root visits、
  simulation数和exploration常数。历史H=2
  exhaustive smoke仅是旧实验事实，不是后续默认方案。
- planner behavior是首动作上的确定性分布。所有planner response token的PPO mask均关闭；
  action token不参加蒸馏、actor PPO或reference KL。Qwen的训练信号来自可微的
  ValueHead critic/WM state路径；critic PPO不直接更新`lm_head`。
- reference KL是actor loss，不改变environment reward、return、advantage或value
  target。冻结reference在独立重放阶段只为采样CoT token写入log-prob；manifest绑定
  reference checkpoint指纹。reward KL尚未实现，任何对应配置会被严格schema拒绝。
- `actor.max_response_tokens`限制完整CoT+协议+action response；实现先扣除协议开销再得到
  reasoning预算。当前VAGEN对齐实验使用512，而不是把512误写成reasoning-only上限。
- `actor.max_state_tokens`限制processor展开图片后的完整Qwen state prefix（截至
  `action_start`）。rollout在超限action执行前截断，训练在Qwen forward前对实际
  `input_ids`重复检查；该门禁不替代对并行拓扑和其他OOM来源的验证。
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

The config-sized online RL smoke uses
`experiments/training/rl/run_vllm_online_ppo_smoke.sh`: vLLM first consumes all
configured GPUs for behavior rollout, exits, and the same allocation then runs
one distributed update. `distributed.world_size` is the number of training
processes; `distributed.gpus_per_rank` is 1 for FSDP or 2 for balanced Qwen
model parallel. Each paired replica registers Qwen, the world model, and any
configured token value head under one RL training-step module. Distributed replicas wrap
that complete multi-device module once with official
`DistributedDataParallel(device_ids=None)`, so one reducer owns every gradient
that contributes to the losses. Planner TD and MC calls use different parameter
subsets, so the official reducer uses dynamic unused-parameter detection; there
is no manual gradient averaging. Physical GPU count is the product of world
size and GPUs per rank. A model-parallel launch validates that every rank's Qwen
placement actually covers both local GPUs; CPU/disk offload and single-GPU
placement are rejected. This process boundary keeps inference ownership out of
the trainer and makes the policy freshness handoff auditable.

Formal online training wraps that single-update boundary with
`run_vllm_online_ppo_full.sh`. Every optimizer step gets a newly fingerprinted
rollout from the preceding checkpoint. `--resume-checkpoint` lets the trainer
load an immutable pre-update snapshot while writing the successor to
`output/latest`. A completed training state is serialized once; periodic and
final names are immutable hard-linked snapshots of those exact bytes.
`--defer-final-checkpoint` postpones the final name until the configured last
iteration. These flags change checkpoint lifecycle only, not the rollout,
loss, gradient, or optimizer semantics.
