# RL 训练与 planning 方案（已退役历史方案）

本文保留截至 2026-07-25 关于 planner distillation、Qwen policy credit
和长期 PPO 方案的历史讨论，不再是当前 planner 的实施依据。

> 2026-07-27 最新人类指令已覆盖本文中所有 planner distillation、
> planner action PPO 和稀疏多步执行方案。当前依据是
> `ai_tasks/ai_progress/2026-07-27_rl_actor_receding_horizon_refactor.md`：
> WM + ValueHead 每个真实 environment step 重新搜索 `k` 步，只执行最佳
> 候选的首动作；不监督 Qwen action prior；每个 transition 使用独立的
> 完整真实 prefix Qwen graph，让 predicted-next-state value loss 经 WM 回传到
> 当前 forward 内的全部历史 token 激活。以下内容仅用于追溯旧设计，
> 不得用于启动、恢复或解释当前 planner 实验。

## 1. 已确认的约束

1. 当前目标实验中的“预测 2 轮”明确指
   `agent.planning.horizon=2`，不指 `predictor.history_size=2`，也不指
   `rl.max_steps_per_episode=2`。
2. 今后自然语言不能唯一映射到配置字段时，必须停止并请人类澄清；禁止猜测参数后
   启动实验。
3. 一条真实 environment rollout 应直接用于监督 ValueHead。真实 rollout return 是
   已执行动作价值的黄金监督信号。
4. Qwen 前向产生 action logits 时，至少要保存 planner 最终选择动作对应的
   `pi_Qwen(action | prompt)`。为了可审计的 entropy、KL、support 和后续 policy
   objective，计划保存完整 action space 的 Qwen 分布以及 planner root 分布。
5. 是否训练 World Model 必须由配置控制。WM predictor、StateProjector、SIGReg 以及
   WM/value loss 是否向 Backbone 传递表征梯度不能被一个含糊开关隐式绑定。
6. RL 训练 grid WM 时必须使用与 SFT2 相同的 state MSE 和 predicted-state DINO MSE。

## 2. 统一用词和参数边界

| 名称 | 精确定义 | 不表示什么 |
|---|---|---|
| `history_size=H` | WM/value 训练窗口中的真实 transition 上下文长度；一个窗口包含 `H` 个 transition 和 `H+1` 个真实 state prompt | planning 未来模拟步数 |
| `planning.horizon=P` | planner 从当前真实 state 起，在 latent space 中模拟的未来动作数 | rollout episode 长度、PPO iteration 数 |
| environment step | 向真实环境执行一次动作 | 一个 planning depth 或一个 token |
| turn | Qwen 在一个 environment step 中生成的完整 CoT 和 action | 含糊的“轮数” |
| behavior policy | 实际产生并执行 trajectory 动作的策略 | 训练时重新计算的 Qwen 分布 |
| Qwen action policy | Qwen 在 action token 位置给出的动作分布 `pi_Qwen(a | s,c)` | planner 实际 behavior 分布 |
| planner root policy | planner 搜索后在根节点形成、并实际用于选择环境动作的分布 `pi_plan(a | s,c)` | Qwen 的 action logits |
| action ValueHead | 当前已有的 `Q(s,a)`，对每个动作输出一个值 | action-independent 的 `V(s)` baseline |
| Turn ValueHead | 建议新增的、在生成本轮 CoT 前预测 `V_turn(s)` 的标量 critic | 当前已有的 `Q(s,a)` |
| action credit | 对该 step 最终动作 policy objective 的监督 | 自动包含 CoT token |
| CoT credit | 对该 turn 中由 Qwen 真正采样且被 loss mask 选中的 reasoning token 分配 credit | 给模板、注入 token 或未采样文本分配梯度 |

根 `README.md` 中的术语表是项目级术语入口；本文只固定当前方案需要的边界。

## 3. 当前真实完成边界

### 3.1 已实现并验证

- direct-policy fresh rollout 已完成一次真实 GPU optimizer-step smoke：vLLM 直接采样
  Qwen behavior token，HF replay 同一 token trace，并执行一次 WM/value/SIGReg/PPO
  update。
- 当前 direct-policy actor 支持 `action` 和 `turn` credit。`turn` credit 是把同一个
  environment-step Monte Carlo advantage 广播给该轮参与 loss 的 reasoning/action
  token；它不是 token-level GAE，也不是 VAGEN Bi-Level GAE。
- direct-policy actor 已增加 `token` credit：trajectory 持久化逐步 reward 和
  `terminated/truncated`；Qwen replay 在每个实际采样 token 前的 hidden state 上运行
  独立 TokenValueHead；每个 turn 内用逐 token GAE 计算 CoT/action advantage。高层监督
  仍是完整真实 environment rollout 的 Monte Carlo return。
- fresh manifest把rollout绑定到policy/planner/trajectory/reference内容指纹；训练在
  optimizer前写入in-progress状态，并只在post-update checkpoint完成后提交消费。

### 3.2 已实现、尚待真实 GPU 验证

- planner behavior 已按已确认方案实现为action distillation，不称为planning PPO：
  action owner是`pi_plan`，Qwen action head拟合完整planner root分布，action token不进入
  PPO；Qwen真实采样的reasoning token使用token-level PPO。
- vLLM rollout通过worker extension从同一次真实多模态forward截取K个latent hidden和
  action boundary hidden，并用已加载模型的`compute_logits`得到8维raw action logits；
  不加载第二份HF Qwen，也不重复处理图片prompt。
- `planning.horizon=2`、`search_mode=exhaustive`时保存全部64条候选及其leaf score；每个
  root score是该根动作下候选score最大值，执行最高分候选的首动作，behavior/teacher
  是该动作上的确定性分布。
- 尚未验证多次迭代的 fresh rollout -> update -> 新 checkpoint -> 新 rollout 在线闭环。
- `token` credit 尚未完成真实 GPU optimizer-step 验证，也没有完成多次在线闭环验证。
- 当前 truncation 只实现了显式 `rl.truncated_bootstrap=zero`；若要从最后真实 state 的
  value bootstrap，仍需实现并由人类确认配置语义。
- 当前 actor advantage 使用 `G_t - Q(s_t,a_t)`。该 baseline 依赖已执行动作，不是标准
  policy-gradient 的 action-independent baseline；该问题仍适用于 `action/turn` 模式，
  `token` 模式改用独立 TokenValueHead。
- 当前 planner 只用预测叶状态上的 `max_a Q(s,a)` 评分，没有 reward head、done head
  或逐步累计 return。增大 horizon 可能只会放大 WM 误差，不能直接解释成更好的长期规划。

## 4. 推荐的长期方案

### 4.1 总体分工

推荐使用“planner action policy 拟合 + Qwen CoT PPO + 真实 rollout value 监督”的组合：

1. planner 是昂贵的 policy improvement 算法，负责在当前 state 和 CoT 条件下搜索；
2. Qwen action head 拟合 planner 的完整 root policy，使部署时可以摊销搜索结果；
3. Qwen 自己采样的 CoT token 使用 PPO 更新；
4. 真实 rollout return 监督 `Q(s,a)`，并监督 CoT PPO 所需的 pre-CoT scalar critic；
5. WM 及其相关表征模块是否训练由独立配置决定。

这一方案更适合未来增加 planning horizon。若每个 PPO minibatch 都重新执行完整 planner
并把它纳入统一 behavior ratio，计算成本、重放一致性和版本管理都会随 horizon 快速变得
复杂。planner root policy 拟合则允许搜索不断变强，同时保留清楚的 teacher/student 边界。

推荐目标函数的概念形式为：

```text
L = lambda_action * KL(stopgrad(pi_plan) || pi_Qwen_action)
  + lambda_cot * L_CoT_PPO
  + lambda_q * L_Q
  + I[train_wm] * lambda_wm * L_WM
  + independently configured representation regularization
```

这里 action 项属于 policy distillation，不称为 on-policy PPO。若以后选择统一的
planner-augmented stochastic behavior policy，必须另行定义并精确重放完整 behavior
分布后，才可以对 planner 动作使用 PPO ratio。

### 4.2 一次 turn 的概率所有权

建议将一次实际 behavior 写成：

```text
mu(c, a | s) = pi_Qwen_CoT(c | s) * pi_plan(a | s, c)
```

- `c` 是 Qwen 实际采样的 CoT，因此其 old/new token log-prob 可以由同一 Qwen policy
  做严格 replay。
- `a` 是 planner root policy 实际采样的动作，因此 action behavior probability 的
  owner 是 planner。不能仅保存 Qwen 对该动作的概率，然后把该动作伪装成 Qwen
  on-policy PPO 样本。
- Qwen action distribution 和 planner root distribution 都应随 trajectory 保存，
  且记录 action space 版本、temperature、top-p、planner 配置和 policy fingerprint。

## 5. ValueHead 与真实 rollout target

### 5.1 action value

已有 action ValueHead 保持 `Q(s,a)` 定义。对于真实 rollout 中第 `t` 个已执行动作：

```text
G_t = r_t + gamma * r_{t+1} + ...
L_Q = loss(Q(s_t, a_t), stopgrad(G_t))
```

必须先补齐以下 trajectory 字段和语义：

- 每一步 `rewards[t]`；
- episode 的 `terminated` 与 `truncated`；
- 对真正 terminal 的 return 不 bootstrap；
- 对时间上限等 truncation 是否从最后真实 state bootstrap，必须显式配置并记录；
- return 的 `gamma` 写入 config、manifest 和 checkpoint metadata。

初版建议关闭目前针对所有未选择动作的 ranking loss。单条 sampled rollout 只监督已执行
动作的 return，不能证明该动作一定优于所有未执行动作。只有存在成对反事实、搜索产生的
可靠排序监督或人类明确批准的假设时，才重新启用 ranking。

### 5.2 CoT baseline

`Q(s_t,a_t)` 依赖最终动作，不适合作为 CoT policy gradient 的无偏 baseline。初版建议
新增独立的 pre-CoT scalar Turn ValueHead：

```text
V_turn(s_t) -> scalar
A_t = G_t - stopgrad(V_turn(s_t))
```

`V_turn` 的输入必须是生成本轮 CoT 之前的 state，避免看到 CoT 或最终动作后发生信息泄漏。
如果不新增该 head，则需要另行确认 group/leave-one-out baseline 或按真实 behavior policy
计算 `sum_a pi(a|s)Q(s,a)`；不能继续默认使用 selected-action Q 作为标准 PPO baseline。

## 6. CoT credit assignment

### 6.1 已保留的 turn-level 基线

在没有 token critic 前，使用 turn-level Monte Carlo credit：

1. 对 step `t` 计算一个 `A_t = G_t - V_turn(s_t)`；
2. 只选择该 turn 中 Qwen 实际采样的 reasoning token；
3. 将同一个 `A_t` 广播给这些 token，并执行 PPO clip；
4. 每个 turn 按参与 credit 的 token 数归一化，使每个 turn 的总权重可比，避免更长 CoT
   仅因 token 更多而获得更大的总梯度；
5. 记录 CoT truncation、finish reason、token mask 和注入 token；注入 delimiter、latent
   query、模板文本与兜底补全不得进入 PPO loss。

这是 turn-level credit，不称为 VAGEN Bi-Level GAE。真正的 Bi-Level GAE 需要至少显式的
turn reward、token critic、`gamma_turn`、`gamma_token` 和对应 lambda 定义，属于后续阶段。

### 6.2 当前 token-level 实现

`actor.credit_assignment=token` 使用独立 TokenValueHead 预测每个实际 sampled token 之前
的 value。对每个 environment turn：

1. 完整真实 trajectory 先按 `rl.gamma` 计算该 step 的 Monte Carlo return `G_t`；
2. 该 turn 最后一个 selected action token 的 immediate reward 设为 `G_t`，更早的
   selected reasoning token reward 设为 0；
3. 用显式 `token_credit.gamma` 和 `token_credit.gae_lambda` 在本 turn 内反向计算 GAE；
4. turn 边界处 reset，不把下一个 environment turn 的 token critic 当作本 turn 的
   bootstrap；
5. PPO 使用 detach 后的全 batch token advantage，TokenValueHead 用未归一化 lambda
   return 做 MSE；模板和 injected token 始终不进入两项 loss。

因此当前算法的精确名称是“真实 environment Monte Carlo return + turn 内 token GAE”。
它已经提供逐 token critic 与 CoT credit，但还不是完整 VAGEN Bi-Level GAE：当前没有独立
的 high-level turn GAE、`gamma_turn/lambda_turn` 或高层 critic。实验报告必须保留这个边界。

### 6.3 CoT 必须真正影响 behavior

只有 CoT 对 planner 最终动作分布有因果影响时，环境 reward 才能合理训练 CoT。当前
实现让真实CoT直接决定latent query hidden，因此planner搜索状态依赖CoT；当前teacher
不融合Qwen action prior。reasoning token由Qwen采样并进入token PPO，planner替换的action
token明确标为loss-mask false。若后续实验证明WM projector抹掉CoT差异，必须停止把环境
reward解释为有效CoT监督，并先量化state/planner分布对CoT扰动的敏感性。

## 7. Planning horizon 扩展

### 7.1 `planning.horizon=2`

当前动作数为 8 时，共有 `8^2=64` 个两步动作序列。正确性阶段建议完整枚举，避免 beam
剪枝让部分根动作变成零 support，并保存：

- 8 维 Qwen action logits/probabilities；
- 8 维 planner root scores 和确定性 behavior/teacher 分布；
- 64 条候选序列的 score 或足以重建 root policy 的审计数据；
- planner 使用的 checkpoint fingerprint、horizon、beam/search 参数。

### 7.2 更长 horizon

未来组合数变大后，可切换到 Qwen-prior-guided MCTS/PUCT 或其他受控搜索，并用 root
visit distribution 或明确的 soft root distribution 监督 Qwen。增加 horizon 前必须具备：

- 明确的预测 reward 和 done/continuation 语义，或其他可验证的累计 return 定义；
- WM 多步 rollout error 指标；
- 相同 rollout budget 下的 planning gain 对照；
- 对模型不确定性、错误累积和搜索分布塌缩的监控。

不能只增加 `planning.horizon` 并继续用未校准的 leaf max-Q，然后把结果解释为长期规划收益。

## 8. 建议的数据契约

每个 environment step 至少持久化：

- pre-CoT state prompt、图片和 prompt fingerprint；
- sampled CoT token IDs、每个 token 的 old log-prob、loss mask、role、finish reason 和
  truncation 状态；
- 完整 8 动作 Qwen logits 或经过明确采样变换后的 probabilities；
- 完整 8 动作 planner root scores/probabilities；
- 实际 action index，以及 Qwen/planner 两套分布中该动作的概率；
- step reward、`terminated`、`truncated` 和 next observation；
- Qwen、WM、ValueHead 和 planner 所用 checkpoint fingerprint；
- `history_size`、`planning.horizon`、action space 版本和所有采样/search 参数。

进入训练前必须验证 token trace 与 assistant response、action token、action index、Qwen
distribution 和 planner behavior distribution 逐项一致。

## 9. 模块职责建议

遵循现有 Nimloth 模块边界，不把所有逻辑塞进 trainer：

| 模块 | 职责 |
|---|---|
| `backbone/qwen25vl` | 同一次多模态 Qwen forward 提取 state hidden、CoT replay logits 和 action logits；不负责 planner 搜索 |
| `agent/planning` | 根据 Qwen prior、WM 和 ValueHead 产生可持久化的 planner root policy；不执行 environment |
| `agent/runtime` / rollout collector | 按真实 behavior 采样 CoT 和 planner action、执行 environment、保存完整 trace |
| `rollout` | trajectory schema、freshness、逐步 rewards/done/truncation、严格一致性验证和窗口采样 |
| `training/rl/credit` | turn/token mask、advantage 广播与每 turn 归一化 |
| `training/rl/token_value` | sampled-token critic 的结构和独立 checkpoint 契约 |
| `training/rl/algorithm` | Q/value、action distillation、CoT PPO、WM/SIGReg loss 及明确的 detach 边界 |
| `training/rl/runtime` | 模型调用、FSDP/paired-GPU 执行、optimizer ownership 和 checkpoint |

Qwen replay 通过 `lm_head` 前向 hook 复用同一次 teacher-forced forward 的 selected-token
hidden states；`logits_to_keep` 只保留 loss-mask 位置，避免为了 token critic 再做一次完整
多模态编码或保存全序列 vocabulary logits。rollout侧则由vLLM worker hook直接截取同一次
生成forward的selected hidden，不存在独立HF Qwen rollout worker。

## 10. 当前已实现配置边界

```yaml
agent:
  planning:
    enabled: true
    horizon: 2
    search_mode: greedy
    device: REQUIRED
actor:
  enabled: true
  action_objective: distillation
  credit_assignment: action
  planner_distillation_weight: REQUIRED
predictor:
  train_wm: REQUIRED
  lambda_wm: REQUIRED
  lambda_dino: REQUIRED
```

distillation weight、planner device及所有token credit数值必须由实验配置显式给定。
`freeze.state_proj`、`lambda_sigreg`和
`gradient.representation_to_backbone`继续独立控制其职责。SFT2和RL调用同一个公共
WM objective；SFT2读取离线DINO cache，RL使用真实next-image的frozen DINO target。

## 11. 实施顺序与验证门槛

### 2026-07-25 实现检查点

- 阶段 A 的逐步 rewards、`terminated/truncated`、真实 terminal return 和显式 truncation
  策略已经实现；token 模式当前只接受明确的 zero bootstrap，尚未实现 learned bootstrap。
- direct-policy 已实现独立 TokenValueHead 与 turn 内 token GAE，并把其结构、optimizer
  参数组、分布式同步、checkpoint/resume metadata 纳入训练生命周期。
- 阶段B/C的代码契约已实现；在真实vLLM TP+图片、真实SFT2 planner checkpoint和GPU
  optimizer step通过前，只能称“实现待验证”，不能称实验完成。

### 阶段 A：修正 rollout/value 基础语义

1. 持久化逐步 rewards、terminated/truncated；修正 return 和 bootstrap。
2. 用真实 rollout return 监督 chosen `Q(s,a)`；默认关闭无依据 ranking。
3. 新增 pre-CoT Turn ValueHead，或先由人类确认替代 baseline。
4. 单元测试覆盖中间奖励、真实 terminal、时间截断、短 episode 和窗口切片。

### 阶段 B：记录 Qwen 与 planner policy

1. 同一次 Qwen forward 输出 state hidden 和完整 action logits。
2. `planning.horizon=2` 对 64 条序列完整枚举。
3. 保存、校验完整 Qwen/planner 八动作分布和 selected action ownership。
4. 保存leaf score到root score的聚合证据，并验证确定性selected action ownership。

### 阶段 C：训练 objective

1. Qwen action policy 对 planner root policy 做 distillation。
2. 对真实采样 CoT 做 per-turn normalized PPO。
3. ValueHead 直接使用真实 rollout target。
4. 按独立 flags 冻结或训练 WM、StateProjector、SIGReg 和 Backbone 表征路径。

### 阶段 D：闭环和长 horizon

1. 先完成 CPU contract tests 和单次 GPU correctness smoke。
2. 验证 fresh rollout -> update -> checkpoint -> fresh rollout 的多 iteration 闭环。
3. 固定 `planning.horizon=2` 做足够长的对照实验；success rate 只在样本量和评估协议
   预先确认后解释。
4. 仅在 reward/done/value calibration 与 WM 多步误差通过门槛后增加 horizon。

任何 GPU 实验都必须在启动前把 exact config key/value、checkpoint、数据 split、节点/GPU
布局、rollout TP、FSDP rank 布局、更新模块和停止条件列给人类确认。

## 12. 尚待人类确认

以下内容仍未确认，不能据此启动实验：

1. 新实验使用的`actor.planner_distillation_weight`、`agent.planning.device`及token
   credit具体数值仍须逐次确认；当前smoke配置中的值不自动成为其他实验默认值。
2. `turn` credit 是否采用每 turn token-count 归一化；`token` credit 当前采用全 batch
   token advantage 标准化，不得与该待确认项混称。
3. 是否新增独立的 scalar Turn ValueHead，或选择 group/leave-one-out baseline。
5. action ValueHead ranking loss是否默认关闭。
6. truncation 是否 bootstrap，以及 bootstrap 使用哪个冻结版本的 value。
7. WM、StateProjector、SIGReg、Backbone representation gradient 的最终配置字段名和默认值。
8. `planning.horizon=2` 长时实验的 episode 数、seed、评估 split、baseline checkpoint、
   success-rate 比较方法和停止条件。
