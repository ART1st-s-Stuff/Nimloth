# Nimloth

Nimloth is a Python machine-learning project for building a **World Model Agent**.

## VAGEN 到 RL 的术语与关键参数

本节固定 VAGEN navigation、SFT1、SFT2 和 RL 之间的公共用词。具体实验值必须从
对应 YAML、checkpoint metadata 和实验 README 读取；这里定义参数的含义、统计单位和
边界，不把某一次实验的取值当作项目默认值。

### 阶段与核心概念

| 阶段 | 标准用词 | 严格含义 | 不得混淆 |
|---|---|---|---|
| VAGEN 运行期 | VAGEN runtime | 提供 navigation environment 和批量 environment client/server 协议的外部运行期 | 不等于某个模型或 checkpoint |
| 环境数据 | dataset asset、`eval_set`、scene、task seed | `eval_set` 选择 VAGEN 环境资产；scene 是环境场景集合；seed 选择具体任务 | `eval_set` 不等于项目自行划分的 train/val/test |
| 环境交互 | episode、environment step、turn | episode 是从 reset 到终止的完整交互；environment step 执行一个动作；turn 是模型的一次完整回复及其动作 | 不使用“轮”同时指 step、turn、WM 预测步数和 PPO iteration |
| 环境输出 | observation、reward、done、success | observation 是当前文本和图片；reward 是环境数值反馈；done 表示 episode 终止；success 表示任务是否成功 | 单步 reward、episode 累计 reward 和 success rate 是不同指标 |
| 动作空间 | action key、action index、action token、token ID | 分别表示环境动作名、稳定动作编号、Nimloth 动作 token 和 tokenizer 词表编号 | 四种表示必须显式校验对应关系，不能统称“action”后直接互换 |
| Behavior rollout | behavior policy、sampling、assistant response、CoT | behavior policy 是实际采集 trajectory 的策略；assistant response 包含该 turn 的真实 CoT 和动作格式 | rollout 会真实执行环境；不是 PPO policy replay |
| Token 来源 | token trace、old log-prob、loss mask、token role、finish reason、truncation | 保存 behavior 生成的 token、旧策略概率、参与 PPO 的位置和生成结束状态 | 自动补全 delimiter、latent query 和未选中的 token 不得默认计入 PPO |
| 轨迹数据 | transcript、trajectory、transition、record | transcript 是完整对话历史；trajectory 是完整 episode；transition 是一个真实状态转移；record 是持久化记录 | JSONL 一行通常是一条 trajectory，不是一条 transition |
| SFT1 数据 | VAGEN rollout record、Nimloth SFT record、supervised prompt、CE label | 将旧动作格式转换为 Nimloth latent/action 格式后，监督 Qwen 输出格式和动作 | `train_all` 包含失败轨迹；`train_success` 只是成功子集，不代表模型成功率 |
| SFT1 协议 | latent query token count `k`、`inject`、`generate` | `k` 是 latent query token 数；`inject` 注入并屏蔽 query token CE；`generate` 监督模型生成 query token | `k` 不等于 `history_size`，也不自动等于图片 patch 数 |
| SFT1 产物 | adapter、merged HF artifact | adapter 只保存增量参数；merged HF artifact 是合并后可供 Transformers、vLLM、SFT2 或 RL 加载的完整策略产物 | 不得把 LoRA adapter 目录当作完整 HF checkpoint |
| SFT2 数据 | `TransitionSample`、trajectory lane、context window、current step、history cache | 一个样本包含同一 trajectory 的连续上下文；旧 state 可来自按时间顺序建立的 detached history cache | SFT2 窗口有多个上下文 step，但主 loss 只监督窗口末端的 current step |
| 状态表征 | Backbone hidden、StateProjector、online state、target state | Qwen hidden 经 StateProjector 得到 WM state；online state 接收训练梯度；target state 按 objective 的 EMA/stop-gradient 规则产生 | prompt history、Qwen hidden 和 WM state 是三种不同对象 |
| Grid WM | grid slot、EMA target encoder、WM predictor、predicted next state | grid state 由多个 slot 表示；predictor 根据真实 state/action context 预测下一状态 | `history_size` 是训练上下文，不是 planner 未来搜索长度 |
| SFT2 目标 | LM CE、WM loss、DINO grid loss、SIGReg、value loss、ranking loss | 分别约束输出格式、latent dynamics、视觉 target、表征分布、执行动作的 return 和动作排序 | DINO loss 属于当前 DINO-grid SFT2 objective；当前 RL objective 不计算 DINO loss |
| ValueHead | action value、chosen action value、`Q(s,a)` | ValueHead 对每个离散动作输出一个 action value；chosen action value 是实际执行动作对应的值 | 不简称为单一 state value `V(s)`；当前实现是 action critic |
| RL fresh rollout | current policy artifact、vLLM behavior rollout、policy fingerprint、fresh manifest | vLLM 使用当前策略生成新 trajectory；manifest 把 trajectory 与策略内容指纹绑定 | fresh manifest 只能被一个 PPO 更新消费一次，不能循环复用 |
| RL 窗口 | `history_size=H`、trajectory window、RL batch | 每个窗口有 `H` 个连续 transition、`H+1` 个真实 state prompt；batch 是若干窗口 | batch size 统计窗口数，不是 episode 数、step 数或 token 数 |
| Return 与 advantage | Monte Carlo return、return target、baseline、advantage | return 先在完整 episode 上计算再切窗口；advantage 是 return 减去 detached chosen action value 后的归一化结果 | 当前不是 GAE，也没有 token critic 或 bi-level critic |
| PPO replay | policy replay、new log-prob、probability ratio、clip、entropy | 对已记录的同一 prompt、图片、采样参数和动作重新计算当前策略概率，并构造 PPO loss | replay 不重新执行环境，也不重新采样动作 |
| Credit assignment | action credit、turn credit、policy token | action credit 只训练 sampled action token；turn credit 把同一 environment step 的 advantage 分给该 turn 中 loss-mask 为真的 reasoning/action token | turn credit 不等于 token-level GAE；也不改变 ValueHead 的 environment-step critic 定义 |
| WM planning | planning horizon、beam width、latent rollout、executed first action | planner 在 WM latent 空间模拟候选动作序列，最后只向环境执行选中序列的首动作 | planning horizon 不等于 `history_size` 或 episode 最大步数 |
| 评估 | training-rollout success、held-out success、success rate、average reward | success rate 的统计单位是 episode；泛化评估必须使用与训练场景不重叠的数据 | 少量训练 rollout 的 success 不能替代 held-out evaluation |
| Checkpoint | initialization、component checkpoint、`latest`、`best`、`final`、resume | initialization 是训练起点；component checkpoint 保存 WM/value 等模块；`latest` 用于恢复；`best` 由显式验证指标选择 | policy artifact 与完整 optimizer-resume checkpoint 不等价 |

navigation v1 动作空间固定为八个 action key：`moveahead`、`moveback`、
`moveright`、`moveleft`、`rotateright`、`rotateleft`、`lookup`、`lookdown`。
动作数量属于 environment action space，不属于 Agent、rollout 或 WM。

### 关键配置参数

#### VAGEN 环境与数据

| 参数 | 含义 | 单位或约束 |
|---|---|---|
| `eval_set` | 选择实际 VAGEN dataset asset | RL 训练只能使用训练场景资产，例如 `*_train`；泛化评估使用不重叠的 eval scene 资产 |
| task seed | 在所选资产内选择具体任务 | 必须和 scene/split 一起记录，不能只记录 seed |
| `max_actions_per_step` | 一次 environment step 允许解析的动作数 | 当前 Nimloth navigation 为 1 |
| `failure_penalty` | 环境报告动作执行失败时的额外惩罚 | reward 标量；属于 adapter 语义 |
| `success_threshold` | VAGEN navigation 成功判定相关阈值 | 属于 environment config；最终仍应保存 `task_success`/success provenance |
| `step_length` | navigation 平移动作的物理步长 | 环境单位 |
| `grounding_reward_weight` | grounding reward 权重 | 环境 reward 配置 |
| `worldmodeling_reward_weight` | world-modeling reward 权重 | 环境 reward 配置，不是 Nimloth WM loss 权重 |

#### SFT1

| 参数 | 含义 | 关键边界 |
|---|---|---|
| `data.train_jsonl`、`data.val_jsonl` | 监督训练和验证记录 | 必须保留真实 split 来源与转换 manifest |
| `latent.token_count` | latent query token 数 `k` | 必须写入 HF checkpoint metadata，并与后续 SFT2/RL 一致 |
| `latent.query_mode` | `inject` 或 `generate` | 决定 query token 是否由模型生成、是否进入 CE label |
| `tuning.mode` | Backbone 参数更新方式 | 例如 LoRA 或 embedding learning-rate 方案 |
| `tuning.lora_r`、`tuning.lora_alpha` | LoRA rank 和缩放 | 只在 LoRA 模式下有效 |
| `train.epochs` | 完整遍历训练数据的次数 | 不等于 optimizer step |
| `train.batch_size` | 每个 microbatch 的记录数 | 与 `grad_accum` 共同决定有效 batch |
| `train.grad_accum` | 梯度累积 microbatch 数 | 有效 batch 还需乘分布式 rank 数 |
| `train.lr`、`train.embedding_lr` | LoRA/模型参数和新增 token embedding 学习率 | 参数组必须区分 |
| `train.max_length` | 单个监督序列最大 token 长度 | 截断必须记录，不能静默改变 action label |
| `train.max_pixels` | 多模态 processor 的图片像素预算 | 会影响视觉 token 数和显存 |

#### SFT2

| 参数 | 含义 | 关键边界 |
|---|---|---|
| `init.sft1_checkpoint` | SFT2 的 Qwen/策略初始化 | 必须是完整、可加载并带正确 latent/action metadata 的 HF artifact |
| `init.wm_predictor_checkpoint`、`init.grid_warmstart` | WM 组件初始化或 warm start | warm start 不自动恢复 optimizer，不等于 resume |
| `data.include_failed_rollouts` | 是否让失败轨迹参与训练 | 会改变 value/WM 数据分布 |
| `tuning.llm_tune`、`tuning.vision_tune` | LLM 和视觉 Backbone 的训练范围 | `freeze`、`full`、LoRA 等模式必须分别记录 |
| `tuning.vision_ema`、`vision_ema_decay` | 是否维护视觉 Backbone EMA 及衰减率 | EMA 权重用于 target/evaluation 的范围必须明确 |
| `train.history_size` | SFT2 最大真实 context 长度 `H` | 一个 batch 样本仍只监督窗口末端 current step |
| `train.batch_mode` | trajectory 上下文的组织方式 | 生产路径为 `trajectory_online_cache` |
| `train.emb_dim` | WM state embedding 维度 | 必须和 StateProjector、WM predictor、ValueHead checkpoint 匹配 |
| `train.state_proj_lr` | StateProjector 学习率 | 与 Backbone、WM、ValueHead 参数组分开 |
| `train.wm_predictor_lr` | WM predictor 学习率 | 训练 latent dynamics |
| `train.dino_decoder_lr` | DINO decoder 学习率 | 只在 DINO decoder 可训练时有效 |
| `train.value_head_lr` | ValueHead 学习率 | 训练每个动作的 `Q(s,a)` |
| `grid.size` | 二维 grid 边长 | slot 数通常为 `grid.size²`；不能与 latent token count 自动画等号 |
| `grid.ema_decay` | grid target encoder EMA 衰减率 | 与视觉 Backbone EMA 是不同 EMA |
| `grid.wm_depth`、`wm_heads`、`wm_dim_head`、`wm_mlp_dim`、`wm_dropout` | temporal-spatial predictor 容量 | checkpoint 加载时必须匹配结构 |
| `loss.lambda_ce` | LM CE 权重 | 监督当前 step 的格式/输出 token |
| `loss.lambda_wm_start`、`lambda_wm_end` | WM loss warmup 起止权重 | 不等于 environment worldmodeling reward weight |
| `loss.lambda_dino` | DINO grid reconstruction/target loss 权重 | 只属于相应 SFT2 objective |
| `loss.lambda_value` | ValueHead loss 权重 | 包含回归及可选 ranking 项 |
| `loss.lambda_sigreg` | SIGReg 权重 | 统计单位和跨 rank 聚合方式必须随实现记录 |
| `loss.value_gamma` | SFT2 action-value target 的折扣率 | 当前 target 来自完整 trajectory 的稀疏 Monte Carlo return |
| `loss.value_rank_margin`、`value_rank_lambda` | chosen action 与其他动作的 ranking margin/权重 | ranking 是辅助约束，不是额外 reward |
| `monitor.checkpoint_metric` | SFT2 checkpoint 选择指标 | `val_wm_mse` 只衡量 WM；不能冒充 agent rollout success |

#### RL、PPO 与 planning

| 参数 | 含义 | 统计单位或约束 |
|---|---|---|
| `agent.prompt_template` | policy/state prompt 的版本化模板 | rollout、state encoding 和 replay 必须使用同一模板 spec |
| `agent.thought` | action-only/state prompt 使用的固定 thought | 不等于 turn-credit behavior 中真实采样的 CoT |
| `agent.planning.enabled` | 是否启用 WM latent planning | 当前 planner behavior 与 PPO actor 不能同时启用 |
| `agent.planning.horizon` | 每个候选序列在 latent 空间向未来模拟的动作数 `P` | 不执行环境；不得用 `history_size` 表达 |
| `agent.planning.beam_width` | 每层保留的候选序列数 | 影响搜索开销，不改变训练 batch size |
| `gradient.representation_to_backbone` | WM/value/SIGReg 是否把表征梯度传回 Backbone | 与 `actor.enabled` 独立 |
| `gradient.backbone_lr` | RL 中 Backbone 的统一学习率 | 实际可训练范围仍由 LLM/vision tune 配置决定 |
| `freeze.state_proj` | 是否冻结 StateProjector | 不决定 ValueHead 或 WM predictor 是否训练 |
| `actor.enabled` | 是否启用 Qwen PPO actor loss | 开启后只接受与当前策略绑定的 fresh rollout |
| `actor.clip_ratio` | PPO probability ratio 裁剪范围 | ratio 为 `exp(new_log_prob-old_log_prob)` |
| `actor.entropy_coeff` | behavior sampling 分布上的 entropy bonus 权重 | entropy 使用相同 temperature/top-p 变换后的分布 |
| `actor.credit_assignment` | `action` 或 `turn` | 决定每个 environment-step advantage 分配到哪些 policy token |
| `actor.max_reasoning_tokens` | turn-credit 最多采样的 reasoning token 数 | 截断和 finish reason 必须持久化 |
| `predictor.history_size` | RL 窗口中的 transition 数 `H` | state 数为 `H+1`；必须和 SFT2 WM checkpoint 的 history 语义兼容 |
| `predictor.emb_dim` | RL WM embedding 维度 | 必须匹配 warm-start 组件 |
| `predictor.lr` | WM predictor 学习率 | 不等于 Backbone 或 ValueHead 学习率 |
| `predictor.lambda_sigreg` | RL SIGReg 权重 | 当前 RL 不计算 DINO loss |
| `value_head.lr` | RL ValueHead 学习率 | 对窗口内 `H` 个 environment step 计算 action value |
| `value_head.rank_margin`、`lambda_rank` | action-value ranking margin/权重 | `lambda_rank=0` 时只有 return regression |
| `rl.iterations` | RL optimizer update 次数 `I` | 在线 PPO 每次更新都需要匹配当前策略的新鲜 behavior rollout |
| `rl.envs_per_iteration` | 每次 iteration 采集的 episode 数 | 不是 transition/window 数 |
| `rl.max_steps_per_episode` | 每条真实 episode 最多执行的环境动作数 `E` | 与 planning horizon、reasoning token 数无关 |
| `rl.gamma` | Monte Carlo return 折扣率 | 先对完整 episode 计算，再切 trajectory window |
| `rl.batch_size` | 每次 optimizer update 采样的 trajectory window 数 | 每个窗口贡献 `H` 个 value/action 位置 |
| `rollout.temperature`、`rollout.top_p` | behavior policy 采样参数 | 必须随 trajectory 保存并在 PPO replay 中复用 |
| `rollout.train_datasets`、`eval_datasets` | 训练和评估环境资产 | 两者必须具有经核实的不重叠 scene 语义 |
| `validation.enabled`、`interval`、`envs` | 是否验证、验证间隔和 episode 数 | `interval` 的单位是 RL iteration；`envs` 的单位是 episode |
| `validation.checkpoint_metric` | `best` checkpoint 的选择指标 | 例如 held-out `success_rate` 或 `avg_reward` |
| `training.seed` | 窗口采样和训练随机种子 | 不替代 environment task seed |
| `training.save_interval` | checkpoint 间隔 | 单位是 RL iteration |

#### 分布式训练与 vLLM

| 参数或用词 | 严格含义 | 关系 |
|---|---|---|
| node | Slurm 分配的物理节点 | 节点可以异构；节点数不直接等于 rank 数 |
| physical GPU | 实际占用的 GPU | 当前配置总数为 `world_size × gpus_per_rank` |
| training process / rank | 一个分布式训练进程 | `distributed.world_size` 是训练 rank 总数 |
| training replica | 持有一份逻辑训练模型并产生本地梯度的执行单元 | 当前一个 rank 对应一个 replica |
| `distributed.gpus_per_rank` | 一个训练 rank 管理的本地 GPU 数 | `1` 可用于单卡 rank/FSDP；`2` 表示一个 rank 内的 Qwen model parallel，不称为“两卡组成一个 FSDP rank” |
| FSDP | 跨 rank 对参数、梯度和 optimizer state 分片的训练方式 | FSDP rank 仍是一个进程、一个主 GPU 执行上下文 |
| intra-rank model parallel | 一个 rank 内把 Qwen 放到多张 GPU | 与 FSDP、跨 rank gradient synchronization 是不同维度 |
| gradient synchronization | 各 training replica 对 trainable gradient 求和/平均 | 多 GPU model-parallel replica 必须使用确定一致的参数顺序 |
| `distributed.rollout_tensor_parallel_size` | 一个 vLLM rollout engine 使用的 tensor-parallel GPU 数 | 只描述 rollout 推理拓扑，不等于训练 `world_size` |

### 禁止歧义的表达

| 不再单独使用 | 必须改写为 |
|---|---|
| “预测 2 轮” | `history_size=2`、`planning.horizon=2` 或 `max_steps_per_episode=2`，三选一并写出参数名 |
| “value” | action value `Q(s,a)`、chosen action value、return target、value loss 或 checkpoint metric |
| “rollout 历史” | 完整 transcript prefix、完整 trajectory、`H` 步 latent window 或 PPO replay window |
| “跑 8 卡” | node 分布、physical GPU 数、`world_size`、`gpus_per_rank` 和 rollout TP |
| “FSDP 两卡 rank” | 单卡 FSDP rank，或两卡 intra-rank model-parallel replica |
| “replay” | PPO policy replay 或 environment trajectory replay；当前 PPO replay 不执行环境 |
| “成功率变了” | 指明 split/scene、checkpoint、采样配置、episode 数、成功数和置信区间 |
| “SFT2 预测 H 步” | SFT2 使用长度不超过 `H` 的 context，但主 loss 监督末端 current step；若指 planner，写 planning horizon |

特别地，RL 中设置 `history_size=2` 表示每个训练窗口含两个真实 transition 和
三个 state prompt，ValueHead 产生形状为 `(B, 2, action_count)` 的 action value；
它不表示 planner 向未来预测两步，也不表示 environment episode 只执行两步。

## Current status

Project initialization is focused on AI collaboration prompts and rules.

## AI collaboration entry

All AI assistants must read [`AGENTS.md`](AGENTS.md). For durable memory, use the memory skill:

```bash
./skill memory search --store all <regex>
./skill memory get --store repo <id>
./skill memory add <title> <content>
./skill memory add --store local <title> <content>
./skill memory set --store repo <id> 'evidence=[{"filename":"...","line_start":1,"total_lines":10}]' 'tags=["..."]'
./skill memory upvote --store repo <id>
```

Human approval of AI-created memories:

```bash
./skill human memory-approve
./skill human memory-approve --store local
```

## Important note

Do not create code structure, model skeletons, training scripts, or experiments unless the human developer explicitly asks for them.
