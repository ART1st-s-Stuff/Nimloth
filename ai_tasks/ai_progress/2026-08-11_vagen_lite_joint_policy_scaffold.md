# VAGEN-Lite joint-policy scaffold

## 目标

在真正上游 VAGEN-Lite `mll-lab-nu/VAGEN main@2936322` 及其固定
`JamesKrW/verl@3fe0a299` 上建立 frozen-Q guided Scheme-B 的可审计训练框架；
旧项目 fork 仅作为移植来源，不再作为运行基线。

## 分支

- Nimloth：`feat/vagen-lite-joint-policy-scaffold`
- VAGEN：`nimloth/upstream-joint-policy-scaffold`
- VAGEN 上游起点：`2936322a6f6c02fbd29ca28e4b6ec37eefefc081`
- VAGEN M1 commit：`45cb9928a8d9316037e1fb86c0dff3d004705097`
- VAGEN M2 contract commit：`25da71df5f1408d54b4b761ff40c985d9118c99c`
- VAGEN confirmed gradient/Q-target contract commit：`0a23ab3923bcef4cbda89380353c312dab77319a`
- 当前父仓库commit：`7b13d622a6361d2e6844b86ef2077b03f8f7e3ee`
- 当前VAGEN commit：`316d9d7bc2a153bd1cecc34d04f752231458892d`
- VERL gitlink：`3fe0a29975e1b02ae2bd1dec249f7807dd7966f5`

## 当前计划

1. M1 decision ledger 已完成并保持不变。
2. 人类已确认每个真实 turn 只执行 guided policy 的第一动作，随后从真实 observation 重新规划；模拟尾部不进入 environment PPO。
3. action prior 是 LLM action-token boundary logits 的 softmax；实际采样的 prior action token log-prob 属于 `pi_LLM`，完整 prior 分布作为 guided policy 条件。
4. M2 暂采用 scheme B：旧 ValueHead 保持 critic，guided logits 为 `alpha * l_prior + beta * stopgrad(frozen_Q)`，没有独立 actor module。
5. rollout/update 内复用 rollout-time frozen critic 的 all-action guidance scores；critic 更新后禁止重算同一批 behavior guidance。
6. 人类已确认 guided actor loss 必须经`l_prior`反传LLM；`backprop_to_llm`保留为可审计合同字段但启用时只能为`true`。Q在actor loss中始终stop-gradient。
7. 人类已确认Q用真实环境reward构造的discounted return训练：只回归实际执行的第一动作，target stop-gradient，首版使用Huber；不把即时reward或advantage本身当Q target，不伪造未执行action监督。terminal bootstrap为0；truncation需rollout-time frozen critic bootstrap。
8. ledger v2、versioned behavior record、reference/Torch guided math 和 contract identity 已实现。
9. `joint_policy.enabled=true`当前显式 fail closed，直到人类决定旧 ValueHead 接收哪一种 state，并完成 Q owner、rollout sampler、replay 和 checkpoint snapshot boundary。

## 已完成

- 创建独立 Nimloth worktree 和 VAGEN-Lite 子模块分支。
- 核验旧 `feat/planner-verl-vagen-scaffold` 仍基于 legacy VAGEN 且绑定旧 PlannerPolicyHead，不能直接复用。
- 将框架计划写入 `external/VAGEN/docs/joint_policy_scaffold.md`。
- 按 TDD 新增 `vagen_decision_ledger_v1`：保存 action-space、完整实际执行动作、来源、actor-policy ownership、turn reward 和 terminal/truncated 状态，并严格拒绝任何未实现的 policy-sampled 声明。
- Navigation 输出可核验的 0-based action-space contract；no-concat agent-loop 将 ledger 原样传入 DataProto，trainer 在 old-log-prob replay 前严格校验并记录覆盖指标。
- system fallback token 从 LLM response mask 排除，同时把 turn reward 锚定到最后一个真实 policy token，避免 reward 被 mask 丢弃。
- latent fallback adapter 仅在 `prompt_format=latent_plan` 启用；remote step transport 不再把字符串 done、布尔 reward 或缺失字段静默强转成合法值。
- M1 两轮独立 code review 的 P1/P2 均已逐项修复；最终独立复审结论为 `APPROVED`。
- M2 合同层新增显式 Scheme-B 配置、dtype-aware 数值合同、Torch 公式、严格 behavior schema/round-trip、action-token/contract/snapshot 绑定与 ledger v2。三轮 review 修复了 silent stock-PPO fallback、伪造 ownership、logprob 容差和 overflow 等问题；最终复审无 blocker。
- 只读核验确认 VAGEN 现有 token critic 不是 `[B,8] Q(s,a)`，transition reward predictor 也不是旧 ValueHead。旧 Nimloth ValueHead 输入 state 与 VAGEN `LatentStateEncoder` state 不同，未获人类决定前禁止直接加载旧权重或用其他模块冒充。
- Git history进一步确认这条`latent z -> LatentStateEncoder -> world_state -> TransitionRewardNet`并非本分支新增：基础类由ARTI5T在嵌套VERL commit `2f291ea`（2026-03-27，`MCTS`）引入；canonical latent提取由`0ca14e2`（2026-04-13，`Step 1&2 prototype`）加入；当前可配置`WorldStatePredictor`及actor wiring由`ae269bd`（2026-04-14，`Add LeWM predictor`）完成。VAGEN顶层由同作者commit `517da7a`固定该gitlink，当前VAGEN-Lite基线`a6b8c8d`继承它。因此它是ARTI5T fork历史中的既有实验路径，不应被当作旧Nimloth ValueHead的输入定义。
- 人类澄清上述fork提交是其本人修改，要求后续以真正上游最新main为基线。只读远端核验：`mll-lab-nu/VAGEN main@2936322a6f6c02fbd29ca28e4b6ec37eefefc081`（2026-07-23）固定`JamesKrW/verl@3fe0a29975e1b02ae2bd1dec249f7807dd7966f5`；该上游不含fork的LeWM predictor路径。
- SFT2兼容性审计结论是可小规模适配，不需重训或转换checkpoint，也不构成退回`vagen-legacy`的理由。VPN恢复后的服务器只读preflight已直接核验corrected ID74 `epoch_001`：完整Qwen2.5-VL HF目录含2个safetensors shard、825个tensor和独立的embedding/lm-head key；config为BF16、hidden 2048、vocab 151691、untied head、K16/inject。Transformers 4.55.4使用上游同类`AutoModelForImageTextToText.from_pretrained`完整CPU加载两片权重，missing/unexpected/mismatched/error均为空，输入/output embedding storage保持独立。
- tokenizer/processor真实加载为`Qwen2TokenizerFast`/`Qwen2_5_VLProcessor`；16个latent token、action边界和8个action token共26个added special tokens占ID 151665--151690，每个均为单token且标为special。vLLM 0.11.0的`ModelConfig`不加载权重即可将该目录解析为multimodal `Qwen2_5_VLForConditionalGeneration` generate runner。SFT2 metadata明确`query_tune=freeze`，因此更准确的兼容结论是HF权重已自包含继承的special-token rows，无需运行时query adapter；不能把它描述为本轮SFT2保存时折叠了一个adapter。
- sidecar真实metadata/shape与旧Nimloth decision-state路径一致：`SharedSlotProjector`逐slot执行2048→2048→1024，输出`[B,16,1024]`；ValueHead先对16 slots mean-pool，再执行1024→1024→8；TemporalSpatialGridPredictor为16 slots、emb1024、action8、history1。`training_state.pt`为step776/epoch1 complete，objective=`decision_state_executed_action_mc_v3`、T4、world16。使用当前Nimloth loader完整加载projector/predictor/ValueHead并执行`[1,16,2048] -> [1,16,1024] -> [1,8]` CPU forward，全部finite。因此Q owner应完整复用这条state提取与mean-pooling路径，不能改接上游文本world state。
- 上游不原生理解Nimloth协议，仍需局部集成：自定义agent-loop强制真实CoT后的16个inject slots/action boundary；vLLM同次forward捕获16个hidden与8-action logits；加载上述sidecar；guided replay、snapshot和checkpoint ownership。上游已有custom agent loop、`extra_fields`、external library/rollout registry与FSDP actor extension点，因此这是联合PPO功能接入，不是SFT2架构迁移。完整GPU vLLM load未在本次只读preflight重跑；历史TP4 rollout证据仍作为GPU证据边界。

## 文件修改

- `external/VAGEN/docs/joint_policy_scaffold.md`：actor 未决边界与 M1/M2/M3 计划。
- `external/VAGEN/vagen/agent_loop/decision_ledger.py`：dependency-light ledger schema、校验、指标与 token ownership helper。
- `external/VAGEN/vagen/agent_loop/{gym_agent_loop_no_concat,agent_loop_no_concat}.py`：ledger producer/DataProto 路径、fallback mask 和 reward anchor。
- `external/VAGEN/vagen/envs/navigation/navigation_env.py`：versioned action-space 与完整 executed action 字段。
- `external/VAGEN/vagen/envs_remote/gym_image_env_client.py`、`vagen/utils/remote_step_protocol.py`：严格 remote step 解码。
- `external/VAGEN/vagen/ray_trainer.py`、`vagen/configs/vagen_multiturn.yaml`：opt-in gate、pre-replay validation 和 metrics。
- `external/VAGEN/vagen/joint_policy/`：Scheme-B config、contract id、behavior schema、reference/Torch math；README明确未完成 ownership。
- `external/VAGEN/tests/test_decision_ledger*.py`、`test_joint_policy_*.py`、`test_remote_step_protocol.py`：单元、可选autograd与 wiring 回归。
- 本文件：任务实时进度。

## 验证

- RED：初次运行因 `vagen.agent_loop.decision_ledger` 不存在失败；review 后新增 action-space/type/reward-anchor 与 remote protocol RED 均先失败。
- M1 GREEN：最初为`30 passed`。
- M2合同初版：`46 passed, 3 skipped`。
- 人类确认梯度/Q target后：`47 passed, 3 skipped`；3项skip均为当前环境缺torch的真实autograd/reference parity/overflow测试。review发现并修复`alpha=0`会切断LLM梯度；启用合同现要求`alpha>0`，最终复审无blocker。
- `python3 -m py_compile`：受影响生产 Python 文件通过。
- `git diff --check`：通过。
- VS Code diagnostics：ledger、gym no-concat 和 trainer 0 diagnostics。
- 旧本地环境的dependency-light门禁已由服务器完整依赖回归补足：精确VAGEN `316d9d7`为`88 passed, 23 subtests`，精确父仓库`7b13d622`的RL/navigation/SFT1 prompt定向为`269 passed`；仅有1条Ray弃用warning和1条既有单样本std warning。未运行真实GPU、多模态rollout、PPO或checkpoint测试，禁止把本阶段表述为joint PPO已完成。
- 远程session现在粘定connect server，state-mutating `step`不做可能重复执行的自动重试；Navigation cache按去除server-owned `gpu_device`后的完整配置隔离，构造/reset失败和agent-loop异常均释放env slot/session。Qwen trailing EOS/PAD只从送入严格environment parser的解码副本移除，原始response IDs/mask/log-probs继续保留给PPO。
- Scheme-B behavior record现在同时绑定action table/token IDs、prior logits、采样prior action的LLM log-prob、rollout-time frozen-Q、guided log-prob和snapshot identity；trainer在最终reward tensor赋值前再次校验ledger reward anchor。`joint_policy.enabled=true`仍在Q owner/guided rollout/replay未接通时fail closed。
- `vagen_eval`兼容profile恢复canonical source-eval wording，仅替换成K16单动作格式；SFT1 converter使用format body避免重复注入“exactly one action”。RL/env launchers不再默认为旧worktree，并校验精确VAGEN短commit。
- 人类已批准并完成首阶段push：VAGEN `nimloth/upstream-joint-policy-scaffold@316d9d7`和Nimloth `feat/vagen-lite-joint-policy-scaffold@a4d5bfad`远端SHA均与本地一致；`.gitmodules`跟踪新VAGEN分支。临时fresh clone成功checkout精确父commit和VAGEN gitlink；未合并或修改main。
- 人类随后批准开始单GPU、无optimizer、无训练的一真实turn smoke。代码审计确认stock `main_ppo val_only`仍构造optimizer、validation丢弃ledger，而generic no-concat loop没有ID74的真实CoT→K16→action生成约束；因此新增专用optimizer-free standalone bridge，而没有运行不可判定的近似smoke。
- VAGEN `1f8ed5f`新增`nimloth_vllm` custom replica和`standalone_one_turn_smoke`：复用Nimloth `TurnGenerationSpec`/`TurnResponseLogitsProcessor`，prompt只预填`<think>`，保留模型实际CoT；K16/action protocol强制token的response mask为0，真实sampled CoT/action为1。输出原子绑定完整environment response、Navigation action-space/source/action token、M1 ledger、reward anchor和finite aligned rollout log-probs；无actor/critic/FSDP/optimizer/checkpoint。
- 本地dependency-light为`84 tests`，服务器完整依赖为`96 passed, 25 subtests`；custom Ray actor继承、registry时序和numpy seed remote JSON序列化等review问题均已修复，最终review`APPROVED`。该VAGEN commit已push，父仓库gitlink待launch contract一并推进。
- 后续runtime/profile加固形成VAGEN`3fc2509`与父`6103945a`：remote config显式等于current Navigation profile以保证prewarm cache identity，standalone强制独立local Ray/temp root并显式传播runtime env；服务器完整依赖`98 passed, 25 subtests`，review批准。1-GPU launch合同固定normal、1×H800同卡Unity+vLLM、16CPU/128GiB/30min、max_envs1、held-out `base` seed0（FloorPlan11/Bread）、ID74内容hash和scoped cleanup。
- ID159 Job`516698.2`在`normal/dgx-52`运行4分28秒后于首token前失败：direct render`5.166s/dynamic246`、real remote prewarm`5.660s/dynamic255`、exact FloorPlan11 cache reuse和ID74 BF16 TP1 vLLM load均通过；fresh parent worktree未初始化tracked `external/le-wm@8edfeb3`，Nimloth policy import经`_vendor_lewm.py`报缺`module.py`。无response/action/environment step/ledger result、optimizer、checkpoint、W&B；cleanup后port/owned processes为空且VRAM回289MiB。ID159不可resume。
- `43a86fc6`补充完整le-wm gitlink/clean及policy/turn/vendor真实CPU import门禁。ID160 step`516748.0`在`normal/dgx-35`以`COMPLETED 0:0`运行8分26秒：direct render`14.302s/dynamic246`、remote prewarm`6.345s/dynamic255`、exact cached FloorPlan11 reuse、ID74 TP1 vLLM和one real turn全部通过。模型生成24 tokens（`<think>move_left</think>`+K16+action0），parser执行唯一`move_forward`；reward0.0锚定sampled index22，ledger source/format/no-fallback通过，environment非terminal且one-turn cap令rollout truncated。atomic result/validator/final/summary、raw token/mask/log-prob/reward校验均通过；peak VRAM48,685MiB回289MiB，owned process/port clean，hold已释放。无sidecar、optimizer/backward/update/checkpoint/W&B/joint-policy；只证明optimizer-free K16 one-turn协议与ledger。
- async same-generation capture bridge现复用既有vLLM worker hook，request-scoped捕获K16 hidden和action-boundary raw 8 logits，并通过两阶段TP prepare/finish避免partial-rank validation进入LM-head collective；失败cleanup只清当前request。sidecar绑定schema/request及latent/action token IDs，经TokenOutput/agent-loop/DataProto传播；DP>1、non-eager或保留engine override均fail closed。服务器CPU为VAGEN`93 passed,27 subtests`、父定向`79 passed`，capture专项为parent`26 passed`及VAGEN`36 passed,9 subtests`，最终review批准。ID161按人类纠正后的normal单节点8×H800/vLLM TP8合同在step`517501.0`/`dgx-39`以`COMPLETED 0:0`运行9分42秒；`mm_encoder_tp_mode=data`后ID74 TP8/DP1/eager加载、held-out base0真实turn、24-token CoT→K16→action0、reward0.0/anchor22/ledger valid均通过。request`1ff645dc78424e9d85b7a6846bdcda81`绑定finite `[16,2048]` hidden和raw `[8]` logits，validator`ALL_OK`；峰值显存50,244--50,915MiB，cleanup后8卡0MiB、端口/owned process为空，hold已释放。结果目录`outputs/experiments/training/rl/2026-08-13/161_smoke_vagenlite_id74_k16_capture_tp8_base0_8g`；无Q/guided action/optimizer，结论只覆盖capture。用户建议的dgx-54 7卡ID162 hold`517551`实际仍PENDING(Priority)，立即取消，elapsed0、无allocation/output/W&B。当前父`01e75245`、VAGEN`6ad75d6`、VERL`084f042b`；TP4 diagnostic明确不能替代TP8证据。

## 待确认问题

- 旧 ValueHead 在 VAGEN 中接收哪种 state：完整复用 Nimloth StateProjector/WM state，还是改用 VAGEN world state 并重新初始化；两者不可伪装为等价。
- `alpha`、`beta`、prior temperature、`gamma`、score dtype、critic loss coefficient、warmup/KL target 的正式实验值。
- 未执行 action slot 的 Q 校准与探索保护。
- frozen critic snapshot 的刷新/checkpoint边界，以及truncation bootstrap所引用的snapshot identity。
- 模拟尾部 action 的生成方式及其非 PPO 辅助目标。
