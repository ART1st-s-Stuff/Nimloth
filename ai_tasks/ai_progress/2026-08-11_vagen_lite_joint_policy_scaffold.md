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
9. `joint_policy.enabled=true`当前显式 fail closed，直到已确认的 Q owner、rollout sampler、replay、optimizer 和 checkpoint snapshot boundary 全部真实接通。
10. 人类已确认生产ownership：rollout coordinator用无状态deterministic keyed draw，不使用worker-local或vLLM RNG；`AgentLoopManager`在workers前创建独立CPU Ray actor持有只读frozen critic snapshot；每个完整global joint update成功后为下一批rollout原子发布新snapshot，同batch/minibatch内不刷新。

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
- async same-generation capture bridge现复用既有vLLM worker hook，request-scoped捕获K16 hidden和action-boundary raw 8 logits，并通过两阶段TP prepare/finish避免partial-rank validation进入LM-head collective；失败cleanup只清当前request。sidecar绑定schema/request及latent/action token IDs，经TokenOutput/agent-loop/DataProto传播；DP>1、non-eager或保留engine override均fail closed。服务器CPU为VAGEN`93 passed,27 subtests`、父定向`79 passed`，capture专项为parent`26 passed`及VAGEN`36 passed,9 subtests`，最终review批准。ID161按人类纠正后的normal单节点8×H800/vLLM TP8合同在step`517501.0`/`dgx-39`以`COMPLETED 0:0`运行9分42秒；`mm_encoder_tp_mode=data`后ID74 TP8/DP1/eager加载、held-out base0真实turn、24-token CoT→K16→action0、reward0.0/anchor22/ledger valid均通过。request`1ff645dc78424e9d85b7a6846bdcda81`绑定finite `[16,2048]` hidden和raw `[8]` logits，validator`ALL_OK`；峰值显存50,244--50,915MiB，cleanup后8卡0MiB、端口/owned process为空，hold已释放。结果目录`outputs/experiments/training/rl/2026-08-13/161_smoke_vagenlite_id74_k16_capture_tp8_base0_8g`；无Q/guided action/optimizer，结论只覆盖capture。用户建议的dgx-54 7卡ID162 hold`517551`实际仍PENDING(Priority)，立即取消，elapsed0、无allocation/output/W&B。
- 新增两项纯TDD里程碑。VAGEN`1e70366`的`replay_guided_behavior_log_probs`逐条重验record并强制expected contract/snapshot，同batch必须同action table/config/snapshot；只从record读取rollout-time all-action frozen Q，current prior logits为唯一梯度入口，返回guided action对应的current与behavior log-prob且没有current-Q输入。VAGEN`f081258`的selected-action Huber helper只读取实际执行slot并detach target，delta/reduction显式必填；dtype fail closed，FP16/BF16以FP32和clamped decomposition避免overflow/NaN gradient。最终真实CPU Torch/joint/ledger定向`65 passed,23 subtests`，review批准。
- 父`9f811fd9`新增严格current critic与内存frozen snapshot：仅加载同root ID74 projector/head并校验outgoing-Q语义、shape/action/dtype，forward为`K16 hidden→shared projector→mean→[8]`；snapshot显式绑定source step/contract并hash live architecture+weights，每次forward前后重验eval/no-grad/identity。mutation与semantic/dtype门禁review批准；本地相关`29 passed`、服务器旧grid/objective`8 passed`，fresh真实ID74 CPU gate得到finite FP32`[1,8]`且旧snapshot在current权重变化后逐bit稳定。actor FSDP、环境动作、return compiler、optimizer/Ray owner/refresh/checkpoint仍未接通，trainer继续在worker前fail closed。VERL仍为`084f042b`。
- capture→frozen-Q纯scoring里程碑已完成并发布：VAGEN`d9451fa`把capture schema升级为v2，episode级`request_id`只负责sticky server，manager用随机namespace+单调counter为每次Nimloth forward生成不可由调用者复用的`generation_id`；vLLM request/capture按generation identity执行，sidecar同时持久化session/generation/token identities。真实one-turn launcher validator同步要求v2和两identity非空互异。
- 父`e0b5ae81`新增`joint_scoring.py`：严格校验capture v2、session/generation/token table、snapshot/contract/action count和finite shape；按snapshot参数dtype喂入K-slot hidden，并将raw prior logits与all-action Q统一量化为snapshot identity中已哈希的`score_dtype`。immutable `FrozenQScoringRecord`的direct/build/replace/from_mapping均重复校验identity/token/finite/alignment并统一量化，禁止绕过scorer伪造精度。
- `FrozenJointCriticSnapshot`现在把`score_dtype`纳入实时identity fingerprint，构造时必须显式声明；critic forward不再无条件窄化FP64为FP32。两轮review先发现dtype/history、generation uniqueness、collapsed identity和launcher v1问题，修复后最终review`APPROVE`。
- fresh服务器CPU回归：parent capture/critic/scoring/launcher为`46 passed`；VAGEN capture/joint-policy/ledger/config为`84 passed,27 subtests`（仅既有Ray/Swig弃用warning）。没有运行新GPU smoke，也没有guided sampling、actor/critic optimizer、snapshot refresh或checkpoint wiring；`joint_policy.enabled=true`继续在worker创建前fail closed。
- VAGEN`3fded6a`完成Navigation-only guided execution授权合同：`GuidedActionExecutionRequest`绑定完整重验后的behavior record、record id和实际raw response UTF-8 SHA-256；环境仍先按Nimloth格式解析原始LLM response并核对prior action，只把behavior中guided action送给AI2-THOR，原始response evidence不被替换。
- remote transport为guided mutation使用独立`step_guided`方法；普通`step`携带guided payload会在mutation前拒绝，旧server不会静默执行raw prior。server与client都重验request，并在动作后核对environment echo中的raw response、action table、guided action id/name和完整request。`step`与`step_guided`均不自动重试；不支持Navigation专用`guided_step` capability的环境在mutation前拒绝。
- review先发现同名free-think text action可冒充Nimloth token prior，现已要求guided override只能用于`prompt_format=nimloth`并有`_exec_action`前失败测试；最终review`APPROVED`。fresh服务器VAGEN全套CPU tests为`118 passed,43 subtests`，仅既有Ray/Swig warning。
- execution envelope尚未由agent loop生产，Q owner/scorer尚未成为Ray rollout service，也未确定guided sampler RNG；actor replay、critic optimizer、snapshot refresh/checkpoint仍未接通。因此本里程碑只关闭“保留raw policy evidence同时执行另一个已授权action”的接口缺口，trainer继续fail closed。
- VAGEN`2ac1dbd`将execution envelope显式升级为v2并加入`response_trace_id`，拒绝v1/missing-field artifact。父`f2f6ad63`新增`NimlothPolicyResponseTrace`和纯assembly helper：trace绑定sticky request、unique generation、caller-pinned generation-spec identity、完整response IDs/mask/log-probs和raw decode文本；trace全文hash进入execution envelope。
- assembly重验scoring/trace的request+generation、snapshot、contract、score dtype、token table与expected generation spec；mask精确复现agent-loop语义，包括reasoning达到上限时forced close token为0；全部log-probs必须finite，sampled action log-prob按float64/float32/bfloat16合同容差核对prior logits。guided action只能作为外部显式ID输入，helper没有RNG、current Q或environment mutation。
- review发现并修复旧helper可混入另一generation同形trace、raw文本未绑定IDs、reasoning prefix mask未全验、action-end由caller自由指定、forced logprob可非finite和dtype容差缺测；最终review`APPROVED`。fresh服务器VAGEN全套`118 passed,43 subtests`，parent critic/scoring/behavior/capture定向`77 passed`（最终新增expected spec identity后behavior组合定向`76 passed`）。
- agent loop尚未构造response trace/scoring/behavior，也没有critic Ray owner、guided RNG或environment call接线；joint trainer继续fail closed。
- VAGEN`6ff224e`新增纯external-draw Scheme-B sampler：调用者显式传`uniform_draw∈[0,1)`，helper以half-open inverse CDF选择action并记录config/contract/action table/tokens/prior logits/rollout frozen Q/guided log-probs/draw/action/logprob。模块不导入RNG、不接受current Q、不调用environment。
- sampler使用probability-space CDF确保常见0.7边界归入下一区间，同时对draw=0单独保留数学上正但`exp`可能下溢的首action。direct/mapping构造均重算derived fields并严格校验类型；所有持久化float及公共config的`beta=-0.0`规范化为`+0.0`，保证相等record拥有相同contract/record ID。
- 两轮review修复log-space边界误选、mapping预tuple化绕过strict sequence、signed-zero导致equal record不同hash及嵌套config signed-zero；最终review`APPROVED`。fresh服务器VAGEN全套`130 passed,65 subtests`。
- RNG owner/seed/stream仍未决定，agent loop未提供draw，也未将draw record接入behavior assembly/environment；trainer继续fail closed。
- VAGEN`3840f2c`把execution envelope升级为v3并新增`action_draw_record_id`。父`0a29e0b9`把behavior assembly收紧为只接受完整`GuidedPolicyActionDrawRecord`，删除裸`config/action_space/guided_action_id`输入；draw经mapping重验后与scoring的contract、action token table、prior logits、rollout frozen Q和score dtype逐项一致，behavior中的selected action/logprob只来自draw record。
- execution envelope现同时持久化response trace与action draw两个audit ID，不能在assembly边界绕过外部draw选择或丢失draw provenance。review只发现文档仍写v2与旧action-table helper死代码，修复后最终`APPROVED`。
- fresh服务器VAGEN全套`130 passed,65 subtests`，parent behavior/scoring/critic/capture相关`76 passed`。agent loop仍未创建snapshot scoring/trace/draw/behavior链，trainer继续fail closed。
- 人类确认此前待定的三个生产ownership：coordinator-owned deterministic keyed draw绑定run seed、global policy step、stable sample/repeat identity、turn、snapshot、contract和RNG schema，使调度/worker重启/基础设施重试不改变同一逻辑decision的draw；`AgentLoopManager`在agent-loop workers前创建独立CPU Ray actor持有active immutable snapshot；trainer在一个完整global joint update成功后stage并原子activate下一snapshot，同一rollout batch及其PPO minibatches不刷新，历史record继续只用持久化旧Q。实现仍按TDD分阶段进行，当前不因此解除fail-closed。
- VAGEN`b8c6f55`实现keyed-draw合同：`GuidedActionDrawKey`完整绑定run seed/policy step/stable sample/repeat/turn/validation/snapshot/contract/schema，canonical JSON SHA-256前53位精确映射到`[0,1)`；public sampler删除裸`uniform_draw`参数，action-draw schema升v2并持久化完整key与derived draw，direct/mapping均重验。父`caefc381`要求assembly同时接收coordinator生成的`expected_draw_key`并做完整相等核验，随后再核对scoring snapshot/contract/tokens/prior/Q；不能把另一step/trajectory/repeat/turn/validation的自洽record混入。
- review先发现assembly只核对snapshot而未核对完整logical decision、文档仍描述旧裸draw API；修复后`APPROVED`。fresh服务器VAGEN全套`137 passed,75 subtests`，parent behavior/scoring/critic/capture相关`81 passed`。生产agent loop尚未构造stable key，CPU Ray Q actor尚未实现，trainer继续fail closed。

- 人类纠正执行方式：资源优化不是赶工或降低正确性。此前把相互依赖的内部helper拆成过多小milestone，并对每个helper重复完整测试/review/提交/push，导致生产主链路接通过晚；已登记`E0096`。后续把CPU owner、batch pin、stable identity和optimizer-free capture→score→draw→behavior→guided env/DataProto wiring组成一个有外部意义的milestone，中间只做必要定位检查，完成后统一跑全套/runtime/review/发布；trainer仍fail closed。
- 该optimizer-free production guided rollout milestone现已形成未验证的完整实现候选：parent新增Ray-safe immutable snapshot transport和`FrozenQSnapshotOwner`，VAGEN新增1CPU/0GPU actor、真实parameter-dtype stage门禁、单线程PyTorch runtime、manager-before-workers lifecycle、整batch pin/unpin、CAS stage/activate与clean checkpoint RPC。dataset生成restart-stable sample id，trainer保留显式repeat index；manager按run seed/policy step/validation/sample/repeat/turn/snapshot/contract预分配全部draw key。
- no-concat Gym loop现于same-generation capture校验后、环境mutation前执行CPU scoring→keyed Scheme-B draw→response trace/behavior/execution assembly，只调用`guided_step`，并将batch pin、scoring、trace、draw、execution、guided ledger、stable identity、原response IDs/mask/log-probs和实际reward持久化到DataProto。guided错误不再被普通environment fallback吞掉；disabled mode不向旧custom worker/agent-loop constructor额外传`None`。standalone one-turn新增显式guided入口，所有未定policy/run/critic参数必须由caller提供，无实验默认值；`RayPPOTrainer`仍在worker前fail closed。
- milestone RED在实现前为`3 failed, 1 skipped`，缺口精确对应manager pin、stable identity和agent-loop production assembly。实现候选本地`py_compile`及parent/VAGEN `diff --check`通过；实现后的统一服务器测试尚未启动，因为首次同步时SSH中断。随后确认直接`rsync`源码违反`.local/SERVER.md`的Git同步规则，已登记`E0097`；旧远程测试worktree不再作为可信验证环境，后续必须以明确parent/VAGEN/VERL candidate SHA在干净worktree测试。
- 后续改用Git candidate refs并为每轮修复创建全新clean服务器worktree。首轮milestone套件发现3项测试assertion问题，修复后为`167 passed,36 subtests`；扩大回归首轮的19项失败全部来自既有`planner_verl_adapter`仍固定capture直接父`3fe0a299`而当前runtime已是`084f042b`，更新exact runtime pin后相关`27 passed`且扩大套件为`513 passed,75 subtests`。
- 独立Claude Opus只读review对parent/VAGEN完整diff结论为`APPROVED`、无P0/P1；提出的具体P2已统一修复：DataProto新增显式0-based`guided_turn_index`并与历史1-based`turn_idx`做运行时关系检查，Ray actor RPC严格拒绝额外字段，pin RPC结果不明时用预构造pin best-effort unpin，critic transport上界改为精确参数元素计数。修复后定向`29 passed`，最终扩大回归在parent`06b993b2`/VAGEN`3a01a2a0`/VERL`084f042b`为`514 passed,75 subtests`；真实local Ray actor lifecycle包含CPU/GPU/thread、pin/CAS/checkpoint和malformed request门禁。最终Opus复审`APPROVED`且无P0/P1/P2 finding。
- 当前milestone的CPU/Ray/审查边界已关闭，TP8 optimizer-free guided one-turn GPU gate尚未运行。该gate需要caller显式给出`alpha/beta/prior_temperature/score_dtype/run_seed/source_step`；正式实验值仍未获人类确认，因此不能自行选择默认值启动。actor replay、critic optimizer、return compiler、global-update snapshot publication和完整checkpoint/resume仍未接通，trainer继续fail closed。
- milestone已按Git发布策略推进feature branches且未修改任何main：测试/复审对应代码SHA为parent`06b993b2`、VAGEN`3a01a2a0`、VERL`084f042b`；随后仅追加验证文档，发布为VAGEN`nimloth/upstream-joint-policy-scaffold@18a04bb1`和parent`feat/vagen-lite-joint-policy-scaffold@007a0314`，parent gitlink精确指向`18a04bb1`，远端SHA已核对。

## 待确认问题

- `alpha`、`beta`、prior temperature、`gamma`、score dtype、critic loss coefficient、warmup/KL target 的正式实验值。
- 未执行 action slot 的 Q 校准与探索保护。
- 磁盘checkpoint频率与truncation bootstrap完整return compiler；active snapshot必须随完整checkpoint保存，但磁盘落盘不要求每个global update一次。
- 模拟尾部 action 的生成方式及其非 PPO 辅助目标。
