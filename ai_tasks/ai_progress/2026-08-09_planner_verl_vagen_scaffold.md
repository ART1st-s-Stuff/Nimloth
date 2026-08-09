# 2026-08-09 PlannerPolicyHead 复用 VERL/VAGEN 脚手架

## 目标

在不改变当前 PlannerPolicyHead 环境动作 PPO、executed-action `Q(s,a)`、WM/DINO 联合目标和 fresh-consumption/checkpoint 语义的前提下，尽可能复用 VERL/VAGEN 的训练与 rollout 脚手架，降低当前逐 transition 完整 prefix 重算和低并发 rollout 带来的耗时。

## 人类确认的边界

- 保留 PlannerPolicyHead、`Q(s,a)`、WM/DINO 目标，不切换为 stock response-token PPO。
- 正在运行的正式 ID147 Job `511059`继续跑完；本任务在独立 worktree 开发，不改变 ID147 runtime commit、输出或训练配置。
- 第一阶段同时覆盖训练后端和 rollout，但按可独立验证的适配层分步落地。

## 已确认事实

- 当前 pinned VAGEN 默认 `actor_rollout_ref.actor.ppo_epochs: 1`；critic 继承同一值。现有 Nimloth VAGEN baseline launcher 没有覆盖它，因此不是4 epochs。
- VERL actor/critic 的每个 epoch 会遍历全部 PPO mini-batch；一个 global step 的 optimizer-step 数还取决于 train batch / mini-batch，而不能只看 `ppo_epochs`。
- 当前 PlannerPolicyHead run 配置为每个 fresh rollout batch 做4 epochs；这是 Nimloth 自定义训练语义，不是从 VAGEN 默认值继承。
- VERL可复用能力包括 DataProto、FSDP、gradient checkpointing、offload、worker/resource orchestration、checkpoint和序列长度平衡。
- pinned VERL 的 actor/critic 只有非多模态 batch 才进入 `rearrange_micro_batches`；检测到 `multi_modal_inputs` 时会优先按固定 micro-batch chunk。因此当前 Qwen-VL planner 不能仅打开 `use_dynamic_bsz`就获得动态 token batching，custom worker必须显式补齐多模态分桶/packing并做显存门禁。
- VAGEN rollout manager会把当前仍active的多个env组成一次 generation batch，并通过 `BatchEnvClient`批量reset/step，这是可复用的并发骨架。
- 当前 Nimloth `PolicyStateCaptureWorkerExtension`明确只支持串行单请求；它不保存request/sequence ID。直接把多条请求交给VAGEN manager会混合latent hidden，不能声称正确。rollout迁移的首个P0是实现并验证per-request vLLM state capture与输出对齐。
- stock VAGEN/VERL actor/critic训练 response token policy/value，不能直接表达当前环境动作分布、完整 decision prefix、PlannerPolicyHead 和 `Q(s,a)`。

## 既有资产

`feat/fsdp-dynamic-rollout` 已实现并真实门禁过一套旧版 VERL/VAGEN 在线路径，包括 `verl_adapter.py`、`vagen_online_rollout.py`、DataProto、FSDP full actor/critic、动态 rollout 和 checkpoint sidecar。该分支基于较早 commit，目标后来演变为 token PPO/critic，且与当前 PlannerPolicyHead 代码大幅分叉。

因此本任务从当前 `dev` commit `79f12f06`建立新分支 `feat/planner-verl-vagen-scaffold`。旧分支只作为经过验证的组件来源；禁止整分支合并或静默恢复已经废弃的 token-PPO/固定 thought/旧 schema 语义。

## 实施计划

1. 记录 ID147 的 rollout、训练、保存分项耗时，建立不改变算法的性能基线。
2. 写 RED contract tests：同一 batch 下 old log-prob、advantage、clipped actor loss、executed-action value、WM/DINO eligibility 和 consumption 边界必须与当前实现一致。
3. 定义 Planner decision batch/DataProto adapter；对可变长多模态 prefix 做显式 token/image-aware packing，禁止截断、补默认 action 或丢弃 terminal CoT。每个FSDP rank必须执行相同数量的forward/backward micro-batch。
4. 实现单一 Nimloth custom VERL FSDP worker/root module：复用worker/resource/checkpoint/offload机制，但调用当前 PlannerPolicyHead、ValueHead、WM/DINO loss，避免多个独立DDP wrapper再次产生collective顺序风险。
5. 为vLLM hidden capture加入per-request identity并先做batch generation↔latent hidden↔action log-prob逐row对齐门禁；通过后再接VAGEN rollout manager的active-env并发和Ray资源编排。保留当前真实 CoT、严格 manifest、seed ownership和可审计输出。
6. 依次通过 CPU parity、单GPU非消费型 batch、分布式非消费型 mechanics、fresh rollout smoke；通过前不替换正式 runner。
7. 用 wall-clock、GPU峰值、transition/s、trajectory/s 和逐项数值 parity 比较旧/新后端，再决定默认 `ppo_epochs`。VAGEN默认1只能作为性能对照，不能未经批准改变 ID147 的4-epoch算法。

## 当前状态

- 独立 worktree：`/workspace/remote2/nimloth-feat-planner-verl-vagen-scaffold`
- 分支：`feat/planner-verl-vagen-scaffold`
- 起点：`79f12f06601ce514dabe8fac957317007804506d`
- 已完成 rollout P0 的CPU接口实现：`PolicyStateCaptureWorkerExtension`使用vLLM V1 `req_ids + query_start_loc`把flattened hidden/token rows按request分流；frontend按显式request ID恢复、检查TP rank parity并拒绝缺失/重复/misaligned identity。`QwenVLLMAgentPolicy.select_responses_with_state()`用一次`engine.generate(requests)`返回逐row CoT、latent hidden和action logits。
- 定向CPU回归：`tests/backbone/qwen25vl/test_vllm_{hidden,policy}.py`为`21 passed`。覆盖两个request在decode forward中换序交错、请求顺序恢复和既有串行capture兼容；`compileall`与`git diff --check`通过。
- 上述结果尚未经过真实vLLM 0.11 TP4多模态generation；在GPU门禁前不能接入正式VAGEN active-env runner或声称rollout提速。
- 训练端完成第一层可逆batch seam：`actor_transition_batch_step()`只把多个完整prefix合并成一次Qwen forward，WM/value/PlannerPolicyHead继续逐transition走原有已验证目标并求和。两条transition的scalar-vs-batch loss、全部metrics及Qwen/StateProjector/WM/ValueHead/PlannerPolicyHead gradients逐项一致；fake builder确认Qwen build/forward从2次降为1次。
- loop新增`training.planner_micro_batch_size`，默认1，故既有ID147/旧config语义不变；大于1时每个micro-batch只做一次backward/barrier，padding row仍zero-weight且不进入metrics。episode/loop/config定向回归`75 passed`。
- 新增严格Planner VERL adapter：运行时必须从当前worktree `external/VAGEN/verl`导入且commit精确为gitlink `65316156...`；DataProto schema保存真实`ExecutedTransition`、MC return、old Q/action log-prob、advantage、loss weight、token count和可选DINO target，不构造或替换CoT。
- 多模态packer按实际padded cost `max(sequence tokens) * rows`做deterministic first-fit-decreasing，直接补足pinned VERL在`multi_modal_inputs`分支绕过dynamic-bsz的缺口；单条prefix超budget时fail closed。
- `PlannerVERLUpdateCore`实现begin→多个DataProto backward→单optimizer step→abort生命周期；`PlannerVERLWorkerMixin`使用VERL原生`ONE_TO_ALL`与`DP_COMPUTE_METRIC` decorator暴露Ray worker调用边界。driver仍拥有fresh consumption，不会被worker提前commit。
- `PlanningPolicy.select_actions()`已支持H=1 PlannerPolicyHead active-env batch，一次调用turn policy的batched Qwen generation并逐row保留真实CoT/state/action trace；非policy search显式拒绝batch。
- active-env生命周期已接通：`AgentRuntime`拆出pending prompt/record decision/terminal prompt边界；Planner与Qwen均支持batch terminal CoT；新collector复用VAGEN `BatchEnvClient`的batch create/reset/step/close，只向仍active env发请求，按显式identity对齐decision/state，且只持久化已完成的连续episode prefix。
- rollout CLI新增显式`--batched-active-envs`，默认关闭；当前只允许H=1 PlannerPolicyHead、`max_episode_attempts=1`且禁止resume，避免把尚未实现的per-request retry/replay伪装成正式等价路径。两env fake VAGEN门禁覆盖1步/2步错峰结束、action batch 2→1、terminal batch、5张图和atomic JSONL。
- 更宽本地CPU回归：`tests/training/rl`为`228 passed, 1 deselected`；`tests/agent + tests/backbone/qwen25vl + batched collector`在忽略需要真实本地vLLM包的既有`test_vllm_logits.py`后为`89 passed`，合计317 passed。唯一deselect是本地临时venv缺完整VAGEN transitive env依赖所影响的既有`vagen_eval` wording测试，与改动路径无关。
- 仍缺真实FSDP worker的模型装配/checkpoint、真实VAGEN/vLLM GPU门禁以及batched per-request retry/resume；未门禁前禁止开启正式flag或把config默认值改大。
- 本地workspace初始可用空间不足导致worktree内测试venv安装失败；已立即删除该venv恢复空间，测试环境改放`/tmp/nimloth-test-venv`，没有删除项目artifact或共享数据。人类随后释放空间，submodule已恢复到exact pins。
- 本分支未改变ID147。SSH恢复后确认ID147 Job`511059`已在06:53:30+08 `COMPLETED 0:0`：global step0→20全部committed并完成held-out120；overall success21/120=0.175。稳态iteration约10.7--15.5分钟，其中two-TP4 rollout关键路径约4.5--8.2分钟、四epoch训练rank关键路径约4.5--6.8分钟，是新脚手架需对照的实际wall-clock基线。

## ID148 real TP4 multimodal request-state gate

- 人类已确认启动`preempt`单节点4×H800门禁。W&B project/run为
  `nimloth-rl/148_smoke_tp4_mm_policy_state_batch2_k16_vllm011`，输出目录为
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-09/148_smoke_tp4_mm_policy_state_batch2_k16_vllm011`。
- 门禁代码commit为`7d290728abb2f654d44bbb5ecf96bcc70bf8defd`，服务器使用独立detached worktree
  `/project/peilab/atst/nimloth/.worktree/feat-planner-verl-vagen-scaffold-8cbc0360`；VAGEN/VERL/le-wm仍固定
  `192c35a9/65316156/8edfeb33`。
- 只读ID147 committed `train/latest` Qwen与其held-out trajectories中两个不同首步
  multimodal prefix。所有模型冻结，无optimizer、无训练、无新rollout或consumption；held-out输入仅用于
  inference integration test。
- `gate_vllm_policy_state_tp4.py`在一个vLLM 0.11 TP4 batch生成两个请求，强制
  temperature/top-p=1，逐请求验证16×hidden latent、8 action logits、finite、TP-rank parity，且
  captured action logits的log-softmax必须在`atol=rtol=2e-3`内复现同请求behavior log-probs；两个请求
  state必须可区分，从而直接检验request identity未交换。
- 资源上限为preempt单节点4 H800、1小时；预计实际5--15分钟。短门禁没有checkpoint/resume；若失败，
  ID148按terminal记录且用新身份重试，绝不修改或续写ID147。

### ID148 terminal result

- Slurm hold Job`511432`分配`preempt/dgx-38:4`。真实gate step`511432.0`在1分31秒内
  `COMPLETED 0:0`；确认成功后主动scancel hold释放四卡，因此parent显示
  `CANCELLED by 3738`是资源清理，不是实验失败。
- vLLM 0.11 TP4、BF16、eager、无prefix cache的batch2实跑通过。两个request分别选择action7/2；
  每个都得到finite latent`[16,2048]`和action logits`[8]`，四个TP rank严格副本parity通过。
- 对两个request，captured action logits的log-softmax与同request behavior action log-probs的
  max error均精确`0.0`。两个request的action-logit最大差为`0.015625`、latent最大差为
  `12.0625`，证明本门禁输入下state可区分且没有request交换。
- W&B run`7n4pwjq8`完成sync，并由正确entity API独立核验为`finished`、summary
  `gate/status=ALL_OK`。输出`result.json`、完整command与日志均在ID148目录。
- runtime打印了可选`torch-c-dlpack`/TVM-FFI extension不可用以及其建议Torch>=2.11的警告；固定环境
  Torch2.8/vLLM0.11仍完成全部generation与parity。该警告不影响本门禁正确性，但在吞吐门禁中需单独记录，
  不能据本次success忽略性能影响。
- request-identity P0已被真实GPU证实；这仍不证明完整VAGEN active-env trajectory、长prefix FSDP
  update或吞吐提升。下一步门禁保持这三项边界。

## Human correction: one complete-objective epoch per global step

- 人类明确否决此前未经要求的`1+3`设计，并确认每个fresh rollout global step只做一个optimizer
  epoch；该次同时训练WM、DINO、ValueHead和PlannerPolicyHead。已登记
  `ai_rules/known_errors/E0090_do_not_invent_ppo_epoch_objective_schedule.md`。
- planner config/schema/runtime统一要求`ppo_epochs=1`并拒绝其他值；全部planner YAML同步为1。
  loop固定`include_world_model=True`，删除多epochmetric平均和auxiliary loss重组。
- VERL DataProto删除`include_world_model`控制字段，custom worker每个micro-batch固定执行完整objective，
  所有micro-batch仅汇成一个optimizer step。
- 旧ID147的四epoch运行与checkpoint是历史事实，不得改写；新语义与其optimizer/checkpoint config
  不兼容。后续如使用ID147权重，必须明确作为weights-only初始化而非resume。
- RED定向套件先得到22个预期失败；GREEN后config/loop/adapter/worker定向`75 passed`，扩大回归
  `269 passed, 1 failed`，唯一失败是临时本地环境缺VAGEN传递依赖SciPy，非本次改动路径。
- ID149 rollout门禁准备新增`planner_policy_h1_active_env_rollout_gate.yaml`：H=1、batch2、
  base_train/common_sense_train、max attempts1、max20 steps、TP4和single epoch。shard launcher新增
  默认关闭的`SHARD_BATCHED_ACTIVE_ENVS`，打开时严格要求H1/policy/attempt1并传递
  `--batched-active-envs`。相关config/launcher与单epoch套件合计`134 passed`。

## ID149 active-env rollout launch contract

- 人类确认`preempt`单节点4×H800、1小时上限。W&B project/run为
  `nimloth-rl/149_smoke_activeenv2_basecs_seed161_h1_k16_tp4_singleep_t20`；输出为
  `/project/peilab/atst/nimloth/outputs/experiments/training/rl/2026-08-09/149_smoke_activeenv2_basecs_seed161_h1_k16_tp4_singleep_t20`。
- runtime commit必须是`215f3a40`或仅增加本launch contract的后继commit；VAGEN/VERL/le-wm保持
  `192c35a9/65316156/8edfeb33`。入口为`run_vllm_rollout_shard.sh`并显式
  `SHARD_BATCHED_ACTIVE_ENVS=true`。
- 模型与planner artifacts只读ID147 committed `train/latest`。全部模块冻结，无optimizer、无
  consumption；这不是对ID147 resume。数据为`base_train/common_sense_train`各一条，
  `seed_per_eval_set=true`、未消费seed161、max20 steps、attempt1、current navigation profile。
- 门禁必须得到两条完整trajectory及terminal真实CoT/state、逐step planner trace、不同record ID、
  严格fresh manifest和batch active-env无丢失证据；W&B只记录rollout summary。短门禁无checkpoint/
  resume，失败时ID149 terminal并使用新identity重试。

### ID149 terminal result

- hold Job`511694`分配`preempt/dgx-16:4`。renderer step`511694.0`以69.403秒通过且
  `COMPLETED 0:0`；gate step`511694.1`为`COMPLETED 0:0`/3:27。成功后主动取消hold释放
  四卡，parent CANCELLED只表示清理。
- 真实active-env batch得到精确2 trajectories/34 transitions：base_train seed161为20步、reward
  -0.7、失败；common_sense_train seed161为14步、reward9.9、成功。聚合success0.5只是门禁诊断。
- 每动作均有真实assistant response与policy planner trace；两条记录均有steps+1 state anchors、
  k16 `[16,2048]` latent和terminal真实CoT/state。日志有连续batch capture；第一环境关闭后，
  第二环境继续单active request，两个结束边界各有terminal batch capture。
- format5 fresh manifest严格绑定ID147 policy fingerprint及WM/StateProjector/ValueHead/
  PlannerPolicyHead四份fingerprint；无consumption。common_sense中3次reasoning按真实length边界截断并
  持久化，不是fixed/invented CoT。
- W&B`zady597f`完成sync且API独立核验`finished/ALL_OK`。active-env rollout P0关闭；仍未验证
  real VERL FSDP worker、single-complete-objective optimizer epoch或吞吐。

## Single-root FSDP objective seam（CPU verified）

- `PlannerObjectiveModule`现在注册完整`Agent`且由自身`forward()`执行既有
  `actor_transition_batch_step()`。后续FSDP root必须调用这个forward，禁止从root外直接调用Qwen或
  WM/head child；这样才能让完整prefix Qwen、WM/DINO、executed-action Q和PlannerPolicyHead PPO
  位于同一FSDP all-gather/reshard边界。
- `wrap_planner_objective_fsdp()`要求CUDA distributed已初始化、显式enabled wrap policy和未预包装的
  Agent，使用FULL_SHARD/`use_orig_params=True`。`initialize_planner_fsdp_update()`保证先wrap后建
  optimizer，并把gradient clipping固定接到root `clip_grad_norm_`。
- DataProto schema升到3并绑定`update_id`。backward dispatch改为仅接收一rank一DataProto list的
  `DP_COMPUTE`合法签名；identity从每个rank batch metadata读取。生命周期显式区分
  ACCUMULATING/STEP_ENTERED/STEPPED，optimizer入口后禁止abort，durable checkpoint前禁止开始下一步，
  同一worker内拒绝completed identity重放，并提供validated checkpoint identity恢复接口。
- 测试：adapter/worker/common optimization定向`21 passed`；扩大training RL+optimization为
  `241 passed, 1 failed`，唯一失败是临时本地环境缺VAGEN/SciPy导致传递import失败，与本改动无关。
  VS Code diagnostics、compileall、`git diff --check`通过。
- server同步`6c6cb731`后，固定`.venv-vagen-main`定向`21 passed`、扩大
  `tests/training/rl + optimization`为`242 passed`，本地缺失的VAGEN/SciPy wording测试也通过。

## Concrete Ray worker / checkpoint transaction（CPU contract complete）

- `774ed47f`新增具体`PlannerVERLFSDPWorker`。Ray actor内初始化NCCL、校验pinned VERL、调用显式
  `module:factory`装配未包装Agent，再建立single-root FSDP和optimizer；不继承stock response-token
  PPO worker，也不调用其actor/critic objective。
- 生产factory只接受显式ID147类weights-only artifact位置；`resume=true`或resume checkpoint在模型加载前
  fail closed。它要求PlannerPolicyHead、single complete epoch、完整prefix recompute、一GPU一FSDP rank、
  frozen vision，并从Qwen声明的transformer classes构造wrap policy。旧optimizer绝不加载。
- driver每轮要求每rank恰有一个非空且等row数batch，并校验schema/objective/total transitions/DINO
  metadata一致，避免nested FSDP因逐transition调用次数不同而collective错序。provisional DataProto identity
  在fresh claim后统一替换为collector的`consumption_id`，该ID同时进入worker replay guard、checkpoint和
  consumption记录。
- 新sharded checkpoint使用`FSDP.optim_state_dict`与`optim_state_dict_to_load`保存/恢复exact-world-size
  model/AdamW/RNG shards。rank-local目录/写入/sidecar/load preflight错误会在后续collective前跨rank汇总。
  driver仅在全部shard验证、临时目录fsync和atomic rename后commit fresh consumption；commit成功后才把
  worker从STEPPED释放到IDLE。
- 本地定向adapter/worker/driver/factory/optimization为`31 passed`；扩大training RL+optimization为
  `251 passed, 1 failed`，唯一失败是本地缺VAGEN/SciPy。review确认可提交和server非GPU回归。
- 新commit已推送，但server同步时SSH再次报timeout/UNKNOWN65535，因此`774ed47f` server回归未完成。
  仍缺真实Ray actor construction、NCCL/FSDP complete-objective数值、checkpoint save/recreate/load/next-step
  parity和long-prefix显存门禁；这些GPU证据通过前不能声称真实FSDP worker完成。

## ID150 Ray/FSDP mechanics gate preparation

- commit`945729f6`新增非正式tiny-model gate入口
  `experiments/training/rl/gate_planner_verl_fsdp_ray.py`。它计划在单节点多GPU上用真实pinned
  RayWorkerGroup/NCCL/FSDP执行一步更新、sharded model/AdamW/RNG checkpoint、第二步、回载第一步、
  重放第二步，并比较完整rank-sharded parameter SHA256而不只比较norm。
- gate同时验证每rank真实hostname/Ray node/CUDA_VISIBLE_DEVICES/GPU UUID、DataProto non-tensor transition
  identity、CPU/CUDA RNG逐rank恢复、checkpoint atomic publish与transaction commit顺序。tiny factory已移入
  可导入的`nimloth.training.rl.planner_verl_gate_factory`；W&B要求显式run ID和`resume=never`，只有
  W&B finish成功后才写`result.json ALL_OK`。
- CPU定向`19 passed`、compile和diff-check通过，review修复了nested BF16 Linear输入dtype问题。该gate
  不读取ID147、不消费ID149、不训练正式模型，不能替代后续long-prefix complete-objective门禁。
- SSH持续在banner阶段timeout，`945729f6`尚未同步server。按实验规则已请求preempt单节点2×H800、
  30分钟上限批准，但人类未选择资源选项；因此没有查询/占用GPU、没有创建ID150 output/W&B/Slurm job。
