# 2026-07-31: 重训 SFT2 value-v3，再启动 H=1/K=1 RL

## 人类决定

人类批准先重训 corrected SFT2，再进行 RL。RL 使用 H=1/K=1，只监督
selected/executed action，禁用 ranking loss 和 PPO。

## SFT2 契约

- 代码基线：2007c661 的 decision_state_executed_action_mc_v3。
- 初始化：SFT1 merged checkpoint；旧 successor-state SFT2 checkpoint 不兼容，
  不作为初始化或 resume 来源。
- 数据：ID52 的 train/val terminal-CoT migrated JSONL，包含成功与失败 rollout。
- cache：只读复用已完整验证的 ID53 compact preprocess cache；ValueHead 的
  decision-state 对齐不改变预处理 cache 内容。提交前只做当前 commit 的读取校验，
  不重建、不覆盖 cache。
- 训练：H=1，T=4，2 epochs，24×H800（preempt 3节点×8卡），per-rank B1，
  GA4，effective global batch 96，fresh optimizer。该 WS24 合同由人类在
  2026-08-01 明确覆盖此前 WS16 合同。
- trainable：Qwen vision、StateProjector、WM predictor、ValueHead。
- frozen：Qwen LLM、DINO teacher/cache、latent query。
- loss：当前步 CE；4 个 successor 的 WM/DINO；4 个 decision state 上对应
  executed action 的 Monte Carlo ValueHead MSE；全局 SIGReg；无 ranking loss。
- 监控：W&B nimloth-sft2，训练/验证 loss 与 val_wm_mse；20 分钟周期
  latest 和 epoch/best/final checkpoint。
- 生命周期：Slurm batch job 自持三节点控制器，不使用登录节点 watcher，
  不在控制器中调用 scancel。

## RL 后续门禁

SFT2 final 通过 checkpoint/invariant/finite-metric 校验后，先做 4-GPU 单次
optimizer-step smoke，再提交正式 RL。RL 固定 predictor.history_size=1、
agent.planning.horizon=1，StateProjector frozen；Qwen、WM predictor 和
ValueHead 接收 executed-action ValueHead 监督梯度。

## 当前状态

- 独立分支/worktree：exp/sft2-value-v3-rl-h1k1。
- 已新增 batch-owned WS16 启动器、节点/rank/H800 门禁、cache 只读 preflight、
  W&B identity 保留与训练完成 checkpoint validator。
- 启动器静态合同 3 项、bash syntax、Python compile 和 diff-check 通过。本地旧
  .venv 的 pytest 入口因解释器链接失效而缺包；superpod clean worktree 固定
  9b0c9ff2，使用 .venv-vagen-main/bin/python3 的完整 SFT2 与 ValueHead objective
  CPU 回归为 114 passed, 1 skipped in 72.22s，skip 仅为显式可选 GPU/NCCL 门禁。
- ID64 只读 preflight 已完成，commit 为 8d9c4b79，W&B run name 为
  64_valuev3_terminalcot_dinogrid_k16_h1_t4_ep2_b1_ga4_ws16_px100352，
  requested run id 为 fcd9b34a。preflight.json 为 status=passed：
  49,638/4,989 个 train/val H1/T4 windows 全量读取；输入哈希、cache
  fingerprints/shards、BF16 materialization、DINO coverage 和 W&B 唯一性均通过。
- WS16/B1/GA4 调度为每 epoch 3,103 microbatches、776 optimizer steps，两 epoch
  共 1,552 steps；每个 global SIGReg microbatch 有 6--16 个有效 states。preflight
  仅在 ID64 新目录写日志/报告，没有修改 cache，没有创建 GPU job、W&B run、
  optimizer 或 checkpoint。
- 正式训练已提交为 Slurm job 500294：normal、2 节点×8 H800、每节点64 CPU/
  800 GiB、world size16、8小时上限。scontrol 核验 ReqTRES=cpu128/mem1600G/
  gres-gpu16，TresPerNode=gres-gpu8。
- 当前为 PENDING(Priority)。提交前 normal 只有15张空闲GPU，test-only 保守预计
  2026-08-04 03:10 UTC 才能启动；preempt 当时有两台完整8卡节点，但人类已确认
  normal，因此没有擅自切换分区。尚无训练输出、W&B run、optimizer/checkpoint；
  allocation 后仍需两节点 H800/rank 门禁和首批 finite optimizer-step 健康检查。
- job 500294 后续在 allocation 前被取消：`Elapsed=0`、无节点、W&B、optimizer 或
  checkpoint。复核发现已提交的 batch/node shell 把运行时变量写成带反斜杠的字面量；
  虽然该错误未在 GPU 上执行，但脚本若获得节点会在模型加载前失败。现登记 E0076，
  修复后必须使用新 commit、新 ID、空输出和新 W&B identity 重做 preflight/提交。
- 人类随后明确要求直接使用 preempt 的三台完整8卡节点，即3×8 H800/WS24；此前
  “只用其中两台组成WS16”的解释失效。WS24 保持 B1/GA4，因此有效全局batch由64
  变为96；49,638个train windows对应每epoch约2,069个microbatches、518个optimizer
  steps，2 epochs预计1,036步，最终以当前commit的生产sampler preflight为准。
- 已新增batch-owned WS24 launcher：3个Slurm task、每节点1个task并在节点内启动8个
  local ranks，global ranks为0--23；preflight拓扑改为显式partition/nodes/GPU参数，
  completion validator显式检查world size24。两个WS16/WS24 launcher静态合同共7项、
  shell syntax、Python compile和diff-check已通过。尚未提交；superpod跳板
  `10.88.0.3`连续两次在SSH握手后立即断开，需连接恢复后完成实时资源、W&B/new-ID、
  cache只读验证、远端clean exact-commit回归和正式提交。
- superpod连接恢复后，最终代码更新到`92efac9c`；远端clean worktree的SFT2/WM/latent/
  planner回归为`141 passed, 1 skipped in 35.16s`，launcher定向回归`8 passed`。
  W&B `nimloth-sft2` live max ID为63，但ID64已被旧preflight和取消作业占用，因此新实验
  使用ID65、run id `6oz3cm0f`，禁止复用ID64。
- ID65首次全量preflight由SSH会话直接拥有；连接在约5分钟后被远端关闭，进程继续到
  约8分钟后消失，但未写出`preflight.json`且stdout/stderr已丢失，无法判定后段assertion
  或session cleanup。该attempt不放行训练，也不重复使用残留进程；问题登记为E0077。
  已新增CPU-only、batch-owned preflight脚本，使用cpu单节点8 CPU/32 GiB、不请求GPU，
  日志写在`RUN_OUTPUT`旁，并把atomic `preflight.json`作为正式训练提交的硬门禁。
  首次误用normal分区的提交被Slurm以`QOSMinGRES`在创建job前拒绝；live `sinfo`确认
  纯CPU分区名为`cpu`，已修正静态合同，未通过申请H800绕过门禁。
  cpu分区的16-CPU请求随后又在创建job前被`QOSMaxCpuPerNode`拒绝；该reader为单进程，
  因此按集群上限修正为8 CPU，不改变全量校验内容。
- 最终训练代码固定为`3a7f81ba`。CPU-only preflight job`500843`在`intel-01`
  运行7分56秒并`COMPLETED 0:0`；ID65 `preflight.json status=passed`：train/val
  49,638/4,989个H1/T4 windows全量加载，DINO missing 0/0，ID53 cache明确只读复用，
  输入hash、BF16 materialization、W&B ID/name唯一性、commit/clean/配额均通过。
- WS24生产sampler实测每epoch2,069 microbatches、518 optimizer steps，两epoch1,036步；
  global SIGReg每个microbatch有6--24个有效states，仅18个padding slots/epoch。
  trainable为Qwen vision、StateProjector、WM predictor、ValueHead；Qwen LLM、DINO
  teacher/cache、latent query冻结；fresh optimizer、20分钟latest checkpoint。
- 正式preempt训练已提交为job`500845`：3节点×8 H800、world size24、B1/GA4、
  192 CPU、2400 GiB、8小时。提交时`dgx-55/56`完整空闲，`dgx-01`被其他用户1-GPU
  job占用，因此`500845`为`PENDING(Priority)`，Slurm候选节点为`dgx-[01,55-56]`；
  尚无allocation/W&B/model load/optimizer step，必须继续监控到24-rank和finite step。
- `500844`释放`dgx-01`后，其他用户array `500847`立即占用该节点7卡并占`dgx-55`
  1卡，job`500855`再占`dgx-55` 1卡。逐节点`AllocTRES`复核表明当前preempt全部
  可用GPU仅16张：`dgx-01` 1、`dgx-55` 7、`dgx-56` 8，其他可响应节点均8/8已分配，
  `dgx-23`为DOWN+NOT_RESPONDING。因此无论拓扑是否放宽都无法立即组成WS24。
  `500845`保持`PENDING(Resources)`，Slurm当前预测最早`2026-08-04 23:43:11`；
  禁止把排队表述为训练已启动，也不得静默降为WS16。
- 人类随后明确改用normal分区凑24卡。live normal共有33张空闲GPU，但按节点分布只有
  5台具备至少4张；为保持torchrun同构local world size，正式拓扑改为6节点×4 H800、
  world24、每节点32 CPU/200 GiB，B1/GA4和1036 steps不变。代码commit`75f0adc4`。
- 新ID66使用空输出和W&B run id`1xjm320d`。CPU preflight job`500864`与normal训练
  job`500865`用`afterok`一次性提交，随后旧preempt `500845`取消（Elapsed=0）。
  `500864`在`intel-01`以`COMPLETED 0:0`运行6分11秒；全量门禁passed。
- `500865`已解除依赖并为`PENDING(Priority)`，请求6节点/24 GPU/192 CPU/1200 GiB，
  Slurm候选`dgx-[09,14,24,26,30,40]`，当前预测`2026-08-02 07:21:13`。尚无allocation、
  W&B或optimizer step。preflight JSON的展示字段`local_ranks`沿用旧常量8，但硬断言与
  launcher均为6×4；该展示bug已在后续commit`d502b88d`修正，已排队job仍固定于
  `75f0adc4`并由运行时GPU_COUNT=4/local ranks0--3门禁。
- 已实现非对称物理拓扑支持，commit`32ccb011`：Slurm heterogeneous allocation可将
  8卡节点拆成两个各见4张卡的torchrun agent，跨het-group单一`srun`使用唯一全局
  `SLURM_PROCID`，保持6个逻辑agent×4卡/world24。模型加载前强制核验6个agent、
  4个物理节点、同机GPU UUID不重叠和全局24个唯一UUID；远端fixture回归`7 passed`。
- live test-only比较：`2×8+2×4`与`1×8+4×4`的1h--8h请求均预测
  `2026-08-04 14:25--14:26`，晚于现有6×4 job`500865`的`2026-08-02 07:21`。
  原因是normal两台完整8卡节点均为`IDLE+PLANNED`，其余未预留节点合计仅16张可调度GPU。
  因此保留更早的`500865`，未提交更慢的异构替代作业。
- 人类改为normal物理`4+4+2+2`后，旧WS24 job`500865`已取消，`Elapsed=0`、无
  allocation/训练产物。commit`03413ed8`新增6个2-GPU agent/world12 launcher、物理
  4+4+2+2与全局12个GPU UUID门禁；远端目标commit回归`11 passed`。
- ID67 CPU preflight `500926`因提交命令手工抄错完整commit hash，在最前置commit门禁
  1秒失败（`FAILED 1:0`），未进入数据、模型、W&B或GPU。ID67不可复用；后续必须从
  `git rev-parse HEAD`取得真实`03413ed8d8260afd973aa44316f67813b1ddb576`并用新identity。
- ID68 full preflight `500929`已`COMPLETED 0:0`（8:07），world12物理4+4+2+2、逻辑
  6×2、49,638/4,989 windows、DINO coverage、W&B freshness均通过，总计2,070 steps。
  首次正式job`500930`因无allocation且预计较晚而取消；替代`500936`立即获得12卡，随后
  因batch将合法0-byte `cache_done.flag`误用`test -s`而1秒失败，未进入controller/model/
  W&B/optimizer。已登记E0078并将sentinel门禁窄修为`test -f`。
- normal出现8+4+4后，人类要求立即占住；heterogeneous hold`500941`已在`dgx-24`
  取得8 H800、`dgx-26/40`各取得4 H800，总计16卡并为`RUNNING`。正式映射采用物理
  8+4+4、逻辑4个4-GPU agent/world16，B1/GA4 effective global batch恢复为64。
- ID69 world16 8+4+4 full preflight`500944`已`COMPLETED 0:0`并确认1,552 steps。
  正式`500945`提交后，hold`500941`释放，因`dgx-24`新reservation只能预计07:21启动。
  替代normal 4+4+4+2+1+1 hold`500950`已取得16 H800；改用16个1-GPU agent保持
  world16/B1/GA4/effective batch64，并直接在该1小时allocation内启动。
- ID70 full preflight`500955`已通过；首次allocation probe未进入模型/W&B，发现多component
  `srun`的`SLURM_PROCID`按component重置，且裸`nvidia-smi`看到物理节点全部allocation，
  不能核验per-task绑定。修正为het-group offsets 0/12/14和按`CUDA_VISIBLE_DEVICES`
  查询唯一GPU；失败ID70不复用。
- 实机trace随后确认`SLURM_PROCID`跨heterogeneous components全局连续，offset结论失效。
  ID71只完成full preflight和16-GPU probe，旧hold到时前未进入模型/W&B。normal 8+4+4
  hold`500977`已取得；ID72 full preflight通过且W&B已建立，但4个4-GPU agent在8卡节点
  拆成两个`torchrun`，DDP参数校验报NCCL `invalid device ordinal`，未到optimizer step。
  启动合同改为16个1-GPU agent，按物理节点显式`map_gpu`后再启动新ID。
- ID73 full preflight和16×1-GPU allocation probe均通过，但16个独立`torchrun` agents
  在同一物理主机仍触发NCCL P2P `invalid device ordinal`，未到optimizer step。后续必须
  每物理节点只启动一个torchrun agent（8/4/4 local world），先跑最小all-reduce验证。
- 8/4/4 variable-local-world最小NCCL probe已16-rank all-reduce通过；正式commit
  `1f6ea55f`远端18 tests通过。ID74 full preflight`500985`通过，W&B `d52u5anf`，已健康
  训练到至少optimizer step23；total/WM/DINO/Value/LM loss均有限，16卡利用率100%。
- ID74已继续推进到至少step93；job`500977`的1小时hold将在04:39:42+08到期，后继
  exact 8+4+4 hold`500990`以`afterany`等待。commit`13fd4320`把同一controller改为
  显式可恢复：resume时必须给出绝对checkpoint路径，并在启动前验证Qwen/StateProjector、
  WM predictor、ValueHead、training state和16份rank history cache完整；当前仅完成
  shell syntax、diff-check和3项静态launcher检查，仍需首个latest落盘后做远端完整验证。
- 第一段job`500977`因1小时hold到时以signal15暂停，最后logged step141；无OOM、
  traceback、NCCL或non-finite。`latest`为完整step117/epoch1/micro468，含optimizer、
  Qwen/StateProjector、vision EMA、WM、ValueHead和16份rank history cache；保存不变量
  明确为world16/B1/GA4/H1/T4/decision-state value-v3。输出README已记录实际srun、输入、
  split、指标、暂停原因和恢复方法。
- 后继hold`500990`于04:40:33+08在相同`dgx-24:8 + dgx-26/40:4+4`启动；commit
  `2c490a3c`远端resume/launcher回归`13 passed`，新allocation验证3个物理节点、3个
  agent和16个唯一H800。ID74已从step117恢复同一W&B `d52u5anf`，日志确认跳过468
  microbatches、恢复optimizer，并产生至少step120的finite真实更新；step118--141为
  checkpoint之后的预期重放，W&B在重新超过旧step140前拒绝重复step，CSV保留两段记录。
- 恢复轨迹已越过旧水位到finite step142；W&B API独立确认同一run `d52u5anf`为
  `running`且summary global step142。原后继空hold`500997`在未分配资源、Elapsed=0时
  取消；替换为依赖`afterany:500990`的batch-owned正式resume job`500999`，同一精确
  8+4+4节点、1小时上限、显式step117 checkpoint门禁，避免下一段依赖login会话启动。
- 已为SFT2完成后的真实RL单步门禁新增`planner_greedy_h1_smoke.yaml`：planner horizon1、
  predictor history1、DINO0.5、StateProjector冻结、direct PPO关闭，ValueHead/WM predictor
  与完整Qwen language body通过full-prefix重算训练；4 GPU为2个同步rank×2 GPU，vLLM
  rollout TP4，4条base_train episode各20步。远端严格RL schema与全部字段assert通过；
  尚未运行RL GPU、rollout或optimizer，必须等ID74完整final门禁。
- 为覆盖当前约10.4秒/step的剩余SFT2时间，已建立batch-owned顺序依赖链：运行中
  `500990`后依次为`500999 -> 501002 -> 501005 -> 501007`，每段只在前段结束后申请
  同一normal 8+4+4、1小时，并在启动前验证当时最新完整checkpoint。四个后继当前均为
  `PENDING(Dependency)`、未占资源；若前段已生成done flag，resume门禁会拒绝重复训练。
- 本段首次周期checkpoint已在step229/epoch1/micro916完整覆盖`latest`；重新加载确认
  optimizer存在，16份rank history cache非空，不变量为world16/B1/GA4/H1/T4、
  DINO-grid和`decision_state_executed_action_mc_v3`。保存期间`training_state.pt`会短暂
  原位变成0字节，因此只在训练继续到step232后把step229认定为新的durable恢复点。
- batch-owned `500999`与`501002`各运行至一小时时限，有限loss日志推进到epoch2 step874；
  W&B `d52u5anf`当前为`crashed`且summary global step873。目标仍为step1552，没有
  `final`/`epoch_002`/done/completion validation，因此RL门禁未打开。
- 当前durable `latest`加载为step785、epoch2、micro36、epoch未完成，optimizer与16份
  history cache完整，训练不变量仍是world16/B1/GA4/H1/T4/DINO-grid/value-v3；后续日志
  786--874需重放。`501005`和`501007`在获得8+4+4后均于2秒exit1，发生在controller
  重定向之前，probe/model/W&B/optimizer均未启动且零字节日志无法证明具体失败test。
- 当前相同门禁逐项重查通过。launcher已前置controller日志，并用`STARTUP_GATE`记录失败
  位置；不改变模型、目标、数据、优化器或resume语义。14:00+08实时normal仅约11卡可用、
  preempt无空闲，暂不能安全组成world16，下一段将保持同一identity从step785排队恢复。
- 修复以`b184a65b`推送并固定到远端，bash syntax和launcher定向回归`4 passed`。normal
  `sbatch --test-only`接受不固定节点的8+4+4合同；preempt估计更迟。正式batch-owned链为
  `502449 -> 502452 -> 502454`，各1小时、同一output/W&B/latest resume；首段当前
  `PENDING(Resources)`且两component估计2026-08-03 07:50/05:30+08，后两段只等dependency，
  尚无GPU/训练启动。随后superpod跳板再次立即断开，队列本身不依赖SSH；恢复连接后须实时
  重查并监控到probe、DDP和finite optimizer step，RL仍等待ID74 final门禁。

## 2026-08-02：ID112取消，ID113定向dgx-46等待调度

- 人类明确批准打开原先“等待ID74 final”的门禁，改用已完整验证的`epoch_001`：global
  step776、`epoch_complete=true`、H1/T4、world16、ValueHead objective
  `decision_state_executed_action_mc_v3`。完整HF权重、StateProjector、WM predictor、
  ValueHead、optimizer和16份rank history cache均存在；val WM MSE为
  `0.4816782568213839`。
- ID112合同为planner greedy H1/history1、DINO loss权重0.5、StateProjector/vision冻结、
  direct PPO/reference KL关闭；训练完整Qwen language body、WM predictor与ValueHead。
  数据为4条`base_train` episode、各最多20步；首个门禁要求2个同步rank各2 GPU、vLLM TP4、
  episode末恰好一次finite optimizer step和完整checkpoint。
- ID112/job`502480`先以2节点×2 H800提交。人类随后指定`dgx-46`，该job在未分配节点、
  `Elapsed=00:00:00`时取消；没有controller log、output、W&B run、Ray/vLLM、rollout、
  DDP、optimizer step或checkpoint，不可resume。复核还发现ID112参数中的checkpoint路径
  漏写`train_ws16/`，若实际启动会在最前置模型门禁失败；终态已写入邻接launch contract
  和`112_*.progress.md`。
- commit`75b21b9ea2bc207f85cea4bec94b9b3ca54333a7`新增
  `planner_greedy_h1_smoke_1x4.yaml`：单物理节点、world2、每rank 2 GPU、总4 GPU、vLLM
  TP4，其余目标/数据/冻结边界不变。远端配置硬断言和31项定向回归通过；用正确
  `train_ws16/epoch_001`路径的CPU preflight通过。项目凭据下W&B entity为
  `art2nd-hong-kong-university-of-science-and-technology`，ID113精确run name查询0命中。
- ID113 output/name为
  `113_smoke_ep1_greedyh1_k16_dino05_qwenwmvalue_ep4x20_1n2r2g_vllmtp4_dgx46`，launch
  contract已落盘。normal job`502499`定向`dgx-46`请求4 H800、64 CPU、160 GiB、2小时；
  提交后仍为`PENDING(Priority)`，最新非约束StartTime为`2026-08-02T23:05:00Z`。
  `dgx-46`实时仅被其他用户占2/8卡，但Slurm尚未backfill本job；当前仍无GPU、Ray/vLLM、
  rollout、真实DDP或optimizer证据。
- 共享workspace并发产生的另一个ID113/job`502498`已主动取消以消除重复排队：sacct为
  `CANCELLED/Elapsed=00:00:00/NodeList=None assigned/ExitCode=0:0`，没有output、W&B、
  Ray/vLLM、rollout、DDP、optimizer或checkpoint。其邻接progress已记录终态；唯一保留
  并监控的canonical ID113是job`502499`。
- 当前normal priority阻塞与旧SFT2链相关：头任务`502449`优先级1088，高于RL的996，两个
  heterogeneous component分别给出约15:03/17:03估计；后继`502452/502454`仍为dependency。
  尝试用标准`hold/holdu`及带`Account=peilab`的Priority0 update可逆暂停头任务，均被站点
  权限插件拒绝且任务状态未改变。固定dgx-46的1h45/1h/30m test-only均进一步推迟到21:13Z，
  说明缩短时限也不能立即backfill。未取消SFT2链、未提交低时限近似实验；canonical
  `502499`保持`PENDING(Priority)`并继续等待健康启动门禁。
- `502499`最终于`2026-08-02T15:06:36Z`由backfill调度到`dgx-46`，实际AllocTRES为4 H800/
  64 CPU/160 GiB，job为RUNNING。四张唯一GPU初始空闲；Ray head
  `10.23.1.117:6741`和Nimloth import probe通过，navigation server 16秒ready，真实
  `base_train` seed1 prewarm耗时10.439秒、图像255×255、dynamic range223。
- vLLM 0.11.0以TP4/NCCL 2.27.3加载corrected epoch1两片权重，四rank连接完成；每rank
  KV cache为3,366,496 tokens，四卡约70--71 GiB，随后打印Supported task generate并开始
  `rl_ep=0`真实trajectory。当前已达到Ray/env/vLLM/rollout健康启动；尚未完成4条episode、
  两rank DDP update、finite global step1或checkpoint，后续结果不得提前宣称。
