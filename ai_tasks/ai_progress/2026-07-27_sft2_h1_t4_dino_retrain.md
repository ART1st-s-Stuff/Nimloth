# SFT2 H=1 / T=4 DINO-grid 重训

## 目标

- 使用单个当前 latent state（`H=1`）作为 WM 输入。
- 严格采用原始 rollout 中连续四个 action，自回归预测四个后续 latent state（`T=4`）。
- 对四个预测状态分别计算 WM latent、DINO-grid 和真实 Monte Carlo value loss。
- ValueHead 只监督各步实际执行的 action，不使用未执行 action 的 rank loss；`gamma=1`。
- 其余训练、冻结、数据和 DINO 监督配置保持当前 DINO-grid SFT2 合同。

## 当前计划

1. 核对 SFT2 sampler、batch/cache、WM rollout、DINO target 和 checkpoint 接口。
2. 增加独立的 `prediction_horizon=4`，实现固定连续四步训练窗口。
3. 补充 action/target 对齐、递归预测、四步梯度和配置回归测试。
4. 完成本地测试后执行远端单卡与正式拓扑 smoke。
5. smoke 通过后启动正式训练并监控首个 optimizer step/checkpoint。

## 已确认合同

- `history_size` 就是输入历史长度 `H`；本次固定为 1。
- `prediction_horizon` 是训练展开长度 `T`；本次固定为 4。
- 四步 action 必须来自同一条原始 rollout 的连续 transition。
- 四步 WM/DINO/value target 都与各自 transition 对齐；禁止 padding 或伪造未来 action/state。
- CE 与 SIGReg 保持当前 SFT2 配置语义。

## 当前状态

- 当前分支：`dev`，起始提交 `837d014e`。
- 工作区开始时仅有用户已有的 `external/le-wm` 和 `readme.md` 未跟踪内容。
- 已实现独立 `prediction_horizon`：`H=1,T=4` 使用同一 rollout 内的固定长度滑窗，
  四个 action、MC return、下一 observation 和 DINO target 均逐位置对齐；不跨 record、
  不跨缺失 step，也不补伪造 transition。
- WM 从真实 `s_t` 开始递归四步，后续输入只使用前一步预测；真实未来 state 只通过
  `no_grad` target encoder 进入 loss。ValueHead 在四个预测 state 上只 gather 原始
  rollout 实际 action 对应的 slot，rank loss 保持删除。
- 新增正式配置 `configs/training/sft2/dino_grid_k16_h1_t4.yaml`：`gamma=1`、两 epoch、
  DINO 权重和其余训练/冻结设置沿用当前 DINO-grid 配置。
- 本地回归：SFT2/WM/Agent 为 `127 passed, 1 skipped`；受共享 WM 接口影响的 RL/grid
  回归另为 `29 passed`。`compileall`、两个 shell 脚本 `bash -n`、`git diff --check`
  均通过。两条 Gloo 测试必须在允许 loopback socket 的环境执行；沙箱内的
  `Operation not permitted` 不属于代码失败。
- 重连后资源查询成功：normal 分区空闲 20/48 GPU，preempt 分区空闲 11/32 GPU；
  当前用户无运行中 Slurm 任务，尚未占用 GPU 或启动训练。
- 代码已提交并推送为 `c1e983ac` 与 `e664c49f`；clean server worktree 固定在
  `e664c49f9cb291de94697ef2c5d0a7ffd04b0280`。远端 exact-environment 定向回归为
  `55 passed in 68.55s`，并通过 compileall、shell syntax 和 diff-check。
- ID50 的 8-train/8-val fresh CPU cache smoke job `494514` 在读取第一条记录时正确失败：
  ID49 已审计 terminal-CoT JSONL 仍是没有 `record_format` 的 legacy trajectory，而当前
  reader 只接受 `nimloth_trajectory_v1`。任务最终状态 `FAILED 1:0`、耗时 `00:01:12`，
  没有 GPU/W&B/optimizer/checkpoint/cache manifest/done marker。
- 先前关于续建 ID49 partial cache（train image shards 32/489）的结论失效。迁移后的
  JSONL fingerprint 必然改变，因此旧 partial cache 不能作为本次训练的 resume source。
  strict reader 与 fingerprint gate 不会放宽。
- 修复提交 `03dd18fc` 已推送；新 clean server worktree 的迁移/H1-T4 定向回归为
  `64 passed in 13.12s`，compileall、shell syntax、diff-check 和 tracked clean gate 通过。
- ID52 CPU migration Slurm `494521` 已 `COMPLETED 0:0`（37秒，`intel-01`，peak RSS
  `512304K`）。train/val逐条验证分别覆盖3211/355 records与59269/6054 transitions；
  IDs唯一，原始rollout action和terminal CoT全部不变，manifest与source/output hash一致。
  migrated JSONL SHA256为train `d43ada06d66c0b5cafa50e9da8ecc354445ca3b9686d1639b18050a981247b97`、
  val `4c092fb4069fb71ad92bca73566d5f20f572569a093bf4712467ca137615212e`。
- ID50 migrated-data fresh cache Slurm `494524` 已 `COMPLETED 0:0`（32秒，`intel-02`）。
  train/val为138/99 transitions、146/107 unique images，fingerprint分别为
  `deacbccea6eec498`/`66186fbb54ef56bf`；格式、BF16、gamma1、terminal-CoT expansion、
  manifests与done flag完整。生产reader全量加载train 114/114、val 75/75个H=1/T=4
  窗口，确认每窗同rollout连续4步、recorded action对齐及1 current+4 next encoding。
- ID50单卡hold `494528`在`dgx-09`启动，W&B `9hcisto1`使用正确run name。17个真实窗口
  已完成step1--15且total/WM/DINO/value-MC/CE全部finite，`step_000015`为完整可训练
  checkpoint；但train step `494528.1`随后`FAILED 1:0`。最后一个T4窗口的terminal-CoT
  next encoding正确没有labels，与前三个含labels next row一起进入include-labels-false
  collate时触发`KeyError: 'labels'`。hold已取消、GPU已释放；没有epoch val/final/done，
  单卡smoke未通过。修复和terminal回归通过后可从step15恢复同一ID50/W&B run。
- 修复提交`4f66b8d3`在无label target路径collate前逐row删除labels，并令CE路径强制所有row
  有labels。远端相关回归`58 passed in 12.52s`；真实ID50最后窗口steps16--19、recorded
  actions`[1,3,1,3]`的label presence为`[true,true,true,false]`，修复后的生产assembler
  已成功输出4-row label-free next batch。代码gate通过，训练smoke仍待从step15恢复完成。
- 首次step15 resume使用hold`494533`/`dgx-21`；W&B `9hcisto1`正确resume，但在任何新
  microbatch前，SFT2 checkpoint loader因调用未导入的`ValueHead`触发`NameError`。
  step `494533.1`在1分02秒`FAILED 1:0`，hold已取消，step15未改变。需增加完整WM-owned
  modules save-load回归与真实step15 loader gate后再恢复。
- resume loader修复提交`63082ac3`已增加`ValueHead`导入和projector/predictor/ValueHead
  完整save-load回归；远端相关CPU回归`76 passed in 14.11s`。对实际ID50
  `step_000015`执行生产`load_world_model_checkpoint()`成功：H=1、K=16、D=1024，三个模块
  权重均finite，checkpoint仍为epoch1/micro-step15未完成。第二次resume的代码与真实
  checkpoint门禁均已通过。
- ID50单卡resume以精确代码`829d9dca`在hold`494535`/`dgx-09`完成；训练step
  `494535.1`为`COMPLETED 0:0`（2分47秒），H800已释放。数据位置严格skip 15/17，重放
  checkpoint之后的step16并完成terminal step17与1个validation batch。W&B同一run
  `9hcisto1`已`finished`/global step17；重复step16的W&B log被拒绝但step17/val完整。
  train step17 WM/DINO/value MSE为`0.267765/0.918687/0.033514`；val为
  `0.578651/1.221469/1.563365`，均finite。`epoch_001/best/final`与`SFT2_DONE`齐全。
  post-validator `494540`为`COMPLETED 0:0`，fresh-process完整加载Qwen、optimizer、EMA、
  H1 WM和ValueHead，并验证4-step rollout/value输出finite。validator `494539`先因错误
  断言completed-epoch micro cursor=17失败；契约实际为0，修正后通过，训练主体未受影响。
  下一门禁是ID51 world-size-8 smoke，以覆盖单卡B1会跳过的global SIGReg/DDP路径。
- ID51分布式smoke已通过提交前preflight：精确代码`936366fe`，4节点×2 H800、world8、
  B1/GA8，使用迁移数据前8条与对应prebuilt cache，从SFT1重新初始化H1 WM/ValueHead/
  optimizer，W&B ID51 live未占用。唯一hold`494549`因当前仅两节点满足每节点2空闲GPU而
  `PENDING(Priority)`，预计本地06:23:16；尚无train输出/W&B run。远端watcher PID
  `2552482`每30秒检查真实状态，allocation到达后运行完整4×2 cgroup/rank/port gate与训练，
  最终自动释放hold；不得提交第二个hold。ID51通过前不得开始ID52正式训练。
- ID51实际获得`dgx-[09,21,27,30]`的4×2 H800 allocation；cgroup/rank/rendezvous、
  Qwen加载、NCCL和sampler gate通过，但step`494549.1`在首次真实forward时`FAILED 1:0`
  （3分31秒）。根因是world8把trainable `wm_predictor`单独包装为DDP，随后
  `simulate_action_sequences()`却在DDP wrapper上调用自定义`rollout_from_history()`；单卡
  不包装因而ID50未暴露。不能用`.module`绕过同步边界。无optimizer step、SIGReg、val、
  checkpoint或done；W&B `6btnjnaw`为`crashed`/空summary，hold已释放且不可resume。
  需修复parameter-owning DDP forward、增加多进程rollout/backward回归，再用新identity
  通过world8 smoke；此前禁止正式SFT2训练及full cache启动。
- DDP rollout修复提交`55a80ad1`令`WorldModel`在每个未来步通过包裹predictor的标准
  `forward()`执行相同的自回归上下文截断；测试fixture action语义校正提交`58f30e98`。
  新增Gloo双进程`static_graph=True`回归，真实grid predictor连续两轮H1/T4 forward、
  backward、optimizer step均完成，且两rank梯度逐元素相同、非零。superpod固定解释器下
  完整SFT2、grid/latent WM、planner回归`114 passed, 1 skipped in 70.40s`，唯一skip是需GPU
  的显式NCCL测试。W&B live max为51，故world8重试使用ID52和全新空输出；原ID52目录中的
  已验证迁移数据保持原路径不变，正式cache/W&B/训练identity顺延为ID53。正式训练继续以
  新smoke通过为前置。
- ID52 world-size-8 smoke 已通过：精确代码`30e5e4f0`，hold `495566`在
  `dgx-[14,18,29,54]`提供4×2 H800；核心step `495566.1`为`COMPLETED 0:0`（5分57秒），
  完成3个finite optimizer steps、validation、完整`epoch_001/best/final` checkpoint和
  `SFT2_DONE`，watcher随后释放8卡。末步train WM/DINO/value/SIGReg为
  `1.175624/1.652063/0.356775/2.562645`，validation WM/DINO/value为
  `1.270767/1.604670/0.271709`，全部为H=1/T=4。global SIGReg batch平均
  `7.75/5.875/5.0`精确反映114个有效window和22个sampler padding，所有调用全局有效
  batch至少5且无skip。W&B `wut6xqhg`已`finished`。独立CPU validator `495571`为
  `COMPLETED 0:0`，fresh-load完整Qwen/optimizer/EMA/8 rank history caches/H1 WM/ValueHead，
  并执行finite的4-state rollout与value计算。ID52分布式门禁解除，正式阶段使用ID53。
- world-size-16提速容量只读核验（commit `9524f0a7`）：核心DDP/data/sampler路径不写死
  world8。全量migrated数据探针在world16下每epoch产生3103 distributed micro-batches，
  完整覆盖49,638 train windows，仅末batch有10个零loss padding；val覆盖4,989/4,989
  windows、无padding。当前ID53 `run_node.sh`/`launch_from_login.sh`/`hold.slurm`仍写死
  4 nodes×2 ranks，不能原样变成world16；world16尚未做GPU/NCCL/end-to-end smoke。
- 优化语义：world8+B1+GA8的effective global batch=64、每epoch776 optimizer steps；world16
  若保持GA8则分别变为128和388。world16+GA4可恢复global batch=64与776 steps/epoch，
  但global SIGReg每microbatch的统计样本仍从约8增加到约16。`world_size`/`grad_accum`/
  `train_micro_batches`是checkpoint invariants，且history cache按rank持久化，因此world8断点
  不允许resume为world16。Preprocessing cache与world size无关，可作为fresh world16 run的输入。
- ID53 CPU cache `495702`已`COMPLETED 0:0`（2:05:09）；train/val fingerprints分别为
  `ac7835348d6eade1`/`d857dc4ef51a70be`，共49,638/4,989 H1/T4 windows。独立
  validator `495754` `COMPLETED 0:0`（5:12），全量生产reader加载、recorded action、
  gamma1 value target与抽样BF16 materialization通过；正式world8 cache gate已打开。
- 资源复盘：job `495702`在224-core `intel-01`上仅申请8 CPU cores，导致全量cache
  耗时超过2小时。人类要求今后全量cache至少64 CPU cores，且必须核验
  `ReqTRES`/`AllocTRES`；不得因为partition/QoS限制而静默降到8 cores。

## 待完成

- 人类要求立即启动已通过分布式smoke的world8+B1+GA8正式SFT2。提交前复核空输出、
  W&B ID53、动态4×2 H800 allocation与刚通过的全量cache；启动后监控首个finite
  optimizer step和首个可恢复checkpoint。
