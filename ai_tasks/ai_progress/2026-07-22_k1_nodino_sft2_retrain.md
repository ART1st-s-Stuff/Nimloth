# k=1、无 DINO SFT2 重训进度

## 任务目标

使用最近 SFT2/RL 统一后的 `history_size=4` 契约重训 SFT2，为后续 k=1 RL、
ValueHead 和 PPO 验证提供兼容 checkpoint；不使用 DINO teacher、feature 或 loss。

## 已确认实验定义

- 代码：superpod clean worktree `/project/peilab/atst/nimloth/.worktree/dev`，
  commit `fb4580b1b6639d6f8f79ade7e3935e9a5571896a`。
- 配置：`configs/training/sft2/latent_wm_value_k1_control.yaml`。
- 数据：`converted_strict_k8_b6c811c/{train_all,val_all}.jsonl`；名称中的 k8 是
  转换实验组标识，记录本身不绑定 latent token 数。训练 3217 条 train split，
  验证 355 条 held-out val split。
- 初始化：k=1 SFT1 run 1 的 `epoch_005/hf_merged`，不从旧 SFT2 resume。
- 模型：`latent_token_count=1`、inject/query adapter、LeWM `history_size=4`、
  `emb_dim=1024`。
- 训练：Qwen language 冻结；vision full + EMA、query adapter、StateProjector、
  WM predictor 和 ValueHead 可训练。目标为 CE、multi-step WM、ValueHead 和
  SequenceSIGReg；没有 DINO 输入或目标。
- 资源：cache 使用 CPU `intel-01`；正式训练使用 preempt 单节点 8 GPU，
  batch 2、grad accumulation 4、10 epochs、20 分钟周期 checkpoint，支持抢占恢复。
- best 指标：held-out `val_wm_mse`；同时监控 WM、value、SIGReg、CE、finite、显存
  和 checkpoint 完整性。
- W&B：project `nimloth-sft2`，ID 34，run
  `34_k1nodino_h4_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_ws8_px100352_img12_bestwm`，
  internal ID `4e78gcir`。
- 输出：`outputs/experiments/vagen_legacy_wm_k8_full/2026-07-10/full_2e66e97/`
  `control_k1/sft2/34_k1nodino_h4_all3217_qadapter_vfull_wmtrain_ep10_b2_ga4_ws8_px100352_img12_bestwm`。

## 当前状态

- 首次 cache job `484432` 在运行 1 分 50 秒时发现 wrapper 隐式回退到 SFT1
  `best` 并准备重复 merge；为保持初始化来源显式，已取消。依赖训练 `484434`
  尚未启动并同时取消。日志和半成品目录保留，未覆盖或删除。
- retry cache job `484435` 使用独占 `preprocess_cache_retry1`，已明确解析到
  `epoch_005/hf_merged`，当前在 CPU partition 健康运行。
- 正式训练 job `484439`：`afterok:484435`，preempt、1 node、8 GPU；当前因依赖
  pending，尚未占用 GPU。

## 后续验证

1. cache 完成后核对 train/val manifest、count、k=1、inject、successor-state cache
   契约、done flag 和无异常日志。
2. 训练启动后核对八个 rank、W&B resume 到 `4e78gcir`、实际 trainable/frozen
   参数、首个 finite optimizer step、显存和错误日志。
3. 持续监控 validation、best/latest checkpoint 和抢占恢复；任务结束时执行
   `on-experiment-end`。

## 2026-07-23：cache 完成，首次 8-GPU 训练 OOM

- cache job `484435` 在 `01:58:24` 后 `COMPLETED 0:0`。train manifest 为
  `dedup_sharded_v2`、59,389 transitions、464 image shards、232 transition
  shards、约 71.2 GB；val 为 6,054 transitions、48+24 shards、约 7.25 GB。
  两者均核实 k=1、inject、BF16、`wm_expand_v1`，cache 可由 retry 只读复用。
- train job `484439` 在 dgx-39 运行 `00:04:11` 后 `FAILED 1:0`。八个 rank 完成
  模型、cache、DDP 和 W&B 初始化，但第一个 forward 在 Qwen causal-LM CE 中
  OOM：rank0 已用约 74.64 GiB、需额外 5.03 GiB；rank2 已用约 76.18 GiB、
  需额外 5.49 GiB。
- `train_step_log.csv` 只有表头，global step 0；没有 training state、StateProjector、
  WM predictor 或 ValueHead checkpoint。因此本次没有训练结果，不能从该目录
  resume，也不能作为 RL 初始化。
- 初步建议：保留并只读复用完成的 v2 cache，先用 `batch_size=1` 做短 GPU smoke；
  若首步 finite，再以 `grad_accum=8` 保持 8-GPU effective batch 64 提交正式 retry。
  该建议随后执行，但未通过，见下一节。

## 2026-07-23：batch1 smoke 仍在单个 H=4 window OOM

- W&B ID35 `35_smoke_k1nodino_h4_b1_ga1_ws8_fullcache_stepgate`，internal ID
  `0ajvq3sy`；job `484846` 在 dgx-42 使用 preempt 8 GPU。
- W&B 实际配置文件核实 `batch_size=1`、`batch_size_per_rank=1`、
  `grad_accum=1`，因此参数覆盖确实生效。此前根据相同显存数值怀疑
  `EXTRA_TRAIN_ARGS` 丢失是错误判断。
- job 在 `00:04:45` 后 `FAILED 1:0`，仍于首个 Qwen causal-LM CE forward OOM；
  rank0/rank2 的占用和额外申请量与 batch2 attempt 基本相同。原因是
  `trajectory_image_budget` 下 batch2 attempt 的首个 microbatch 也已被图片预算
  限制为一个 window；而单个 H=4 window 含四个连续 prefix 状态，当前 sequence
  forward 一次处理全部时间位置，单 window 已无法容纳全词表 FP32 CE logits。
- CSV 仍只有表头、global step 0、无 checkpoint。仅增加 grad accumulation 无法
  解决；下一步需要在人类确认后选择保持数学语义的时间位置 chunking/低内存 CE
  实现，或改变 max_pixels/max_length/lambda_ce 等实验语义并重建对应 cache。

## 2026-07-23：获批实现 H 行分段 Qwen forward

- 人类确认采用保持实验语义的低显存实现。k=1 control 配置新增
  `backbone_rows_per_forward: 1`：单个 H=4 window 的四个 prefix 分别执行 Qwen，
  latent 按原顺序重组为 `(B,H,...)` 后再统一计算 WM、ValueHead 和 SIGReg。
- CE 没有关闭或降权；每段 HF mean loss 按 shifted non-ignore label 数加权，恢复
  原始全 batch CE reduction。图片 grid 与 pixel rows 根据每个文本行的 image token
  数量同步切分，避免文字行和视觉输入错配。
- 已增加视觉切分、row 顺序、CE 数值及 CE 梯度测试；`compileall` 与
  `git diff --check` 通过。本地 pytest 环境缺少 `_pytest`，数值测试仍需在远端
  环境运行后才能提交 GPU smoke。

## 2026-07-23：chunk=1 GPU smoke 仍 OOM

- commit `aaf16ba` 已由人类推送并同步到 superpod clean worktree。首次提交
  `484881` 在 dgx-17 获得 8 卡后因漏传 `SKIP_SFT1_DONE=1` 于 1 秒退出；没有加载
  模型、没有 W&B run，也没有 GPU 实验结果。
- 正确 retry job `484885` 使用 preempt dgx-17 8 卡，W&B ID36、internal ID
  `0s8tcq0y`。实际配置核实 `history_size=4`、`backbone_rows_per_forward=1`、
  batch1/GA1、k=1 inject、vision full、无 DINO，并复用完成的 v2 cache。
- job 在 `00:05:10` 后 `FAILED 1:0`。chunking 消除了原先全 batch CE 一次额外申请
  约 5--5.5 GiB 的失败形态，但四个在线 chunk 的 autograd graph 仍同时保留；
  rank1 在 Qwen MLP 仅剩 12 MiB 时申请 52 MiB，rank3 在后续 chunk LM head 申请
  930 MiB，rank0 在之后的 no-grad target forward DDP sink 申请 20 MiB 时 OOM。
- `train_step_log.csv` 仍只有表头、global step 0、没有 checkpoint，不能 resume，
  仍不能开启 RL。下一步应在不改变 loss 的前提下 offload online chunk 保存的
  activation，或重构为可分段 backward 并精确保留下游跨 H 梯度；需要再次实现和
  GPU 验证。
- 已实现下一版 activation offload：chunk forward 通过 saved-tensor hooks 仅把
  autograd 保存的非 leaf CUDA tensor 搬到 CPU；模型参数、latent 输出和 loss 留在
  GPU，backward 时按原图恢复。k=1 control 显式启用该选项；当前只通过 compileall
  和 diff check，尚未获得 GPU 数值/显存验证。

## 2026-07-23：activation-offload smoke 首步成功

- 人类推送 commit `9d29929` 后，superpod 训练 Python 环境中的 standalone 核心
  断言通过：多模态 row/grid/pixel 切分、chunk 输出顺序、全局 CE 数值和 CE 梯度。
- ID37 job `484906`，W&B project `nimloth-sft2`，run
  `37_smoke_k1nodino_h4_chunk1_cpuoffload_b1_ga1_ws8_stepgate`，internal ID
  `1ogp76s3`；preempt dgx-17 单节点 8 卡，commit `9d29929`，B=1、GA=1。
- W&B/日志实际配置核实 H=4、k=1 inject、`backbone_rows_per_forward=1`、
  `offload_backbone_chunk_activations=true`、vision full + EMA、query adapter、WM 和
  ValueHead 可训练、无 DINO、复用完成的 v2 cache。
- 首个 optimizer step finite：total 9.304960、WM MSE 0.275662、value total
  0.191877（reg 0.015217、rank 0.176660）、CE 9.085517；B=1 下 SIGReg 按既有小
  batch guard 跳过。forward 44.80s、backward 54.12s、optimizer 2.30s。
- PyTorch step peak allocated 31.02 GiB、peak reserved 31.08 GiB；实时采样各 rank
  约 23--49 GiB，显著低于 ID36 的 77--79 GiB。达到 step gate 后主动取消并释放
  8 卡；取消前 checkpoint 尚未落盘，因此 ID37 不用于 resume 或 RL 初始化。
- 结论：低显存路径已真实越过 forward、backward 和 optimizer step；可以基于同一
  commit 提交正式 B=2、GA=4、10 epoch 重训，正式训练需验证 B=2 下 SIGReg finite
  和周期 checkpoint 后，才可开启 RL。

## 2026-07-23：正式 ID38 已启动并完成首步

- 正式 job `484910`，W&B run
  `38_k1nodino_h4_chunk1_cpuoffload_all3217_b2_ga4_ws8_bestwm`，internal ID
  `zc0y6j3c`；preempt dgx-17 单节点 8 卡，B=2、GA=4、10 epochs、20 分钟周期
  checkpoint，抢占后由输出目录最近 training state 恢复。
- 首个 GA=4 optimizer step finite：total 7.707960、WM 0.274856、value 0.211024、
  CE 7.469451；四个 microbatch 合计 forward 152.94s、backward 230.34s，step peak
  allocated/reserved 48.81/48.98 GiB。
- 首步 `sigreg_loss` 仍为空，说明图片预算打包出的这四个 rank-local microbatch
  实际窗口数不足 2，触发小 batch guard；需继续观察后续 batch 是否产生 finite
  SIGReg。任务保持 RUNNING，首个周期 checkpoint 尚未到时，RL 尚未开启。

## 2026-07-23：ID38 因不可接受的 ETA 停止并删除产物

- 只读 sampler 统计确认 46,524 个有效 H=4 windows 被图片预算打包成 46,524 个
  microbatches，实际 B 全部为 1；因此 SIGReg 在整个 run 都不会执行。每 rank 每
  epoch 5,816 microbatches、GA4 后 1,454 optimizer steps，10 epochs 共 14,540
  steps。
- 前 8 steps 平均约 339 秒，连续运行 ETA 约 57 天；48 小时 allocation 仅能完成
  约 500 steps。人类明确判定该速度不可接受并要求立即停止。
- job `484910` 于 `00:58:20` 被取消，dgx-17 已恢复 idle、8 GPU 全部释放。随后按
  人类要求永久删除 ID38 输出目录（删除前约 14 GB）；`latest` checkpoint、CSV、
  本地 W&B 文件及日志均已删除，任务不可 resume，也不能用于 RL。
- 共享 v2 preprocess cache、ID37 smoke、SFT1 初始化和 W&B 云端 run `zc0y6j3c`
  未删除。

## 2026-07-23：纠正隐藏的 multi-step loss 重复计权

- 人类明确指出旧实现错误：`Agent.forward_sequence()` 把重叠 window 的 B*H 行
  隐藏成一个 `lm_loss`，但实际对同一 transition 重复计算 CE，WM/value 也对 H
  个位置重复取 loss；这不是标准 SFT 的逐 step 语义。
- sampler 已改为每个拥有真实 next state 的 transition 恰好拥有一次 current-step
  loss。episode 开头使用 T=1/2/3 的真实短 context，之后最多 T=H=4；同一
  microbatch 只打包相同 T，不丢弃开头 step、不伪造 padding。
- algorithm 显式调用 `forward_step_from_history`，只使用最后一项 action、value
  target 和 next target。CE 只属于每个 context 的最后一行；旧 Backbone row 在
  no-grad 下编码，projected history 在进入 WM predictor 前 detach。ValueHead
  只读取当前 `s_t`，其早期历史由累积 Agent prompt 提供。WM/value 各
  计算一次，SIGReg 每 step 只计算一次在线 `(s_t,s_{t+1})`。
- Qwen label 构造已核对为 `last_assistant_span_v1`：即使 current prompt 累积了
  更早 assistant turn，CE label 也只覆盖最后一个当前 action，不会在单行内
  重复监督旧 action。
- 所有 Qwen 路径（history/current、target、SIGReg next）均支持 row=1 顺序 forward；
  sampler 的 image budget 在该模式下按真实单 row 峰值计算，不再把 H 个累积 prefix
  图片数相加。新增实际 B 分布启动日志，并在启用 SIGReg 但所有 microbatch B=1
  时直接拒绝训练。
- 新增 known error `E0039_do_not_hide_repeated_step_losses_inside_history_forward.md`
  及 current-step ownership、CE 选择、旧 history 零梯度、短 context、SIGReg pair
  测试。当前仅 `compileall` 与 `git diff --check` 通过，尚未在远端 PyTorch/pytest
  和 GPU 上验证，禁止据此启动正式训练。

## 2026-07-23：current-step-once 回归与 B2/GA4 OOM smoke 通过

- 语义修复提交 `e31ee89`，变长 trajectory sampler 测试的解码修复提交
  `367e834`，均已由 agent 推送并同步到 superpod。远端定向 PyTorch 回归覆盖
  current-only CE、WM/value/SIGReg ownership、history detach、变长 sampler、
  compact-cache metadata 和 selected-row Qwen CE：`38 passed in 11.26s`。
- ID39 `39_smoke_k1nodino_h4_currentonce_chunk1_cpuoffload_b2_ga4_ws8_oomcheck`
  使用 preempt job `485076`、dgx-04、8 GPU、B=2/GA=4/H=4/row1/CPU activation
  offload、k=1 inject、无 DINO；初始化仍为 k=1 SFT1 epoch5 merged，复用完成的
  compact cache，不从旧 SFT2 resume。W&B internal ID `go89t9yi`。
- 启动审计明确输出 `sft2_step_ownership=current_step_once_v1`；56,172 个 current
  steps 全部组成 28,086 个 B=2 microbatches，没有实际 B=1。首个完整 optimizer
  step finite：total 7.222023、CE 6.902349、WM 0.267141、SIGReg 1.011569、value
  0.191802（reg 0.016271、rank 0.175531），证明 B=2 SIGReg 未被 guard 跳过。
- step timing 为 dataloader 0.566s、forward 271.456s、backward 171.523s、optimizer
  1.476s。PyTorch peak allocated/reserved 23.313/25.566 GiB；实时最高单卡约
  27.9 GiB。CPU offload 很重，节点 RAM 实时采样最高约 569 GiB，但仍剩约 1.4 TiB。
- 达到 step1 stop gate 后主动取消，Slurm 状态 `CANCELLED`、elapsed `00:11:39`；
  无 OOM、CUDA error、NaN、Inf 或 traceback，dgx-04 已恢复 idle、8 GPU 全释放。
  smoke 禁用 checkpoint，因此不可 resume，也不能直接开启 RL。
- `scancel` 杀死异步 W&B client 后，使用原 internal ID `go89t9yi` 只补传已落盘的
  step1 指标并 clean finish；最终 state=`finished`、summary 中
  `smoke_status=goal_reached_then_cancelled`，没有创建重复 run。
- 结论：旧 formal 的 B=2/GA=4/world8 并行参数在修正语义后已通过 OOM gate；但
  首步约 445 秒，若有代表性则正式 10 epochs 仍不可接受。下一步应先由人类决定
  是否提交多步 throughput gate，不能直接把本 smoke 当作正式重训或 RL 产物。

## 2026-07-23：改为在线 detached history cache，删除 OOM 应急路径

- 人类明确要求删除 row1/activation-offload 应急路径，并选择在线 cache：trajectory
  在 rank 内按时间顺序推进，`s_t` 只在当前 step 执行一次在线 Qwen，detached 后
  缓存；未来 H=4 窗口读取缓存 state，不重新执行历史 Qwen。
- current CE、WM、ValueHead 与 SIGReg 仍以 current transition 为唯一统计单位；
  history 只进入 WM predictor，不能把梯度传回旧 optimizer version 的 Backbone/
  StateProjector。target next 与 SIGReg online-next 继续各自按 current transition
  计算一次。
- 生产代码和配置已移除 `backbone_rows_per_forward`、
  `offload_backbone_chunk_activations`、旧 image/row budget及旧 batch modes。新 sampler
  固定完整 trajectory lane 的 rank ownership；DDP 对齐只使用零 loss padding batch，
  不复制真实 transition loss。
- partial checkpoint 将保存每个 rank 的 history cache，并与 microbatch cursor 一起
  恢复。当前实现已通过 compileall/diff check，远端 PyTorch 回归和 8-GPU B2/GA4
  smoke 尚未执行；在这两项通过前仍不能开始正式 SFT2 或 RL。

## 2026-07-23：online-cache 回归通过，但 B2 在长 prefix 仍 OOM

- 实现/测试/观测提交 `0f1412a`、`ad6846e`、`0d030bd` 已由 agent 推送；superpod
  PyTorch 2.8 扩展回归 `110 passed`，最终观测指标定向回归 `16 passed`。
- ID40 `40_smoke_k1nodino_h4_onlinecache_b2_ga4_ws8_oomspeed`，W&B internal ID
  `qf82rxkq`，preempt hold job `485157` 在 dgx-40 使用8 GPU、B2/GA4、H4、k1、
  无DINO并复用ID34 compact cache。全局 sampler 分布为 B2=28,072、B1=28、
  零loss padding=4；56,172个真实current transition完整且不重复。
- step1遍历T1..4，cache entries平均5，finite total/CE/WM/SIGReg/value为
  7.338828/6.962769/0.277349/1.034820/0.244841；耗时约24.6秒，peak allocated
  47.626GiB。step2全为T4，cache entries平均13，finite total 7.625444；耗时约
  16.3秒，但长累计image prefix使peak allocated升至76.952GiB。
- 第三个accumulation周期在SIGReg的online `s_{t+1}` Qwen forward全rank OOM：
  allocated 77.23--77.35GiB、仅剩16--98MiB，失败申请20--102MiB。该问题来自
  current/online-next两份有梯度activation随真实prefix增长，不是历史cache miss、
  历史重算或allocator碎片。
- job已取消，dgx-40恢复idle且8卡释放。ID40无checkpoint，不可resume、不可开启
  RL；B2/GA4正式训练被否决。W&B因进程终止仍显示running且只含step1，step2和
  OOM证据已保存在输出CSV/log/README。下一步若获批，使用B1/GA8保持effective
  batch做短smoke；不能恢复已删除的row/offload应急路径。

## 2026-07-23：SIGReg 仅对在线 next state 反传

- 人类选择保留 B2/GA4 并修改梯度生命周期：SIGReg 仍计算在线
  `(s_t,s_{t+1})`，但 `s_t` 强制 detach，只让 `s_{t+1}` 接收 SIGReg 梯度。
- 提交 `6ccca36` 已推送。每个 microbatch 先执行唯一一次 CE/WM/value forward 与
  backward，释放当前 Qwen 图后才执行 SIGReg online-next forward/backward；没有
  恢复 row1、activation offload 或历史重算。
- 新增回归保护阶段顺序、current-state detach、online-next 梯度、B1 DDP 零 loss
  和合并后的 total loss。superpod PyTorch 2.8 定向测试 `22 passed in 6.47s`。
- superpod PyTorch 2.8 的 SFT2、Agent、Qwen、WM 和 config 扩展回归为
  `112 passed in 19.28s`。仍需 8-GPU B2/GA4 长 prefix smoke 证明真实 DDP
  static-graph 可运行且峰值显存不再随双图叠加 OOM。通过前不启动正式重训，也没有
  可供 RL 使用的新 checkpoint。

## 2026-07-23：ID41 staged-SIGReg 长 prefix smoke 仍在主 CE OOM

- ID41 `41_smoke_k1nodino_h4_stagedsigreg_b2_ga4_ws8_longprefix`，commit
  `7f952e4`，W&B `3l0hlbou`，preempt hold job `485173` 在 dgx-39 使用8 H800、
  B2/GA4/H4/k1/no DINO。输入、SFT1 epoch5初始化与ID34只读compact cache均与ID40
  一致；无resume、无checkpoint。
- staged primary/SIGReg forward/backward在真实8-rank DDP static graph上正常。
  step1/2/3均finite，peak allocated为31.402/49.752/64.694GiB；ID40前两步是
  47.626/76.952GiB，因此分阶段确实释放了双Qwen图，且通过了ID40第三周期失败点。
- 第四个accumulation周期在主阶段current Qwen forward的标准
  `ForCausalLMLoss -> cross_entropy`全rank OOM，尚未进入SIGReg。每卡PyTorch已分配
  约73.53--74.55GiB，full CE尝试再申请4.07--4.19GiB，只剩2.73--3.59GiB。
  这证明单个更长B2 current multimodal prefix本身仍超过80GB，不是SIGReg双图或
  allocator碎片。
- job失败后立即取消hold；sacct hold=`CANCELLED` elapsed00:02:10、train step
  `FAILED` elapsed00:01:36，dgx-39恢复idle且8卡全释放。W&B state=`failed`，输出
  README/CSV/log和实验组progress已更新；无checkpoint，不可resume或初始化RL。
- B2/GA4继续被否决。裸B1/GA8也不可直接用于正式训练：当前SequenceSIGReg在
  per-rank `B<2`时跳过，必须配套另行设计的可微跨rank SIGReg。另一条路径是保持
  B2并单独批准标准CE数学等价的低显存实现；禁止恢复已经删除的row-by-row/
  activation-offload应急路径。

## 2026-07-23：实现 per-rank B1 + global SIGReg B8

- 人类批准B1/GA8，并让SIGReg对每个microbatch的全局有效state计算。world8时SIGReg
  B最多为8；GA8只累积八个独立global-B8 loss，不保留state图凑成B64。
- 提交 `5a3eea4` 已推送。current state全局汇聚后保持detach；online-next state使用
  可微all-gather，backward将global state梯度送回来源rank，再由DDP平均得到一次全局
  batch loss对共享Qwen/StateProjector参数的正确梯度。
- collective支持不同rank本地B不等；先补齐到max local B，再用global valid mask排除
  补齐和整rank sampler padding。所有rank用相同可恢复microstep seed生成SIGReg随机
  projection，避免把目标隐式变成多组不同随机loss。
- K1 control配置已改B1/GA8；checkpoint invariant与日志新增global SIGReg scope/B。
  superpod定向回归 `27 passed in 8.80s`，其中两进程Gloo+DDP解析测试证明参数梯度
  等于单次global valid batch参考。SFT2、Agent、Qwen、WM、config扩展回归
  `113 passed, 1 skipped in 21.31s`；原skip为需GPU allocation的NCCL门槛。
- ID42使用preempt/dgx-40、hold job `485236`。真实两卡CUDA/NCCL门槛复测
  `1 passed, 1 deselected in 8.25s`，CPU/Gloo复测`1 passed, 1 deselected in 7.90s`。
  首轮NCCL测试失败是测试实例的SequenceSIGReg buffer仍在CPU，并非collective或梯度
  错误；提交`948079c`仅令测试模块跟随worker device，正式trainer原本已显式放置。
- ID42 8-GPU B1/GA8 long-prefix smoke完成11个finite optimizer step后按计划取消，
  超过4-step门槛并覆盖完整20-action trajectory prefix。所有step均为per-rank B1、
  global SIGReg B8；total loss `7.7422--9.7174`、CE `7.0504--8.6608`、WM MSE
  `0.03611--0.26906`、SIGReg `3.5761--4.2034`、ValueHead total
  `0.02548--0.62099`。最大step peak allocated/reserved显存`53.216/54.932 GiB`，无
  OOM/traceback/NCCL-DDP错误/NaN/Inf。hold job`485236`与train step已取消并离开
  squeue；无checkpoint、不可resume，尚无可用于RL的新checkpoint。下一步可按同一
  B1/global-SIGReg配置提交正式SFT2重训。

## 2026-07-23：epoch_001 RL H=4 smoke ID3 启动前失败

- ID43 `epoch_001` 已核验为完整 k=1/inject HF checkpoint，WM predictor
  `history_size=4`，StateProjector/ValueHead 产物齐全。
- RL smoke 实验提交 `2b6211c`：新增 H=4/PPO 配置，并让端到端启动器接受
  config、episode 数和 max steps。配置经远端当前 schema 解析通过，启动器
  `bash -n` 与 `git diff --check` 通过。
- hold job `485290` 在人类指定的 preempt/dgx-40 占用2 GPU。ID3
  `3_smoke_k1ep1_h4_base4x5_fsdp2_iter2` 在环境服务、rollout、W&B和训练前
  被启动器拒绝：外层控制日志预先写入 `RUN_OUT`，正确触发禁止复用非空输出的
  fail-fast。ID3 不可 resume，仅保留 README 和控制日志作为失败证据；新的
  服务器 RL 实验组实时 `progress.md` 已有 ID65，本地旧记录的 ID1/2 编号已失效。
  ID66 将把控制日志放在输出目录外后重试。

## 2026-07-23：epoch_001 RL H=4 smoke ID66 rollout通过、PPO训练契约拒绝

- ID66 在 dgx-40 使用 ID43 `epoch_001` 完成 `base_train` seeds1..4 的真实
  rollout：4 trajectories、20 transitions、每条5 steps，足以构造H=4 windows。reward为
  `-0.4/0.0/-0.4/-0.2`，success0/4只是smoke现象，不解读为policy质量。
- 两rank FSDP在模型加载和W&B初始化前fail-fast：当前运行时明确禁止
  `actor.enabled=true` 与 static JSONL collector，PPO要求从当前policy fresh采样。
  本轮无optimizer step、W&B run或checkpoint，不可resume。
- hold `485290` 已取消并释放dgx-40。后续不能擅自关闭PPO后声称原测试通过；
  需人类选择两卡actor-disabled H=4 WM/value离线smoke，或单卡direct-online PPO smoke。
