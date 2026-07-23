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
