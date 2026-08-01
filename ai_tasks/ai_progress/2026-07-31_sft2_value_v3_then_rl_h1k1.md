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
