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
- 训练：H=1，T=4，2 epochs，16×H800，per-rank B1，GA4，fresh optimizer。
- trainable：Qwen vision、StateProjector、WM predictor、ValueHead。
- frozen：Qwen LLM、DINO teacher/cache、latent query。
- loss：当前步 CE；4 个 successor 的 WM/DINO；4 个 decision state 上对应
  executed action 的 Monte Carlo ValueHead MSE；全局 SIGReg；无 ranking loss。
- 监控：W&B nimloth-sft2，训练/验证 loss 与 val_wm_mse；20 分钟周期
  latest 和 epoch/best/final checkpoint。
- 生命周期：Slurm batch job 自持两节点控制器，不使用登录节点 watcher，
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
- ID64正式job 500294原请求normal、2节点×8 H800、WS16/B1/GA4；人类随后要求
  改为单节点8卡。该job在PENDING时取消，sacct为CANCELLED by 3738、elapsed
  00:00:00、Start=None且无AllocTRES。
- ID64没有训练目录、W&B run、optimizer、checkpoint或done marker，不能resume；
  cache未修改。服务器ID64 README和实验组progress已记录取消原因与证据。
- 新重试使用ID65独立output/W&B identity，normal单节点8 H800、WS8/B1/GA8；
  effective global batch保持64，预期仍为每epoch776、两epoch1,552 optimizer steps。
- 已新增WS8 batch入口并把node launcher参数化为可校验的nodes/local ranks/GA合同；
  修复正式batch/node入口误写为字面量反斜杠加美元符号、导致运行时变量不会展开的
  问题。job 500294没有allocation，因此没有执行过该错误路径。5项静态合同、4个
  shell入口语法、3个Python文件编译和diff检查通过。superpod clean worktree固定
  `77acc219`后，显式`.venv-vagen-main/bin/python3`完整CPU回归为116 passed、
  1 skipped（48.68s），skip仅为可选GPU/NCCL门禁；全量只读preflight和实际8卡
  健康门禁仍待执行。
