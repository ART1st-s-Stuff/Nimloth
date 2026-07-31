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
  .venv 的 pytest 入口因解释器链接失效而缺包；完整相关 CPU 回归转到 superpod
  固定解释器执行。
- 尚未提交 Slurm job、创建训练输出或创建 W&B run。
