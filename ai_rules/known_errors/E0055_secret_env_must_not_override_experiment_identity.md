# E0055 — secret `.env`不能覆盖显式实验identity

## 已发生的错误

ID31 launcher显式传入`WANDB_PROJECT=nimloth-rl`，但rank wrapper随后source共享`flower/.env`，其中的默认值把project改成`flower`。问题在进程命令中被发现，任务在checkpoint shard加载0%时取消。ID52的新online shell重复了同一模式；Ray打印resolved config时显示`trainer.project_name=flower`，但在W&B初始化前另一个config assertion先终止。

## 正确做法

- source secret文件前保存显式`WANDB_PROJECT`、run name和run ID。
- source只用于取得密钥；之后恢复显式实验identity。
- 模型加载前审计实际进程参数中的project/name/id。
- identity不符时立即停止，禁止让任务写入错误W&B project。

## 证据

- `experiments/training/rl/run_verl_exact_replay_rank.sh`
- `experiments/training/rl/run_verl_online_world8_smoke.sh`
- ID31 README：`outputs/experiments/training/rl/2026-07-18/31_smoke_verl455_fullactor_fullcritic_exactreplay_id22traj0_cudaordinalfix_maskedgae/README.md`
