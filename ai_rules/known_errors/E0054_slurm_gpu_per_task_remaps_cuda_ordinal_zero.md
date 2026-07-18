# E0054 — `--gpus-per-task=1`时不能把`SLURM_LOCALID`当CUDA ordinal

## 已发生的错误

ID30使用8个Slurm task，每个task通过`--gpus-per-task=1 --gpu-bind=single:1`只看到一张GPU。Slurm在每个task内把该GPU重映射为CUDA ordinal0，但wrapper把`SLURM_LOCALID=0..7`传给`torch.cuda.set_device`，tasks1–7报`invalid device ordinal`。

## 正确做法

- 全局分布式rank继续使用`SLURM_PROCID`。
- 每task只暴露一张GPU时固定`LOCAL_RANK=0`并调用`torch.cuda.set_device(0)`。
- 保留`SLURM_LOCALID`为审计字段，不能用作task内CUDA ordinal。
- 正式模型启动前用同一8-task srun直接验证每个task的`device_count==1`、current device0和唯一物理GPU绑定。

## 证据

- `experiments/training/rl/run_verl_exact_replay_rank.sh`
- `experiments/training/rl/run_verl_exact_replay_worker_gate.py`
- ID30 README：`outputs/experiments/training/rl/2026-07-18/30_smoke_verl455_fullactor_fullcritic_exactreplay_id22traj0_submodulefix_maskedgae/README.md`
