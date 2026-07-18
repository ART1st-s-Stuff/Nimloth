# E0056 — 本集群单节点NCCL不能使用one-GPU-per-task隔离

## 已发生的错误

ID32用8个Slurm task和`--gpus-per-task=1`，每个进程只见重映射`cuda:0`。即使显式`init_process_group(..., device_id=cuda:0)`，独立8-rank all-reduce仍报NCCL `Cuda failure 101 invalid device ordinal`；Slurm task GPU隔离阻断了NCCL所需的同节点peer设备访问。

## 正确做法

- 单节点world8只启动一个Slurm task并把8张GPU全部交给它。
- 在该task内用`python -m torch.distributed.run --nproc-per-node=8`。
- 每个child看到CUDA ordinals0..7，使用torchrun的`LOCAL_RANK`设置device并显式绑定NCCL process-group device。
- 正式模型加载前必须用同样布局通过8-rank NCCL all-reduce/barrier direct gate。

## 证据

- `experiments/training/rl/run_verl_exact_replay_torchrun.sh`
- `experiments/training/rl/launch_verl_exact_replay_in_hold.sh`
- ID32 README：`outputs/experiments/training/rl/2026-07-18/32_smoke_verl455_fullactor_fullcritic_exactreplay_id22traj0_wandbidentityfix_maskedgae/README.md`
