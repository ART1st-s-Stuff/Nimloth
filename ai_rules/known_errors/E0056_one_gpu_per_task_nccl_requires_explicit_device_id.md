# E0056 — one-GPU-per-task NCCL必须显式绑定device ID

## 已发生的错误

ID32的每个Slurm task只看到重映射后的`cuda:0`，但VERL worker用`init_process_group(backend="nccl")`且未传`device_id`。首次barrier警告rank→GPU mapping未知，随后NCCL内部按无效ordinal访问并报`Cuda failure 101`。

## 正确做法

- 创建VERL worker前由runner先调用`torch.cuda.set_device(0)`。
- 显式`init_process_group(backend="nccl", device_id=torch.device("cuda:0"))`，使worker复用已初始化的group。
- 模型加载前用同一8-task布局执行NCCL all-reduce和`barrier(device_ids=[0])`direct gate。

## 证据

- `experiments/training/rl/run_verl_exact_replay_worker_gate.py`
- ID32 README：`outputs/experiments/training/rl/2026-07-18/32_smoke_verl455_fullactor_fullcritic_exactreplay_id22traj0_wandbidentityfix_maskedgae/README.md`
