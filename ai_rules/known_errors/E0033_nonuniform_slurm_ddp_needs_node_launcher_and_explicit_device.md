# E0033 — Non-uniform Slurm DDP needs node launchers and explicit devices

## 错误路径

为使用 normal 分区碎片，SFT2 需要在每个节点使用不同 GPU 数量。

1. Job `476443` 使用 one Slurm task per GPU；每个 task 只看见重编号后的
   `cuda:0`。NCCL 同节点 peer 初始化报 `invalid device ordinal`。
2. Jobs `476453`/`476464` 改为每节点一个 launcher，保留所有本地 GPU 可见，
   但 `init_process_group(backend="nccl")` 仍让 ProcessGroupNCCL 猜测 barrier
   device。非均匀 local world 下 communicator 卡住，没有 train step。

## 正确做法

- 每节点只启动一个 Slurm task，让它看见该 component 的全部 GPU；launcher 再按
  `LOCAL_RANK=0..N-1` 启动本地 Python workers，并显式分配连续 global ranks。
- 在 `torch.cuda.set_device()` 后调用
  `init_process_group(backend="nccl", device_id=torch.device(...))`；不要依赖首个
  barrier 的 rank/device 猜测。修复 commit：`5e2b454`。
- 多节点训练使用共同 bootstrap interface；本集群 `ibp41s0f0` 已验证。
- 不要为正式多节点训练设置 `NCCL_IB_DISABLE=1`。实测 dgx32↔dgx54 的
  NET/IB GDRDMA 20×64 MiB all-reduce 为约 33.5 GiB/s；Socket 路径更慢。
- 提交后必须看到所有 ranks `Init COMPLETE`、train log finite rows，再判为健康。
