# E0072 — Ray-remapped vLLM在Torch2.8必须禁用symmetric-memory allreduce

## 已发生的错误

ID53通过env/data/config/Ray和8-rank actor checkpoint load，随后每个Ray worker各自只见一个物理GPU为ordinal0。vLLM0.11/Torch2.8 symmetric-memory communicator在TP8 rendezvous把不同worker的ordinal0误判为overlapping device，actor-rollout init失败；critic/W&B/rollout/update未开始。

## 正确做法

- 当前Ray单GPU可见性布局显式设置`VLLM_ALLREDUCE_USE_SYMM_MEM=0`，回退正常NCCL communicator。
- 该设置不改变模型/采样语义；仍须direct验证TP8 generation和FSDP↔vLLM weight sync。
- 不得改回one-GPU-per-Slurm-task布局；Ray内部单GPUworker remap与Slurm task隔离是不同层次。

## 证据

- `experiments/training/rl/run_verl_online_world8_smoke.sh`
- ID53 vLLM stack trace。
