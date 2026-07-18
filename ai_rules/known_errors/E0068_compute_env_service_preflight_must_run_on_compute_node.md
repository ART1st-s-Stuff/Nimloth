# E0068 — Compute env service preflight必须从compute node执行

## 已发生的错误

ID49在normal hetero hold的env节点成功启动VAGEN server并写出`10.23.*` URL；但外部nohup launcher运行在login节点，直接curl该compute service地址时不可达，launcher终止并留下孤立env srun step。没有trainer/model/W&B/update。

## 正确做法

- Hold外部launcher只负责Slurm编排；对compute-node `10.23.*`服务的health/create/reset/close必须通过trainer allocation内的overlap `srun`执行。
- 失败后显式写`trainer_done.flag`并等待/清理自己的env step；保留hold前需确认没有孤立step。
- 不得把env节点本地listen/health成功当成跨节点可达性证据。

## 证据

- `experiments/training/rl/launch_verl_online_in_hold.sh`
- ID49 env/server/driver logs。
