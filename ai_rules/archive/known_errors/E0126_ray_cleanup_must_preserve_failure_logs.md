# E0126 — Ray cleanup must preserve local failure logs

## 已发生的错误

ID183 retry2 Job `521163`的Ray head在报告`Ray runtime started`约10秒后退出1，导致
4节点bootstrap失败。launcher cleanup随后删除各节点`RAY_CLUSTER_ROOT`；共享目录只保留
Ray CLI日志，没有持久化`ray_process_exit.log`、raylet/GCS/agent日志，因此无法从已结束
allocation确定head子进程退出原因。

## 原因

cleanup只保存owned-process audit，然后直接删除节点本地Ray临时目录。它把成功清理和
失败诊断错误地绑定在同一个删除动作中。

## 正确做法

- 删除`RAY_CLUSTER_ROOT`前，在每个节点把关键Ray session日志复制到job-keyed共享
  control目录；复制失败本身必须留下明确证据。
- 日志持久化必须有大小边界，避免长期任务无限复制本地日志。
- 在取得真实Ray退出证据前，禁止猜测根因并修改训练语义。

## 证据

- `AI_branch_progress.md`中的Job `521163`记录。
- `outputs/experiments/training/rl/slurm/id183-ray-521163-p1/dgx-29.log`。
