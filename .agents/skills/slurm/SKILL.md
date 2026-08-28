---
name: slurm
description: >-
  按机器专用的.local服务器合同指导Nimloth Slurm与远程GPU操作。资源查询、hold allocation、srun、远程worktree或任何Slurm实验均须使用。
---

# Slurm

## 触发条件

SSH/服务器访问、Slurm资源查询/提交、hold allocation、`srun`、远程GPU/长job或远程worktree同步均须使用本skill。

## 权威合同与机器边界

执行任何远程操作前，必须阅读：

- [实验索引](../../../.trellis/spec/experiments/index.md)；
- [启动/生命周期合同](../../../.trellis/spec/experiments/launch-and-lifecycle.md)；
- [实验任务合同](../../../.trellis/spec/experiments/task-contract.md)；
- `.local/SERVER.md`，获取当前主机别名、远程路径、凭据、partitions和机器专用命令。

本仓库skill只包含可移植行为。禁止把主机名、绝对服务器路径、凭据、当前节点清单或临时集群事实复制到本文件；这些信息必须留在`.local/`下。

## 拒绝门禁

- 远程/GPU/Slurm工作必须使用含有`task.json.meta.kind = "experiment"`的Trellis实验任务。
- 必须执行`on-experiment-start`；参数或数据/checkpoint/output语义缺失时必须停止操作。
- 必须请人类确认partition和GPU资源总量。
- 必须展示精确命令、train/freeze/objectives、checkpoint、output、恢复方式、监控方式以及资源/时间估算，并取得单独的启动审批。
- 本地修改必须已经commit，远程worktree必须指向该精确commit。禁止直接在服务器上修改生产代码。

## 连接与资源

使用`.local/SERVER.md`当前记录的命令/别名。如果连接超时，且本地文档指出VPN可能是原因，必须停止并请人类恢复连接；禁止循环重试。

提交前以及替换或启动等待中job的紧邻时刻，必须查询集群状态。存在仓库本地封装脚本时优先使用：

```bash
.local/scripts/query-resources.sh
.local/scripts/query-resources.sh --only-free-gpu
```

禁止根据过期记录或先前命令推断当前可用性。

## Hold allocation 与执行

除非获批任务要求其他拓扑，否则优先申请一个bash/hold allocation，并通过`srun`在其中启动工作。单个hold可减少脚本失败后的requeue浪费；多个并发hold可能触发QoS争用。

```bash
srun --jobid <approved-job-id> --pty <command>
srun --jobid="${HOLD_JOB}" --overlap --nodes=1 --ntasks=1 -w <allocated-node> bash -lc '<approved-command>'
```

禁止仅为方便而硬编码节点或固定拓扑。必须使用人类批准的资源总量和当前可用资源，同时保持训练/runtime拓扑合同。

## 监控与结束

必须监控调度器状态、日志、资源、指标、输出创建和实验标识，直到job健康运行或进入终止状态。完成、失败、取消或暂停必须立即触发`on-experiment-end`；在当前对话中记录调度器/runtime证据、输出、指标/限制、checkpoint/恢复方式、任务进度和实验组进度。
