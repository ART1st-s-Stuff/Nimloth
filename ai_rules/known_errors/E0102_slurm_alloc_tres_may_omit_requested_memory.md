# E0102: Slurm `AllocTRES` may omit requested memory

## 已发生的错误

ID165 hold Job `519083`已获得`normal`单节点8GPU、64CPU、256GiB allocation，
strict launcher却要求`AllocTRES`包含`mem=256G`。本cluster的`scontrol`/`sacct`
把内存记录在`ReqTRES`、`MinMemoryNode`和`ReqMem`，`AllocTRES`只包含CPU/GPU/node。
launcher因此在`srun`前错误退出。首次修复只改了launcher，遗漏phase runner中的同一重复
assertion，导致后续Job `519090`进入`srun`后仍在output创建前退出。

## 正确做法

在这个cluster上核验job memory时，必须同时检查请求合同`ReqTRES=...mem=256G`和
实际job字段`MinMemoryNode=256G`（结束后也可核对`sacct ReqMem`）。CPU/GPU仍从
`AllocTRES`核验。禁止假设所有Slurm部署都会把memory复制进`AllocTRES`；共享同一allocation
合同的launcher与runner必须同时搜索和修复，测试也必须覆盖两个文件。

## Evidence

- `experiments/training/rl/launch_vagen_joint_update_gate_on_hold.sh`：修正后的memory gate。
- `tests/training/rl/test_vagen_joint_update_gate_launcher.py`：cluster字段回归合同。
- 服务器`outputs/experiments/training/rl/slurm/id165-hold-519083.metadata.md`：真实job字段与失败记录。
