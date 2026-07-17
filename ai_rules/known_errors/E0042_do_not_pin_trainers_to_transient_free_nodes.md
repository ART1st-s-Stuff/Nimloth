# E0042: 不要把 trainer 固定到瞬时空闲节点

## 已发生错误

ID20提交前虽然查询了空闲GPU，但launcher把trainer固定为dgx-09和dgx-27。提交后其他任务立刻占用dgx-09，导致任务被固定节点锁住并长期PENDING；人类明确指出应在每次提交前检查全部空闲卡且不要锁死节点。

## 原因

把“提交前资源快照”错误地固化成长期调度约束。节点名只在必须保证已验证的env语义或人类明确指定时才应固定；trainer只需要满足world size、partition、显存/主机资源和排除列表，不需要特定机器身份。

## 正确做法

- 每次提交前重新查询目标partition全部节点的GPU、CPU、主机内存和`PLANNED`状态。
- trainer launcher默认不写固定`--nodelist`；让Slurm在满足fragment形状和硬排除的节点间选择。
- 只固定已经通过bounded semantic preflight且确实需要节点身份的env节点。
- 提交后立刻核对`ReqTRES`、`ExcNodeList`、`SchedNodeList`和实际分配；PENDING replacement继续遵守E0026。
