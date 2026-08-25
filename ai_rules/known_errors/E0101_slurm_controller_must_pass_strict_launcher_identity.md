# E0101: Slurm controller must pass strict launcher identity

## 已发生的错误

ID165 hold Job `519040` 获得8卡 allocation 后，外部queue controller直接调用
`launch_vagen_joint_update_gate_on_hold.sh`，但没有设置该launcher要求的`REPO`、
`EXPECTED_PARENT_COMMIT`、`EXPECTED_VAGEN_COMMIT`和`EXPECTED_VERL_COMMIT`。
launcher在进入`srun`前fail closed，controller随后取消了allocation。

## 原因

controller只保存了production worktree路径，却错误地假设strict launcher会从路径推断
repo和Git identity。launcher故意要求caller显式提供这些身份，避免对错误worktree运行。

## 正确做法

任何异步queue controller在调用strict launcher时必须显式传入全部required env，并在提交前
逐项核对production worktree的实际SHA。不能因为controller知道worktree路径，就省略launcher
的身份合同。

## Evidence

- `experiments/training/rl/launch_vagen_joint_update_gate_on_hold.sh`：入口处的四个required env gate。
- `AI_branch_progress.md`：ID165 Job 519040的5秒pre-launch failure记录。
- 服务器`outputs/experiments/training/rl/slurm/id165-hold-519040.metadata.md`：`sacct`和controller证据。
