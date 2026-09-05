# E0089: srun 非零返回后仍须验证完整 rank 结果

## 现象

两个跨节点 `srun` steps 返回 `1:0`，外层 controller 因此标记 gate 失败；但四个 rank
已经分别写出完整 `status=passed` JSON，并完成四个 PPO epochs、梯度同步、参数同步和
冻结模块断言。日志中没有 traceback、OOM、NCCL error 或 assertion。

## 原因

Slurm step 返回码只能说明 step 外壳非零，不能区分计算过程失败与所有 rank 结果落盘后的
清理期非零。只看 `srun` 状态会把完整通过的计算证据误报为失败。

## 正确做法

- 任一 rank 结果缺失时继续 fail closed，并输出 node logs。
- 四份结果都存在时，仍必须逐份解析并验证 rank/world size/PPO epochs、非零梯度与参数
  delta、冻结边界和 replica 差异；只有全部断言通过才允许记录
  `passed_with_srun_warning`。
- 不得仅因文件存在而忽略其内容，也不得把 controller warning 隐藏为普通 clean pass。

对应事件：ID145。对应启动器：
`experiments/training/rl/run_planner_policy_gpu_gate_4x4_on_hold.sh`。
