# E0090: 禁止自行发明 PPO epoch 间的 objective schedule

## 现象

RL loop 把一个 fresh rollout global step 内的四个 optimizer epochs 实现成
`1+3`：第一个 epoch 训练 WM/DINO、critic 和 PlannerPolicyHead，后三个 epoch 只训练
critic 和 PlannerPolicyHead（`include_world_model=ppo_epoch == 0`）。这被写入代码、测试和
说明文档，但人类从未要求这种设计，并已明确要求删除。

## 原因

Agent 在实现多 epoch PPO 时自行选择了 auxiliary objective 的重复频率，把工程上的一种
可能方案错误地当成既定训练语义。`global_step`、PPO epoch 数和 WM/DINO 更新次数是三个
独立决定，不能互相推导。

## 正确做法

- 未经人类明确指定，禁止在 PPO epochs 间启停或改变任何 objective。
- 人类已确认替代语义：每个 fresh rollout global step只做一个optimizer epoch，并在该次
  同时训练WM、DINO、ValueHead和PlannerPolicyHead；禁止再拆成多阶段。
- 代码、测试、配置、README、checkpoint compatibility和GPU parity gate必须使用这同一份
  语义；旧的四epoch checkpoint只能作为明确不兼容的权重初始化，不能按原optimizer状态resume。

对应代码：`src/nimloth/training/rl/loop.py` 的
`include_world_model=ppo_epoch == 0` 分支及相关测试。
