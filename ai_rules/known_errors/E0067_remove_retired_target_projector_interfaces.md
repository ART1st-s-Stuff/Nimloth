# E0067：WM 的固定监督值不能训练共享 projector

## 已发生的错误

SFT2 很早就允许下一状态监督分支更新共享 projector，后来又用
`project_target_state()` 包装这一语义。移除 GridWM 的 EMA target encoder 后，该方法
只会转调同一个 `project_state()`，进一步掩盖了“监督值本身也在向预测值移动”的问题。

## 正确规则

- WM 的下一状态监督值必须同时与 Backbone 和 StateProjector 计算图分离；
- StateProjector 在同一状态作为 current/start state 时训练，不需要监督分支重复更新；
- planner RL 直接使用 rollout 保存的真实终点 anchor state，不能重新投影终点 hidden
  并保留梯度；
- 没有独立 target 参数时，不要保留 target-projector 假接口；
- 梯度测试应把 predictor 到 current state 的 Jacobian 置零，确认固定监督值不会产生
  StateProjector 梯度。
