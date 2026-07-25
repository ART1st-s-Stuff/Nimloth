# E0047：planner 必须保留 checkpoint variant 的 state 形状

## 已确认错误

planner-distillation 首版只接受 `(B,L,D)` state history，并只给标准 latent predictor
实现了 history rollout；实际 corrected SFT2 epoch1 使用 grid WM，其 state 为
`(B,L,N,D)`。rollout-time loader 还把 grid predictor 装进普通 `WorldModel`，遗漏了
grid ValueHead 的 slot mean-pooling。

## 原因

测试只使用 `Identity` projector 和标准 latent fake predictor，没有从真实 grid
checkpoint loader 一直覆盖到 H-step planner search。

## 正确做法

- planner 公共逻辑只拥有 batch/time 轴，必须保留 variant-specific state tail。
- 每个 WM predictor variant 都必须实现同一 `rollout_from_history` 契约。
- rollout-time 轻量 loader 可以省略 DINO/EMA 辅助模块，但不能省略 grid value pooling
  等实际推理语义。
- 启动 GPU 前必须用目标 variant checkpoint 构造 state，并覆盖完整候选搜索的回归测试。

相关代码：`src/nimloth/agent/planning.py`、`src/nimloth/wm/grid.py`、
`src/nimloth/training/rl/planning_loader.py`。
