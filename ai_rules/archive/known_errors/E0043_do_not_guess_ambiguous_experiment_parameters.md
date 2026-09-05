# E0043：实验参数不明确时禁止猜测后启动

## 已确认错误

人类提出“让 ValueHead 预测 2 轮”时，agent 曾自行把它解释为
`predictor.history_size=2`。人类随后明确实际参数是
`agent.planning.horizon=2`。两者改变不同的模型语义；错误解释在实验设计阶段发生，
所幸尚未据此启动 GPU 作业。

## 错误原因

自然语言中的“轮”可能指真实 environment step、训练 history context、WM planning
horizon、PPO iteration 或生成 turn。agent 没有先把描述映射到唯一配置字段并请求确认。

## 正确做法

- 实验参数必须使用完整配置字段名和数值确认。
- 一个描述可能对应多个字段时，立即停止，列出歧义及影响并请求人类澄清。
- 得到澄清前不得改实验配置、提交 Slurm、启动 GPU 任务或把猜测写成实验结论。
- 本次实验已确认的是 `agent.planning.horizon=2`，不是
  `predictor.history_size=2`；其他未明确参数仍不得推断。

## 相关实现边界

- `src/nimloth/agent/planning.py`：planning horizon 控制未来 latent 搜索深度。
- `src/nimloth/wm/predictor.py`：history size 控制 predictor 最大因果上下文。
- `src/nimloth/training/rl/rollout_runtime.py`：当前 planning 与 PPO actor 不能同时开启。
