# RL Actor receding-horizon 重构

## 人类确认的目标

- Actor是当前代码中的World Model + ValueHead。每个真实environment step向前搜索
  `k=planning.horizon`步，只执行value最高候选的首动作；候选尾部不实际执行，下一步
  基于真实observation重新运行Qwen和Actor。
- 取消planner对Qwen action prior的蒸馏或PPO监督。
- 对transition `t`，训练重新计算
  `Qwen(完整真实prefix_t) -> s_t -> WM(s_t, a_t) -> hat{s}_{t+1} -> ValueHead`。
  历史token/CoT是固定输入，但当前这次完整prefix forward中处理历史的Qwen激活参与
  value loss反传。每个environment step使用独立graph；不连接或保留以前step的旧graph。

## 已实现

- `PlanningPolicy`删除planned-action queue。每次`select_action`都生成当前真实CoT/Qwen
  state、执行greedy/exhaustive/beam搜索，并只提交最佳候选第一步；history只保存真实
  Qwen投影state和实际执行action。
- planner trace删除Qwen action分布、`ActionTrainingTrace`和distillation/PPO objective。
  planner response仍保存真实Qwen reasoning和Actor action，但所有token的old log-prob为
  `None`、loss mask为`False`，trajectory credit明确为`none`。
- planner rollout改为每个实际action都有独立search trace、真实Qwen hidden和投影state，
  terminal observation额外保存真实CoT/state。旧稀疏segment planner记录无法忠实迁移，
  migration会要求重新采集。
- planner训练单元改为每个真实`ExecutedTransition`：
  1. 使用`build_state_prompt(t)`重建包含全部真实历史和当前CoT的prefix；
  2. 在有梯度模式下重新执行Qwen和StateProjector；
  3. 用最近`history_size`个真实state作为WM上下文，并用重算的`s_t`替换最后一项；
  4. WM预测一步`hat{s}_{t+1}`，真实`s_{t+1}`和DINO next-image target保持固定；
  5. ValueHead对`hat{s}_{t+1}`评分，只回归执行action slot到完整episode的`G_t`；
  6. 联合loss按本轮真实transition数归一化，每个step单独backward，最后一次optimizer step。
- 删除planner Qwen action replay、action distillation loss、独立detached MC ValueHead pass和
  对应metric/config/checkpoint字段。planner checkpoint写入
  `planner_training_objective=receding_horizon_transition_mc_v1`，旧objective不能直接resume。
- planner配置强制：`actor.enabled=false`、`gradient.state_source=recompute`、
  `gradient.representation_to_backbone=true`、`predictor.train_wm=true`、
  `predictor.lambda_sigreg=0`、`value_head.lambda_rank=0`。五份planner YAML已同步；正式
  greedy配置仍保持greedy，没有擅自切换搜索模式。
- 更新了rollout schema/validation、launcher、README和定向测试。新增梯度测试覆盖：
  value输入确实是预测next state；梯度到达ValueHead、WM、StateProjector和完整prefix
  Qwen；历史部分对应的Qwen参数列收到非零梯度；未执行action slot没有MC梯度；policy
  replay没有被调用。

## 当前验证

- `python -m compileall -q src/nimloth tests experiments/training/rl/rollout_env.py`通过。
- `bash -n experiments/training/rl/run_vllm_online_ppo_smoke.sh`通过。
- `bash -n experiments/training/rl/smoke_test.slurm`通过。
- `git diff --check`通过。
- 当前本地Python环境没有PyTorch/pytest，因此尚未执行CPU单元测试；没有运行GPU、vLLM、
  DDP或数值训练验证，也没有启动实验。

## 保留的模型边界

- 当前ValueHead仍输出每个离散action一个value，没有未经确认改成scalar head。planner
  transition在`hat{s}_{t+1}`上选择实际执行的`a_t` slot，并以`G_t`监督；planner配置关闭
  ranking loss，所以未执行slot没有ground-truth监督。
- 当前搜索仍以候选叶节点的最大action value作为启发式score。模型没有reward/done head，
  不能把该score表述成精确的k步environment return。
