# 2026-07-29 SFT2 MCTS pre-RL evaluation

## 目标

实现独立于 RL optimizer 的真实 VAGEN rollout 评估：从完整 SFT2 checkpoint 读取
`history_size=1` 和 `prediction_horizon=K`，每个真实环境 step 用 WM 做 K 步
UCT-MCTS，执行选中根动作，再以真实 environment success 汇总成功率。

## 计划

1. 在当前 receding-horizon planner 中增加 MCTS search mode。
2. MCTS leaf score 使用 SFT2 实际监督的
   `Q_tilde(predicted_state_K, final_simulated_action)`，不累加多步 MC return。
3. rollout JSONL 保存 MCTS visit/value/search 参数，独立入口严格读取 SFT2 checkpoint
   invariants，拒绝 H/K/value-objective 不匹配。
4. 汇总总体及每个 eval-set 的 success rate、reward 和 episode length。
5. 完成 CPU 单元测试与静态检查；本任务不启动 GPU/Slurm evaluation。

## 当前状态

- 已实现 `WorldModelPlanner(search_mode="mcts")`。每个 simulation 从唯一真实 H=1
  state 出发，严格执行 K 个 predicted transitions；UCT exploitation 为 backed-up
  leaf mean，exploration 为显式常数，最终 root action 先按 visit count、再按均值稳定选择。
- leaf score 是当前 SFT2 实际训练的
  `Q_tilde(predicted_state_K, final_simulated_action)`；不读取未执行 slot，不累计不同
  时间位置的 MC-return prediction。
- planner trace 持久化 unique K-step candidate、candidate/root visit count、backed-up
  mean value、simulation 数和 exploration constant；旧 greedy/exhaustive/beam trace
  仍可读取。
- `rollout_env.py` 已接入 MCTS 参数，并在真实 collector 完成后输出总体及逐 eval-set
  `success_rate/avg_reward/avg_steps`，同时持久化到`rollout_summary.json`。
- collector 新增 per-eval-set seed stream，使每个 held-out dataset 使用同一 seed 范围，
  并把 eval-set 写入 episode id，避免图片/record id 冲突。
- 新入口 `experiments/training/sft2/eval_mcts_rollout.py` 从完整、epoch-complete SFT2
  checkpoint 的 `training_invariants` 自动读取 H/K，严格要求 DINO-grid、H=1 和
  `predicted_rollout_executed_action_mc_v2`，且要求 simulations/UCB 常数、sampling、
  seed、dataset split 等影响结果的参数显式传入。
- checkpoint门禁额外校验WM predictor和ValueHead的action count一致，避免直到真实
  rollout中途才暴露动作空间不兼容。

## 修改与验证

- 修改：
  - `src/nimloth/agent/{planning.py,policy.py}`
  - `src/nimloth/rollout/schema.py`
  - `src/nimloth/environment/navigation/collector.py`
  - `src/nimloth/training/sft2/mcts_evaluation.py`
  - `experiments/training/{rl/rollout_env.py,sft2/eval_mcts_rollout.py}`
  - 对应 Agent/RL/SFT2 tests
- 本地 Python 3.13 定向回归：`60 passed, 1 warning`；Agent、rollout、RL、SFT2
  完整相关CPU回归在允许Gloo loopback的环境为`271 passed, 1 skipped, 1 warning`。
  skip是无CUDA环境下的既有测试，warning是既有B=1 unbiased std断言。MCTS、trace
  roundtrip、H/K及action-count checkpoint门禁、balanced seeds、per-dataset success
  aggregation、grid checkpoint loader和CPU分布式测试均通过。
- 变更文件`py_compile`和`git diff --check`通过；受影响的`.codeabs`导航条目已刷新。
- 尚未运行真实 GPU/vLLM/VAGEN rollout；本任务当前没有成功率结果。
