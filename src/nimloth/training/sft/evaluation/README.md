# SFT 环境 rollout 评估

`EvaluationConfig` 是显式、经过校验的评估请求；`eval_direct(config)` 和
`eval_wm(config)` 均执行真实 VAGEN episode，而不是离线训练 loss validation。

```bash
python -m nimloth.training.sft.evaluation \
  --mode direct --checkpoint /path/to/full-hf-checkpoint \
  --env-url http://environment-server --output-dir /path/to/new-output \
  --eval-sets base common_sense --split test --episodes-per-eval-set 60 \
  --seed-offset 1 --max-steps 20 --temperature 0 --top-p 1 \
  --max-response-tokens 512 --tensor-parallel-size 4
```

WM评估将 `--mode direct` 换成 `--mode wm`，并显式提供
`--num-simulations`、`--exploration-constant`、`--planner-device`。
WM checkpoint 必须通过 SFT3（旧SFT2）的完整、epoch-complete、H=1、
DINO-grid / outgoing-action MC value 合同；预测深度来自该 checkpoint 的
`prediction_horizon`。格式/Query阶段 checkpoint 不可充当 WM checkpoint。

- `cli.py`：把同一 episode、seed、生成和步数配置接到两种路径；保存并校验
  `evaluation_contract.json`，resume 改变输入或策略参数时拒绝。
- `rollout.py`：迁入原 RL rollout producer，复用 QwenVLLMAgentPolicy、
  VAGENNavigationRolloutCollector、AgentRuntime 和 EpisodeRunner；原
  `experiments/training/rl/rollout_env.py` 保留兼容转发。
- direct：每个真实观测生成回答并执行解析出的动作，不加载 WM。
- WM：现有 PlanningPolicy 在每个真实观测生成对应真实 CoT/state，调用现有
  MCTS，只执行首动作；下一观测重新生成并搜索。末边评分为
  `Q(predicted_state[K-1], action[K])`，不累加各深度 MC-return prediction。
- 两种路径都保存真实 terminal CoT，不执行 terminal draft action。
- `rollout_summary.json` 共享 overall / by_eval_set 的 success_rate、avg_reward、
  avg_steps；WM模拟步不计入环境步数。episode顺序、每集合seed和上限一致。

命令是运行接口示例，不表示已执行评估。真实执行需要 CUDA/vLLM、完整模型和
环境服务，并遵循项目独立实验授权；CPU接口检查只验证接线与调用合同。
旧SFT3的 `stage3.evaluate.evaluate` 仍为离线 loss validation。
