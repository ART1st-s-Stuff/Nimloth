# 2026-07-25 direct-policy token-level credit

## 人类要求

- 停止 SFT2，立即转向 RL。
- 阅读既有 RL task 与 VAGEN 实现，在 `dev` 直接实现 token-level credit。
- 参数不明确时停止确认，禁止猜测后启动错误实验。

## 实现边界

- trajectory 新增逐步 `rewards`、`terminated`、`truncated`；真实 terminal 从 0
  bootstrap，truncation 必须显式选择策略。当前只实现 `zero`。
- 新增独立 `TokenValueHead`，输入为 Qwen replay 中每个 selected sampled token 之前
  的 hidden state；模板和 injected token 不参与 critic 或 PPO。
- 每个 environment step 的完整 rollout Monte Carlo return 放在该 turn 最后的 action
  token；更早 reasoning token reward 为 0，在 turn 内按显式 token gamma/lambda 计算
  GAE，turn 边界 reset。
- action ValueHead 继续用真实 rollout return 监督 `Q(s,a)`；token PPO 不再使用
  selected-action Q 作为 baseline。
- token head 已接入 optimizer、双卡副本的手工 gradient sync、单卡/FSDP路径的 DDP、
  checkpoint 保存/恢复与配置 metadata 校验。

精确算法名称是“真实 environment Monte Carlo return + turn 内 token GAE”。当前没有
high-level turn GAE、`gamma_turn/lambda_turn` 或 planner root policy，因此不能称完整
VAGEN Bi-Level GAE，也不能声称 planning PPO 已完成。

## 配置门禁

`actor.credit_assignment=token` 时必须显式提供：

- `token_credit.gamma`
- `token_credit.gae_lambda`
- `token_credit.value_lr`
- `token_credit.value_loss_weight`
- `token_credit.hidden_dim`
- `rl.truncated_bootstrap`

缺失字段会在配置解析阶段失败。未得到人类逐项确认前不启动 GPU RL。

## 验证

- 本地 `compileall` 与 `git diff --check` 通过；本机 Python 没有 pytest。
- 服务器定向测试：`56 passed`。
- 扩大回归首次得到 `134 passed, 1 failed`；唯一失败是测试 fake policy 选择
  reasoning+action token 却未显式声明 `turn` credit。生产校验正确拒绝该不一致；测试
  fixture 补齐显式契约后，完整 `tests/training/rl tests/agent
  tests/backbone/qwen25vl` 回归为 `135 passed, 1 expected warning`。
- 尚未启动 GPU experiment、rollout、W&B 或 optimizer step。

## 2026-07-25 planner-distillation RL 启动门禁

- 人类已明确“可以开始 RL”，但尚未给出新路径强制要求的数值和实验规模；按此前
  “参数未明确必须停下来确认”的规定，尚未提交 Slurm、GPU、rollout 或训练任务。
- 已确认可继承：corrected SFT2 `epoch_001` lineage、`planning.horizon=2`、64 条
  exhaustive candidate、planner distribution 采样环境动作、CoT token PPO、action
  distillation、DINO loss 关闭，以及 terminal observation 使用同 checkpoint/同采样
  参数生成到 `action_start` 并持久化真实 CoT、丢弃草稿 action。
- 仍需人类明确：`agent.planning.teacher_temperature`、
  `actor.planner_distillation_weight`、`agent.planning.device`、
  `predictor.train_wm`，全部 token-credit 数值、实验 episode/iteration/step 预算、
  rollout sampling 数值、partition 和物理 GPU/TP/world-size/gpus-per-rank 布局。
- 新 vLLM selected-hidden worker extension 当前只有 compile/static 边界；启动 GPU 前还要
  在远程 vLLM 0.11 环境完成 CPU/interface regression，再做一轮真实图片 GPU correctness
  smoke。未通过 smoke 前不得直接解释长期 success rate。
