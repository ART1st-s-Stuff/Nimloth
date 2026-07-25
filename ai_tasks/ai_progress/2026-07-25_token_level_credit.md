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
  reasoning+action token 却未显式声明 `turn` credit。生产校验正确拒绝该不一致，测试
  fixture 已补齐显式契约，等待最终重跑。
- 尚未启动 GPU experiment、rollout、W&B 或 optimizer step。
