# VAGEN-Lite joint-policy scaffold

## 目标

在 actor 参数化尚未确定时，从项目 fork 的 VAGEN-Lite `origin/main@a6b8c8d`
建立不会提前选择 logits/PPO 公式的训练框架，并把计划保存在 VAGEN 子模块。

## 分支

- Nimloth：`feat/vagen-lite-joint-policy-scaffold`
- VAGEN：`nimloth/vagen-lite-joint-policy-scaffold`
- VAGEN 起点：`a6b8c8d03cedca169637a2e8cec9d868f5b5ad35`
- VAGEN M1 commit：`45cb9928a8d9316037e1fb86c0dff3d004705097`
- VERL gitlink：`ae269bda8ef43fad44796254146471e89d89894a`

## 当前计划

1. 保存 joint-policy scaffold 设计和明确的 deferred decisions。
2. 先用失败测试定义 no-concat decision ledger。
3. 记录每个 turn 的完整实际执行动作、来源、actor-policy ownership、reward 与终止语义。
4. 贯通 `AgentLoopOutput.extra_fields → DataProto.non_tensor_batch → trainer` 严格校验和指标。
5. 不实现 actor logits、joint behavior log-prob、PPO loss 或新的可训练模块。
6. 完成定向单元测试、编译和 diff 检查后分别提交 VAGEN 与 Nimloth gitlink。

## 已完成

- 创建独立 Nimloth worktree 和 VAGEN-Lite 子模块分支。
- 核验旧 `feat/planner-verl-vagen-scaffold` 仍基于 legacy VAGEN 且绑定旧 PlannerPolicyHead，不能直接复用。
- 将框架计划写入 `external/VAGEN/docs/joint_policy_scaffold.md`。
- 按 TDD 新增 `vagen_decision_ledger_v1`：保存 action-space、完整实际执行动作、来源、actor-policy ownership、turn reward 和 terminal/truncated 状态，并严格拒绝任何未实现的 policy-sampled 声明。
- Navigation 输出可核验的 0-based action-space contract；no-concat agent-loop 将 ledger 原样传入 DataProto，trainer 在 old-log-prob replay 前严格校验并记录覆盖指标。
- system fallback token 从 LLM response mask 排除，同时把 turn reward 锚定到最后一个真实 policy token，避免 reward 被 mask 丢弃。
- latent fallback adapter 仅在 `prompt_format=latent_plan` 启用；remote step transport 不再把字符串 done、布尔 reward 或缺失字段静默强转成合法值。
- 两轮独立 code review 的 P1/P2 均已逐项修复；最终独立复审结论为 `APPROVED`，无 blocker。

## 文件修改

- `external/VAGEN/docs/joint_policy_scaffold.md`：actor 未决边界与 M1/M2/M3 计划。
- `external/VAGEN/vagen/agent_loop/decision_ledger.py`：dependency-light ledger schema、校验、指标与 token ownership helper。
- `external/VAGEN/vagen/agent_loop/{gym_agent_loop_no_concat,agent_loop_no_concat}.py`：ledger producer/DataProto 路径、fallback mask 和 reward anchor。
- `external/VAGEN/vagen/envs/navigation/navigation_env.py`：versioned action-space 与完整 executed action 字段。
- `external/VAGEN/vagen/envs_remote/gym_image_env_client.py`、`vagen/utils/remote_step_protocol.py`：严格 remote step 解码。
- `external/VAGEN/vagen/ray_trainer.py`、`vagen/configs/vagen_multiturn.yaml`：opt-in gate、pre-replay validation 和 metrics。
- `external/VAGEN/tests/test_decision_ledger*.py`、`test_remote_step_protocol.py`：单元与 wiring 回归。
- 本文件：任务实时进度。

## 验证

- RED：初次运行因 `vagen.agent_loop.decision_ledger` 不存在失败；review 后新增 action-space/type/reward-anchor 与 remote protocol RED 均先失败。
- GREEN：`PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py' -v`：`30 passed`。
- `python3 -m py_compile`：7 个受影响生产 Python 文件通过。
- `git diff --check`：通过。
- VS Code diagnostics：ledger、gym no-concat 和 trainer 0 diagnostics。
- 当前本地环境缺少 torch/Ray/OmegaConf/httpx/PIL，未运行真实 DataProto/Ray、多模态 rollout、PPO、checkpoint 或 GPU 测试；禁止把本阶段表述为 joint PPO 已完成。

## 待确认问题

actor logits、action prior 的随机变量语义、多步 action 的真实执行方式、PPO ratio 粒度和新增状态的 checkpoint ownership 均留待人类后续决定。
