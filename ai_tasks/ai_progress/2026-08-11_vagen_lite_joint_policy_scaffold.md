# VAGEN-Lite joint-policy scaffold

## 目标

在 actor 参数化尚未确定时，从项目 fork 的 VAGEN-Lite `origin/main@a6b8c8d`
建立不会提前选择 logits/PPO 公式的训练框架，并把计划保存在 VAGEN 子模块。

## 分支

- Nimloth：`feat/vagen-lite-joint-policy-scaffold`
- VAGEN：`nimloth/vagen-lite-joint-policy-scaffold`
- VAGEN 起点：`a6b8c8d03cedca169637a2e8cec9d868f5b5ad35`
- VAGEN M1 commit：`45cb9928a8d9316037e1fb86c0dff3d004705097`
- VAGEN M2 contract commit：`25da71df5f1408d54b4b761ff40c985d9118c99c`
- VAGEN confirmed gradient/Q-target contract commit：`0a23ab3923bcef4cbda89380353c312dab77319a`
- VERL gitlink：`ae269bda8ef43fad44796254146471e89d89894a`

## 当前计划

1. M1 decision ledger 已完成并保持不变。
2. 人类已确认每个真实 turn 只执行 guided policy 的第一动作，随后从真实 observation 重新规划；模拟尾部不进入 environment PPO。
3. action prior 是 LLM action-token boundary logits 的 softmax；实际采样的 prior action token log-prob 属于 `pi_LLM`，完整 prior 分布作为 guided policy 条件。
4. M2 暂采用 scheme B：旧 ValueHead 保持 critic，guided logits 为 `alpha * l_prior + beta * stopgrad(frozen_Q)`，没有独立 actor module。
5. rollout/update 内复用 rollout-time frozen critic 的 all-action guidance scores；critic 更新后禁止重算同一批 behavior guidance。
6. 人类已确认 guided actor loss 必须经`l_prior`反传LLM；`backprop_to_llm`保留为可审计合同字段但启用时只能为`true`。Q在actor loss中始终stop-gradient。
7. 人类已确认Q用真实环境reward构造的discounted return训练：只回归实际执行的第一动作，target stop-gradient，首版使用Huber；不把即时reward或advantage本身当Q target，不伪造未执行action监督。terminal bootstrap为0；truncation需rollout-time frozen critic bootstrap。
8. ledger v2、versioned behavior record、reference/Torch guided math 和 contract identity 已实现。
9. `joint_policy.enabled=true`当前显式 fail closed，直到人类决定旧 ValueHead 接收哪一种 state，并完成 Q owner、rollout sampler、replay 和 checkpoint snapshot boundary。

## 已完成

- 创建独立 Nimloth worktree 和 VAGEN-Lite 子模块分支。
- 核验旧 `feat/planner-verl-vagen-scaffold` 仍基于 legacy VAGEN 且绑定旧 PlannerPolicyHead，不能直接复用。
- 将框架计划写入 `external/VAGEN/docs/joint_policy_scaffold.md`。
- 按 TDD 新增 `vagen_decision_ledger_v1`：保存 action-space、完整实际执行动作、来源、actor-policy ownership、turn reward 和 terminal/truncated 状态，并严格拒绝任何未实现的 policy-sampled 声明。
- Navigation 输出可核验的 0-based action-space contract；no-concat agent-loop 将 ledger 原样传入 DataProto，trainer 在 old-log-prob replay 前严格校验并记录覆盖指标。
- system fallback token 从 LLM response mask 排除，同时把 turn reward 锚定到最后一个真实 policy token，避免 reward 被 mask 丢弃。
- latent fallback adapter 仅在 `prompt_format=latent_plan` 启用；remote step transport 不再把字符串 done、布尔 reward 或缺失字段静默强转成合法值。
- M1 两轮独立 code review 的 P1/P2 均已逐项修复；最终独立复审结论为 `APPROVED`。
- M2 合同层新增显式 Scheme-B 配置、dtype-aware 数值合同、Torch 公式、严格 behavior schema/round-trip、action-token/contract/snapshot 绑定与 ledger v2。三轮 review 修复了 silent stock-PPO fallback、伪造 ownership、logprob 容差和 overflow 等问题；最终复审无 blocker。
- 只读核验确认 VAGEN 现有 token critic 不是 `[B,8] Q(s,a)`，transition reward predictor 也不是旧 ValueHead。旧 Nimloth ValueHead 输入 state 与 VAGEN `LatentStateEncoder` state 不同，未获人类决定前禁止直接加载旧权重或用其他模块冒充。
- Git history进一步确认这条`latent z -> LatentStateEncoder -> world_state -> TransitionRewardNet`并非本分支新增：基础类由ARTI5T在嵌套VERL commit `2f291ea`（2026-03-27，`MCTS`）引入；canonical latent提取由`0ca14e2`（2026-04-13，`Step 1&2 prototype`）加入；当前可配置`WorldStatePredictor`及actor wiring由`ae269bd`（2026-04-14，`Add LeWM predictor`）完成。VAGEN顶层由同作者commit `517da7a`固定该gitlink，当前VAGEN-Lite基线`a6b8c8d`继承它。因此它是ARTI5T fork历史中的既有实验路径，不应被当作旧Nimloth ValueHead的输入定义。

## 文件修改

- `external/VAGEN/docs/joint_policy_scaffold.md`：actor 未决边界与 M1/M2/M3 计划。
- `external/VAGEN/vagen/agent_loop/decision_ledger.py`：dependency-light ledger schema、校验、指标与 token ownership helper。
- `external/VAGEN/vagen/agent_loop/{gym_agent_loop_no_concat,agent_loop_no_concat}.py`：ledger producer/DataProto 路径、fallback mask 和 reward anchor。
- `external/VAGEN/vagen/envs/navigation/navigation_env.py`：versioned action-space 与完整 executed action 字段。
- `external/VAGEN/vagen/envs_remote/gym_image_env_client.py`、`vagen/utils/remote_step_protocol.py`：严格 remote step 解码。
- `external/VAGEN/vagen/ray_trainer.py`、`vagen/configs/vagen_multiturn.yaml`：opt-in gate、pre-replay validation 和 metrics。
- `external/VAGEN/vagen/joint_policy/`：Scheme-B config、contract id、behavior schema、reference/Torch math；README明确未完成 ownership。
- `external/VAGEN/tests/test_decision_ledger*.py`、`test_joint_policy_*.py`、`test_remote_step_protocol.py`：单元、可选autograd与 wiring 回归。
- 本文件：任务实时进度。

## 验证

- RED：初次运行因 `vagen.agent_loop.decision_ledger` 不存在失败；review 后新增 action-space/type/reward-anchor 与 remote protocol RED 均先失败。
- M1 GREEN：最初为`30 passed`。
- M2合同初版：`46 passed, 3 skipped`。
- 人类确认梯度/Q target后：`47 passed, 3 skipped`；3项skip均为当前环境缺torch的真实autograd/reference parity/overflow测试。review发现并修复`alpha=0`会切断LLM梯度；启用合同现要求`alpha>0`，最终复审无blocker。
- `python3 -m py_compile`：受影响生产 Python 文件通过。
- `git diff --check`：通过。
- VS Code diagnostics：ledger、gym no-concat 和 trainer 0 diagnostics。
- 当前本地环境缺少 torch/Ray/OmegaConf/httpx/PIL，未运行真实 DataProto/Ray、多模态 rollout、PPO、checkpoint 或 GPU 测试；禁止把本阶段表述为 joint PPO 已完成。

## 待确认问题

- 旧 ValueHead 在 VAGEN 中接收哪种 state：完整复用 Nimloth StateProjector/WM state，还是改用 VAGEN world state 并重新初始化；两者不可伪装为等价。
- `alpha`、`beta`、prior temperature、`gamma`、score dtype、critic loss coefficient、warmup/KL target 的正式实验值。
- 未执行 action slot 的 Q 校准与探索保护。
- frozen critic snapshot 的刷新/checkpoint边界，以及truncation bootstrap所引用的snapshot identity。
- 模拟尾部 action 的生成方式及其非 PPO 辅助目标。
