--------
本文为 AI 起草、供人类审阅。当前仅创建任务，不代表设计已获确认或功能已经实现。
--------

# RL：k>1 latent query 与 WM+ValueHead 连续动作适配任务

日期：2026-07-20

状态：本地实现与单元验证完成；按人类要求暂缓 smoke/服务器任务

## 1. 背景与当前边界

当前 Nimloth RL 已具备 SFT2 warm-start、独立 rollout → JSONL → FSDP training、WM/value loss、可选 Qwen PPO、完整 checkpoint/resume 等基础链路，但存在两个明确边界：

1. RL action/encoding runtime 在 `src/nimloth/training/rl/rollout.py::validate_rl_policy_protocol` 中只允许 `k=1 + inject`；prompt、token 注册、latent 提取和 `StateProjector` 构造也仍按单 query 编写。
2. `LatentWMPredictor.rollout_states()`、`Planner` 和 `WMAgent` 已有多步推演/fast-path 代码，但 RL 轨迹仍由 Qwen action prior 每步产生，训练仍使用单步 dynamics target；现有两步 RL smoke 不能作为 WM+ValueHead 连续动作验证。

## 2. 目标

在不破坏现有 k=1 路径和两阶段 FSDP 安全边界的前提下，为 RL 接入：

### Feature A：k>1 latent query

- 从 SFT2/HF checkpoint metadata 读取 `latent_token_count` 与 `latent_query_mode`。
- 人类已确认首期只支持 `inject`，至少支持当前主路径 `k=8`；`generate` 不在本次范围。
- rollout action policy、RL trajectory encoding、PPO forward、StateProjector、checkpoint/resume 全链路使用同一个显式 k。
- k=1 继续保持兼容，并对 metadata/token/shape 不一致 fail-fast。

### Feature B：WM+ValueHead 连续动作

- 一次 Qwen slow-path 编码后，允许连续若干步由 WM state 与 ValueHead/planner 选择动作。
- 每个动作仍逐步发送给环境并记录真实 observation、reward、done；不得伪造环境反馈，也不得把多个动作冒充成一个环境 step。
- fast-path 中通过 `WMPredictor(s_t, a_t)` 得到下一 predicted state，并在配置的 horizon/resync 边界重新使用真实 observation 经 Qwen 对齐。
- 为连续 fast-path 增加真实的多步 dynamics rollout loss，避免只训练单步 MSE 却把递归推演描述成已训练的多步模型。

## 3. 术语与已确认的连续动作语义

- `latent_token_count=k`：每个 observation 导出 k 个有序 query hidden，shape 为 `[B,k,H]`；StateProjector 展平为 `[B,kH]` 后映射到固定维度 WM state。
- `fast_path_horizon=N`：一次 Qwen real-observation sync 后，最多连续执行 N 个由 WM+ValueHead 产生的动作。
- 人类已确认：fast path 开始时使用 Qwen GT state；整个 fast-path segment 内不再用环境 observation 覆盖 state，而是持续以 `WMPredictor(s_t,a_t)` 的 predicted state 作为下一步输入；segment 结束后，再从当前真实 observation 经 Qwen 得到新的 GT state，开始后续过程。
- 每个动作仍逐步发送给真实环境，并记录 observation/reward/done；环境 done 会立即终止 segment。
- 人类已确认首期 planner 只支持 greedy。Beam search 不在本次范围内，因此本任务不依赖尚待复核的 sequence-score 语义。

## 4. 关键正确性约束

### 4.1 k>1 协议必须由 metadata 驱动

- 禁止在 RL 中另写一套 k=8 token 名称或顺序；复用 `nimloth.latent` 的 latent block/token API。
- tokenizer 必须注册完整 k-token block；prompt 中必须插入完整且有序的 block。
- latent 提取必须使用 block API，严格验证每个样本恰有 k 个连续 query token。
- `StateProjector` 必须按 checkpoint 的 `qwen_hidden_dim × k` 构造；禁止继续硬编码 `qwen_hidden_dim=2048, k=1`。
- checkpoint save/resume 必须保存并核对 k、query mode、query token IDs、projector input dimension 和 WM embedding dimension。

### 4.2 行为策略与 PPO log-prob 必须一致

WM planner 产生的动作不是 Qwen rollout policy 的采样结果。不得把这些动作配上 Qwen 的 log-prob 后直接进入当前 PPO ratio，否则 `old_log_prob` 不代表真实行为策略。

人类最终确认正式 fast-path RL 使用 hybrid `policy=qwen_wm`：

- 每个 segment 的首步从 Qwen GT k-query state 出发，由 Qwen 行为策略真实采样动作并记录 behavior log-prob；
- ValueHead 对 Qwen step 提供 critic，使用标准 advantage `A=G-Q(s,a)` 进入 clipped PPO；
- segment 后续 step 由 deterministic greedy ValueHead 在连续 WM predicted state 上产生，不进入 Qwen PPO；
- WM dynamics 与 ValueHead 使用 segment 的全部 step；训练 Qwen language full、WM predictor和ValueHead，冻结vision与StateProjector；
- 保留纯 `qwen`/纯 `wm_value` 模式用于对照，但正式配置使用 `qwen_wm`。

### 4.3 多步 dynamics 必须使用连续 trajectory window

建议同时保留：

- 单步 teacher-forced loss：`pred(s_t,a_t) -> target(s_{t+1})`；
- 多步 rollout loss：从真实 `s_t` 开始，递归使用 predicted state 展开 `1..H` 步，并分别对齐真实 `s_{t+j}`；
- 每个 horizon 的权重、最大 unroll steps 和是否 stop-gradient 必须配置化并记录到 checkpoint。

不得随机拼接来自不同 trajectory 或不连续 step 的状态/动作。

### 4.4 分布式安全边界保持不变

- `world>1` 时仍使用独立 inference rollout backend 生成固定 JSONL，FSDP trainer 不直接做动态环境交互。
- 所有 rank 必须消费相同的 trajectory/window 顺序并执行相同次数的 Qwen/FSDP forward。
- rollout 只能使用明确的 `*_train` dataset；validation 必须使用独立 split。

## 5. 拟修改范围

### RL runtime

- `src/nimloth/training/rl/rollout.py`
  - metadata-driven protocol validation；
  - k-token prompt；
  - `policy_source`、sync/fast-path 信息与合法 behavior log-prob 记录。
- `src/nimloth/training/rl/trainer.py`
  - k-token latent block encoding；
  - policy-source-aware PPO mask；
  - 连续 trajectory window 与多步 dynamics loss；
  - k/horizon 相关 metrics。
- `src/nimloth/training/rl/loss.py`
  - 多步 dynamics loss；
  - 明确单步/多步 loss 权重和指标。
- `src/nimloth/training/rl/cli.py`
  - 从 checkpoint metadata 构造 tokenizer、StateProjector 和 runtime protocol；
  - 新增显式 fast-path/planner/multi-step 配置校验。
- `src/nimloth/training/rl/checkpoint.py`
  - 保存并验证 k、query mode、fast-path、多步 loss 和 optimizer ownership。

### 共享 WM/agent 代码

- 优先复用 `src/nimloth/latent/`、`src/nimloth/wm/predictor.py`、`src/nimloth/wm/state_proj.py` 和 `src/nimloth/wm/planning.py`。
- 只有在 RL 所需接口无法由现有公共 API 表达时才修改共享模块。
- `WMAgent` 当前也按单 latent token 编码；如果将其作为正式 rollout backend，必须同步改为 metadata-driven k-token extraction。

### 配置、实验入口与文档

- `configs/training/rl/`：增加独立的 k>1 + WM fast-path 配置，不覆盖已验证的 k=1 smoke config。
- `experiments/training/rl/rollout_env.py`：接入真实 WM/value checkpoint 和 fast-path policy。
- `src/nimloth/training/rl/README.md`、`experiments/training/rl/README.md`：明确行为策略、PPO mask、sync 语义和当前验证范围。

## 6. 建议实施顺序（TDD）

### 阶段 0：冻结协议与 baseline

1. 记录当前 k=1 RL 测试基线和 checkpoint metadata。
2. 为现有 k=1 prompt、action log-prob、latent shape、checkpoint resume 补齐回归测试。
3. 人类确认第 9 节中的设计问题后再进入实现。

### 阶段 1：k>1 RED → GREEN

1. 先写 k=8/inject protocol、token block、latent extraction 和 projector shape 的失败测试。
2. 将 RL runtime 改为 metadata-driven k。
3. 验证 k=1 与 k=8 的 prompt/action位置、hidden shape、projector加载和 checkpoint gate。

### 阶段 2：WM fast-path rollout RED → GREEN

1. 用 counting Qwen/predictor/value fake module 验证：一次 sync 后能连续执行 N 步，fast-path 不额外调用 Qwen。
2. 环境每步仍真实执行，正确处理 done、失败动作和最后 observation。
3. JSONL 为每步记录 `policy_source`、sync index、predicted-state step、planner配置和合法的 behavior log-prob ownership。

### 阶段 3：多步 dynamics training RED → GREEN

1. 构造同 trajectory 连续 window 测试。
2. 验证递归输入使用 predicted state，而非每步偷用真实 state。
3. 验证 horizon mask、短 episode、done boundary、有限 loss 和 predictor 梯度。
4. 保留单步 loss，对单步/多步指标分开记录。

### 阶段 4：PPO ownership 与 checkpoint

1. 验证 Qwen PPO 只消费 `policy_source=qwen` 且具备真实 old log-prob 的样本。
2. 全 WM-policy batch 下 actor loss 应明确跳过，而不是填 0 log-prob 冒充。
3. 验证保存/恢复 k、planner、horizon、多步 loss、Qwen/WM/value optimizer state。

### 阶段 5：真实端到端 smoke

需另行遵守实验启动审批规则。建议最小 gate：

1. 使用真实 SFT2 `k=8/inject` checkpoint；
2. train split 轨迹，至少一次 Qwen sync 后连续 2 个 WM+ValueHead 动作；
3. JSONL schema、image/action/reward 对齐通过；
4. 两卡 FSDP 完成一次 update，并由新进程恢复再完成一次 update；
5. loss/grad/checkpoint tensor 全 finite、非空；
6. 参数 delta 符合冻结策略；
7. 只宣称机械可行，不从小样本 success rate 推断效果。

## 7. 测试与验证建议

本地静态与单元测试：

```bash
python -m compileall -q src/nimloth tests
PYTHONPATH=src python -m pytest \
  tests/training/rl \
  tests/test_wm_predictor_rollout.py \
  tests/test_wm_planning.py -q
bash -n experiments/training/rl/*.sh experiments/training/rl/*.slurm
```

需要新增或扩展的测试至少覆盖：

- k=1/k=8 token 注册、prompt block、action-start位置和 extraction shape；
- metadata、token ID、projector input shape 冲突 fail-fast；
- fast-path Qwen call count、连续动作数、resync 和 early done；
- policy-source-aware PPO mask 与 behavior log-prob ownership；
- 多步 rollout loss 的递归语义、trajectory boundary 和梯度；
- JSONL round-trip 保留新 metadata；
- k>1 full checkpoint save/resume；
- 分布式所有 rank 的固定 forward 次数。

## 8. 完成标准

当前已满足本地实现、回归单测和静态检查；真实 checkpoint/env/GPU/FSDP 条目因人类要求暂缓 smoke，尚未验证。因此任务不能宣称端到端完成。

只有同时满足以下条件才可将任务标记为完成：

- k=1 回归测试保持通过；
- 真实 k>1/inject checkpoint 可完成 rollout encoding、WM/value update 和 checkpoint resume；
- 至少两个连续真实环境动作由 WM+ValueHead 路径产生，且 Qwen call count 证明中间 fast-path 未重编码图像；
- 多步 dynamics loss 真实参与训练并产生 predictor 梯度；
- Qwen PPO 未错误消费 WM planner 行为数据；
- 两阶段 FSDP 安全 gate、train/val split gate、finite/shape/delta gate 全部通过；
- 文档清楚区分“代码支持”“smoke 可行”和“效果验证”。

## 9. 已确认决策与剩余配置边界

1. **已确认**：k>1 首期只支持 `inject`；`generate` 不在本次范围。
2. **已确认**：fast path 从 Qwen GT state 开始，segment 内连续使用 WM predicted state；segment 结束后从当前真实 observation 经 Qwen 重新取得 GT state。
3. **已确认**：正式使用 `qwen_wm` hybrid segment；首步Qwen sampled action以`A=G-Q(s,a)`做合法PPO，后续WM/value action不进入PPO；全部step训练WM/value。
4. **已确认**：Qwen language full、WM predictor、ValueHead可训练；vision与StateProjector冻结。
5. `fast_path_horizon` 与 multi-step loss horizon 均配置化；首期默认2。
6. **已确认**：WM planner只使用greedy，beam search不在本次范围。
