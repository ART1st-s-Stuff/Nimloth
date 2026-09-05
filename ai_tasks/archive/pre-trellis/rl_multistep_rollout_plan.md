# RL 多步 Rollout 实现计划

日期：2026-06-24  
分支：`feat/rl`  
状态：计划阶段，待人类确认后执行

---

## 目标

实现基于 World Model（LatentWMPredictor + ValueHead）的在线 RL 训练，并支持多步 fast-path rollout：
- **Slow path（Qwen）**：Qwen policy 与环境交互，每步提取 latent state。
- **Fast path（WM）**：后续 N 步用 LatentWMPredictor 自回归预测 state，用 ValueHead 评估 action，无需每步过 Qwen。

## 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| `LatentWMPredictor` | 已有，单步 | `predict_next_emb(state_emb, action) → next_state_emb`，`history_size=1` |
| `ValueHead` | 已有 | `state_emb → [v0..v7]` per-action scalar values |
| `StateProjector` | 已有 | Qwen hidden dim → WM emb dim |
| `TransitionSample` + `discounted_action_value_targets` | 已有 | MC return target，基于 trajectory-level sparse reward |
| `training/rl/` | 空目录 | 待实现 |
| VAGEN 环境 | 已有 | Navigation (EB-Nav)，通过 `external/VAGEN` 子模块 |
| SFT2 训练管线 | 参考实现 | `training/sft2/trainer.py` 可作为 RL trainer 模板 |

## 实现步骤

### 1. 多步 Predictor Rollout（`wm/predictor.py`）

扩展 `LatentWMPredictor` 支持多步自回归展开：

- **`history_size` 扩展**：从 1 改为可配置，默认 4，让 ARPredictor 看到更长上下文。
- **新增 `rollout_states` 方法**：
  ```python
  def rollout_states(
      self,
      state_emb: Tensor,          # (B, emb_dim) 初始状态
      action_sequences: Tensor,   # (B, num_steps) 动作序列
  ) -> Tensor:                    # (B, num_steps, emb_dim) 预测的状态序列
  ```
  自回归地：`state_t → action_t → predictor → state_{t+1}`，不经过 Qwen。
- **单元测试**：验证多步 rollout 输出的 shape 和数值范围。

### 2. 在线 RL 训练管线（`training/rl/`）

参照 `training/sft2/` 的结构，新建：

```
training/rl/
├── __init__.py
├── README.md
├── trainer.py        # 在线 RL 主循环（rollout → 编码 → train → 循环）
├── rollout.py        # Qwen policy 与环境交互采集轨迹
├── loss.py           # Value loss + Predictor loss
├── checkpoint.py     # Checkpoint 保存/加载
└── cli.py            # 命令行入口
```

#### 在线 RL 主循环（`trainer.py`）

```
for iteration in 1..N:
    1. Rollout: Qwen policy 与环境交互，采集 trajectories
       → 每条 trajectory = (image_0..T, action_0..T-1, reward)
    2. Encode: Qwen forward 每帧 → StateProjector → WM latent states (s_0..T)
       → 生成 transitions: (s_t, a_t, s_{t+1}, discounted_return_t)
    3. Train:
       a. Predictor: s_t + a_t → pred_s_{t+1}，MSE loss 对 Qwen 编码的 s_{t+1}
       b. ValueHead: s_t → action_values，MSE loss 对 discounted_return
    4. Checkpoint 保存
```

- 先单步训练（`rollout_steps=1`）：Predictor 只预测下一帧 state。
- DDP 支持（复用 `training/common/dist.py`）。
- 所有模块冻结策略通过 config 控制。

#### Rollout（`rollout.py`）

- 启动 VAGEN 环境（复用已有 baseline 环境配置）。
- Qwen policy 生成 actions（当前阶段使用 Qwen 自身 action prior，即 `<|action_start|>` 位置 logits argmax）。
- 收集 trajectories：`(image_path_0..T, action_index_0..T-1, success, reward)`。
- 写 JSONL（兼容已有 `TransitionSample` 格式）。

#### Loss（`loss.py`）

- **Predictor Loss**：`MSE(predictor(state_emb, action) → next_state_emb, actual_next_state_emb)`
  - Actual next state 来自 Qwen 对下一帧的编码（trainer encode 阶段）。
- **Value Loss**：`MSE(value_head(state_emb)[action_taken], discounted_return)`
  - Target = `discounted_action_value_targets`（MC return, gamma=0.99）。

#### 配置项（`configs/training/rl/`）

```yaml
# 模块冻结策略
freeze:
  qwen: true             # Qwen backbone（只 forward 提取 latent state）
  state_proj: true       # StateProjector

# Predictor 配置
predictor:
  lr: 1e-3
  history_size: 4        # ARPredictor 上下文帧数
  rollout_steps: 1       # 训练时 rollout 步数（先单步跑通）

# ValueHead 配置
value_head:
  lr: 1e-3

# RL 训练配置
rl:
  iterations: 1000         # 在线 RL 迭代次数
  envs_per_iteration: 8    # 每次迭代采集的并行环境数
  max_steps_per_episode: 20
  gamma: 0.99
  batch_size: 32
  train_steps_per_iteration: 10

# 输出
training:
  log_interval: 10
  save_interval: 50       # iterations
```

### 3. 搜索/规划（`wm/planning.py`）

推理时用 predictor + value head 做搜索，选最优 action。**通过 config 选择算法**：

```yaml
planner:
  algorithm: beam_search    # "beam_search" | "greedy" | "mcts"（后续）
  beam_width: 4             # beam_search 使用
  rollout_depth: 4          # 搜索深度（fast-path 步数）
  num_simulations: 100      # mcts 使用（后续）
```

#### Greedy（优先实现）

直接取 `ValueHead(state_emb)` 中 value 最高的 action，无搜索。

#### Beam Search

1. 初始 state_emb，展开全部 8 个 action，predictor 预测 8 个 next state，value head 评分。
2. 保留 top-K（beam_width）序列。
3. 每步从 K 个序列各展开 8 个 action，共 K×8 候选，value head 评分后保留 top-K。
4. 到达 rollout_depth 后，选 value 最高的序列的第一个 action。

#### MCTS（后续）

用 predictor 做树展开，value head 做 leaf evaluation，UCB 做节点选择。

### 4. Qwen + WM 推理循环（`agent/`）

编排 slow path 和 fast path：

1. 环境给出一帧图像 → Qwen encode → 提取 `<|latent_state|>` hidden state → `StateProjector` → WM state_emb。
2. Planner（greedy/beam_search）选 action。
3. 执行 action → 环境返回下一帧，回到步骤 1。

---

## 已确认决策

1. **数据来源**：在线 rollout（Qwen policy 与环境实时交互采集）。
2. **Trainer 先行**：先单步训练跑通（`rollout_steps=1`），后续再扩展多步。
3. **在线 RL**：rollout 采集 → 编码 → 训练 → 循环，不依赖离线数据集。
4. **阶段顺序**：先实现 RL 训练管线（步骤 1-2），再实现搜索/规划（步骤 3-4）。
5. **Qwen policy**：RL 阶段 Qwen 自身 action prior 作为 policy（`<|action_start|>` logits argmax），待 WM 训练后可替换为 WM planner。
