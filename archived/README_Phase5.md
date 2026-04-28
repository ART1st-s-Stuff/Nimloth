# Phase 5: RL 训练流程

**日期**: 2026-04-26
**状态**: 已实现并验证通过

---

## 概述

Phase 5 实现了一个完整的 PPO (Proximal Policy Optimization) 强化学习训练流程，基于 World Model (WM) 产生的 latent 表示进行策略学习。

### 核心设计思想

1. **Latent Space RL**: 直接在 WM 的 latent 空间中进行决策，避免在高维图像空间的复杂性
2. **Actor-Critic 架构**: 使用 PolicyModel (Actor) 和 ValueNetwork (Critic) 联合学习
3. **PPO 算法**: 使用近端策略优化保证训练稳定性
4. **向量化环境**: 支持并行环境加速数据收集

### 文件结构

```
src/rl/
├── __init__.py          # 模块导出
├── policy_model.py      # 策略网络 (Actor)
├── value_network.py     # Value 网络 (Critic)
├── storage.py           # RolloutStorage - 经验回放
├── vec_env.py           # LatentVecEnv - 向量化环境
├── ppo_learner.py       # PPO 训练器
└── train_rl.py          # 训练入口

configs/
├── rl_default.yaml      # 默认配置文件
```

---

## 核心组件

### 1. LatentVecEnv (`vec_env.py`)

**用途**: 基于预编码的 latent cache 构建 RL 环境

**关键特性**:
- 自动检测 latent 维度 (num_patches, token_dim)
- 支持真实数据 (LatentVecEnv) 和测试数据 (DummyVecEnv)
- 每个环境对应一个 episode，支持并行采样

**数据格式**:
```python
# Latent cache 格式 (.pt 文件)
{
    "latents": {
        ".../floorplan6_ep0001_step0000.png": torch.Tensor([16, 384]),
        ".../floorplan6_ep0001_step0001.png": torch.Tensor([16, 384]),
        ...
    },
    "latent_dim": 6144
}

# Episode 结构 (自动从路径解析)
# 图像路径格式: .../floorplan6_ep0001_step0156.png
# -> episode_key = "floorplan6_ep0001", step = 156
```

**关键方法**:
```python
class LatentVecEnv:
    def reset(self) -> tuple[Tensor, Tensor | None]:
        """重置所有环境，返回初始观察"""
        # z_history: [num_envs, history_len, num_patches, token_dim]
        return self.z_history, self.semantic

    def step(self, actions: Tensor) -> EnvStepResult:
        """执行动作，返回 (obs, reward, done, info)"""
        # rewards: [num_envs]
        # dones: [num_envs]
        return EnvStepResult(...)
```

**自动维度检测**:
```python
# 加载 cache 时自动检测实际维度
sample = list(latents.values())[0]  # [16, 384]
self.num_patches, self.token_dim = sample.shape  # 16, 384
```

---

### 2. PolicyModel (`policy_model.py`)

**用途**: 策略网络 (Actor)，输出动作分布

**架构**:
```
输入: z_history [B, H, P, D]
  │
  ├─ patch_token_proj: Linear(D -> hidden_dim)
  ├─ patch_pool: Linear(hidden_dim -> hidden_dim), mean pooling
  ├─ pos_embedding: 可学习位置编码
  │
  ├─ TransformerEncoder: num_layers 层
  │
  ├─ (可选) VLM 语义融合:
  │    semantic_proj -> gate_proj -> hidden * gate + s * (1-gate)
  │
  └─ 动作分布头:
       mean_head -> [B, action_dim]
       log_std -> 可学习参数 [action_dim]
```

**输出**:
```python
def forward(self, z_history, semantic=None):
    mean, std = self.policy(z_history, semantic)
    # mean: [B, A], std: [B, A]

def act(self, z_history, semantic=None, deterministic=False):
    # 用于收集经验
    action, log_prob, entropy = self.policy.act(...)
    # action: [B, A], log_prob: [B], entropy: [B]
```

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| latent_dim | P * D | latent 总维度 |
| hidden_dim | 256 | Transformer 隐藏维度 |
| num_layers | 4 | Transformer 层数 |
| num_heads | 4 | 注意力头数 |
| action_std_init | 0.5 | 初始动作标准差 |

---

### 3. ValueNetwork (`value_network.py`)

**用途**: 状态价值估计 (Critic)，用于 GAE 优势函数计算

**架构**: 与 PolicyModel 类似，但输出标量

```python
def forward(self, z_history, semantic=None):
    value = self.value_head(hidden)  # [B]
    return value
```

---

### 4. RolloutStorage (`storage.py`)

**用途**: 存储 PPO 训练过程中的 rollout 数据

**数据结构**:
```python
class RolloutStorage:
    # 循环缓冲区 [num_steps + 1, num_envs, ...]
    z_history: Tensor      # [T+1, B, H, P, D]
    semantic: Tensor        # [T+1, B, D_s] 或 None
    actions: Tensor        # [T, B, A]
    rewards: Tensor        # [T, B]
    dones: Tensor          # [T, B]
    values: Tensor         # [T+1, B]
    log_probs: Tensor      # [T, B]
    advantages: Tensor    # [T, B]
    returns: Tensor        # [T, B]
```

**核心方法**:
```python
def insert(self, z_history, action, reward, done, value, log_prob, semantic=None):
    """插入一个时间步的数据"""
    self.z_history[self.step] = z_history
    self.step = (self.step + 1) % self.num_steps

def compute_returns(self, gamma=0.99, gae_lambda=0.95):
    """计算 GAE 优势函数和 TD 回报"""
    # A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
    # R_t = r_t + gamma * R_{t+1}

def feed_forward_generator(self, mini_batch_size):
    """生成 mini-batch 用于 PPO 更新"""
    yield {
        "z_history": ...,
        "actions": ...,
        "old_log_probs": ...,
        "advantages": ...,
        "returns": ...,
    }
```

---

### 5. PPOLearner (`ppo_learner.py`)

**用途**: PPO 训练逻辑封装

**核心方法**:
```python
class PPOLearner:
    def collect_experience(self, env, storage, num_steps=None):
        """与环境交互收集 rollout"""
        for step in range(num_steps):
            action, log_prob, _ = policy.act(obs_z, obs_s)
            result = env.step(action)
            storage.insert(...)
        storage.compute_returns(gamma, gae_lambda)
        return {"reward_mean": ..., "num_episodes": ...}

    def update(self, storage) -> PPOTrainingStats:
        """执行 PPO 更新"""
        for epoch in range(num_epochs):
            for batch in storage.feed_forward_generator(mini_batch_size):
                # 1. 计算新 log_prob 和 entropy
                new_log_prob, entropy = policy.evaluate_actions(...)
                ratio = exp(new_log_prob - old_log_prob)

                # 2. PPO 裁剪损失
                surr1 = ratio * advantages
                surr2 = clip(ratio, 1-eps, 1+eps) * advantages
                policy_loss = -mean(min(surr1, surr2))

                # 3. Value 损失
                value_loss = MSE(value_net(batch_z), batch_returns)

                # 4. 熵正则
                entropy_loss = -mean(entropy)

                # 5. 总损失
                loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

                optimizer.step()
```

---

## 训练流程

### 完整训练循环

```python
for iteration in range(num_iterations):
    # 1. 收集经验
    collect_stats = learner.collect_experience(env, storage)

    # 2. PPO 更新
    train_stats = learner.update(storage)

    # 3. 日志和保存
    if iteration % eval_every == 0:
        print(f"Iter {iteration} | Reward={reward:.3f} | PolicyLoss={policy_loss:.4f} ...")

    if iteration % save_every == 0:
        learner.save_checkpoint(f"checkpoint_{iteration:06d}.pt")
```

### 运行命令

```bash
# 默认配置（使用真实 latent cache）
uv run python src/rl/train_rl.py

# 自定义训练
uv run python src/rl/train_rl.py \
  rl.num_iterations=100 \
  rl.num_envs=16 \
  rl.num_steps=128 \
  rl.num_epochs=10

# 完整配置示例
uv run python src/rl/train_rl.py \
  rl.num_iterations=1000 \
  rl.num_envs=16 \
  rl.num_steps=128 \
  rl.lr=3e-4 \
  rl.epsilon=0.2 \
  rl.gamma=0.99 \
  rl.gae_lambda=0.95 \
  env.action_dim=3 \
  rl.hidden_dim=256 \
  rl.num_layers=4
```

---

## 配置说明

### `configs/rl_default.yaml`

```yaml
# === RL 算法配置 ===
rl:
  algorithm: ppo

  # 并行环境数 (数据收集并行度)
  num_envs: 8

  # 每次收集的步数 (rollout 长度)
  num_steps: 64

  # PPO 超参数
  gamma: 0.99           # 折扣因子
  gae_lambda: 0.95      # GAE 参数
  epsilon: 0.2           # PPO 裁剪范围
  value_coef: 0.5       # Value loss 权重
  entropy_coef: 0.01    # 熵正则权重
  lr: 3e-4              # 学习率

  # 训练控制
  num_iterations: 1000
  eval_every: 10
  save_every: 50

# === 环境参数 ===
env:
  # 对于 cfm_dinov2m: num_patches=16, token_dim=384
  # 对于 lewm: num_patches=16, token_dim=32
  num_patches: 16
  token_dim: 32
  action_dim: 3
  history_len: 4

# === 数据源 ===
# latent cache 路径 (需要是 .pt 文件)
manifest_path: datasets/ai2thor/2026-04-22_14-50-39/manifest.latents.cfm_dinov2m.pt
latent_cache_dir: datasets/ai2thor/2026-04-22_14-50-39/manifest.latents.cfm_dinov2m.pt
```

---

## 数据格式

### Latent Cache 格式

```python
# 位置: datasets/ai2thor/{split}/{run}/2026-04-24_14-47-16.latents.{wm_name}.pt
# 示例: datasets/ai2thor/val/2026-04-24_14-47-16/2026-04-24_14-47-16.latents.cfm_dinov2m.pt

{
    "latents": {
        "datasets/ai2thor/train/2026-04-24_14-47-16/images/floorplan14_ep0084_step0117.png": tensor([16, 384]),
        ...
    },
    "latent_dim": 6144
}
```

### Episode 结构

图像路径自动解析为 episodes:
```
floorplan14_ep0084_step0117.png
  -> episode_key = "floorplan14_ep0084"
  -> step = 117
```

---

## 已验证结果

**测试配置**:
- cfm_dinov2m latent: 16 patches × 384 tokens = 6144 维度
- 200 episodes, 40000 latent entries
- 4 envs, 32 steps/iteration, 2 iterations

**训练输出**:
```
[2026-04-26 21:16:11,995][INFO] - 加载了 40000 个 latent cache 条目, 200 个 episodes
[2026-04-26 21:16:13,642][INFO] - Policy 参数: 3391494, Value 参数: 1811457
[2026-04-26 21:16:17,231][INFO] - Iter 1 | Reward=0.000 | PolicyLoss=0.0110 | ValueLoss=0.0474 | EntropyLoss=-5.7557
[2026-04-26 21:16:19,875][INFO] - Checkpoint saved to models/rl/rl_ppo_default/20260426_211609/checkpoint_final.pt (step=2)
[2026-04-26 21:16:19,876][INFO] - 训练完成！总计 2 iterations, 256 steps, 6.2 秒
```

---

## 已知限制与待完善

### 当前简化

1. **奖励函数**: 使用简化的动作惩罚 reward = -||action|| * 0.01
   - 待实现: 基于任务成功、WM 预测误差、语义一致性的真实奖励

2. **VLM 集成**: 未启用 semantic_dim
   - 待实现: VLM Adapter 提供语义特征，reward_shaping 模块

3. **WM 预测**: 未使用 WM 预测下一 latent
   - 待实现: WM-conditioned RL 奖励塑形

4. **GRPO 支持**: 仅实现 PPO
   - 待实现: GRPOLearner 支持组内相对优势更新

### 未来扩展

```python
# 1. 真实奖励塑形
class RewardShaper:
    def compute(self, obs_z, obs_s, action, pred_z_next, gt_z_next):
        r_task = self.task_reward_fn(action, ...)       # 任务奖励
        r_wm = cosine_sim(pred_z_next, gt_z_next)       # WM 一致性
        r_semantic = vlm_judgment_reward(...)            # 语义奖励
        return r_task + 0.2 * r_wm + 0.1 * r_semantic

# 2. System 1/2 切换
class SystemSwitch:
    def select_action(self, obs_z, obs_s):
        uncertainty = estimate_divergence(self.wm, ...)
        if uncertainty > threshold:
            return self.vlm_deliberate(obs_z, obs_s)  # System 2
        return self.pm_fast(obs_z, obs_s)             # System 1

# 3. WM 在线更新
if global_step > wm_freeze_steps:
    wm_optimizer.step()
```

---

## 参考资料

1. Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
2. Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
3. Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis"
4. DeepSeek, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs" - GRPO

---

## 附录: 关键公式

### PPO 裁剪损失
$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]$$

### GAE 优势函数
$$A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

### Value 函数损失
$$\mathcal{L}^{VF}(\phi) = \mathbb{E}_t\left[(V_\phi(s_t) - \hat{R}_t)^2\right]$$

### 熵正则
$$\mathcal{L}^{ENT} = -\beta \cdot H(\pi_\theta)$$