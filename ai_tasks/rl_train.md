# Phase 5 RL 训练计划

**日期**: 2026-04-26
**状态**: 规划中

---

## 背景与目标

### 当前架构（Phase 2-3）
- **WM**（World Model）: CFM/LeWM，基于 latent 的世界模型
  - CFM: `src/wm/predictor/cfm.py` - 条件流匹配，预测速度场 v_θ
  - LeWM: `src/wm/predictor/lewm.py` - 自回归 Transformer，预测下一 latent
- **IDM**（Inverse Dynamics Model）: `src/wm/inverse_dynamics.py` - 从 latent 序列预测动作
- **ActionMapper**: `src/wm/action_mapper.py` - 动作空间映射 MLP
- **SIGReg**: `src/wm/sigreg_modules.py` - 分布正则化损失
- **Uncertainty**: `src/wm/uncertainty.py` - 散度估计用于不确定度计算
- **VLM Adapter** (Phase 3): 将 WM latent 对齐到 VLM 语义空间

### Phase 5 目标
使用 PPO/GRPO 联合优化 VLM + WM + PM，实现：
1. WM 预测准确率提升（latent 重建质量）
2. PM 策略在物理环境中的成功率
3. VLM 提供高层语义指导，PM 提供快速反应

---

## 核心算法

### 5.1 PPO（Proximal Policy Optimization）

**优势函数估计**：
$$A_t = \sum_{i=0}^{\infty} \gamma^i r_{t+i} - V_\phi(s_t)$$

**PPO 裁剪损失**：
$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$。

**Value Function 损失**：
$$\mathcal{L}^{VF}(\phi) = \mathbb{E}_t\left[(V_\phi(s_t) - \hat{R}_t)^2\right]$$

**熵正则**：
$$\mathcal{L}^{ENT} = -\beta \cdot H(\pi_\theta)$$

### 5.2 GRPO（Group Relative Policy Optimization）

**参考 DeepSeek GRPO**：
$$\mathcal{L} = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(\frac{\pi_\theta(o_i)}{\pi_{ref}(o_i)}A_i,\text{clip}(\cdot)A_i\right) - \lambda \cdot D_{KL}(\pi_\theta||\pi_{ref})\right]$$

其中 $G$ 是每个 prompt 的采样组数，$A_i$ 是组内相对优势。

**优势计算**：
$$A_i^{GRPO} = \frac{r_i - \mu_G}{\sigma_G}$$

其中 $\mu_G, \sigma_G$ 是组内均值和标准差。

### 5.3 WM--conditioned RL

**状态表示**：
$$s_t = [\text{WM\_latent}; \text{VLM\_semantic}] = [z_t; s_t^{VLM}]$$

**奖励塑形**：
$$r_t = r_t^{task} + \alpha_1 \cdot r_t^{WM} + \alpha_2 \cdot r_t^{semantic}$$

其中：
- $r_t^{task}$: 任务完成奖励
- $r_t^{WM}$: WM 预测一致性奖励（预测 latent vs 真实 latent 的相似度）
- $r_t^{semantic}$: 语义一致性奖励（VLM 判断结果与目标的对齐程度）

### 5.4 不确定度感知切换

**System 1/2 架构**：
```
if uncertainty(z_t) > threshold:
    action = VLM_deliberate(z_t, s_t)  # System 2: 慢速深思
else:
    action = PM_fast(z_t, s_t)          # System 1: 快速反应
```

**不确定度估计**（来自 `uncertainty.py`）：
```python
divergence = estimate_divergence(wm, z_history, action_history, noise_scale=0.1, num_samples=8)
uncertainty = divergence.mean()
```

---

## 关键配置

### 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rl.algorithm` | "ppo" | 算法选择：ppo / grpo |
| `rl.gamma` | 0.99 | 折扣因子 |
| `rl.lambda_` | 0.95 | GAE 参数 |
| `rl.epsilon` | 0.2 | PPO 裁剪范围 |
| `rl.value_coef` | 0.5 | Value loss 系数 |
| `rl.entropy_coef` | 0.01 | 熵正则系数 |
| `rl.wm_reward_weight` | 0.2 | WM 一致性奖励权重 |
| `rl.semantic_reward_weight` | 0.1 | 语义奖励权重 |
| `rl.num_envs` | 16 | 并行环境数 |
| `rl.num_steps` | 128 | 每次收集的步数 |
| `rl.mini_batch_size` | 64 | Mini batch 大小 |
| `rl.num_epochs` | 10 | 每次收集后的更新轮数 |
| `rl.max_grad_norm` | 0.5 | 梯度裁剪 |
| `rl.uncertainty_threshold` | 0.5 | System 1/2 切换阈值 |
| `rl.wm_freeze_steps` | 1000 | WM 预热步数 |

### GRPO 特定参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rl.grpo.group_size` | 4 | 每组采样数 |
| `rl.grpo.kl_weight` | 0.01 | KL 散度惩罚 |
| `rl.grpo.adv_normalize` | true | 组内优势归一化 |

---

## 实现计划

### 5.1 数据结构定义

**RL Experience Batch**：
```python
@dataclass
class RLExperienceBatch:
    """RL 训练experience batch。"""
    obs_z: Tensor[B, H, P, D]      # WM latent 历史
    obs_s: Tensor[B, D_s]          # VLM 语义状态
    actions: Tensor[B, A]          # 动作
    rewards: Tensor[B]              # 奖励
    dones: Tensor[B]                # 完成标志
    values: Tensor[B]               # Value 估计
    log_probs: Tensor[B]           # 旧策略 log prob
    hidden_states: Optional[Tensor]# RNN 隐藏状态
```

**PM-ready 数据格式**（`export_pm_ready_features.py`）：
```python
# 导出字段
{
    "z_t": latent_tensor,           # [P, D] 当前 latent
    "s_t": semantic_tensor,        # [D_s] VLM 语义特征
    "a_t": action_tensor,          # [A] 动作向量
    "z_next": next_latent_tensor,  # [P, D] 下一 latent
    "reward": float,
    "done": bool,
}
```

### 5.2 模块实现

#### 5.2.1 `src/rl/policy_model.py` - PM 策略模型

基于 `InverseDynamicsModel` 扩展，支持：
- 接收 `[z_t; s_t]` 作为输入
- 输出动作分布 $\pi(a_t | z_t, s_t)$
- 支持 Gumbel-Softmax（离散动作）或 Gaussian（连续动作）

**架构**：
```
Input: [z_t; s_t] -> [B, P*D + D_s]
  -> Linear -> Hidden
  -> 6x TransformerBlock (AdaLN-Zero)
  -> Linear -> action_dim * 2 (mean, std)
```

#### 5.2.2 `src/rl/ppo_learner.py` - PPO 训练器

**核心接口**：
```python
class PPOLearner:
    def collect_experience(self, envs: VecEnv, policy: PM, wm: WM, vlm: VLM) -> RLExperienceBatch:
        """与环境交互收集经验。"""

    def update(self, batch: RLExperienceBatch, policy: PM, critic: ValueNetwork) -> dict:
        """执行 PPO 更新。"""

    def compute_advantages(self, rewards: Tensor, values: Tensor, dones: Tensor) -> tuple[Tensor, Tensor]:
        """计算 GAE 优势函数和回报。"""
```

#### 5.2.3 `src/rl/grpo_learner.py` - GRPO 训练器

```python
class GRPOLearner:
    def update(self, group_batch: dict, policy: PM) -> dict:
        """组内相对优势更新。"""
        # 1. 计算组内 reward 均值和标准差
        # 2. 计算组内相对优势 A_i^GRPO
        # 3. 执行策略更新（带 KL 约束）
        # 4. 返回训练指标
```

#### 5.2.4 `src/rl/reward_shaping.py` - 奖励塑形

```python
class RewardShaper:
    """组合多种奖励信号。"""

    def __init__(
        self,
        task_reward_fn: Callable,      # 任务奖励函数
        wm_reward_weight: float = 0.2,
        semantic_reward_weight: float = 0.1,
        uncertainty_threshold: float = 0.5,
    ):
        ...

    def compute(
        self,
        obs_z: Tensor,
        obs_s: Tensor,
        action: Tensor,
        pred_z_next: Tensor,
        gt_z_next: Tensor,
        vlm_judgment: Optional[str] = None,
    ) -> Tensor:
        """计算组合奖励。"""
        r_task = self.task_reward_fn(action, ...)
        r_wm = cosine_sim(pred_z_next, gt_z_next)  # WM 一致性
        r_semantic = self._semantic_reward(vlm_judgment) if vlm_judgment else 0
        return r_task + self.wm_reward_weight * r_wm + self.semantic_reward_weight * r_semantic
```

#### 5.2.5 `src/rl/system_switch.py` - System 1/2 切换

```python
class SystemSwitch:
    """不确定度感知的 System 1/2 切换。"""

    def __init__(
        self,
        wm: WM,
        pm: PM,
        vlm_adapter: VLMAdapter,
        uncertainty_threshold: float = 0.5,
    ):
        ...

    @torch.no_grad()
    def select_action(
        self,
        obs_z: Tensor,
        obs_s: Tensor,
        deterministic: bool = False,
    ) -> tuple[Tensor, bool]:
        """
        返回: (action, used_vlm)
        - used_vlm=True 表示使用 VLM（System 2）
        - used_vlm=False 表示使用 PM（System 1）
        """
        uncertainty = estimate_divergence(self.wm, ...)
        if uncertainty > self.uncertainty_threshold:
            action = self._vlm_action(obs_z, obs_s)
            return action, True
        else:
            action = self._pm_action(obs_z, obs_s, deterministic)
            return action, False
```

### 5.3 训练流程

#### Phase 5.1: PM 基础训练（Behavior Cloning）

使用 `export_pm_ready_features.py` 导出的数据，先做行为克隆：
```bash
uv run python src/train/train_pm.py \
    data.manifest_path=data/phase4/pm_train_manifest.json \
    train.batch_size=128 \
    train.lr=1e-4 \
    train.num_epochs=20
```

#### Phase 5.2: PPO 联合训练

```bash
uv run python src/train/train_rl.py \
    rl.algorithm=ppo \
    rl.num_envs=16 \
    rl.num_steps=128 \
    rl.num_epochs=10 \
    rl.wm_freeze_steps=1000 \
    wm.ckpt_path=models/wm/cfm_dinov2m/latest/checkpoint.pt \
    pm.ckpt_path=models/pm/bc_pretrain/latest/checkpoint.pt
```

**训练循环**：
```
for iteration in range(max_iterations):
    # 1. 收集经验
    for _ in range(num_steps_per_iteration):
        actions, values, log_probs = policy.act(obs_z, obs_s)
        pred_z_next = wm.predict(z_history, action)
        rewards = reward_shaper.compute(obs_z, obs_s, actions, pred_z_next, gt_z_next)
        envs.step(actions)

    # 2. 计算优势
    advantages, returns = compute_gae(rewards, values, dones)

    # 3. PPO 更新
    for _ in range(num_epochs):
        for batch in get_mini_batches(experience, mini_batch_size):
            policy_loss, value_loss, entropy = ppo_update(policy, critic, batch)
            optimizer.step()

    # 4. WM 在线更新（可选）
    if global_step > wm_freeze_steps:
        wm_optimizer.step()
```

#### Phase 5.3: GRPO 对比训练

```bash
uv run python src/train/train_rl.py \
    rl.algorithm=grpo \
    rl.grpo.group_size=4 \
    rl.grpo.kl_weight=0.01 \
    rl.num_envs=8
```

### 5.4 环境接口

**VecEnv 抽象**（支持并行数据收集）：
```python
class LatentVecEnv:
    """基于预编码 latent 的向量化环境。"""

    def __init__(self, manifest_path: str, wm: WM, batch_size: int):
        ...

    def reset(self) -> dict:
        """返回初始观察。"""
        return {"obs_z": z_0, "obs_s": s_0}

    def step(self, actions: Tensor) -> tuple[dict, Tensor, Tensor, dict]:
        """执行动作，返回 (obs, reward, done, info)。"""
        # 1. 更新 latent history
        # 2. 使用 WM 预测下一 latent
        # 3. 获取 VLM 语义状态
        # 4. 计算奖励
        obs = {"obs_z": z_t, "obs_s": s_t}
        return obs, reward, done, info
```

---

## 关键指标

| 指标 | 目标 | 说明 |
|------|------|------|
| Episode Return | > baseline + 20% | 相比 BC baseline 提升 |
| WM Prediction MSE | < 0.01 | 预测 latent 误差 |
| Policy Entropy | > 1.0 | 探索程度 |
| KL(π || π_ref) | < 0.05 | 策略偏离度 |
| VLM Switch Rate | 5-15% | System 2 调用频率 |
| Value Loss | < 0.1 | Critic 预测精度 |

---

## 文件结构

```
src/rl/
├── __init__.py
├── policy_model.py      # PM 策略网络
├── value_network.py     # Value 函数网络
├── ppo_learner.py       # PPO 训练器
├── grpo_learner.py      # GRPO 训练器
├── reward_shaping.py    # 奖励塑形
├── system_switch.py     # System 1/2 切换
├── vec_env.py           # 向量化环境
├── storage.py           # Experience Replay
└── train_rl.py          # 训练入口

configs/rl/
├── ppo_default.yaml     # PPO 默认配置
├── grpo_default.yaml    # GRPO 默认配置
└── reward_shaping.yaml  # 奖励权重配置
```

---

## 已实现清单

- [x] `src/rl/policy_model.py` - PM 策略网络
- [x] `src/rl/value_network.py` - Value 函数网络
- [x] `src/rl/ppo_learner.py` - PPO 训练器
- [x] `src/rl/vec_env.py` - 向量化环境
- [x] `src/rl/storage.py` - Experience Replay
- [x] `src/train/train_rl.py` - RL 训练入口
- [x] `configs/rl_default.yaml` - 默认配置
- [x] `src/rl/joint_trainer.py` - WM + PM + VLM 联合训练器
- [x] `src/rl/train_joint.py` - 联合训练入口
- [x] `configs/rl_joint.yaml` - 联合训练配置
- [x] `dev/test_joint_train.py` - 联合训练测试脚本

## 待实现清单

- [ ] `src/rl/grpo_learner.py` - GRPO 训练器
- [ ] `src/rl/reward_shaping.py` - 奖励塑形
- [ ] `src/rl/system_switch.py` - System 1/2 切换
- [ ] `src/train/train_pm.py` - PM 行为克隆训练
- [ ] `configs/rl/grpo_default.yaml`
- [ ] `export_pm_ready_features.py` - 导出 PM-ready 数据

---

## 参考资料

1. Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
2. DeepSeek, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs" - GRPO
3. Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" - 熵正则
4. Janner et al., "Planning with Diffusion for Flexible Behavior Synthesis" - Diffusion RL
