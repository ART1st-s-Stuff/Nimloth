# RL 训练管线

在线 RL 训练：Qwen policy 与环境交互采集轨迹 → Qwen 编码 latent state → 训练 WM predictor + value head。

## 模块

| 文件 | 职责 |
|------|------|
| `rollout.py` | 环境交互，采集 RolloutTrajectory |
| `loss.py` | Predictor MSE loss + ValueHead regression loss |
| `trainer.py` | 在线 RL 主循环：rollout → encode → train → checkpoint |
| `checkpoint.py` | 保存/加载 state_proj + predictor + value_head + optimizer |
| `cli.py` | 命令行入口与配置加载 |

## 入口

```bash
python -m nimloth.training.rl.cli --config configs/training/rl/defaults.yaml
```

覆盖输出目录：

```bash
python -m nimloth.training.rl.cli --config configs/training/rl/defaults.yaml --output-dir outputs/experiments/training/rl/2026-06-25/test
```

从已有 WM checkpoint 热启动：

```bash
python -m nimloth.training.rl.cli \
  --config configs/training/rl/defaults.yaml \
  --wm-checkpoint outputs/experiments/training/sft2/.../best/wm_predictor \
  --value-head-checkpoint outputs/experiments/training/sft2/.../best/value_head
```

## 训练流程

```
for iteration in 1..N:
    1. 采集 rollout（Qwen policy + VAGEN env）
    2. Qwen 编码每帧 → StateProjector → WM latent states
    3. 构建 transitions: (s_t, a_t, s_{t+1}, discounted_return_t)
    4. 训练 predictor + value head
    5. Checkpoint
```

## 冻结策略

通过 config 控制：

```yaml
freeze:
  qwen: true        # Qwen backbone（仅 forward，不训练）
  state_proj: true  # StateProjector（不训练）
```

Predictor 和 ValueHead 始终训练。

## 与 SFT2 的关系

- SFT2 是离线监督训练（从已有 rollout JSONL 学习 WM dynamics + value）。
- RL 训练管线是在线 RL（持续采集新数据、持续训练），是 SFT2 后续阶段。
- 二者共享 WM 模块（LatentWMPredictor、ValueHead、StateProjector）、公共训练工具（dist、metrics、logging）。
- SFT2 的 checkpoint 可作为 RL 的 warm-start。
