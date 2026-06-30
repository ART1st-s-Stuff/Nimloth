# RL 训练管线

在线 RL 训练：Qwen policy 与环境交互采集轨迹 → Qwen 编码 latent state → 训练 WM predictor + value head。

## 运行模式

| 模式 | `--env-url` | `--use-jsonl-rollout` | 适用 |
|------|-------------|----------------------|------|
| 单 GPU 在线 | 需要 | 否 | `world == 1` 调试 |
| JSONL 离线 | 不需要 | 是 | 分布式 FSDP；独立 rollout 生成 JSONL |

**分布式/FSDP (`world > 1`) 训练禁止使用 `EnvRolloutCollector`**。trainer 会在启动时检测并报错，要求使用 `--use-jsonl-rollout --jsonl-sources <路径>`。

JSONL collector 支持从指定文件/目录读取轨迹，按 iteration 轮转消费（数据耗尽自动循环），所有 rank 得到相同轨迹序列，保证 FSDP forward 触碰次数一致。Batch 选择使用 per-iteration 确定性 generator (`seed + iteration`)。

## 总览

```
┌─────────────────────────────────────────────────────────┐
│                     Online RL Loop                       │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ Rollout  │ →  │ Encode   │ →  │ Train             │  │
│  │ Qwen+Env │    │ Qwen→WM  │    │ predictor+value   │  │
│  └──────────┘    └──────────┘    └──────────────────┘  │
│       ↑                                   │             │
│       └───────────────────────────────────┘             │
│              checkpoint / resume                         │
└─────────────────────────────────────────────────────────┘
```

## 算法

### 初始化

```
Input:
  - Qwen model θ_qwen (base + optional LoRA/full-tune LLM or vision)
  - StateProjector f_proj : R^d_qwen → R^d_wm
  - LatentWMPredictor f_pred : R^d_wm × {0..7} → R^d_wm
  - ValueHead V : R^d_wm → R^8
  - Environment env (AI2-THOR navigation)
  - Hyperparams: γ (discount), N_iter, K_envs, T_max, B, S_train

Freeze:
  - θ_qwen:   requires_grad = False  (unless --llm-tune full or --vision-tune full)
  - f_proj:   requires_grad = False  (unless freeze.state_proj = false)

Trainable:
  - f_pred:   requires_grad = True
  - V:        requires_grad = True
```

### Online RL Loop

```
for iteration = 1, 2, …, N_iter:

    # ============================================================
    # Phase 1: Rollout — collect trajectories
    # ============================================================
    trajectories = []

    for env_i = 1, 2, …, K_envs (parallel):
        env.reset(seed_i)
        obs_0 = initial observation (image)
        τ = []   # trajectory for this episode

        for step t = 0, 1, …, T_max-1:
            # Qwen policy: action prior from <|action_start|> logits
            prompt_t = build_prompt(obs_0..obs_t, action_0..action_{t-1})
            logits = θ_qwen(prompt_t)              # Qwen forward
            a_t = argmax(logits at <|action_start|> position)
                   restricted to 8 action tokens

            obs_{t+1}, r_t, done = env.step(a_t)
            τ.append((obs_t, a_t, r_t))

            if done: break

        # Episode-level sparse reward
        R = env.compute_reward() if done else 0

        trajectory = {
            image_paths:  [save(obs_0), …, save(obs_T)],  # T = len(τ)
            action_indices: [a_0, …, a_{T-1}],
            reward: R,
            success: done ∧ (distance < threshold),
            messages: [system, user_0, assistant_0, …, user_{T-1}, assistant_{T-1}],
        }
        trajectories.append(trajectory)

    # ============================================================
    # Phase 2: Encode — extract WM latent states
    # ============================================================
    transitions = []

    for each trajectory in trajectories:
        # Encode each frame independently through Qwen
        hiddens = []  # T+1 vectors in R^d_qwen
        for img in trajectory.image_paths:
            h = θ_qwen.encode(img)                    # extract hidden state
            h_latent = h[<|latent_state|> position]   # Nimloth latent token
            hiddens.append(h_latent)

        # Compute discounted MC returns
        for t = 0, …, T-1:
            G_t = trajectory.reward · γ^(T-1-t)      # MC target

        # Build transitions
        for t = 0, …, T-1:
            transitions.append({
                qwen_hidden_current: hiddens[t],      # ∈ R^d_qwen
                qwen_hidden_next:    hiddens[t+1],    # ∈ R^d_qwen
                action_index:        a_t,             # ∈ {0..7}
                value_target:        G_t,             # ∈ R
            })

    # ============================================================
    # Phase 3: Train — update predictor + value head
    # ============================================================
    for train_step = 1, 2, …, S_train:
        batch = sample(transitions, B)   # random batch of size B

        for each (h_cur, h_next, a, G) in batch:

            # ---- Predictor loss (dynamics) ----
            s_cur  = f_proj(h_cur)                   # current WM state
            s_next = f_proj(h_next).detach()          # target: next WM state (no grad)
            ŝ_next = f_pred(s_cur, a)                # predicted next WM state

            L_pred = MSE(ŝ_next, s_next)

            # ---- Value loss (critic) ----
            values = V(s_cur)                         # ∈ R^8
            L_reg  = MSE(values[a], G)                # regression on taken action

            # Optional ranking loss: penalise when any unchosen action
            # scores higher than the chosen one
            max_other = max_{j≠a} values[j]
            L_rank = ReLU(rank_margin + max_other - values[a])
            L_value = L_reg + λ_rank · L_rank

            # ---- Total loss ----
            L = L_pred + L_value

        # Update
        optimizer.zero_grad()
        L.backward()
        clip_grad_norm(max_norm=1.0)
        optimizer.step()
        if vision_ema enabled:
            vision_ema.update(θ_qwen)

    # ============================================================
    # Phase 4: Checkpoint
    # ============================================================
    if iteration % save_interval == 0:
        save_checkpoint(
            state_proj, wm_predictor, value_head,
            model=θ_qwen (full or LoRA adapter),
            optimizer, iteration, global_step,
        )
    if value_loss improved:
        save_checkpoint(best/)
```

### 推理：Slow Path / Fast Path

训练后的 predictor 和 value head 可用于加速推理（见 `agent/` 模块）：

```
slow_path_steps = 4   # Qwen re-sync interval

agent.reset(first_image):
    s_wm = f_proj(θ_qwen.encode(first_image)[<|latent_state|>])
    steps_since_sync = 0

agent.act(current_image):   # called at each env step
    if steps_since_sync >= slow_path_steps:
        # Slow path: re-align WM state with real observation
        s_wm = f_proj(θ_qwen.encode(current_image)[<|latent_state|>])
        steps_since_sync = 0
    else:
        # Fast path: predict next state without Qwen
        s_wm = f_pred(s_wm, previous_action)
        steps_since_sync += 1

    action = planner.select_action(s_wm)   # greedy or beam_search
    previous_action = action
    return action
```

## 模块

| 文件 | 职责 |
|------|------|
| `rollout.py` | `RolloutTrajectory` 数据结构，`EnvRolloutCollector`（单 GPU 在线），`JSONLRolloutCollector`（离线/分布式，支持多源文件轮转） |
| `loss.py` | `compute_predictor_loss`（MSE dynamics），`compute_value_loss`（MSE + ranking），`compute_advantages`（unbiased=False，避免 batch size=1 NaN），`compute_actor_loss`（PPO clipped） |
| `trainer.py` | `train_rl` — 在线 RL 主循环，含 Qwen 加载、FSDP/DDP、resume、分布式 guard |
| `checkpoint.py` | `save_rl_checkpoint` / `load_rl_wm_checkpoint` / `load_lora_adapter_state` |
| `cli.py` | 命令行入口，`--llm-tune` / `--vision-tune` / `--resume` / `--jsonl-sources` 等参数 |

## 入口

```bash
# 基本用法
python -m nimloth.training.rl.cli \
  --config configs/training/rl/defaults.yaml \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --output-dir outputs/experiments/training/rl/2026-06-27/test

# SFT2 warm-start（LLM LoRA + Vision Full）
python -m nimloth.training.rl.cli \
  --config configs/training/rl/sft2_warmstart.yaml \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --llm-tune lora --vision-tune full \
  --wm-checkpoint outputs/.../sft2/.../best/wm_predictor \
  --state-proj-checkpoint outputs/.../sft2/.../best/state_proj.pt \
  --value-head-checkpoint outputs/.../sft2/.../best/value_head \
  --output-dir outputs/experiments/training/rl/.../test

# Resume
python -m nimloth.training.rl.cli \
  --config configs/training/rl/sft2_warmstart.yaml \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --llm-tune lora --vision-tune full \
  --resume \
  --output-dir outputs/experiments/training/rl/.../existing_run
```

## Loss 函数

### Predictor Loss（Dynamics）

```
L_pred = MSE(ŝ_{t+1}, s_{t+1})

where:
  s_{t+1} = f_proj(θ_qwen.encode(img_{t+1})[<|latent_state|>])  # target (no grad)
  ŝ_{t+1} = f_pred(s_t, a_t)                                      # prediction
  s_t     = f_proj(θ_qwen.encode(img_t)[<|latent_state|>])
```

### Value Loss（Critic）

```
L_reg  = MSE(V(s_t)[a_t], G_t)

L_rank = ReLU(rank_margin + max_{j≠a_t} V(s_t)[j] - V(s_t)[a_t])

L_value = L_reg + λ_rank · L_rank

where:
  G_t = R · γ^(T-1-t)    # Monte Carlo return (sparse terminal reward)
```

## 配置参考

```yaml
# configs/training/rl/defaults.yaml

freeze:
  qwen: true             # True = no gradients for Qwen (overridden by --llm-tune)
  state_proj: true       # True = StateProjector frozen

predictor:
  lr: 1e-3
  emb_dim: 128           # WM embedding dimension
  history_size: 4        # ARPredictor context window (frames)
  rollout_steps: 1       # training rollout steps (1 = single-step first)

value_head:
  lr: 1e-3
  rank_margin: 0.1       # margin for ranking loss
  lambda_rank: 0.0       # 0 = regression only; > 0 enables ranking term

rl:
  iterations: 1000         # number of online RL iterations
  envs_per_iteration: 8    # parallel environments per iteration
  max_steps_per_episode: 20
  gamma: 0.99              # discount factor
  batch_size: 32
  train_steps_per_iteration: 10

training:
  seed: 42
  log_interval: 10         # log every N iterations
  save_interval: 50        # checkpoint every N iterations
```

## 调参模式

通过 CLI 控制（覆盖 config 中的 freeze 设置）：

| `--llm-tune` | `--vision-tune` | 说明 |
|-------------|----------------|------|
| `freeze` | `freeze` | Qwen 全冻结，仅 forward（默认） |
| `lora` | `freeze` | LLM LoRA，vision 冻结 |
| `freeze` | `full` | LLM 冻结，vision 全参数 |
| `lora` | `full` | LLM LoRA + vision 全参数（SFT2 常用） |
| `full` | `freeze` | LLM 全参数，vision 冻结 |

`--lora` 是 `--llm-tune lora --vision-tune freeze` 的快捷方式。

## Checkpoint 结构

```
best/
├── config.json                  # Qwen (full HF) 或 adapter_model.safetensors (LoRA)
├── state_proj.pt                # StateProjector weights
├── wm_predictor/                # LatentWMPredictor (predictor.pt + config.json)
├── value_head/                  # ValueHead (value_head.pt)
├── vision_ema.pt                # VisionEncoderEMA shadow (可选)
├── rl_state.pt                  # {iteration, global_step, best_value_loss, optimizer}
└── processor/                   # tokenizer + image processor files
```

## 与 SFT2 的关系

- **SFT2**：离线监督训练，从已有 rollout JSONL 学习 WM dynamics + value。数据不来自在线交互。
- **RL**：在线 RL，持续采集新数据、持续训练。是 SFT2 的后续阶段。
- 二者共享 WM 模块（`LatentWMPredictor`、`ValueHead`、`StateProjector`）、公共训练工具（`dist`、`metrics`、`logging`），以及 Qwen 调参基础设施（`configure_qwen_tuning`）。
- SFT2 的 `best/` checkpoint 可作为 RL 的 warm-start（通过 `--wm-checkpoint` 等参数）。
