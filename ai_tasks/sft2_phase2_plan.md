--------
本文为 AI 起草、供人类审阅。如需修改需要得到人类同意。
--------

# SFT2 阶段 2 实现计划（Qwen ↔ LeWM 对齐）

对应任务：`ai_tasks/sft2_exp.md`

## 目标

在 SFT1 checkpoint 基础上，让 Qwen 在 `<|latent_state|>` 位置的 hidden state 含有足够信息，使 **冻结的 LeWM predictor** 能预测下一步 state embedding。主 loss 为 predictor MSE；同时保留 Qwen 的格式与任务能力。

## 总体阶段

| 阶段 | 内容 | 代码位置（计划） |
|------|------|------------------|
| 1 | 在 navigation rollout 像素数据上预训练 LeWM | `src/nimloth/wm/` + `experiments/navigation_baseline/pretrain_lewm_navigation.py` |
| 2 | 冻结 LeWM，用 WM MSE + 辅助 CE 训练 Qwen + `state_proj` | `src/nimloth/sft2/` + `experiments/navigation_baseline/train_sft2_qwen25vl.py` |
| 3（可选，暂缓） | 极小 LR 联合微调 LeWM 子模块 | 待阶段 2 指标稳定后再定 |

**原则：新逻辑尽量放在 `src/nimloth/`，`external/VAGEN` 仅保留 eval/rollout 必需的最小改动。**

## 阶段 1：LeWM 预训练（navigation）

### 数据

- 来源：SFT1 rollout 转换后的 train split（`convert_sft1_rollouts_to_nimloth.py` 产物）。
- 样本单位：transition `(o_t, a_t, o_{t+1})`。
  - `o_t`, `o_{t+1}`：`image_paths` 中相邻观测帧（LeWM 专用 resize/normalize）。
  - `a_t`：`action_indices[t]`，8 类离散动作 → 固定 one-hot `[8]`（与阶段 2 一致）。
- Split：仅 train split；是否与 SFT1 一样只用 successful trajectory 待人类确认（默认可做 ablation）。

### 模型与 loss

- 架构：LeWM JEPA（`encoder` + `ARPredictor` + `action_encoder` + `projector`/`pred_proj`）。
- Loss（标准 LeWM）：
  - `pred_loss = MSE(predictor(z_t, φ(a_t)), z_{t+1})`
  - `sigreg_loss = SIGReg(encoder outputs)`
  - `loss = pred_loss + λ * sigreg_loss`
- 产物：`checkpoints/lewm_navigation/` 下完整 JEPA object checkpoint + config。

### 实现要点

- LeWM 代码可作为 `external/le-wm` submodule 或 vendored 最小子集；封装入口在 `src/nimloth/wm/lewm.py`。
- 图像预处理与 Qwen processor 分离，不共用 tensor pipeline。

## 阶段 2：Qwen 对齐（本计划重点）

### 初始化

- Qwen：`SFT1 best` checkpoint（`experiments/navigation_baseline/runs/.../best`）。
- LeWM：阶段 1 checkpoint，**全部 freeze**（`requires_grad=False`）。
- 新增可训练模块：`state_proj: ℝ^{d_qwen} → ℝ^{d_lewm}`（Linear 或小型 MLP）。

### 数据

- 同样从 rollout jsonl 展开 transition 三元组：
  - `prefix_messages`：到第 t 步 assistant 为止的多模态历史；
  - `action_index`；
  - `next_image_path`。
- 可选 replay：同一 batch 混合完整 trajectory（算 CE）与 transition（算 WM MSE）。

### 前向

1. Qwen2.5-VL forward，`output_hidden_states=True`。
2. 用 `nimloth.latent.extraction` 定位 `<|latent_state|>` index，取最后层 hidden `h_t`。
3. `z_t = state_proj(h_t)`，shape 对齐 LeWM embedding dim。
4. `act_emb = lewm.action_encoder(one_hot(a_t))`。
5. `pred_emb = lewm.predict(ctx_emb=z_t, act_emb)`（首版 `history_size=1`）。
6. `tgt_emb = stopgrad(lewm.encode(next_image))`。

### Loss

```text
wm_mse  = MSE(pred_emb, tgt_emb)
lm_ce   = assistant-span CE（与 SFT1 相同 mask 逻辑）
total   = λ_wm * wm_mse + λ_ce * lm_ce + λ_kl * KL(π_θ || π_sft1)   # KL 可选
```

**权重策略（默认）：**

- 起始：`λ_ce=1.0`, `λ_wm=0.1`
- 前 30% steps 将 `λ_wm` cosine 提升到 `1.0`
- 不对 Qwen latent 使用 LeWM 的 SIGReg

### 保能力措施

1. 必须从 SFT1 best 初始化。
2. 默认 **LoRA**（LLM 部分，`r=8~16`）+ 全参 `state_proj`；vision tower 首轮冻结。
3. 辅助 CE + 可选 trajectory replay。
4. 可选 KL anchor 到 SFT1（action token 位置）。
5. LR 不超过 SFT1；`state_proj` 可用略高 LR。

### 监控与 checkpoint

训练日志（CSV / wandb）至少包含：

| 指标 | 含义 |
|------|------|
| `train_wm_mse` / `val_wm_mse` | predictor 对齐 |
| `train_lm_ce` / `val_lm_ce` | 格式/语言 |
| `val_success_rate` | VAGEN `prompt_format=nimloth` eval（与 SFT1 同流程） |

**早停规则：** `val_success_rate` 下降超过阈值或 `val_lm_ce` 持续恶化时，降低 `λ_wm` 或停止。

Best checkpoint 按 `val_success_rate` 选取（MSE 仅作辅助）。

## `src/nimloth/` 模块划分（计划）

```text
src/nimloth/
  latent/          # 已有：latent state / action prior 提取
  wm/
    __init__.py
    lewm.py        # LeWM 封装：load, encode, predict, freeze
    dataset.py     # TransitionDataset from nimloth jsonl
    preprocess.py  # LeWM 图像预处理
  sft2/
    __init__.py
    loss.py        # compute_sft2_wm_loss, combined loss
    collate.py     # Qwen VL batch + transition metadata
    metrics.py     # MSE / CE logging helpers
```

`experiments/navigation_baseline/` 仅保留薄封装脚本（argparse、DDP launch、Slurm 入口），调用 `src/nimloth`。

## 与 VAGEN 的边界

| 放在 Nimloth `src/` | 留在 VAGEN（已有/最小改动） |
|----------------------|------------------------------|
| LeWM 封装、transition dataset、SFT2 loss | `nimloth_format.py` prompt 格式 |
| `state_proj`、LoRA 训练逻辑 | eval/rollout `prompt_format=nimloth` |
| latent extraction（已有） | success rate eval slurm 脚本 |

阶段 2 **不**在 VAGEN 训练 loop 内嵌 WM loss；独立 `train_sft2_qwen25vl.py` 调用 `src/nimloth`。

## 实验脚本（计划新增）

- `experiments/navigation_baseline/pretrain_lewm_navigation.py`
- `experiments/navigation_baseline/pretrain_lewm_navigation.slurm`
- `experiments/navigation_baseline/train_sft2_qwen25vl.py`
- `experiments/navigation_baseline/train_sft2_vagen79.slurm`（或复用 SFT1 资源模板）
- `experiments/navigation_baseline/eval_sft2_valtest.slurm`（复用 SFT1 eval，换 checkpoint 路径）

## 测试（计划）

- `tests/test_wm_transition_dataset.py`：jsonl → transition 展开、action/image 对齐。
- `tests/test_sft2_loss.py`：fake Qwen hidden + fake LeWM，验证 MSE 梯度回传到 `state_proj`。
- 复用 `tests/test_latent_extraction.py`。

## 风险与待确认

1. **LeWM 预训练数据量**：6060 条 rollout 展开 transitions 是否足够；可能需要数据增强或更长训练。
2. **Successful-only vs all train**：SFT1 训练用 success-only；SFT2 文档仅写 train set。
3. **显存**：每 transition 一次 Qwen forward；需 `grad_accum` 或 trajectory 级 batch 优化。
4. **阶段 3 联合微调**：默认不做，避免 LeWM 目标漂移。

## 实现顺序

1. `src/nimloth/wm/dataset.py` + tests
2. `src/nimloth/wm/lewm.py` + `pretrain_lewm_navigation.py`
3. `src/nimloth/sft2/loss.py` + `collate.py`
4. `train_sft2_qwen25vl.py` + Slurm
5. 接入现有 val success eval
