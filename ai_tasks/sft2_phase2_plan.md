--------
本文为 AI 起草、供人类审阅。人类已要求按 `ai_tasks/sft2_exp.md` 更新本计划（2026-06-17）。
--------

# Training 模块与 Phase 2（SFT2）实现计划

对应人类规格：`ai_tasks/sft2_exp.md`  
关联规格：`ai_tasks/sft1_exp.md`（Phase 1）

## 0. 架构原则（2026-06-17 修订）

**SFT2 不再作为独立顶层模块。** 所有阶段性训练逻辑归入统一 `training` 模块，按 **phase** 划分子包、配置与实验入口。

| 原则 | 说明 |
|------|------|
| 单一 `training` 模块 | `src/nimloth/training/` 承载 Phase 0/1/2 的训练逻辑；`sft2/` 等旧包迁移后删除 |
| Phase 分目录 | `experiments/training/phase{0,1,2}_*/` 放各阶段 Slurm/提交脚本；`configs/training/phase{0,1,2}_*/` 放默认超参 |
| 薄实验层 | 实验目录只保留 argparse、Slurm、submit；核心 loss/dataset/tuning 在 `src/` |
| VAGEN 边界 | `external/VAGEN` 仍跑 rollout/RL；**默认参数与 Nimloth 封装** 可放在 `configs/training/phase0_vagen/` |
| 渐进迁移 | `experiments/navigation_baseline/` 保留至各 phase 脚本迁完；旧路径用 README 指向新位置 |

### 目标目录树（计划）

```text
src/nimloth/training/
  __init__.py
  common/                 # 跨 phase：qwen_tuning, schedules, metrics, wandb helpers
  phase0_vagen/           # VAGEN 相关 Nimloth 封装（默认配置加载、rollout 后处理钩子）
  phase1_sft/             # SFT1：LM CE、assistant mask、LoRA 保存
  sft2/                   # SFT2：WM predictor、Value head、组合 loss、collate

configs/training/
  phase0_vagen/
    defaults.yaml         # VAGEN navigation 默认 env/train 参数（从现有 slurm 提炼）
  phase1_sft/
    qwen25vl_lora.yaml
    qwen25vl_full.yaml
  sft2/
    latent_wm_value.yaml  # 默认：LLM freeze + vision full + vision EMA

experiments/training/
  phase0_vagen/           # rollout / VAGEN train slurm（自 navigation_baseline 迁入）
  phase1_sft/             # train_sft1_*, convert_rollouts, eval slurm
  sft2/                   # train SFT2, pretrain predictor init, eval slurm

tests/
  training/
    phase1_sft/
    sft2/
```

### 自 `navigation_baseline` 迁移映射（计划，未一次性搬完）

| 现路径 | 目标 |
|--------|------|
| `train_sft1_qwen25vl.py` | `experiments/training/phase1_sft/train.py` |
| `train_sft2_qwen25vl.py` | `experiments/training/sft2/train.py` |
| `pretrain_lewm_navigation.py` | `experiments/training/sft2/pretrain_predictor.py`（可选 init） |
| `convert_sft1_rollouts_to_nimloth.py` | `experiments/training/phase1_sft/convert_rollouts.py` |
| `sft1_rollouts_*.slurm`, `train_sft1_*.slurm` | `experiments/training/phase1_sft/` |
| `train_sft2_*.slurm`, `submit_sft2_*.sh` | `experiments/training/sft2/` |
| `resume_retry2_*.slurm`, `dgx*_train_*.slurm` | `experiments/training/phase0_vagen/` |
| `src/nimloth/sft2/*` | `src/nimloth/training/sft2/*` + `common/qwen_tuning.py`（**已删除** shim） |

---

## 1. Phase 2 目标（对齐 `sft2_exp.md`）

在 **Phase 1 SFT1 checkpoint** 上继续训练，使：

1. **`<|latent_state|>` hidden** 含有足够信息，供 **LeWM ARPredictor**（仅用 predictor，不用 pixel encoder）预测 **下一步 Qwen latent**。
2. **Value head**：输入当前 state embedding，输出 **所有 action 的 value**。
3. **数据**：train split rollout，**包含失败 trajectory**（先对齐 Qwen 偏好；性能优化留给后续 RL）。
4. **监控**：WM predictor 曲线 + val success rate 曲线（CSV / wandb）。

### 与旧计划差异（必须遵守人类规格）

| 旧实现/计划 | 现行规格 |
|-------------|----------|
| `src/nimloth/sft2/` 独立包 | 迁入 `training/sft2/`（**已删除** 旧包） |
| LeWM encoder 作 WM target | **禁止**；target = 下一步 **Qwen latent**（`state_proj` 后 stop-grad） |
| `pretrain_step` / pixel JEPA loss | **移除**；仅 predictor + value losses |
| 仅 success rollout | **train 含失败 run** |
| 无 Value head | **新增** Value head + 排序损失 |
| LLM LoRA + vision 冻结 | **3×3**：LLM / vision 各 `freeze \| lora \| full`；vision 可选 **EMA** |
| 默认 LoRA LLM | **默认：LLM 冻结；vision 全量微调 + EMA** |

---

## 2. Phase 2 模型与前向

### 2.1 模块

| 模块 | 来源 | 训练默认 |
|------|------|----------|
| Qwen2.5-VL | Phase 1 init | 见 §3 参数矩阵 |
| `state_proj` | `ℝ^{d_qwen} → ℝ^{d_wm}` | 全参 |
| `LatentWMPredictor` | LeWM **ARPredictor** 子集 | 可训（可从 phase2 pretrain init） |
| `ValueHead` | 新建 `ℝ^{d_wm} → ℝ^{|A|}` | 全参 |

### 2.2 WM 前向（latent 空间）

1. Qwen forward（current prefix）→ `h_t` @ `<|latent_state|>`。
2. `s_t = state_proj(h_t)`。
3. `pred_{t+1} = WM_predictor(s_t, a_t)`。
4. Qwen forward（next prefix，`no_grad`）→ `h_{t+1}` → `target = stopgrad(state_proj(h_{t+1}))`。
5. `L_wm = MSE(pred_{t+1}, target)`。

仅当 transition 存在 **下一 assistant turn**（`next_prefix_messages`）时计算 `L_wm`。

### 2.3 Value head 前向与监督（待实现）

1. `v = ValueHead(s_t)`，shape `(B, |A|)`。
2. **监督 (1)**：真值 action value（来自 rollout reward / MC return；字段与归一化待人类确认）。
3. **监督 (2)**：排序损失 — Qwen 在 step `t` 选择的 `a_t` 的 value 应高于同 step 未选 action：
   - `L_rank = Σ_{a' ≠ a_t} max(0, margin + v[a'] - v[a_t])`（或 pairwise CE）。
4. `L_value = L_reg + λ_rank * L_rank`（`L_reg` 形式待确认：MSE to target value 或 Huber）。

### 2.4 辅助 CE

- Assistant-span LM CE（与 Phase 1 相同 mask），权重 `λ_ce`，防止格式遗忘。

### 2.5 总 loss

```text
L = λ_wm * L_wm + λ_value * L_value + λ_ce * L_lm_ce
```

`λ_wm`：cosine ramp（0.1 → 1.0，前 30% steps），与现实现一致。

---

## 3. Phase 2 参数矩阵（`sft2_exp.md`）

### 3.1 Qwen 子模块

| | freeze | lora | full |
|---|--------|------|------|
| **LLM backbone** | ✓ 默认 | 可选 | 可选 |
| **Vision encoder** | 可选 | 可选 | ✓ 默认 |

- 实现：`training/common/qwen_tuning.py`（自 `sft2/qwen_tuning.py` 迁入）。
- **Vision EMA**（仅当 vision 可训时）：shadow weights `θ_ema ← τ θ_ema + (1-τ) θ`；推理 / target 可选走 EMA（细节待实现时定）。

### 3.2 默认配置（`configs/training/phase2_align/latent_wm_value.yaml`）

```yaml
llm_tune: freeze
vision_tune: full
vision_ema: true
vision_ema_decay: 0.999
train_wm_predictor: true
include_failed_rollouts: true
lambda_wm_start: 0.1
lambda_wm_end: 1.0
lambda_ce: 1.0
lambda_value: 1.0
```

---

## 4. Phase 2 数据

- 来源：Phase 1 收集并转换的 Nimloth jsonl（`convert_rollouts` 产物）。
- Split：**train** 用于训练（**含失败**）；**val** 用于 success rate / WM / value 监控。
- Transition 展开：`wm/dataset.py`（共用）；`next_prefix_*` 供 next latent。
- Collate：`training/phase2_align/collate.py`。

---

## 5. 各 Phase 摘要

### Phase 0 — VAGEN baseline & rollout

- 人类流程见 `sft1_exp.md` 步骤 1。
- Nimloth 侧：`configs/training/phase0_vagen/defaults.yaml` 收录 navigation 默认 env、并行度、checkpoint 路径模板。
- 脚本：`experiments/training/phase0_vagen/`（从 `resume_retry2_*`, rollout slurm 迁入）。

### Phase 1 — 格式 SFT（SFT1）

- 人类流程见 `sft1_exp.md`。
- 逻辑：`training/phase1_sft/`（LM CE、checkpoint、LoRA merge 钩子）。
- 脚本：`experiments/training/phase1_sft/`。
- 配置：`configs/training/phase1_sft/*.yaml`。

### Phase 2 — WM + Value 对齐（SFT2）

- 人类流程见 `sft2_exp.md`。
- 逻辑：`training/phase2_align/`（predictor、value head、loss、collate）。
- 脚本：`experiments/training/phase2_align/`。
- 配置：`configs/training/phase2_align/*.yaml`。

### Phase 2 前置：Predictor 初始化（可选）

- 历史 `pretrain_lewm_navigation.py`（pixel JEPA）**不再作为 SFT2 主路径**。
- 仅可作为 `LatentWMPredictor` 权重 warm-start；SFT2 监督均在 Qwen latent 空间。

---

## 6. 监控、checkpoint、早停

| 指标 | 用途 |
|------|------|
| `train/val wm_mse` | WM predictor 对齐 |
| `train/val lm_ce` | 格式保持 |
| `train/val value_*` | Value regression / ranking |
| `val_success_rate` | VAGEN `prompt_format=nimloth` eval |

- Best checkpoint：**优先 `val_success_rate`**（MSE 为辅）。
- 日志：CSV + wandb；Phase 2 上传脚本归 `experiments/training/phase2_align/`。
- Checkpoint 内容：`adapter/full HF`、`state_proj.pt`、`wm_predictor/`、`value_head.pt`、`training_state.pt`。

---

## 7. 实现状态（2026-06-17，收尾更新）

| 项 | 状态 |
|----|------|
| Qwen latent WM loss（无 encoder） | ✅ `training/sft2/loss.py` + `predictor.py` |
| LLM / vision 分调 `freeze\|lora\|full` | ✅ `training/common/qwen_tuning.py` |
| `next_prefix` transition | ✅ `wm/dataset.py` |
| Value head + ranking loss | ✅ `training/sft2/value_head.py` + `loss.py` |
| Vision EMA | ✅ `training/common/vision_ema.py` |
| 含失败 rollout 训练 | ✅ 默认 `train_all.jsonl`，`--success-only` 可选 |
| YAML 配置加载 | ✅ `configs/training/sft2/latent_wm_value.yaml` + `cli.py` |
| wandb 训练内日志 | ✅ `training/common/wandb_logging.py` |
| best checkpoint 按 val_success_rate | ✅ `val_rollout_success_rate` + `early_stop_metric` |
| 脚本迁出 `navigation_baseline` | ✅ `experiments/training/sft2/`（**已删除** navigation_baseline 内 SFT2 冗余脚本/shim） |
| VAGEN 在线 val eval | ⏸ 离线 jsonl 成功率已接入；在线 VAGEN greedy eval 沿用 SFT1 slurm，可按需 wrapper |
| `training/phase2_align` 命名 | ✅ 统一为 `training/sft2` |

---

## 8. 实现顺序（修订）

1. **目录与配置骨架**：`src/nimloth/training/`、`configs/training/`、`experiments/training/`（✅）。
2. **代码迁移**：`sft2/*` → `training/sft2/` + `common/`；更新 import 与测试（✅）。
3. **Value head**：`sft2/value_head.py` + loss + tests（✅）。
4. **Vision EMA**：`common/vision_ema.py` + config 开关（✅）。
5. **数据**：`include_failed_rollouts`；确认 value 标签字段。
6. **实验脚本迁移**：`train_sft2` → `experiments/training/sft2/train.py`；Slurm/submit 同步迁（✅）。
7. **Phase 1 脚本迁移**（可并行）：`train_sft1` → `phase1_sft/`。
8. **Phase 0 默认配置**：从现有 VAGEN slurm 提炼 `phase0_vagen/defaults.yaml`。
9. Val success eval 接入 Phase 2 训练循环 / watcher（离线已接入）。

---

## 9. 风险与待人类确认

1. **真值 action value** 定义：step reward、折扣回报、还是 VAGEN 内部 value？字段名？
2. **失败 rollout** 是否与 success 混采比例需要上限？
3. **Vision EMA**：仅 encoder 还是含 merger？target 网络是否必须用 EMA？
4. **Phase 0 配置**：`defaults.yaml` 是否覆盖现有全部 retry2 slurm 变体，还是只文档化核心子集？
5. **迁移窗口**：`navigation_baseline` 内 SFT2 脚本已清理；SFT1/VAGEN 脚本仍保留至 phase0/1 迁移完成。

---

## 10. 测试（计划）

- `tests/training/sft2/test_sft2_loss.py`（自 `test_sft2_loss.py` 迁）
- `tests/training/sft2/test_value_ranking_loss.py`（新）
- `tests/training/common/test_schedules.py`（自 `test_sft2_schedules.py` 迁）
- `tests/test_wm_transition_dataset.py`（保留，测 `next_prefix`）
- `tests/training/common/test_qwen_tuning.py`
