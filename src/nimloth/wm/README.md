# World Model (`nimloth.wm`)

Nimloth 世界模型层：组合后的 `WorldModel`、LeWM predictor、state/value 头和规划。

LeWM 核心算子来自 `external/le-wm`，经 `wm/_vendor_lewm.py` 以最小子集 vendoring；Nimloth 不在运行时 import `external/le-wm` 脚本。

**当前状态**：Nimloth 尚不是完整的 pixel encoder JEPA，但 predictor / loss 复用 LeWM core（ARPredictor / Embedder / MLP / SIGReg）并采用 LeWM-style projector / pred_proj。

## 模块

| 文件 | 内容 |
|------|------|
| `dataset.py` | Nimloth jsonl → `TransitionSample`；折扣 action value target |
| `statistics.py` | 不运行模型的 rollout 数据集描述性统计 |
| `_vendor_lewm.py` | LeWM `ARPredictor` / `Embedder` / `MLP` / `SIGReg`（上游子集） |
| `lewm.py` | `LeWMConfig`、`action_one_hot`、`freeze_module` |
| `model.py` | `WorldModel`：组合 `StateProjector`、`LatentWMPredictor`、`ValueHead`，并提供公共 loss 成员方法 |
| `predictor.py` | `LatentWMPredictor`（Qwen-latent 动力学，无 pixel encoder） |
| `state_proj.py` | `StateProjector`：LeWM-style MLP (LayerNorm) Qwen hidden → WM emb |
| `value_head.py` | `ValueHead`：state emb → 每 action 的 value |
| `reconstruction.py` | `WMImageDecoder`：post-hoc reconstruction diagnostic decoder（不参与 SFT2/RL loss） |

### LeWM 结构对齐

- **ARPredictor**：`input_dim=emb_dim`, `hidden_dim=predictor_hidden_dim`, `output_dim=predictor_hidden_dim`（LeWM 风格：不直接输出 emb_dim）。
- **pred_proj**：LeWM `MLP(predictor_hidden_dim → predictor_hidden_dim → emb_dim)`，使用 `LayerNorm` 归一化。
- **StateProjector**：LeWM `MLP(qwen_hidden_dim → projector_hidden_dim → emb_dim)`，默认 `projector_hidden_dim=2048`，使用 `LayerNorm`。
- **SIGReg**：Sketch Isotropic Gaussian Regularizer（LeWM §3.3），对 projected embeddings 施加正则化，默认 `lambda_sigreg=0.1`。
- `WorldModel.compute_dynamics_loss()` 和 `compute_action_value_loss()` 使用本模块
  持有的子网络。SFT2 与 RL 仍在各自 `algorithm.py` 中决定 stop-gradient：
  SFT2 更新当前/下一两侧 StateProjector，RL 把下一状态作为 target。

## 与 training 的边界

- **本包**：组合后的世界模型、模型定义、规划和公共 loss 成员方法。
- **`nimloth.model`**：`NimlothModel(llm, wm)` 完整模型组合。
- **`nimloth.backbone.qwen25vl`**：transition → Qwen messages 的适配。
- **`nimloth.training.sft2`**：训练循环（`trainer.py`）、loss 组装、checkpoint、验证。

SFT2 实验入口：`experiments/training/sft2/train.py` → `nimloth.training.sft2.trainer`。
