# World Model (`nimloth.wm`)

Nimloth 世界模型层：transition 数据、LeWM predictor 封装、state/value 头，以及与 Qwen latent 的桥接。

LeWM 核心算子来自 `external/le-wm`，经 `wm/_vendor_lewm.py` 以最小子集 vendoring；Nimloth 不在运行时 import `external/le-wm` 脚本。

**当前状态**：Nimloth 尚不是完整的 pixel encoder JEPA，但 predictor / loss 复用 LeWM core（ARPredictor / Embedder / MLP / SIGReg）并采用 LeWM-style projector / pred_proj。

## 模块

| 文件 | 内容 |
|------|------|
| `dataset.py` | Nimloth jsonl → `TransitionSample`；折扣 action value target |
| `collate.py` | transition batch → Qwen messages + metadata |
| `_vendor_lewm.py` | LeWM `ARPredictor` / `Embedder` / `MLP` / `SIGReg`（上游子集） |
| `lewm.py` | `LeWMConfig`、`action_one_hot`、`freeze_module` |
| `predictor.py` | `LatentWMPredictor`（Qwen-latent 动力学，无 pixel encoder） |
| `state_proj.py` | `StateProjector`：LeWM-style MLP (BatchNorm1d) Qwen hidden → WM emb |
| `value_head.py` | `ValueHead`：state emb → 每 action 的 value |
| `reconstruction.py` | `WMImageDecoder`：post-hoc reconstruction diagnostic decoder（不参与 SFT2/RL loss） |

### LeWM 结构对齐

- **ARPredictor**：`input_dim=emb_dim`, `hidden_dim=predictor_hidden_dim`, `output_dim=predictor_hidden_dim`（LeWM 风格：不直接输出 emb_dim）。
- **pred_proj**：LeWM `MLP(predictor_hidden_dim → predictor_hidden_dim → emb_dim)`，使用 `BatchNorm1d` 归一化。
- **StateProjector**：LeWM `MLP(qwen_hidden_dim → projector_hidden_dim → emb_dim)`，默认 `projector_hidden_dim=2048`，使用 `BatchNorm1d`。
- **SIGReg**：Sketch Isotropic Gaussian Regularizer（LeWM §3.3），对 projected embeddings 施加正则化，默认 `lambda_sigreg=0.1`。
- **MSE target** 使用 detached target embedding（stop-gradient），SIGReg 对 state_proj 的当前和下一状态投影**均有梯度**。

## 与 training 的边界

- **本包**：模型定义、transition 数据与 collate。
- **`nimloth.training.sft2`**：训练循环（`trainer.py`）、loss 组装、checkpoint、验证。

SFT2 实验入口：`experiments/training/sft2/train.py` → `nimloth.training.sft2.trainer`。
