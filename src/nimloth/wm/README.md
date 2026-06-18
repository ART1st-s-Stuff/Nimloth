# World Model (`nimloth.wm`)

Nimloth 世界模型层：transition 数据、LeWM predictor 封装、state/value 头，以及与 Qwen latent 的桥接。

LeWM 核心算子来自 `external/le-wm`，经 `wm/_vendor_lewm.py` 以最小子集 vendoring；Nimloth 不在运行时 import `external/le-wm` 脚本。

## 模块

| 文件 | 内容 |
|------|------|
| `dataset.py` | Nimloth jsonl → `TransitionSample`；折扣 action value target |
| `collate.py` | transition batch → Qwen messages + metadata |
| `_vendor_lewm.py` | LeWM `ARPredictor` / `Embedder` / `MLP`（上游子集） |
| `lewm.py` | `LeWMConfig`、`action_one_hot` |
| `predictor.py` | `LatentWMPredictor`（Qwen-latent 动力学，无 pixel encoder） |
| `state_proj.py` | `StateProjector`：Qwen hidden → WM emb |
| `value_head.py` | `ValueHead`：state emb → 每 action 的 value |

## 与 training 的边界

- **本包**：模型定义、transition 数据与 collate。
- **`nimloth.training.sft2`**：训练循环（`trainer.py`）、loss 组装、checkpoint、验证。

SFT2 实验入口：`experiments/training/sft2/train.py` → `nimloth.training.sft2.trainer`。
