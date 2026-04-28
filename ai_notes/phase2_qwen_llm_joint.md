# Phase 2 Qwen LLM Joint Training

## 概述

在 `ai-dev-qwen-joint-training` 分支上实现，使用 Qwen LLM 的 vision embedding 作为 WM 的 latent space。

## 架构

```
Image → Qwen Vision Encoder → vision_embedding (last patch)
                                      ↓
                         LeWM(latent, action) → next latent prediction
                                      ↓
                              SIGReg (adaptive warmup)
```

## 关键文件

| 文件 | 修改内容 |
|------|---------|
| `src/vlm/qwen_adapter.py` | 新增 `get_image_hidden_state()` 方法 |
| `src/wm/encoder/qwen.py` | 新增 `QwenLLMLatentEncoder` 类 |
| `src/wm/encoder/factory.py` | 支持 `qwen_llm` encoder 类型 |
| `src/wm/predictor/factory.py` | 支持 `num_patches=1, token_dim=4096` 配置 |
| `configs/wm/lewm_qwen_llm_joint.yaml` | 新建训练配置 |

## 配置

```yaml
wm:
  name: lewm_qwen_llm_joint
  latent_dim: 4096
  num_patches: 1
  token_dim: 4096
  encoder:
    name: qwen_llm
```

## SIGReg 配置

- 在 `lewm.*.sigreg_*` 字段配置
- `sigreg_enabled: true`
- `sigreg_latent_dim: 4096`
- `warmup_steps: 1000` (adaptive warmup)

## 训练命令（待测试）

```bash
uv run python src/train/train_wm.py \
    config=wm/lewm_qwen_llm_joint \
    pipeline.train.sigreg.enabled=true \
    pipeline.train.sigreg.weight=0.1 \
    pipeline.train.sigreg.warmup_steps=1000
```

## 状态

- [x] 实现 `get_image_hidden_state()`
- [x] 实现 `QwenLLMLatentEncoder`
- [x] 创建训练配置
- [x] 支持 SIGReg 在 encoded latent 上应用
- [x] 修复 LeWMModel SIGReg warmup bug
- [x] 测试验证训练流程

## 相关 commit

- `e056427`: feat: add Qwen LLM hidden state as WM latent space
- `03aecb8`: refactor: simplify get_image_hidden_state
