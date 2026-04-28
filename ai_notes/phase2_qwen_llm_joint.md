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

## EB-Nav 数据集信息

下载自：https://huggingface.co/datasets/EmbodiedBench/EB-Nav_trajectory_dataset

**文件结构**：
- `datasets/EB-Nav/eb-nav_dataset_single_step.json` - 单步数据 (~97MB)
- `datasets/EB-Nav/eb-nav_dataset_multi_step.json` - 多步数据 (~58MB)
- `datasets/EB-Nav/images.zip` - 图像 (~14GB, 已解压)

**数据格式**：
```json
{
  "model_name": "claude-3-5-sonnet-20241022_additional_nav_no_c_h",
  "instruction": "navigate to the Bread in the room and be as close as possible to it",
  "trajectory": [
    {
      "visual_description": "I can see a kitchen environment...",
      "reasoning_and_reflection": "...",
      "language_plan": "1. Move forward...",
      "executable_plan": [
        {
          "step_id": 1,
          "img_path": "images/.../episode_1_step_1.png",
          "action": [0, "Move forward by 0.25"],
          "action_success": true,
          "env_feedback": "Last action MoveAhead executed successfully."
        }
      ]
    }
  ]
}
```

**动作空间**：8 个离散动作
- 0: Move forward by 0.25
- 1: Move backward by 0.25
- 2: Move rightward by 0.25
- 3: Move leftward by 0.25
- 4: Rotate to the right by 90 degrees
- 5: Rotate to the left by 90 degrees
- 6: Tilt the camera upward by 30 degrees
- 7: Tilt the camera downward by 30 degrees

**包含内容**：
- 导航指令（instruction）
- 每个步骤的 CoT（reasoning_and_reflection）
- 图像（images/）
- 动作标签

**用于**：
- Phase 3 Value Head 训练
- 验证 LLM 对物理场景的理解
