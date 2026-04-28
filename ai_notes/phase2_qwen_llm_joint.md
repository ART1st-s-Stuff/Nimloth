# Phase 2 Qwen Vision Encoder Joint Training

## 架构澄清（2026-04-28）

### 背景
- 之前的测试表明 Qwen vision encoder 不能很好地保留物理信息
- DINO 可以保留物理信息，但用户不想用（contribution 不大）
- Phase 2 主要目的是训练 WM，同时也要训练 vision encoder
- 担忧：vision encoder 训练后输出可能偏移，导致 LLM backbone 无法理解

### 正确的架构

```
Prompt + Image → Qwen Vision Encoder → vision tokens
                                            ↓
                            Qwen LLM backbone (FROZEN)
                                            ↓
                                   [optional CoT]
                                            ↓
                                latent token embedding
                                            ↓
                            WM(latent, action) → next latent prediction
                                            ↓
                                        Loss 反传
                                            ↓
                    更新 Vision Encoder + WM
                    (LLM backbone FROZEN，不更新)
```

### 组件状态

| 组件 | 状态 | 说明 |
|------|------|------|
| Qwen LLM backbone | **FROZEN** | 固定不动，保持语义理解能力 |
| Qwen Vision Encoder | **可训练** | 联合训练，学习物理信息 |
| WM | **可训练** | Phase 2 主要目标 |

### 为什么 LLM backbone 要 frozen

1. **防止输出偏移**：Vision Encoder 训练时输出可能偏移，FROZEN LLM 保证 latent space 与预训练对齐
2. **防止 mode collapse**：有 LLM backbone 约束
3. **保持语义理解**：Vision Encoder 学到的物理信息可以被 LLM 理解

## 关键文件

| 文件 | 修改内容 |
|------|---------|
| `src/vlm/qwen_adapter.py` | `get_image_hidden_state()` 已修改为返回 LLM hidden state |
| `src/wm/encoder/qwen.py` | `QwenLLMLatentEncoder` 已更新，新增 `use_vision_only` 和 `llm_backbone_trainable` 参数 |
| `src/wm/encoder/factory.py` | 支持新参数 |
| `configs/wm/lewm_qwen_llm_joint.yaml` | 训练配置 |
| `configs/wm/lewm_qwen_llm_joint.yaml` | 训练配置 |

## 配置

```yaml
wm:
  name: lewm_qwen_llm_joint
  latent_dim: 4096
  num_patches: 1
  token_dim: 4096
  encoder:
    name: qwen_llm
    model_name: Qwen/Qwen2.5-VL-7B-Instruct
lewm:
  sigreg_enabled: true
  sigreg_latent_dim: 4096
```

## 当前实现状态

### 已完成
- [x] `QwenLLMLatentEncoder` 类
- [x] `qwen_llm` encoder 类型支持
- [x] 训练配置 `lewm_qwen_llm_joint.yaml`
- [x] SIGReg adaptive warmup
- [x] 修改 `get_image_hidden_state`：让 vision tokens 通过 LLM backbone，返回 hidden state
- [x] 新增 `use_vision_only` 参数（只用 Vision Encoder，不经过 LLM）
- [x] 新增 `llm_backbone_trainable` 参数（预留接口）
- [x] `_set_llm_backbone_trainable()` 方法：冻结/解冻 LLM backbone
- [x] **验证通过**：Vision Encoder (390 params) 可通过 LLM hidden state loss 反传获取梯度

### 待完成
- [ ] 完整训练测试（需要 GPU 显存足够）

## 状态
- [x] 实现 `get_image_hidden_state()` （通过 LLM backbone）
- [x] 实现 `QwenLLMLatentEncoder`
- [x] 创建训练配置
- [x] 支持 SIGReg 在 encoded latent 上应用
- [x] 修复 LeWMModel SIGReg warmup bug
- [x] LLM backbone hidden state 获取
- [ ] 完整训练（需要 Qwen 模型）


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

## 数据集适配器

**文件**：`src/data/eb_nav_dataset.py`

**类**：
- `EBNavDataset`：单帧数据集，返回 (image_path, action, instruction, cot)
- `EBNavSequenceDataset`：序列数据集，返回历史帧和未来预测目标

**动作映射**：
- 0: Move forward → [0.25, 0, 0]
- 1: Move backward → [-0.25, 0, 0]
- 2: Move right → [0, 0, -0.25]
- 3: Move left → [0, 0, 0.25]
- 4: Rotate right 90° → [0, -90, 0]
- 5: Rotate left 90° → [0, 90, 0]

**测试结果**：
- 单帧数据集：56,920 样本
- 序列数据集：43,720 样本
