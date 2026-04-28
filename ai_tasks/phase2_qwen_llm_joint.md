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

### 完整训练测试
- [x] **已完成**：1 epoch 训练成功，loss 从 0.1573 降到 0.0001
- 训练参数：batch_size=2, lr=1e-5, 21860 steps
- Vision Encoder (676M) + WM (127M) 联合训练成功

## 状态
- [x] 实现 `get_image_hidden_state()` （通过 LLM backbone）
- [x] 实现 `QwenLLMLatentEncoder`
- [x] 创建训练配置
- [x] 支持 SIGReg 在 encoded latent 上应用
- [x] 修复 LeWMModel SIGReg warmup bug
- [x] LLM backbone hidden state 获取
- [x] 完整训练（1 epoch 验证成功，loss 0.1573 → 0.0001）


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

## Phase 2 前置阶段：Qwen 格式与 Action Prior LoRA 微调

### 目标
在正式进入 Phase 2 的 WM + Vision Encoder joint training 之前，先使用 EB-navigation 数据对 Qwen 做一轮 LoRA 微调，使 VLM 能稳定输出后续训练所需的中间格式：

```
[VLM] Text prompt + Image -> CoT / planner trigger -> latent state z_t
[Action Prior] P(a|z_t)：根据当前 latent state 提供候选动作先验
```

### 训练目标
- [ ] 让 Qwen 在给定导航 instruction + 当前图像时，按固定 schema 输出：
  - `cot` / `reasoning_and_reflection`
  - `planner_trigger`
  - `latent_state` 或用于抽取 `z_t` 的 special token / hidden state anchor
  - `action_prior`：8 个离散动作的候选先验分布或 top-k 动作
- [ ] 使用 EB-navigation 的 step-level 数据构造 SFT 样本：
  - 输入：`instruction` + 当前 step image
  - 输出：CoT / planner trigger / latent anchor / action prior
  - action prior target 可先由当前 expert action 构造成 one-hot 或 label-smoothed distribution
- [ ] 训练方式采用 LoRA：
  - 冻结 Qwen base model 主体参数
  - 只训练 LoRA adapter
  - 优先覆盖 language side 的 attention / MLP target modules；是否覆盖 vision projector 作为可选配置

### 输出格式建议
```json
{
  "cot": "<reasoning_and_reflection>",
  "planner_trigger": true,
  "latent_state": "<LATENT_STATE>",
  "action_prior": {
    "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "top_actions": [
      {"action_id": 0, "name": "Move forward by 0.25", "score": 0.0}
    ]
  }
}
```

### 实现任务
- [ ] 新增 EB-navigation SFT 数据构造脚本：
  - 从 `eb-nav_dataset_single_step.json` / `multi_step.json` 读取 instruction、image、CoT、expert action。
  - 生成统一的 prompt / response JSONL。
  - 校验所有 response 可被 JSON parser 正确解析。
- [ ] 新增 Qwen LoRA SFT 训练入口或配置：
  - 训练输出格式与 action prior。
  - 保存 LoRA adapter checkpoint。
  - 记录格式解析成功率与 action prior accuracy。
- [ ] 新增推理/验证脚本：
  - 输入 instruction + image。
  - 验证输出是否符合 schema。
  - 抽取 `latent_state` 对应 hidden state 作为 `z_t`。
  - 抽取 `action_prior` 作为 `P(a|z_t)`。
- [ ] 将该 LoRA adapter 接入后续 Phase 2：
  - Phase 2 初始化时加载此 adapter。
  - WM 输入 latent 使用格式微调后的 hidden state anchor。
  - action prior 可作为 planner 候选动作先验或辅助 loss。

### 验收标准
- [ ] 格式解析成功率达到可用阈值（例如 > 95%）。
- [ ] action prior top-1 / top-k accuracy 明显高于随机 8 动作基线。
- [ ] 能稳定抽取 `z_t` hidden state，shape 与 WM latent_dim 对齐。
- [ ] 加载 LoRA adapter 后不破坏后续 Phase 2 的 Qwen hidden state 获取流程。

## 训练实现最新更新（2026-04-28）

### 训练入口与数据流
- `src/train/train_wm_joint.py` 已改为两阶段入口：
  - `pipeline.train.stage=stage1_wm_vision`：执行 WM + Qwen visual encoder 训练
  - `pipeline.train.stage=stage2_value_head`：占位（暂不执行）
- Stage1 数据已切换为 AI2-THOR 已采集数据：
  - 优先自动解析 `dataset.manifests.train/test` 对应 run_dir
  - 失败时回退固定 run 路径（带 TODO）

### Loss 与训练编排
- 已从手写单步 MSE 切换为 `LeWMModel.train_step` 口径（对齐 DINOv2 训练路径）。
- 训练日志已包含：
  - `loss`
  - `loss_recon`
  - `loss_action`
  - `loss_sigreg`
  - `sigreg_weight`
  - `loss_kl`
  - `loss_total_with_kl`

### Qwen visual encoder 训练策略扩展
- 新增配置组 `pipeline.train.qwen_encoder`：
  - `train_mode: full | lora`
  - `lora: r/alpha/dropout/target_modules`
  - `kl: enabled/weight/temperature/max_images_per_batch`
  - `ema: enabled/decay/use_ema_for_eval`
- 目前支持：
  - `full`：visual 全参数训练（LLM backbone 冻结）
  - `lora`：仅 visual 相关 LoRA 参数训练
- 依赖新增：`peft`

### KL 与 EMA
- 已实现 vision token 级 KL（teacher-student）：
  - teacher：冻结原始 Qwen visual encoder
  - student：当前训练中的 Qwen visual encoder
- 已实现 Qwen visual encoder EMA：
  - 每 step 更新 visual EMA
  - checkpoint 保存 `vision_encoder_ema_state`
  - 可视化可选使用 EMA 权重

### 可视化与 WandB
- run 名包含时间戳（例如 `train_wm_joint_YYYYMMDD_HHMMSS`）。
- wandb config 中附加完整 hydra 配置快照。
- 训练后可视化：支持原始 latent 空间 + SIGReg encoder 空间（可开关）。
- 新增训练中周期可视化：
  - `pipeline.train.post_visualization_every_n_steps=10`（默认）
  - 每 N step 上传一次可视化，训练结束后再补一次。

### TUI / 终端交互
- 训练主循环切换为 Rich Live 面板 + 进度条。
- 支持快捷键：
  - `1` 当前 step 指标
  - `2` GPU 负载
  - `3` 最近 loss 列表
  - `4` 控制面板
  - `p` 暂停并保存断点

### 多卡支持
- 新增 `pipeline.train.multi_gpu` 配置：
  - `enabled`
  - `device_ids`
- 启用后对 WM/IDM/mapper 使用 `DataParallel`。
- `LeWMModel` 已兼容 DataParallel 的 `predict_next`/`compute_sigreg` 调用与 state_dict 保存。

## 新增计划（2026-04-28）

### 1. EB-navigation reward 标注 + WM reward head

**目标**：让 WM 不只预测 next latent，还能从 `(latent, action)` 预测该动作/状态转移的 reward，用于后续规划、rollout 评估或 value head 训练。

**数据侧任务**：
- [ ] 为 EB-Nav / EB-navigation 数据集生成 step-level reward 标注。
- [ ] reward 标注优先基于已有字段构造：
  - `action_success`：动作成功给正向基础 reward，失败给负 reward。
  - `env_feedback`：解析碰撞、失败、接近目标、完成导航等反馈。
  - trajectory 终止状态：到达目标或最终更接近目标时给 terminal / progress reward。
- [ ] 将 reward 写入新的 manifest/cache，避免每次训练动态解析原始 json。
- [ ] 扩展 `src/data/eb_nav_dataset.py`：
  - `EBNavDataset` 返回 `reward`。
  - `EBNavSequenceDataset` 返回每个 transition 的 `reward` 序列。
- [ ] 增加 reward 统计脚本/日志：
  - 样本数
  - reward 均值/方差
  - 正负 reward 比例
  - terminal reward 比例

**模型侧任务**：
- [ ] 给 WM 添加 `reward_head`：
  - 输入：WM transition feature / predicted latent feature。
  - 输出：标量 `reward_pred`。
- [ ] 在 `LeWMModel.train_step` 中加入 reward loss：
  - 默认 `MSE(reward_pred, reward_target)`。
  - 配置项：`lewm.reward.enabled`, `lewm.reward.weight`, `lewm.reward.loss_type`。
- [ ] 训练日志新增：
  - `loss_reward`
  - `reward_pred_mean`
  - `reward_target_mean`
- [ ] checkpoint 保存/加载 reward head 参数。

**验收标准**：
- [ ] EB-Nav reward cache 可复现生成。
- [ ] dataloader batch 中包含 reward tensor，shape 与 transition 对齐。
- [ ] reward head 参与反传，`loss_reward` 正常下降或保持数值稳定。

### 2. Perceptual loss：latent 到原始图像

**目标**：约束 latent 保留视觉可重建信息，避免 joint training 只优化 latent transition 而丢失原始图像细节。

**实现方向**：
- [ ] 增加 latent-to-image decoder / reconstruction head：
  - 输入：Qwen LLM hidden state latent 或 WM predicted latent。
  - 输出：重建图像 `image_recon`，尺寸与训练配置对齐。
- [ ] 增加 perceptual loss：
  - 使用冻结视觉特征网络提取 `image_recon` 与原始图像特征。
  - 默认候选：LPIPS / VGG perceptual / frozen Qwen visual feature。
  - loss：`||phi(image_recon) - phi(image_target)||`。
- [ ] 配置项：
  - `lewm.perceptual.enabled`
  - `lewm.perceptual.weight`
  - `lewm.perceptual.backbone`
  - `lewm.perceptual.image_size`
  - `lewm.perceptual.use_predicted_latent`
- [ ] 在 `LeWMModel.train_step` 中并入总 loss：
  - `loss_total = loss_recon + loss_action + loss_sigreg + loss_kl + loss_reward + loss_perceptual`
- [ ] 训练日志新增：
  - `loss_perceptual`
  - `loss_image_recon`
- [ ] 可视化新增：
  - 原始图像
  - latent reconstruction 图像
  - predicted latent reconstruction 图像（可选）

**验收标准**：
- [ ] 单 batch 前向可产生非空 reconstruction。
- [ ] perceptual backbone 冻结，不参与训练。
- [ ] `loss_perceptual` 有限且可反传到 decoder / latent path。
- [ ] wandb 或本地可视化能对比原图与重建图。

### 更新后的 Phase 2 优先级

0. Phase 2 前先完成 Qwen LoRA SFT，使 VLM 按固定格式输出 CoT / planner trigger / `z_t` anchor / action prior。
1. 完成 EB-Nav reward cache 与 dataset 返回字段，保证训练数据闭环。
2. 添加 WM reward head 与 `loss_reward`，先用小 batch 验证反传和日志。
3. 添加 latent-to-image decoder，先用 reconstruction / perceptual 单 batch 验证。
4. 将 perceptual loss 并入正式 joint training，观察是否稳定。
5. 在 reward + perceptual 都稳定后，再考虑启用 stage2 value head。

### 当前已知问题与排查结论
- 在 `full + KL + EMA + SigReg` 组合下，早期 step 可能出现 NaN（常见于大模型全参微调数值不稳定）。
- `Qwen2.5-VL` 不同 transformers 版本接口存在差异：
  - 某些版本无 `get_image_features`
  - 已改为兼容路径：优先 `get_image_features`，否则回退 `self.visual(...)`
