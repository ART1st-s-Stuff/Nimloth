# Qwen 联合训练约定

- 重要性: high
- L: 1
- D: 2

## 训练口径

- `src/train/train_wm_joint.py` 的 Phase2 联合训练采用 `LeWMModel.train_step`，不再使用手写单步 MSE。
- SIGReg 的 loss 开关与权重以 `pipeline.train.sigreg.*` 为准，结构参数由 `wm.*` 提供。
- 两阶段入口保持：
  - `stage1_wm_vision`: 训练 WM + Qwen visual encoder
  - `stage2_value_head`: 占位，不执行训练

## Qwen Visual Encoder 策略

- 训练模式可切换：
  - `pipeline.train.qwen_encoder.train_mode=full`
  - `pipeline.train.qwen_encoder.train_mode=lora`
- LoRA 当前阶段仅作用在 Qwen visual encoder，不扩展到 LLM backbone。
- KL loss 为 vision token 级 teacher-student KL：
  - teacher: 冻结原始 Qwen visual encoder
  - student: 当前训练中的 Qwen visual encoder
- Qwen visual encoder 支持可选 EMA，并可配置训练后可视化是否使用 EMA 权重。

## 可视化约定

- 训练后默认在 test split 生成 rollout 图并上传 wandb。
- 除原始 latent 空间外，可额外生成 LeWM 内部 encoder（SIGReg 前）空间的轨迹图。
