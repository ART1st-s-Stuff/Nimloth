# E0065：不得把第二阶段 grid projector 擅自前移到 SFT1

## 已确认错误

人类明确：`SharedSlotProjector` 本应属于 SFT2，由第二阶段训练；SFT1 不拥有也不应
训练这个 projector。历史提交 `6c1828a0` 却在 SFT1 DINO-grid 脚本中创建它、加入
optimizer、用 DINO grid MSE 训练并随 SFT1 checkpoint 导出。后续 SFT2 又把它当作
“SFT1 projector”冻结，并额外增加 `LeWMGridEncoder` 和对应 EMA target encoder。

## 影响

- “冻结 SFT1 projector、训练第二层 online encoder”的阶段归属错误；
- 为第二层 online encoder 引入 Grid EMA 的设计理由随之失效；
- 基于这条路径产生的 DINO-grid SFT2/RL checkpoint 只能证明该错误实现的机械行为，
  不能作为人类指定的 SFT1→SFT2 语义正确性证据。

## 正确做法

- SFT1 只交付其规定的 Qwen/query 表征，不训练或导出 SFT2 的 grid projector；
- SFT2 明确创建并训练 grid projector；
- target state、stop-gradient、是否需要 EMA 必须按人类确认的 SFT2 目标重新设计，
  禁止通过再加一层 encoder 来掩盖错误的阶段归属；
- 修复前先审计现有 SFT1/SFT2 checkpoint schema 和下游 loader，禁止把旧权重静默解释成
  新语义。

## 证据

- 历史提交：`6c1828a0`
- `experiments/training/sft1/train_dino_grid.py`（历史版本）
- `src/nimloth/wm/grid.py`
- `src/nimloth/training/sft2/trainer.py`
