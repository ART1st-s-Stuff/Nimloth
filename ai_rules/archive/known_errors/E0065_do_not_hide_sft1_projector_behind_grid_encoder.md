# E0065：不得冻结 SFT1 projector 后再增加一套 grid encoder

## 已确认错误

先前错误地认定 DINO-grid projector 不应在 SFT1 训练。人类随后明确：SFT1 使用 DINO
监督预训练 projector 是合理的；真正的问题是 SFT2 冻结该 projector，又串接一层
`LeWMGridEncoder`，并为第二层引入 EMA target encoder 和 DINO decoder。

## 影响

- SFT1 已有的 DINO-aligned state 被第二层 encoder 改写；
- WM state、EMA target 和 decoded DINO output 成为三套不同表征；
- 基于该结构产生的旧 DINO-grid SFT2 结果及其下游 RL 结果不证明当前 state 语义。

## 正确做法

- SFT1 可以用 DINO grid 监督并导出 `SharedSlotProjector`；
- SFT2 加载后继续训练同一个 projector，其输出直接作为 WM state；
- predicted state 直接接受 DINO grid MSE，不再增加 state encoder、WM EMA 或 decoder；
- 旧 checkpoint 保留但不兼容新结构，不得静默转换或作为新语义的验证证据。

## 证据

- `experiments/training/sft1/train_dino_grid.py`
- `src/nimloth/wm/grid.py`
- `src/nimloth/training/sft2/trainer.py`
- `src/nimloth/training/sft2/dino_grid.py`
