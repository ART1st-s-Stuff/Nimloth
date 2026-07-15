# E0032 — Factorized WM dummy path needs explicit dtype conversion

## 错误

给 `LatentWMPredictor` 新增 `8192 → dynamics_dim` 的 LayerNorm/Linear 输入层后，
只测试了普通 FP32 forward，没有覆盖 SFT2 terminal-only dummy forward。

## 实际结果

Job `476351` 前 9 步正常；首个 terminal-only batch 在 autocast 外把
StateProjector 的 BF16 输出传给 FP32 LayerNorm，报
`expected scalar type BFloat16 but found Float`。无 checkpoint，未进入正式训练。

## 正确做法

Factorized input 层必须把外部 State 显式转换为该层参数 dtype，不能依赖外层
autocast。测试必须包含“FP32 predictor 参数 + BF16 State 输入”的
`predict_next_emb` 路径，以及 terminal-only dummy batch。

修复：commit `41f8778`。
