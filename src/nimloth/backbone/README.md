# Backbone

`nimloth.backbone` 定义 Agent 的可训练语言/视觉 backbone 边界。

| 模块 | 职责 |
|------|------|
| `base.py` | `Backbone(nn.Module)`、`BackboneInputBuilder` 和装配结果 |
| `qwen25vl/` | Qwen2.5-VL 的模型、processor、policy 和 checkpoint 适配 |
| `dino_grid.py` | frozen DINO spatial-grid supervision cache 的身份校验与只读访问 |

训练代码只依赖公共接口，不直接调用 Qwen 类。input builder 只做 Agent prompt
到模型张量的转换；return、窗口采样、target 对齐和 terminal mask 属于 rollout 或
具体训练阶段。processor、DataLoader cache 和 EMA 不属于 `Backbone.state_dict()`。
DINO cache 模块不定义 SFT2 loss，也不持有可训练 WM 参数。
