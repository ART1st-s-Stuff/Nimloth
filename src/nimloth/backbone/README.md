# Backbone

`nimloth.backbone` 定义 Agent 的可训练语言/视觉 backbone 边界。

| 模块 | 职责 |
|------|------|
| `base.py` | `Backbone(nn.Module)`、输入输出和 factory 装配结果 |
| `qwen25vl/` | Qwen2.5-VL 的模型、processor、policy、rollout 和 checkpoint 适配 |

训练代码只依赖 `Backbone` 与独立适配器协议，不直接调用 Qwen 类。processor、
DataLoader cache 和 EMA 是运行期对象，不属于 `Backbone.state_dict()`。
