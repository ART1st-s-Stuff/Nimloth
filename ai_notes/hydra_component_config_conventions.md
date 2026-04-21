importance: high
L: 1
D: 2

# Hydra 组件化配置约定

- 根配置 `configs/config.yaml` 仅做组件组合（defaults），不承载大段具体参数。
- 组件分组固定为：`dataset/`、`wm/`、`pm/`、`vlm/`、`pipeline/`。
- 配置访问优先单层路径：`cfg.dataset.xxx`、`cfg.wm.xxx`、`cfg.pipeline.<task>.xxx`。
- `pipeline` 只描述流程参数与运行参数，不放模型实现细节。
- 旧结构 `configs/data|train|calib|rollout` 已硬下线，不再使用旧键路径。
