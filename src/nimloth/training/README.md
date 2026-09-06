# 训练模块

Nimloth 各训练阶段的实现入口。跨模块合同见[世界模型与训练](../../../.trellis/spec/domains/world-model-and-training.md)，三阶段 SFT 的使用和阅读顺序见 [`sft/`](sft/README.md)。

## 职责划分

- `training/sft/`：回答格式监督、Query 对齐、WM/value 三阶段训练，以及 direct/WM 环境 rollout 评估。
- `training/rl/`：RL 损失、rollout 迭代、验证和 checkpoint。
- `training/reconstruction/`：冻结 Qwen/WM 后的图像解码器诊断训练。
- `training/common/`：训练阶段间共用的基础功能。

训练依赖的模型与运行组件保持各自所有权：`backbone/` 负责 Qwen 批处理、状态提取、微调和视觉 EMA；`wm/` 负责世界模型及预测头；`agent/` 负责状态/动作接口与规划；`rollout/` 负责轨迹数据、采集和存储；`config/` 负责经过校验的配置；`util/` 负责分布式、调度、缓存和指标等公共工具。

## 阶段与实验边界

原 WM/value 训练迁入 `sft/stage3/`，新 SFT2 为 `sft/stage2/` 的 Query 对齐。原 `training/sft1/`、`training/sft2/` Python 包已移除，调用者须使用统一后的路径。旧 checkpoint 字段及 `config/sft2` 名称继续表示 WM/value，目录迁移不改变其含义。

`experiments/training/sft1/` 和 `experiments/training/sft2/` 中既有启动脚本按原实验命名保留，其活动导入指向新实现。专项 canary、动作头修复、KV/整轨迹研究原型和特征审计放在 `experiments/training/sft/diagnosis/`；生产训练不依赖这些工具。
