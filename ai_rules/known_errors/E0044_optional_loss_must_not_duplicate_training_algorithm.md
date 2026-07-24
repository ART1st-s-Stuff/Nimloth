# E0044：可选 loss 不得复制核心训练算法

## 错误

发现 DINO-grid 接入复制了完整的 SFT2 current/target、CE、WM、value 和 SIGReg
流程后，又错误地把它进一步包装成完整 training variant，而没有消除第二套算法。

## 后果

- 相同训练语义存在两份实现，修复梯度、cache、padding 或分布式逻辑时容易遗漏一份。
- “模块化”被误解成按配置复制整条训练路径；模块边界变多，但核心行为仍然分叉。
- 新增一个监督项被迫拥有模型、batch、algorithm、trainer/checkpoint 组装，复杂度失控。

## 已发生证据

- 原实现包含 `DINOGridSFT2Algorithm._grid_step()`，与 `SFT2Algorithm._step()` 重复。
- AI 首次修复又创建了完整 variant/loader registry；人类明确指出目标是把 DINO 作为
  configurable loss 加入唯一 SFT2 核心。错误提交已由 `5e086e6` 撤回。

## 正确做法

1. SFT2 只有一个 `SFT2Algorithm`，统一拥有 current/target、CE、WM、value、SIGReg。
2. 可选监督实现小型 loss component，只消费核心 step 已有 prediction 和附加 target。
3. 公共 batch 可携带具名 auxiliary target，但不能为每个 loss 复制 batch/algorithm。
4. loss 权重由配置控制；component 只返回 raw/weighted loss 和自身指标。
5. 对不同 state topology 的统计表示通过 WorldModel 公共方法适配，不复制 SIGReg 流程。
