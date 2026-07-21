# E0034：不要把 multi-step WM 训练展平成独立 transition

## 已确认错误

RL 配置允许 `history_size > 1`，但旧实现先把 trajectory 展平成独立 transition，
再把每个 `(B,D)` 状态临时扩成 `(B,1,D)` 送入 predictor。结果只有第一个位置参与
训练，配置中的 history size 只改变了模型容量，没有形成真实多步训练。

## 错误原因

把“模型最多接受 H 个位置”和“训练数据实际提供 H 个连续位置”混为一谈。单独
保留 current/next tensor 无法恢复 trajectory 边界和时间顺序，随机 transition
采样也无法构造合法上下文。

## 正确做法

- 编码后继续保留 trajectory 边界和 step 顺序。
- 一步预测偏移下，每个 H-step 训练窗口必须包含同一 trajectory 的 `H+1` 个
  连续状态与 H 个动作。
- predictor 返回 H 个因果预测，依次监督 `[s_1, ..., s_H]`。
- SIGReg 使用完整的 `(T=H+1,B,D)` 状态序列。
- episode 开头使用真实的短前缀，禁止重复初始状态或填充虚构动作来冒充历史。
- `history_size` 必须保持可配置，禁止用强制设为 1 掩盖数据/算法错误。
