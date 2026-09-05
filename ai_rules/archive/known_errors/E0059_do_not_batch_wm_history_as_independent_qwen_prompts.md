# E0059：不要把 WM history 展平成一批独立 Qwen prompt

## 已确认错误

RL 采样一个 `history_size=H` 的连续窗口后，曾把每条窗口的 `H + 1` 个完整多模态
prefix 展平成 `B * (H + 1)` 个独立样本，一次送进 HF Qwen。ID106 的正式20步
trajectory在首个state-sequence forward因此额外申请约38.52GiB并OOM；训练尚未产生
loss、backward或optimizer step。

同时，我们后来又错误假定history里的每个state都应有一份Qwen hidden cache。人类明确
指出：WM一次预测多个step时，中间state本来就是WM输出，不应为了凑齐`H + 1`行而重新
运行Qwen或伪造Qwen hidden。

## 正确做法

- planner trajectory分开保存两种数据：segment起点和terminal的稀疏Qwen anchor hidden；
  以及每个动作前state加terminal组成的稠密真实/预测混合WM state序列。
- 训练只对当前segment起点和终点anchor执行StateProjector；history中的其余预测state
  作为detached常量输入WM。不得把`history_size`解释为需要调用Qwen的次数。
- detached rollout hidden只能用于`representation_to_backbone=false`。若表征loss需要
  回传Qwen，必须重新forward，不能把cache伪装成可训练图。
- cache只替代state target编码；action distillation仍按锚点保存的真实response/token IDs
  重放当前Qwen，不能拿behavior hidden替代current-policy action logits。
- planner trajectory缺少任一anchor或任一稠密WM state时必须fail closed。旧的无cache非planner数据
  若仍需在线编码，应按时间位置拆成`H + 1`次、每次`B`个prompt，不恢复一次性
  `B * (H + 1)` batch。
