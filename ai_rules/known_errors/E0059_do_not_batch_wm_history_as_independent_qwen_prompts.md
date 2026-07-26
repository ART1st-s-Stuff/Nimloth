# E0059：不要把 WM history 展平成一批独立 Qwen prompt

## 已确认错误

RL 采样一个 `history_size=H` 的连续窗口后，曾把每条窗口的 `H + 1` 个完整多模态
prefix 展平成 `B * (H + 1)` 个独立样本，一次送进 HF Qwen。ID106 的正式20步
trajectory在首个state-sequence forward因此额外申请约38.52GiB并OOM；训练尚未产生
loss、backward或optimizer step。

同时，planner rollout本来已经从每次真实vLLM CoT forward取得当前observation的latent
hidden，却只用于在线规划，trajectory落盘时将它丢弃。训练随后再次从完整历史prompt
重算相同state，既浪费显存，也割裂了rollout与WM使用的state来源。

## 正确做法

- planner trajectory保存每个动作state及额外terminal state的`T + 1`个
  pre-StateProjector Qwen latent hidden；它们必须来自各自真实CoT生成的同一次forward。
- RL窗口只切片所需的`H + 1`个hidden，并重新执行当前StateProjector，使projector的
  freeze/gradient语义保持显式。
- detached rollout hidden只能用于`representation_to_backbone=false`。若表征loss需要
  回传Qwen，必须重新forward，不能把cache伪装成可训练图。
- cache只替代WM/value/SIGReg的state编码；PPO仍按保存的token IDs和mask重放当前Qwen，
  不能拿behavior hidden替代new-policy log-prob。
- planner trajectory缺少任一state cache时必须fail closed。旧的无cache非planner数据
  若仍需在线编码，应按时间位置拆成`H + 1`次、每次`B`个prompt，不恢复一次性
  `B * (H + 1)` batch。
