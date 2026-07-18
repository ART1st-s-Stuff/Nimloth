# E0065 — FSDP hook-captured hidden loss必须依赖forward返回树

## 已发生的错误

ID42已用`no_sync()`解决首个PPO gradient过早shard的问题，但WM loss仅依赖final-norm forward hook捕获的hidden，丢弃actor forward返回的logits。FSDP root只在返回树中的tensor参与backward时注册/触发pre-backward状态；因此parameter post-backward hook执行时root仍为`TrainingState.IDLE`，报期望`FORWARD_BACKWARD`。

## 正确做法

- hook捕获hidden的auxiliary loss必须增加对FSDP forward实际返回tensor的零值graph依赖，例如`model_output.logits.sum() * 0.0`。
- 该anchor数值和loss语义均不变，但确保FSDP root pre-backward hook先执行。
- 仍须在distributed GPU direct gate核对真实backward、optimizer和checkpoint；普通`nn.Module` CPU test无法复现FSDP runtime state machine。

## 证据

- `src/nimloth/training/rl/verl_wm_aux.py`
- ID42 worker log。
