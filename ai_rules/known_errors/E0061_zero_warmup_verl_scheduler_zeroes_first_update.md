# E0061 — pinned VERL零warmup scheduler会把首次更新LR设为0

## 已发生的错误

ID38在AdamW.step内部逐rank确认：gradient非零、optimizer state step从无到1，但参数即时前后fingerprint完全相同。worker在scheduler.step后报告LR=1e-5，容易误以为optimizer使用了该LR。

Pinned VERL的`get_constant_schedule_with_warmup()`使用`step / max(1, warmup_steps)`。`warmup_steps=0`时，LambdaLR构造阶段在step0把optimizer LR设为0；首次`optimizer.step()`因此只初始化Adam state而不更新参数，随后scheduler才把LR改回配置值。

## 正确做法

- 当`num_warmup_steps==0`时，scheduler lambda从step0起恒为1。
- 正warmup配置保持原语义。
- actor与critic共享该helper，必须在两类worker构建前安装修复。
- 更新成功必须由optimizer-step即时fingerprint、post-worker fingerprint和非零后续policy变化共同证明。

## 证据

- `src/nimloth/training/rl/verl_gate.py::install_verl_zero_warmup_scheduler_patch`
- ID38 `CRITIC_UPDATE_AUDIT`。
