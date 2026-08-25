# E0094：capture TP门禁必须使用目标TP规模

## 已确认错误

在async same-generation capture bridge已实现两阶段TP prepare/finish之后，曾继续沿用旧ID160
单卡TP1 one-turn smoke合同，准备只用1×H800做capture GPU gate。该资源规模只能验证TP1路径，
不能验证新增的多rank collective顺序、TP一致性和partial-rank fail-closed语义。

## 正确做法

- 若当前门禁目标包含TP capture协议，GPU smoke必须使用计划验证的真实TP规模；本阶段为单节点
  8×H800、vLLM TP8。
- 必须明确区分“环境/协议TP1 smoke”和“capture TP8 gate”，禁止因旧launcher可复用而默认
  继承旧资源规模。
- 先提交8卡hold，再在allocation内用`srun --overlap`执行；主体失败时保留hold供诊断，
  不得自动退回TP1冒充TP8证据。
