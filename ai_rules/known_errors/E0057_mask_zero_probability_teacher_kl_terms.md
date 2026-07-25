# E0057：确定性teacher的零概率KL项必须避免`0 * -inf`

## 已确认错误

greedy planner用`0/-inf`表示确定性teacher action log-prob。直接计算
`p_teacher * (log p_teacher - log p_qwen)`时，未选动作会产生`0 * -inf = NaN`。
ID103因此把`action_distillation_kl`记录为NaN，尽管用于优化的交叉熵式
`action_distillation_loss`和总loss都是有限值。

## 正确做法

- KL计算必须先屏蔽零概率support，或把非有限teacher log-prob替换为有限占位后再乘零概率。
- 不得把诊断指标NaN误报成优化loss NaN，也不得因为优化loss有限而声称所有指标有限。
- deterministic teacher回归必须检查KL等于被选动作的负Qwen log-prob，并检查梯度全部有限。
