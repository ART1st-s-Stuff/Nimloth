# E0112：不得擅自平衡内生动作分布

## 错误

看到SFT2训练数据八个动作的频数差异后，AI把“所有动作均有覆盖”错误推成“必须加入
类别平衡action CE”。

## 后果

class weight或按动作过采样会改变环境状态分布与behavior policy共同产生的经验动作分布，
使训练出的LLM prior不再匹配数据occupancy。

## 正确做法

- 保留原始trajectory/action频率，不擅自加class weight或动作均衡采样。
- 可以分别报告各动作的样本数、loss、accuracy和logit统计，但监测不得改变训练权重。
- 只有人类明确改变目标分布时才允许重加权。
