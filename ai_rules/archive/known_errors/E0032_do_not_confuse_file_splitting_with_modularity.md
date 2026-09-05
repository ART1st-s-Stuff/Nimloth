# E0032：不能把横向拆文件当作模块化

## 已确认错误

SFT2/RL 重构曾把一次 batch 的计算横向拆成 `algorithm.py`、`objective.py`、
`schedule.py`、`update.py` 和宽泛的 `components.py`。这些文件没有独立领域边界，
只负责继续转发同一组对象和 tensor；开发者必须来回跳转才能读懂一次训练更新。

## 错误原因

拆分依据是代码片段的形式名称，而非可独立解释的职责。文件数增加了，但模型装配、
梯度边界和 loss 组合仍然互相纠缠，复杂度只是被搬到了调用链上。

## 正确做法

- 一个阶段的 `algorithm.py` 应能顺序展示一次 batch 的模型前向、梯度边界、loss
  和 optimizer update；只把真正跨 batch/iteration 的生命周期留给 loop。
- trainer 按实际启动顺序显式装配 Agent、distributed runtime、optimizer 和 resume，
  禁止用含义宽泛的 `Components` 容器隐藏依赖。
- 只有当代码拥有独立契约、独立调用方或可独立替换的实现时才拆成文件；单函数
  objective、schedule 或 update wrapper 不构成模块边界。
