# E0124：自定义advantage不得标成stock GRPO

## 已发生错误
joint训练实际由`prepare_joint_training_batch()`计算behavior-time Frozen-V GAE，并让custom actor读取`joint_advantages`；stock `compute_advantage()`被绕过，但ID165–ID183配置仍写`algorithm.adv_estimator: grpo`。人类指出该名字会错误描述训练公式。

## 原因
把VERL schema中的非GAE枚举当成无害占位值，忽略了配置名本身也是算法合同和审计证据。

## 正确做法
使用明确的`joint_frozen_v_gae`身份，并在driver与Ray trainer两层拒绝stock estimator。即使某字段只用于routing，也必须准确命名真实算法，不能靠“代码不会读它”解释误导性配置。
