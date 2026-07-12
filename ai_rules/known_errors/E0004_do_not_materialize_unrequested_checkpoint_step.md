# E0004 — 不要为了编号补造用户不需要的 checkpoint

## 错误
看到 legacy trainer 的 `TOTAL_STEPS=330` 最终保存的是 `global_step_329`，就额外安排了一次训练来“补造” `global_step_330`。

## 问题
用户要的是继续训练效果，不要求 checkpoint 编号必须为330。额外补造 checkpoint 增加了等待时间和流程复杂度。

## 正确做法
- 先确认用户是否真的需要特定编号的 checkpoint；
- 本次应直接保留完整的 `global_step_329`；
- 删除 `global_step_301` 到 `global_step_328`；
- 从 step329 继续额外30个数据 epoch，即540次 PPO 更新，训练/save step330到869。
