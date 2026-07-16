# E0033 — Dynamic rollout不能把system prompt当作task instruction

## 错误

环境创建后调用`get_system_prompts_batch()`，把返回的通用动作说明保存为`nav_instruction`；随后只从`reset/step` observation提取图片，丢弃`obs_str`中的具体任务、feedback、reward和done。

## 后果

策略只看到“instruction会在first observation提供”这类通用说明和图片，却看不到实际目标（例如“navigate to the Pot”）。更糟的是，该system prompt带有具体`<answer>rotateright</answer>`示例，并被错误地重复放入每轮user消息，直接把策略推向rotateright。rollout/PPO虽内部使用同一个错误prompt，但都偏离SFT transcript。导航成功率和RL质量结论因此无效；只能保留FSDP、collective、checkpoint等mechanics结论。

## 正确做法

1. 分开保存`system_prompt`、initial task observation text和每步observation/feedback text。
2. 从`reset`返回的`obs_str`强制提取并验证唯一的`Human Instruction:`；不得只检查通用system prompt非空。
3. trajectory schema必须保存policy实际使用的每步文本，使PPO能够逐字重放rollout prompt。
4. 在真实env integration test中断言不同seed/task产生不同task instruction，并与dataset task一致。
5. 修复后先做固定heldout baseline；修复前的成功率或“无提升”结果不得用于判断模型质量。

## 已确认事件

2026-07-17 ID11中80条train/heldout trajectory的`nav_instruction`只有1个唯一值，0条包含`Human Instruction:`；真实`base` seeds1–20对应20个具体物体导航任务。ID11的mechanics仍有效，质量结论作废。
