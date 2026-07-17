# E0033 — Dynamic rollout不能把system prompt当作task instruction

## 错误

环境创建后调用`get_system_prompts_batch()`，把返回的通用动作说明保存为`nav_instruction`；随后只从`reset/step` observation提取图片，丢弃`obs_str`中的具体任务、feedback、reward和done。

同时又自行构造了与SFT不同的prompt和固定thought，使用`prompt_format=wm`、旧动作别名、0.5m/1.5m几何、success=10、额外碰撞惩罚，并把trajectory总reward当terminal return。旧smoke的mock observation只有图片，恰好绕过真实`obs_str`结构，所以不能发现根因。

## 后果

策略只看到“instruction会在first observation提供”这类通用说明和图片，却看不到实际目标。ID11使用的generic system prompt还包含具体`<answer>rotateright</answer>`示例，并被每轮重复加入；这与缺任务文本共同造成尖锐的rotateright分布。因此此前所谓“top-p放大已有偏置”不是独立的次要问题：该尖锐分布本身就是错误prompt协议造成的结果之一。

rollout/PPO虽内部重放同一个错误prompt，但都偏离SFT transcript。导航成功率和RL质量结论无效；只能保留FSDP、collective、checkpoint等mechanics结论。

## 正确做法

1. 分开保存`system_prompt`、`task_instruction`、initial observation和每步observation/feedback/reward/done。
2. `reset/step` observation必须同时含非空`obs_str`和`multi_modal_data['<image>']`；禁止image-only fallback。
3. 从initial `obs_str`提取`Human Instruction:`，并与每步`info.instruction`逐字比较。
4. source-eval→Nimloth prompt rewrite必须与SFT converter共用一个函数；不能自行替换动作名或XML。
5. policy必须真实生成`<think>...</think>`，框架随后注入query/action prefix；禁止teacher-force reference thought或固定generic thought冒充runtime。
6. trajectory schema必须保存policy实际使用的文本、assistant responses和逐步reward；PPO/latent encoding逐字重放。
7. 奖励使用VAGEN每步reward与final reward原值，不额外加碰撞惩罚，不按reward阈值猜success。
8. 真实env gate必须将每条seed的task与dataset精确比对。数据本身允许不同seed有相同instruction（heldout seeds1–20只有9个唯一文本），所以不能用“20个instruction必须全不同”代替逐seed核验；小smoke应选已知文本不同的seed。
9. 修复后先做固定heldout evaluation-only baseline；修复前的成功率或“无提升”结果不得用于判断模型质量。

## 已确认事件

2026-07-17 ID11中80条train/heldout trajectory的`nav_instruction`只有1个唯一值，0条包含`Human Instruction:`。ID11的mechanics仍有效，质量结论作废，禁止作为quality experiment resume。
