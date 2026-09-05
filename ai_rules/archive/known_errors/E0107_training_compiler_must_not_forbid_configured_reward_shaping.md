# E0107: Training compiler must not forbid configured reward shaping

## 已发生的错误

ID168 rollout携带Navigation默认`per_turn_format_reward=0.01`。训练compiler把“当前
smoke想使用纯结果reward”的实验配置选择错误地实现成通用硬门禁，拒绝所有nonzero
intermediate reward。人类指出这个gate设计不合理：后续实验可能需要保留中间奖励。

## 正确做法

- dataset/environment config决定reward shaping。
- return和Frozen-V GAE compiler保留每个真实turn的有限`env_turn_reward`并正常折扣。
- compiler只校验trajectory topology、terminal/truncation事实、identity和数值有限性，
  不硬编码reward必须为零或正数。
- 需要纯结果reward的具体实验必须在自己的dataset config显式设置，例如ID169：
  `per_turn_format_reward=0`、`format_reward=0`、`success_reward=1`。
- integration preflight应核验该实验自己的reward字段，但这不能限制未来其他配置。

## Evidence

- ID168 Job `519165`在optimizer前被错误的通用reward-policy gate阻断。
- 人类当前直接prompt要求修改ID169配置并删除通用gate。
