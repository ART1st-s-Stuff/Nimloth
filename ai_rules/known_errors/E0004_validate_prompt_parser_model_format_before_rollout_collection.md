# E0004 — 全量 rollout 前必须验证 prompt / parser / 模型输出格式一致

## 错误

曾在 Plan B retry15 中使用 `eval_mode` 配置采集数据，但没有先验证 prompt、parser 和模型实际输出是否一致。结果模型大量输出 plain text 或多动作，而环境 parser 按 XML single-action 格式解析，导致大量 invalid action 和重复图片。

## 问题

retry15 的 prompt 同时包含互相冲突的信息：

- single-action 约束：

```text
You must take exactly one action in each response. Do not output multiple actions and do not use '|'.
Your response should be in the format of:
<think>...</think><action>...</action>
```

- 旧 navigation hint 仍保留 multi-action 暗示：

```text
You can take multiple actions at a time ...
```

- user prompt 仍写：

```text
Decide your next action(s).
```

同时，`eval_mode` parser 期望 XML action 格式，但模型实际大量输出：

```text
think ...
action moveahead, moveleft
```

或多行：

```text
action moveahead
action moveleft
action moveahead
```

这些输出与 parser/单动作要求不匹配，造成 `action_is_valid` 很低、环境不动、图片重复。

## 正确做法

任何 rollout collection 扩大前，必须做 prompt/parser/model-output smoke，至少检查：

1. 打印完整 system prompt 和首轮 user prompt；
2. 确认没有互相冲突的动作说明；
3. 抽样模型前几轮原始 assistant 输出；
4. 统计 XML/action-token/ plain action 比例；
5. 统计单动作 vs 多动作比例；
6. 检查 parser 输出 `action_is_valid`、`action_is_effective`；
7. 打开图片序列确认有效动作后视角确实变化。

如果 prompt 要求单动作，prompt 中不能再出现 “multiple actions”、“action(s)” 等误导；parser 也必须和模型实际输出格式一致。

## 修复方向

- 清理 navigation prompt：移除 multi-action hints 和 `action(s)`；
- 或使用 checkpoint 训练时真实匹配的 prompt/parser；
- 在 full rollout 前用小样本统计 action validity 和格式正确率。
