# E0045：禁止 AI 自行发明 fixed CoT

## 错误

Coding agent 把 prompt 的结构前缀与 CoT 内容混为一谈，自行引入
`What should I do next?`，随后又把它标准化为模板默认值。SFT2 terminal state 直接
调用该默认值，PlanningPolicy 也因此使用了与真实 rollout 不同的 state 定义。

## 原因

- 没有向人类确认 state 是否允许固定 thought；
- 把“需要稳定的 latent query 位置”错误推导成“需要固定的 CoT 文本”；
- 用 `canonical`、`policy-independent` 等命名掩盖了未经审查的训练语义决策。

## 后果

- SFT2 每条轨迹的 terminal WM/SIGReg target 含伪造 CoT；
- 训练 state 与真实 rollout/deployment state 分布不一致；
- 依赖该 state 定义的历史实验结论需要重新审计，不能继续作为有效基线引用。

## 正确做法

1. 普通 state 只能读取该轮真实 assistant response 中的 CoT。
2. terminal observation 使用人类指定 checkpoint 与明确生成参数额外生成一次真实
   CoT并持久化；对应动作不执行。
3. 缺少 CoT 或生成协议时立即询问人类；不得提供 fixed fallback，也不得静默注入。
4. 删除错误路径并使旧数据/cache 明确失效，不能仅通过配置开关将其隐藏。

相关代码：`src/nimloth/agent/templates/nimloth.py`、
`src/nimloth/rollout/transitions.py`。
