# Qwen2.5-VL Backbone

| 文件 | 职责 |
|------|------|
| `model.py` | `Qwen25VLBackbone`：Qwen forward 与 latent query 提取 |
| `factory.py` | 阶段无关的模型加载、tuning 与独立能力构造 |
| `input.py` | Agent 消息/图片到 `BackboneBatch` 的通用输入适配 |
| `batch.py` | chat rendering、图片处理、CE label 与 tensor collate |
| `policy.py` | Qwen direct policy score 与 masked-token PPO replay 适配器 |
| `vllm_policy.py` | 独立 vLLM staged CoT/action behavior backend；不承担训练 |
| `checkpoint.py` | PEFT 与 full vision artifact |
| `latent.py` | final hidden 捕获与 latent query 提取 |
| `tuning.py` | LLM/vision `freeze | lora | full` 配置 |
| `vision_ema.py` | 可训练视觉参数 EMA |
| `monkey_patch.py` | 只供诊断脚本启用的局部实验 patch |

Qwen 代码不计算 return、不采样 trajectory window，也不创建 SFT2/RL batch。
训练代码通过 `nimloth.backbone` 的公共构造入口获取 Backbone、input builder、
policy、PPO replay 和 EMA，各能力彼此独立。

vLLM turn mode 先采样 CoT，再注入 latent query/action 边界，最后在八个 action
token 上采样动作；trajectory 只给真实采样 token 保存 old log-prob。训练 replay
用 `logits_to_keep` 只计算 loss-mask 位置的 vocabulary logits，reasoning 使用屏蔽
Nimloth 注入 token 的词表，action 使用八 token 词表。
