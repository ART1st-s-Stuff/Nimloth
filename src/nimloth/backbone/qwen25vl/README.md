# Qwen2.5-VL Backbone

| 文件 | 职责 |
|------|------|
| `model.py` | `Qwen25VLBackbone`：Qwen forward 与 latent query 提取 |
| `factory.py` | 阶段无关的模型加载、tuning 与独立能力构造 |
| `input.py` | Agent 消息/图片到 `BackboneBatch` 的通用输入适配 |
| `batch.py` | chat rendering、图片处理、CE label 与 tensor collate |
| `policy.py` | Qwen direct policy score 与 masked-token PPO replay 适配器 |
| `vllm_policy.py` | 独立 vLLM 单请求 CoT/action behavior backend；不承担训练 |
| `turn_generation.py` | turn continuation 的可测试 token 状态机与 logits mask |
| `vllm_logits.py` | 把 turn 状态机接入 vLLM V1 per-request logits processor |
| `checkpoint.py` | PEFT 与 full vision artifact |
| `latent.py` | final hidden 捕获与 latent query 提取 |
| `state_training.py` | SFT1-v2：验证真实 archived response/CoT，并在同一 Qwen forward 返回 K16 hidden 与 action-boundary 八动作 logits |
| `tuning.py` | LLM/vision `freeze | lora | full` 配置 |
| `vision_ema.py` | 可训练视觉参数 EMA |
| `monkey_patch.py` | 只供诊断脚本启用的局部实验 patch |

Qwen 代码不计算 return、不采样 trajectory window，也不创建 SFT2/RL batch。
训练代码通过 `nimloth.backbone` 的公共构造入口获取 Backbone、input builder、
policy、PPO replay 和 EMA，各能力彼此独立。

vLLM turn mode 在一个多模态 request 中采样 CoT、约束注入 latent/action 边界并在
八个 action token 上采样动作，图片只经过一次 vLLM processor。trajectory 只给
真实采样 token 保存 old log-prob，并绑定 action token mapping、assistant response、
reasoning finish reason 和 truncation 状态。训练 replay 用 `logits_to_keep` 只计算
loss-mask 位置的 vocabulary logits，reasoning 使用屏蔽 Nimloth 注入 token 的词表，
action 使用八 token 词表；注入或强制补全的 token 不进入 PPO。
SFT1-v2 state-training 是独立的可微能力：输入必须携带每行真实 archived assistant
response/CoT provenance。多轮prefix也可能在system prompt中包含格式示例；所有结构pair必须
完整相邻，只选择消息序列最后一个current pair，并让其K16 hidden和八动作logits来自同一次
模型forward。普通 `Backbone.forward()`、Agent policy与PPO replay的既有输出不因此改变。
`Qwen25VLBackbone.save_pretrained()`可接收官方FSDP聚合后的完整state dict；query delta
只在该完整副本中materialize到embedding行，训练中的sharded参数保持不变，adapter私有key
不会进入HF artifact。
latent query的注入边界按tokenizer解码后的字面`</think>`匹配，而不是假设该文本只有
一种token ID切分；达到reasoning上限时才强制补入canonical close token序列。
