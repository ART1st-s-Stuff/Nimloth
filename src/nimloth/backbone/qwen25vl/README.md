# Qwen2.5-VL Backbone

| 文件 | 职责 |
|------|------|
| `model.py` | `Qwen25VLBackbone`：Qwen forward 与 latent query 提取 |
| `factory.py` | 阶段无关的模型加载、tuning 与独立能力构造 |
| `input.py` | Agent 消息/图片到 `BackboneBatch` 的通用输入适配 |
| `batch.py` | chat rendering、图片处理、CE label 与 tensor collate |
| `policy.py` | 在线动作 policy 与独立 PPO replay 适配器 |
| `checkpoint.py` | PEFT 与 full vision artifact |
| `latent.py` | final hidden 捕获与 latent query 提取 |
| `tuning.py` | LLM/vision `freeze | lora | full` 配置 |
| `vision_ema.py` | 可训练视觉参数 EMA |
| `monkey_patch.py` | 只供诊断脚本启用的局部实验 patch |

Qwen 代码不计算 return、不采样 trajectory window，也不创建 SFT2/RL batch。
训练代码通过 `nimloth.backbone` 的公共构造入口获取 Backbone、input builder、
policy、PPO replay 和 EMA，各能力彼此独立。
