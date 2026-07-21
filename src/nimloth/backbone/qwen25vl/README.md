# Qwen2.5-VL Backbone

| 文件 | 职责 |
|------|------|
| `model.py` | `Qwen25VLBackbone`：Qwen forward 与 latent query 提取 |
| `factory.py` | SFT2/RL 模型加载、tuning 与运行期适配器装配 |
| `batch.py` | chat rendering、图片处理、CE label 与 tensor collate |
| `transition.py` | `Qwen25VLBatchBuilder`、下一状态去重与 cache 适配 |
| `policy.py` | 在线动作 policy 与独立 PPO replay 适配器 |
| `rollout.py` | 独立 rollout encoder |
| `checkpoint.py` | PEFT 与 full vision artifact |
| `latent.py` | final hidden 捕获与 latent query 提取 |
| `tuning.py` | LLM/vision `freeze | lora | full` 配置 |
| `vision_ema.py` | 可训练视觉参数 EMA |
| `monkey_patch.py` | 只供诊断脚本启用的局部实验 patch |

训练阶段不会通过一个万能 backend 间接访问这些能力。batch builder、policy、
rollout encoder、PPO replay 和 EMA 是职责独立的对象。
