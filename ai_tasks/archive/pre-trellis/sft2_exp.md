--------
本文为人类编写。如需修改需要得到人类同意。
--------

# 第二阶段SFT实验
在第一阶段SFT的基础上，让<|latent_state|>对应的隐状态保存足够的信息，能够被WM predictor正确预测.
除此之外, 添加一个Value head, 输入state, 输出所有的action value. 其监督信号来自(1) 真实的action value (2) Qwen选择的action,排序要高于其他未被选择的action.
在SFT2的阶段, 我们允许失败的run一同进入训练, 先让value head和Qwen的偏好对齐, 之后的RL阶段再进行性能优化.

WM predictor使用LeWM: https://github.com/lucas-maes/le-wm

## 数据集
这一阶段仍然使用之前收集到的rollout数据，选取train set进行训练，但是加入predictor的loss和Value head的loss.

## 参数选择
关于LLM backbone和Qwen vision encoder, 需要添加 冻结/LoRA/全量微调 共3组方案. Qwen vision encoder添加EMA更新参数的选项.

默认参数为冻结LLM backbone; 全量微调Qwen vision encoder, 并使其EMA更新.

## Misc 
需要记录训练过程中predictor的曲线，以及在val set上的成功率曲线.

## 实现注意事项

### Action value 监督（轨迹级回报）

Value head 的回归监督当前使用 **轨迹级 `reward` 的折扣回报**，而非逐步环境 reward：

- 数据来源：Nimloth rollout jsonl 中的 `reward` 字段（convert 脚本从 VAGEN 的 `reward` / `score` 写入）。
- 对轨迹中第 `t` 步（0-based）实际执行的 action，监督目标为  
  `target_t = reward * γ^(T - 1 - t)`  
  其中 `T` 为轨迹 action 步数，默认 `γ = 0.99`（见 `wm/dataset.py::discounted_action_value_targets`）。
- 仅对 Qwen 所选 action 做 MSE 回归；其余 action 通过排序损失约束（所选 value 应高于未选 action）。
- 若后续引入 step-level reward 或更精确的价值估计，需同步更新 dataset 中的 target 计算，并在此注明。

### WM predictor 与 Qwen latent

- 仅使用 LeWM **ARPredictor**（及 action encoder / pred_proj），**不使用** pixel encoder。
- WM 监督在 Qwen latent 空间：当前 `<|latent_state|>` hidden 经 `state_proj` 后由 predictor 预测下一步 latent；target 为下一步 prefix 的 `<|latent_state|>` hidden（`state_proj` 后 stop-grad）。

### Vision EMA

当 vision encoder 可训练（`vision_tune` 为 `full` 或 `lora`）时：

- 维护 vision 子模块可训练参数的 shadow：`θ_ema ← τ · θ_ema + (1 - τ) · θ`，默认 `τ = 0.999`。
- **WM 下一步 latent target** 的前向与 **验证** 默认使用 EMA vision 权重，以降低 target 抖动。
- 训练前向仍使用在线（非 EMA）权重；每个 optimizer step 后更新 EMA。
- checkpoint 另存 `vision_ema.pt`；resume 时一并加载。

### 数据与 split

- 训练使用 **train split**；允许失败 rollout 进入训练（与 SFT1 仅 success 不同）。
- 验证成功率在 val split 上统计，不得与 train 混用。
