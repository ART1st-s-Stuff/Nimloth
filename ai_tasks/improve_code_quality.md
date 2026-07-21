# RL 代码质量与正确性改进清单

更新时间：2026-07-21

范围：`src/nimloth/training/rl/` 及其直接依赖的 Qwen2.5-VL policy、checkpoint、配置和 rollout 路径。

本文记录当前审计发现，供后续分阶段修复。除特别标注外，这些问题均来自源码静态审计；当前 RL 测试主要覆盖 JSONL schema、轮转确定性和 W&B，没有覆盖 policy prompt 等价性、PPO 行为概率、checkpoint 完整恢复、独立 validation 或 FSDP EMA。

状态说明：

- **已确认错误**：源码能够直接证明训练语义或公开接口不正确。
- **已确认缺陷**：功能边界、配置或可恢复性不完整，但不一定在所有运行模式下触发。
- **待运行验证风险**：源码存在明确风险，需要对应环境的测试确认实际表现。
- **待设计决策**：当前行为可以有多种合理定义，必须先确定目标语义。

## P0：PPO 正确性

### 1. Rollout 使用当前图片伪造全部视觉历史

- 状态：**已确认错误**
- 位置：`src/nimloth/training/rl/rollout.py`，`_select_action_nimloth()`。
- 现状：函数只接收当前图片和历史动作。构造历史 turn 时，每个历史 user turn 都重复使用当前图片。
- 影响：真实轨迹 `I0 --a0--> I1 --a1--> I2` 会被编码成近似 `I2 --a0--> I2 --a1--> I2`。动作历史与视觉历史矛盾，序列越长误差越大。
- 修复方向：统一 policy-input builder，显式接收 `images[:t+1]`、`actions[:t]` 和 instruction。rollout、PPO 重算、WM state 编码必须复用同一实现。

### 2. old/new log probability 使用不同长度的 prompt

- 状态：**已确认错误**
- 位置：`src/nimloth/training/rl/rollout.py::_select_action_nimloth()`；`src/nimloth/training/rl/trainer.py::compute_new_log_probs_for_batch()`。
- 现状：rollout 使用完整动作历史；trainer 重算 `new_log_prob` 时只保留最后四步历史。
- 影响：第 5 步以后，PPO 计算的是 `exp(log pi_new(a|prompt_B) - log pi_old(a|prompt_A))`。该值不是同一条件分布上的策略概率比，prompt 差异会被误当作参数更新。
- 修复方向：rollout 和 trainer 必须共用完全相同的 prompt 构造及截断规则；轨迹中需要保存可精确重放行为输入的信息。

### 3. 保存的 old log probability 不是实际采样分布

- 状态：**已确认错误**
- 位置：`src/nimloth/training/rl/rollout.py::_select_action_nimloth()`。
- 现状：先保存原始 `log_softmax(action_logits)`，再对 logits 应用 temperature 和 top-p，并从变换后的分布采样。
- 影响：动作来自 `q_old = top_p(softmax(logits / temperature))`，保存的却是 `p_old = softmax(logits)`。PPO 分母不是行为策略的真实概率；top-p 排除动作在两个分布中的概率甚至分别为正数和零。
- 修复方向：选择并统一 policy 定义。可以固定 `temperature=1, top_p=1` 使用原始 categorical，也可以让 rollout 与 trainer 都计算并记录 temperature/top-p 后的真实分布。

### 4. 推理异常和缺失概率被伪装为合法 PPO 样本

- 状态：**已确认错误**
- 位置：`src/nimloth/training/rl/rollout.py::EnvRolloutCollector.collect()`；`src/nimloth/training/rl/trainer.py::build_rl_transitions()`。
- 现状：动作推理异常时强制选择 `moveahead` 并写入 `[0.0] * 8`；缺失概率时同样把 taken action 的 `old_log_prob` 补成 `0.0`。
- 影响：log probability 为零表示概率为 1；八个动作同时为 1 不是合法分布。训练会把模型没有选择的兜底动作当作旧策略以 100% 概率执行的动作。
- 修复方向：推理失败时丢弃 step/trajectory 或明确终止；JSONL 加载时严格验证概率长度、有限性、归一性和 taken action 索引，禁止默认补零。

### 5. 固定 JSONL 轨迹被无限循环用于 PPO

- 状态：**已确认缺陷**
- 位置：`src/nimloth/training/rl/rollout.py::JSONLRolloutCollector`。
- 现状：collector 首次读取后缓存全部轨迹，数据耗尽便从头循环，不会刷新行为策略或概率来源。
- 影响：WM/value 可以使用离线数据，但 PPO 会在当前策略持续变化后，长期复用旧策略轨迹和旧 `old_log_prob`，逐渐变成严重离策略训练。JSONL 也没有记录或校验 policy checkpoint、prompt 版本和采样配置。
- 修复方向：actor 开启时周期性用当前策略生成新 rollout，或改用明确支持离策略数据的算法；轨迹 schema 增加 policy provenance 和 sampling configuration。actor 关闭时可以继续把 JSONL 用作离线 WM/value 数据。

## P0：Checkpoint 完整性

### 6. LoRA + Vision Full checkpoint 丢失完整视觉权重

- 状态：**已确认错误**
- 位置：`src/nimloth/training/rl/checkpoint.py::save_rl_checkpoint()`、`load_lora_adapter_state()`；`src/nimloth/backbone/qwen25vl/tuning.py::configure_qwen_tuning()`。
- 现状：LoRA 模式下 `modules_to_save` 为空，checkpoint 对 `PeftModel` 调用 `save_pretrained()`。该调用保存 adapter，但普通 full-vision 参数不属于 adapter。恢复时也只加载 adapter。
- 影响：`--llm-tune lora --vision-tune full` 训练得到的 LLM LoRA 可以恢复，Vision Full 更新会丢失。`vision_ema.pt` 只保存 EMA shadow，恢复代码没有把它应用到主模型，不能代替主视觉权重。
- 修复方向：复用 `src/nimloth/backbone/qwen25vl/checkpoint.py`，独立保存和恢复 adapter 与 `vision_full_state.pt`；增加测试，确保每个 `requires_grad=True` 参数都能 checkpoint round-trip。

## P1：Validation 与 checkpoint 选择

### 7. Validation 复用训练 collector 和训练数据来源

- 状态：**已确认错误**
- 位置：`src/nimloth/training/rl/trainer.py` validation rollout；`src/nimloth/training/rl/cli.py` collector 创建。
- 现状：训练和 validation 调用同一个 collector。Env 模式将其配置为 `split="train"` 和 `*_train` datasets；JSONL 模式则从同一 cursor 继续读取静态轨迹。
- 影响：Env 模式没有 held-out 环境；JSONL 模式根本不运行当前策略，`val_success_rate` 只是下一批离线记录已有的 success 标签。
- 修复方向：建立独立的 `train_collector` 和 `eval_collector`，分别配置数据集、seed 范围与采样策略。离线评估应命名为 held-out dataset metric，不能冒充当前策略 rollout 指标。

### 8. `best/` 根据训练 minibatch value loss 选择

- 状态：**已确认错误**
- 位置：`src/nimloth/training/rl/trainer.py` checkpoint 部分。
- 现状：`current_val` 实际取自当前随机训练 minibatch 的 `value_loss`，没有使用 `val_success_rate`、`val_avg_reward` 或 held-out loss；best 比较只在 `save_interval` 触发。
- 影响：`best/` 的真实含义是“保存时刻训练 minibatch value loss 最低”，容易被采样噪声支配，也无法表示策略质量。
- 修复方向：显式配置 checkpoint metric 及 mode，例如最大化 held-out `eval/success_rate` 或最小化 held-out `eval/value_loss`；统一 validation 和 checkpoint 的触发契约。

## P1：配置与启动契约

### 9. YAML 非严格解析，大量字段静默无效

- 状态：**已确认缺陷**
- 位置：`src/nimloth/training/rl/cli.py::load_rl_config()`、`merge_config_overrides()`。
- 现状：配置仅用 `yaml.safe_load()`。当前未被实现消费的字段包括 `qwen.*`、`dataset.*`、`predictor.rollout_steps`、`rl.train_steps_per_iteration` 和 `training.output_dir`。
- 影响：配置看起来已经设置模型、验证集或每轮训练步数，实际运行行为不变，容易产生错误实验结论。
- 修复方向：建立 RL-owned 严格 schema；未知字段直接报错；每个字段有唯一 owner；明确 YAML/CLI 覆盖优先级并输出最终配置。

### 10. Actor 是否启用与 Qwen 是否可训练由两套开关控制

- 状态：**已确认错误**
- 位置：`src/nimloth/training/rl/trainer.py` actor setup；`src/nimloth/backbone/qwen25vl/tuning.py`。
- 现状：是否计算 actor loss 由 YAML `freeze.qwen` 决定；参数 `requires_grad` 由 CLI `--llm-tune/--vision-tune` 决定。
- 影响：`freeze.qwen=true + --llm-tune lora` 会创建 LoRA 但不计算 actor loss；`freeze.qwen=false + tune=freeze` 会记录 actor loss但策略参数不会更新。
- 修复方向：只保留一套权威 tune mode；根据最终可训练参数决定 actor 是否启用；遇到矛盾配置直接报错。

### 11. README 中的 base Qwen 启动与协议校验冲突

- 状态：**已确认错误**
- 位置：`src/nimloth/training/rl/README.md` 启动示例；`src/nimloth/training/rl/rollout.py::validate_rl_policy_protocol()`；`src/nimloth/training/rl/trainer.py` model setup。
- 现状：README 允许直接传入标准 Qwen base model，但 trainer 在 resize special tokens 及 tuning 前要求 config 已包含 `nimloth_latent_query_mode="inject"`。
- 影响：标准 Qwen config 缺少该字段，会在启动早期直接报错。README 的 “SFT2 warm-start（LLM LoRA + Vision Full）” 示例也只加载 WM/state/value，没有加载 SFT2 Qwen adapter 和 full-vision state。
- 修复方向：明确区分“完整 SFT2 HF export”与“base Qwen + SFT2 adapter + full vision state”两种启动模式，并让 CLI 对应地完整加载和验证。

### 12. StateProjector 输入维度硬编码为 2048

- 状态：**已确认缺陷**
- 位置：`src/nimloth/training/rl/cli.py` WM module setup。
- 现状：Qwen 加载前就创建 `StateProjector(qwen_hidden_dim=2048, ...)`。
- 影响：更换 Qwen 尺寸时会出现 checkpoint shape 或 forward 矩阵维度错误。
- 修复方向：先加载模型，再从实际 model config 推导 hidden size；把 Qwen 及依赖其维度的模块放入统一 component factory。

## P1：State 语义与梯度边界

### 13. WM state 与 policy state 使用不同 prompt

- 状态：**已确认错误**
- 位置：`src/nimloth/training/rl/trainer.py::encode_trajectory_hiddens()`。
- 现状：每帧使用 generic、独立的 image prompt 编码 latent，没有 instruction、历史图片或历史动作；policy 选动作时使用另一套带 instruction/history 的 prompt。
- 影响：WM predictor/value head 学习的 state 与 policy 实际决策 state 语义不同，也可能破坏从 SFT2 warm-start 的 latent 表示契约。
- 修复方向：建立唯一的 Qwen policy-state encoder，让 rollout、PPO、WM predictor 和 value head 共享完全相同的状态定义。

### 14. Value loss 不会更新解冻后的 StateProjector

- 状态：**待设计决策**
- 位置：`src/nimloth/training/rl/trainer.py` value loss 前的 `wm_state = state_proj(...).detach()`。
- 现状：predictor loss 可以更新未冻结的 StateProjector；value loss 因 detach 只能更新 ValueHead。
- 影响：当 `freeze.state_proj=false` 时，value supervision 无法塑造 state representation。若设计目标本来就是只由 dynamics loss 更新 projector，则当前行为合理，但必须明确记录。
- 修复方向：先确定 StateProjector 的梯度 ownership，再保留或删除 detach，并增加梯度路径测试。

## P1：分布式风险

### 15. FSDP FULL_SHARD 与 Vision EMA 的参数形状可能不兼容

- 状态：**待运行验证风险**
- 位置：`src/nimloth/training/rl/trainer.py` EMA/FSDP setup；`src/nimloth/backbone/qwen25vl/vision_ema.py`。
- 现状：EMA 在 FSDP 包装前用完整视觉参数初始化；FSDP FULL_SHARD 包装后，EMA 在 forward 外直接遍历 `param.data` 更新 shadow。
- 风险：`use_orig_params=True` 在 forward 外可能暴露 local shard 或空参数，导致 full shadow 与 shard shape 不一致、仅更新局部分片或 copy 失败。
- 修复方向：增加真实多卡 FSDP+vision-full EMA 测试；根据结果使用 FSDP full-param context、sharded EMA，或只在可安全收集 full state 的阶段维护 EMA。

## P2：公开接口与模块边界

### 16. VAGENRolloutCollector 是公开但不可用的入口

- 状态：**已确认缺陷**
- 位置：`src/nimloth/training/rl/rollout.py::VAGENRolloutCollector`；`src/nimloth/training/rl/cli.py`。
- 现状：CLI 暴露 `--vagen-config/--vagen-checkpoint` 并可构造 collector，但 `collect()` 永远抛出 `NotImplementedError`。
- 影响：形成死入口，增加使用者和维护者判断成本。
- 修复方向：删除 CLI 暴露及 package re-export，或完整实现并增加端到端测试。

## 建议实施顺序

1. 先统一 policy prompt、真实历史图片、采样分布和轨迹 schema，补齐 PPO 行为概率契约测试。
2. 修复异常样本处理和 JSONL policy provenance；明确多卡 actor 的新 rollout 机制。
3. 修复 LoRA + Vision Full checkpoint，并做所有可训练参数的 round-trip 测试。
4. 拆分 train/eval collector，定义真实 validation 与唯一 checkpoint metric。
5. 引入严格 RL config，消除 YAML/CLI 双重开关和硬编码维度。
6. 统一 policy state 与 WM state；决定 StateProjector 梯度 ownership。
7. 用真实多卡测试验证 FSDP + Vision EMA，再确定实现。
8. 最后清理死入口，并按 `config / components / engine / collectors` 拆分 `trainer.py` 和 `rollout.py`。

## 完成标准

- rollout 与 PPO 对相同轨迹 step 生成逐 token 相同的 prompt/input，且 taken action 的 old/new probability 来自同一分布定义。
- 任何缺失、非有限、长度错误或来源不明的行为概率都会在进入训练前失败。
- actor 开启时，训练数据有可验证的行为策略来源和有限复用周期。
- LoRA、Vision Full、WM heads、optimizer 和 EMA 的保存/恢复均有 round-trip 测试。
- train/eval 数据源、seed 与指标完全分离，`best/` 只由显式 validation metric 选择。
- 所有 YAML 字段要么被消费，要么在解析时被拒绝。
- StateProjector 的梯度来源有明确设计和自动化测试。
- FSDP + Vision EMA 通过真实多卡测试，或在不支持时启动即报错。
