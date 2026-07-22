# RL 代码质量与正确性改进清单

更新时间：2026-07-22

范围：`src/nimloth/training/rl/` 及其直接依赖的 Qwen2.5-VL policy、checkpoint、配置和 rollout 路径。

本文记录当前审计发现及分阶段修复结果。当前测试已覆盖多步 WM 窗口、SIGReg、
policy prompt/probability、独立 validation 与核心梯度边界；真实多卡 FSDP + EMA
仍受启动保护，尚未解除。

状态说明：

- **已确认错误**：源码能够直接证明训练语义或公开接口不正确。
- **已确认缺陷**：功能边界、配置或可恢复性不完整，但不一定在所有运行模式下触发。
- **待运行验证风险**：源码存在明确风险，需要对应环境的测试确认实际表现。
- **待设计决策**：当前行为可以有多种合理定义，必须先确定目标语义。

## P0：PPO 正确性

### 1. Rollout 使用当前图片伪造全部视觉历史

- 状态：**已修复（2026-07-21）**
- 修复：公共 `Agent` 保存每一步真实的 observation text/image/action；注册式 `NimlothPromptTemplate` 按原始顺序绑定 `images[:t+1]`。`EpisodeRunner` 负责真实环境交互，旧 `_select_action_nimloth()` 已删除。
- 验证：Agent prompt 测试检查多步真实图片顺序；rollout schema 要求 observations/images 均为 actions + 1。

### 2. old/new log probability 使用不同长度的 prompt

- 状态：**已修复（2026-07-21）**
- 修复：rollout 保存每一步完整的 `policy_messages`；PPO 不再截断最后四步，而是绑定该步全部历史图片后重放。进入训练前会从结构化 transcript 重建 prompt，并与保存的审计副本做完全相等比较。
- 验证：schema 测试覆盖 prompt 缺失、版本过期和 transcript/prompt 不一致。

### 3. 保存的 old log probability 不是实际采样分布

- 状态：**已修复（2026-07-21）**
- 修复：`backbone/qwen25vl/policy.py` 统一计算 temperature/top-p 后的行为分布；rollout 保存该分布和采样参数，PPO 用同一变换重算。entropy 也基于变换后的分布。
- 验证：测试覆盖无变换、top-p mask/重新归一化、greedy 和 `-inf` entropy。

### 4. 推理异常和缺失概率被伪装为合法 PPO 样本

- 状态：**已修复（2026-07-21）**
- 修复：推理失败会丢弃不完整 trajectory；trainer 禁止缺失概率和补零。Agent 公共校验器检查 8-way 长度、taken action、NaN/+inf、归一性，并允许 top-p 的 `-inf`。落盘以标准 JSON `null` 表示 `-inf`。
- 验证：schema 与 JSONL round-trip 测试覆盖非归一分布、缺失分布和 masked action。

### 5. 固定 JSONL 轨迹被无限循环用于 PPO

- 状态：**已修复（2026-07-21）**
- 修复：`JSONLRolloutCollector` 明确只承担离线 WM/value 数据源；任一 Qwen tune mode 启用 actor 时，trainer 在读取数据前直接拒绝静态 JSONL。CLI 还要求显式选择 env 或 JSONL 模式。
- 验证：严格配置/CLI 测试覆盖 rollout 模式契约；`actor.enabled` 与 Backbone tune
  mode 分别校验。

## P0：Checkpoint 完整性

### 6. LoRA + Vision Full checkpoint 丢失完整视觉权重

- 状态：**已修复（2026-07-21）**
- 修复：RL 和 SFT2 共用 `backbone/qwen25vl/checkpoint.py`；LoRA + Vision Full 会额外保存 `vision_full_state.pt`，恢复 adapter 时同步恢复视觉塔。
- 验证：Qwen checkpoint 测试覆盖视觉塔定位、保存和 adapter + full-vision 恢复。

## P1：Validation 与 checkpoint 选择

### 7. Validation 复用训练 collector 和训练数据来源

- 状态：**已修复（2026-07-21）**
- 修复：CLI 分别创建 train/eval collector；环境模式使用独立 dataset、split 和 seed range，JSONL 模式要求独立 held-out source。trainer 不再复用训练 cursor。

### 8. `best/` 根据训练 minibatch value loss 选择

- 状态：**已修复（2026-07-21）**
- 修复：`validation.checkpoint_metric` 显式选择最大化 `success_rate` 或 `avg_reward`；`best/` 只在独立 validation 后更新，`latest/` 单独保存可恢复进度。

## P1：配置与启动契约

### 9. YAML 非严格解析，大量字段静默无效

- 状态：**已修复（2026-07-21）**
- 修复：配置迁到 `nimloth.config.rl` 的不可变 dataclass schema；未知 section/field、字符串布尔值、非法概率和无效 validation 数量均直接报错，CLI override 返回新配置对象。

### 10. Actor 是否启用曾与 Qwen tune mode 错误耦合

- 状态：**已修复（2026-07-21）**
- 修复：删除含义重叠的 `freeze.qwen`，并明确三个独立契约：
  `--llm-tune/--vision-tune` 决定可训练参数，`actor.enabled` 决定是否计算 PPO，
  `gradient.representation_to_backbone` 决定 WM/value/SIGReg 是否回传 Backbone。
  actor 开启但 Backbone 全冻结时会在启动阶段直接报错。

### 11. README 中的 base Qwen 启动与协议校验冲突

- 状态：**部分修复，仍有 artifact 设计缺口**
- 位置：`src/nimloth/training/rl/README.md`、`src/nimloth/training/rl/cli.py`、`src/nimloth/training/rl/trainer.py`。
- 已修复：README 与 CLI 现在明确要求完整 `k=1 inject` HF checkpoint，不再声称 plain base Qwen 或 standalone PEFT adapter 可直接启动。
- 剩余问题：SFT2 产出的 PEFT adapter、materialized query embedding、Vision Full 和基础模型引用尚未形成一个可由 RL 直接消费的统一 manifest。需要先定义跨阶段 artifact 契约，再实现自动装载，不能把仅加载 WM heads 称为完整 SFT2 warm-start。

### 12. StateProjector 输入维度硬编码为 2048

- 状态：**已修复（2026-07-21）**
- 修复：`training/rl/trainer.py` 先加载 Backbone，再通过公共
  `backbone_hidden_size()` 兼容读取顶层或 `text_config.hidden_size`，随后创建
  StateProjector。

## P1：State 语义与梯度边界

### 13. WM state 与 policy state 使用不同 prompt

- 状态：**已修复（2026-07-21）**
- 修复：`rollout/windows.py` 直接从 `RolloutTrajectory.build_policy_prompt()`
  取得窗口内 `H+1` 个状态。online rollout、PPO 与 WM 都使用同一个
  `AgentPromptTemplate`；Qwen 层不再单独重建 RL transition。
- 验证：结构化 SFT2 transition 测试确认当前 supervised prefix 与下一状态 policy prefix 使用共享模板。

### 14. Value loss 不会更新解冻后的 StateProjector

- 状态：**已修复（2026-07-22）**
- 修复：ValueHead 直接消费 `state_context`，不再无条件 detach。
  `freeze.state_proj` 唯一控制 StateProjector 是否训练；新增
  `gradient.representation_to_backbone` 独立控制 WM/value/SIGReg 是否回传
  Backbone，`actor.enabled` 独立控制 PPO。
- 验证：已新增梯度测试，分别覆盖 joint 模式、frozen representation 模式、WM
  target stop-gradient 与 value 对 Backbone/StateProjector/ValueHead 的梯度；本地
  CPU 回归已通过。

## P1：分布式风险

### 15. FSDP FULL_SHARD 与 Vision EMA 的参数形状可能不兼容

- 状态：**已加启动保护，尚未实现多卡 EMA**
- 修复：多 rank 与 Vision EMA 同时启用时在 FSDP 包装前直接报错，避免进入未验证的 shard 更新路径。未来只有在真实多卡测试覆盖后才应解除限制。

## P2：公开接口与模块边界

### 16. VAGENRolloutCollector 是公开但不可用的入口

- 状态：**已修复（2026-07-21）**
- 修复：删除永远抛出 `NotImplementedError` 的公开入口和对应 CLI；真实 collector
  归入 `nimloth.environment.navigation.VAGENNavigationRolloutCollector`，并通过
  通用 `AgentPolicy` 注入模型行为。

### 17. `history_size > 1` 仍按独立 transition 训练 WM

- 状态：**已修复（2026-07-21）**
- 修复：原始 trajectory 保留连续 step；RL 先按 `H=history_size` 采样同一
  trajectory 的 H-step window，再按显式梯度模式编码 H+1 个状态，以它们监督 H 个
  因果预测。SIGReg 消费完整 `(T=H+1,B,D)` 状态序列，episode 开头使用真实短前缀。
- 验证：窗口边界、shape、target 对齐、梯度 stop、可变 history 和 SIGReg 时间轴
  均有单元测试。禁止再用强制 `history_size=1` 或虚构 padding 掩盖问题。

## 建议实施顺序

1. 先统一 policy prompt、真实历史图片、采样分布和轨迹 schema，补齐 PPO 行为概率契约测试。
2. 修复异常样本处理和 JSONL policy provenance；明确多卡 actor 的新 rollout 机制。
3. 修复 LoRA + Vision Full checkpoint，并做所有可训练参数的 round-trip 测试。
4. 拆分 train/eval collector，定义真实 validation 与唯一 checkpoint metric。
5. 引入严格 RL config，消除 YAML/CLI 双重开关和硬编码维度。
6. 统一 policy state 与 WM state；决定 StateProjector 梯度 ownership。
7. 用真实多卡测试验证 FSDP + Vision EMA，再确定实现。
8. 最后清理死入口，并按可顺序阅读的算法、训练生命周期和 collector 边界整理；
   禁止再用宽泛 `components` 或单函数 `objective/schedule/update` 文件横向切碎流程。

当前架构进展：公共神经网络 `Agent(backbone, wm)`、episode `AgentRuntime`、
rollout 和 Agent/Rollout config 已迁出 training。Qwen policy/replay 位于
`backbone/qwen25vl`，VAGEN collector 位于 `environment/navigation`。
2026-07-22 进一步删除 Qwen rollout encoder：原始连续窗口由 `rollout/windows.py`
采样，RL runtime 再通过公共 input builder 执行 joint/no-grad Backbone forward。
SFT2 的 next-state 对齐进入 `training/sft2/batch.py`，Qwen 代码不再认识训练阶段。
`WorldModel` 只组合 StateProjector、WMPredictor、ValueHead 并负责神经网络计算。
SFT2/RL 的单批 forward、loss 和梯度边界分别集中在各自 `algorithm.py`；二者
都是不持有模型/optimizer 的普通算法对象，通过阶段 runtime 显式接收 Agent、
target 或 policy replay。公共 `OptimizationRuntime` 管理 backward、no_sync、
裁剪、optimizer step 与 EMA callback。SFT2 loop 的 reporting/checkpoint 触发已
拆到完整职责模块。原先宽泛的 `components` 以及只做转发的
`objective/schedule/update` 层已删除。

## 完成标准

- rollout 与 PPO 对相同轨迹 step 生成逐 token 相同的 prompt/input，且 taken action 的 old/new probability 来自同一分布定义。
- 任何缺失、非有限、长度错误或来源不明的行为概率都会在进入训练前失败。
- actor 开启时，训练数据有可验证的行为策略来源和有限复用周期。
- LoRA、Vision Full、WM heads、optimizer 和 EMA 的保存/恢复均有 round-trip 测试。
- train/eval 数据源、seed 与指标完全分离，`best/` 只由显式 validation metric 选择。
- 所有 YAML 字段要么被消费，要么在解析时被拒绝。
- StateProjector 的梯度来源有明确设计和自动化测试。
- FSDP + Vision EMA 通过真实多卡测试，或在不支持时启动即报错。
