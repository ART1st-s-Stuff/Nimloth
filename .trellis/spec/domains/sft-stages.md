# 三阶段SFT接口

## 1. 范围
适用于`training/sft/`的训练、checkpoint阶段识别与环境评估接线。算法概要在该模块spec.md；实现职责和入口见README。

## 2. 入口
- `python -m nimloth.training.sft.stage1`：格式训练。
- `python -m nimloth.training.sft.stage2`：Query/DINO对齐。
- `python -m nimloth.training.sft.stage3`：迁移后的原SFT2训练。
- `python -m nimloth.training.sft.evaluation --mode direct|wm`：显式环境rollout。
- Python评估入口：`eval_direct(config: EvaluationConfig) -> int`及`eval_wm(config: EvaluationConfig) -> int`；返回执行退出码，指标保存在配置指定输出目录。

## 3. 数据和兼容合同
新stage2不是历史`sft2`。旧`training.sft1`、`training.sft2` Python 包已移除，活动调用使用`training.sft.stage1`或`stage3`。历史checkpoint字段仍指WM/value阶段；不改写已保存的历史产物。stage3迁移保持旧目标、类型与恢复协议。

stage2需要同观测的真实回答/CoT、完整有序的query slots和冻结DINO grid；projector输出形状必须精确匹配teacher目标。空间grid、teacher身份和projector参数随checkpoint保存，供stage3校验读取。

评估配置显式指定checkpoint、episode集合/split/数量、seed、步数和生成参数。WM模式还要求搜索参数和现有完整checkpoint合同。direct不加载WM规划组件。环境反馈定义成功和回报，不用离线loss代替。

## 4. 验证与错误
- query与DINO轴、维度或token顺序不符：拒绝，不广播或补slot。
- 目标回答被截断、缺少同观测CoT：拒绝，不填固定文本。
- 不匹配的阶段/teacher/projector恢复：拒绝，不静默推断新语义。
- direct携带planner参数、wm缺少planner参数：配置校验拒绝。
- evaluation输出已存在但非显式resume、或resume合同不一致：拒绝覆盖。

## 5. 正常与异常情形
正常格式样本只监督回答token；正常query样本同时训练query/projector并保持DINO无梯度。正常H=1多步窗口对每步未来state及动作前Q进行监督。缺失query、错配teacher和把旧SFT2解释为query阶段均属于异常。

## 6. 必须验证
训练mask、DINO冻结、query/projector梯度、checkpoint字典/张量往返、canonical导入路径、训练包不依赖实验诊断；评估mode接线、resume合同、真实观测驱动重新规划与环境步计数。隔离环境替身仅验证接口，真实GPU/环境验证遵循实验合同。

## 7. 错误与正确
错误：因目录从sft2迁到stage3，重写WM目标、改变Q评分时刻或把旧checkpoint标为query。
正确：保留stage3旧计算和恢复语义，用明确的新stage2实现Query对齐，通过版本/阶段字段验证交界。

## 模块边界补充
`stage3.algorithm`拥有损失和反传顺序，`stage3.sigreg`拥有跨rank有效状态汇聚及同步随机投影，提取时不改变collective顺序或梯度。canary、动作头专项修复、packed/KV原型及依赖它们的特征审计归`experiments/training/sft/diagnosis`，不得由生产训练导入。Python旧路径不再兼容；保存的tensor/state_dict、目标标识及恢复语义保持不变。

`stage1.cli.parse_args(argv=None, *, stage="format")`负责入口参数校验；`stage1.checkpoint`负责保存及恢复阶段校验；`stage1.trainer`负责模型构建与训练生命周期，不再动态转发数据模块中的任意属性。数据调用者直接依赖`stage1.data`。
