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

## 经审查的选择性 LM 监督
Stage2 使用全部轨迹的回答对齐 DINO，只对成功轨迹的回答计算 LM；Stage3 使用全部窗口的 WM/value/DINO，只对成功轨迹起点回答计算 LM。success 必须来自完整轨迹的显式布尔字段，缺失拒绝。LM 分母为成功回答/窗口数，其他监督分母为全部有效回答/窗口数，跨累积组和 rank 分别归约；全失败组 LM 为图连接的零。恢复身份须拒绝旧全部轨迹 LM 目标的优化器状态。

## Stage3 outcome 与有限时域转换

### 范围和入口
Stage3 可选 `--outcome-head --lambda-outcome 1 --outcome-head-lr 1e-4`；
对照组同样实例化 head，但系数0、参数冻结，不宣称具备 outcome 能力。
`python -m nimloth.rollout.tail_drop` 将已审计 SFT view 转为真实 T+1 观测的截断记录。

### 数据与梯度合同
`action_successes` 与 outgoing actions 等长，标签仅取对应下一观测的环境反馈。
`action_value_targets` 必须带 `finite_horizon_provenance`；完整 original_rewards/dones
先计算 return，再删除最后一个监督位置。原始任务终点 bootstrap=0 不意味着人为切点
未来回报为0。保留原轨迹 success 用于 LM mask，不推断未执行动作的结果。
Outcome head 复用预测 grid，同一 WM 前向产生 logit；普通 BCE 按跨rank/累积组的
有效动作计数平均，不按类别加权、不按初始loss归一化。无标签和padding不参与。
`query_tune=selected_rows` 保持未选词表行冻结、选行 FP32 master 和准确恢复sidecar。

### 验证与错误
缺失/错位reward、done、query顺序或源反馈：拒绝转换；历史raw hash无法核实时明确
保留为未验证声明，不冒充当前输入hash。恢复必须核对outcome开关、系数、head和选行身份。
联合训练下目标state可漂移，跨组质量主要比较固定DINO；copy基线也使用固定teacher的
当前观测。LM CE只按成功窗口平均，无成功窗口不报告伪零。分类准确不能替代状态预测改善。

### 正常、边界与反例
正常20步失败轨迹保留前19个transition及原20步return；提前真实成功也先用完整reward。
单动作记录经完整校验后无可训练transition；两类不全时AUC等指标不可用。
错误：先删除末步reward再算return，或比较各组不同state目标的MSE来宣称DINO改善。
正确：从完整原轨迹计算目标，按固定teacher、同一trajectory配对比较。

### 必测合同
覆盖转换不改原数据、gamma/done/回报一致、后继outcome对齐、padding全局归约、
控制组等价、选行冻结及两步优化保存恢复。训练LM与独立SIGReg反传的head参与范围不同；
static DDP须有真实多rank交替反传测试，CPU测试不放行未经验证的Qwen/FA2八卡训练。
