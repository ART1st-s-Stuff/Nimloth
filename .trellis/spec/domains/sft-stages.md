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

## Stage 1 标准训练与早期阶段评估

Stage 1 标准运行入口为 `experiments/training/sft1/train.py`（委托 `stage1.workflow`），标准配置 `configs/training/sft1/format.yaml`。顺序为 action-token prompt 派生、语义动作 token 初始化、预处理缓存、训练/完整恢复；训练核心仍属于 `stage1.trainer`。动作边界初始化来自 EOS，动作编号来自对应语义词的原始 embedding/head；不得重置无关 tensor。Stage 2 不继承 Stage 1 专用默认值。

独立的早期 success rate 评估通过 `nimloth.training.sft.evaluation --stage vagen|stage1|stage2` 唯一入口路由，各 stage 的 eval 接口负责阶段验收。VAGEN 保留语义动作协议；Stage 1 训练和评估共用 action-token prompt 与动作协议；Stage 2 加载 checkpoint 指定的 query generate/inject 协议。直接评估不运行 WM/value/MCTS，不通过动作约束或强制动作前缀掩盖格式错误。inject 只能在同观测实际生成的 CoT 边界插入训练定义的 query。

早期服务使用原 batch navigation 协议，明确校验服务类型；环境动力学参数与模型 prompt 协议分开配置。模型原文、注入 token、解析结果、实际服务输入及环境反馈均持久化；无效输出不可修成有效动作。成功率来自完整 episode 的真实环境 success，部分完成须报告分母与完成范围。Stage 3/RL 路径本轮不迁移，保持其现有服务与规划合同。

Stage 1 现有训练入口可用 `--success-eval-env-url` 启用保存 epoch 后的环境评估，复用上述环境与动作协议，不增加可执行入口。`--success-eval-concurrency` 独立控制每 rank 的并发，不继承格式验证 batch size；部署前核验环境容量覆盖总并发。各 rank 使用当前模型处理互斥的 episode 集合，合同绑定 checkpoint（LoRA 包含基座）、采样配置、world size 和分配表。改变并行布局不得混用旧结果。

rank 独占记录目录；全局统计校验身份、重复记录、stage 和 success 类型，完整分母仍是请求的全部 episode。缺失记录只能产生部分结果；任何 rank 失败必须传播，不能将其排除后报告完成。长环境阶段通过独立 CPU 控制组同步，不挂起 NCCL collective；返回训练前恢复参数包装、模型模式与 RNG。成功率不隐式改变训练的收敛目标。

必须验证互斥且完整的任务分配、部分/完整汇总、已完成记录恢复、合同不匹配拒绝、跨 rank 错误传播和训练恢复。逐阶段耗时区分模型生成与环境操作；真实多卡吞吐和显存须远程核验。错误做法是重复评估相同样本再平均各 rank 成功率；正确做法是按唯一 episode 的成功数及完成数汇总，并明确是否全部完成。

早期评估已有 CLI 可通过 `--episode-manifest-parquet` 使用原 VAGEN test.parquet
的有序 extra_info 任务，而非连续 seed。必须校验集合/每组数量、唯一 identity、
测试 split 和支持的环境配置；原始文件 SHA256、有序 identity 与实际环境参数绑定
恢复合同，变更即拒绝。显式 manifest 不受连续 seed 的 60 条上限限制，不伪称
standard_heldout120，也不以 test 文件名推断训练独立性。省略参数保持原行为。
