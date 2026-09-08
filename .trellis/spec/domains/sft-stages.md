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

## SFT1显式续训段与PEFT导出

1. 范围：已完成epoch的scheduler已走完时，以新输出目录开始额外训练；普通`--resume`不接受改变world、数据或目标。
2. 入口：`--new-scheduler-segment-from PATH --epochs N --early-stopping-patience P --early-stopping-min-delta D`；`--keep-resume-checkpoints 2`限制本次恢复点数量。
3. 合同：新段保留模型和optimizer moments，明确重建scheduler并恢复正的初始LR。仅允许在完整epoch边界调整world与grad_accum且保持effective batch；段内resume校验完全相同的identity。保存scheduler_segment和segment_bad_epochs。新段保留最新/最佳epoch及原子best指针，不修改来源实验。
4. 错误：来源无COMMITTED、optimizer缺失、数据/目标不符、effective batch改变、同段身份变化均拒绝。PEFT导出input/output同时出现plain与modules_to_save别名时，选择后者训练副本；同优先级多候选或input/output混用别名类型仍拒绝。
5. 情形：world4/batch1/GA8完整epoch可开启world8/batch1/GA4新段；mid-epoch不能借新段跳过同world恢复限制。patience衡量离线val_loss停滞，不证明held-out质量收敛。
6. 验证：源LR为0而新段发生有效更新；moments保留；patience保存/恢复；拓扑变化保持batch；旧默认保存行为保留；retention不得越过新输出目录；PEFT两个别名数值不同时必须导出训练副本。
7. 错误与正确：错误是放宽普通resume identity或以零LR继续；正确是显式新段记录来源，并对同段恢复继续严格校验。

## SFT1只读格式诊断

1. 范围：标量格式正确率异常时，检查相同首轮生成单位，不修改训练和checkpoint。
2. 入口：`experiments/training/sft/evaluation/diagnose_stage1_format.py --model BASE --adapter CKPT --val-jsonl VAL --output-dir NEW`，默认32样本、128生成token、K1。
3. 合同：复用训练词表与严格adapter恢复；分别报告自由生成、参考thought条件动作生成、首个assistant的teacher-forcing。仅输出结构、类别、概率和运行身份，增量写samples.jsonl。
4. 错误：无GPU、非正预算、输出目录已存在、词表不同、adapter roundtrip不符、teacher目标截断均失败。
5. 情形：目标参考通过而自由生成不通过时，查看EOS/长度/marker分类；条件动作通过不能替代自由生成通过。
6. 验证：有效格式、仅动作、缺失闭合marker分类；EOS与长度上限独立计数；诊断结果不包含私有文本。
7. 错误与正确：参考文本短不能证明实际生成未截断；必须读取本次生成结构证据。离线loss不代替自由生成格式证据。

## SFT1 LoRA重试启动合同

SFT1旧LoRA对照配置为`--lora --lr 2e-4 --embedding-lr 5e-4`；`1e-6/5e-6`来自旧非LoRA全参分支，不得因迁移入口而混用。Stage2配置独立审查。

`run_stage1_fresh_retry.sh`通过ALLOCATION_JOB_ID/WORLD_SIZE/EXPECTED_NODE/REPO/EXPECTED_COMMIT/RUN_ROOT绑定已分配资源。WORLD_SIZE必须整除32，batch1/GA32÷world，完整一轮训练后单卡32样本格式诊断。RUN_ROOT必须新建，不能读取错误低LR旧optimizer状态。节点/GPU数/源码不符、端口占用和已存在输出均拒绝；中断清理本进程组，不自动重提。回归检查覆盖LoRA学习率配对、Stage2不被修改、无旧checkpoint恢复、shell语法及内嵌Python编译。诊断结果而非训练loss单独决定是否修复了格式问题。

续训launcher的`WORLD_SIZE`默认8、`EXPECTED_NODE`默认dgx-56；world必须整除32，GA=32/world，CPU=12×world、内存=60G×world。source仍限定已审查world4/batch1/GA8完整epoch；新run identity固定node/world/GA/LR，不静默接纳缺字段的旧identity。CPU门禁分别执行world4/8真实checkpoint preflight和内嵌Python编译；trainer严格恢复合同保持不变。

## SFT2旧数据CoT审计合同

### 1. 范围 / 触发

旧SFT1转换数据用于query/DINO Stage2前，必须先审计每个回答是否保留同观测的真实、非空CoT。旧转换器会把缺少`<think>`和空`<think></think>`都归一化为空CoT；不能等到GPU collator才发现，也不能补写CoT。

### 2. 命令

- 生成：`python -m experiments.training.sft.evaluation.stage2_inputs materialize --train-source TRAIN --val-source VAL --output-root NEW`
- 核验：`python -m experiments.training.sft.evaluation.stage2_inputs validate --input-root NEW --dino-cache-root CACHE --expected-train-source TRAIN --expected-val-source VAL`

### 3. 合同

派生目录使用`nimloth_stage2_nonempty_cot_inputs_v1`，包含`train.jsonl`、`val.jsonl`、`exclusions.jsonl`和`manifest.json`。任一回答无真实非空CoT时排除整条轨迹；sidecar记录split、record ID、输入行、assistant ordinal/message index、原始source行和`missing_think_tag`或`empty_think_body`。manifest记录原始/输出绝对路径及SHA256、前后轨迹/回答/图片计数、整轨迹排除数和sidecar hash。Stage2 run identity同时绑定manifest、源和输出hash。

### 4. 校验与错误矩阵

- train或val保留集为空 -> 拒绝生成/启动。
- 源路径/hash、输出路径/hash、计数、排除sidecar或重算结果不同 -> 拒绝。
- 转换后CoT与可读原始响应不同，或原始响应无法证明同一非空CoT -> 排除整条轨迹并记录原因。
- 保留图片缺少DINO grid4缓存 -> 拒绝。
- 派生目录已存在 -> 不覆盖；使用新的版本目录。

### 5. Good / Base / Bad

- Good：全部回答的转换CoT与原始响应一致；轨迹保留，DINO路径完整。
- Base：部分轨迹有空CoT；完整记录排除后，其余轨迹顺序和正文逐字保留。
- Bad：逐回答跳过、借用相邻CoT、填固定文本、只改manifest hash或让collator运行时随机发现。

### 6. 必须测试

测试需断言缺tag与空body分类、整轨迹排除、非空split、重复/缺失ID、原始与转换CoT不一致、源/输出/sidecar篡改、预期源路径绑定、DINO缺图，以及启动器在`srun`和模型加载前调用核验并把数据身份写入run identity。

### 7. 错误与正确

错误：把SFT1动作格式有效等同于SFT2 query输入有效，或放宽`answer_examples`的非空CoT门禁。正确：保留运行时fail-closed门禁，在GPU申请/使用前生成并核验可追溯的Stage2专用派生数据。
