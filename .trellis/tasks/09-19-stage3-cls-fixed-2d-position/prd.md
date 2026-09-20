# Stage3 CLS 与固定二维位置编码实验

## Goal

为世界模型提供 DINOv2 CLS 全局视觉目标及固定二维空间位置编码，优先检验显式全局
场景信息与二维几何是否能改善未来空间 state/DINO 预测及其 reconstruction。Stage3 完成后
从头训练一个显式 K64 spatial + K1 CLS 双分支 CFM，并以 CLS 消融确认 decoder 是否使用
全局槽位；本任务不增加新的
ValueHead/OutcomeHead pooling 对照。训练结果只作为评估证据，不作为正式模型或后续
RL 起点。

## Confirmed facts

- 当前 DINO teacher 只截取 patch tokens 并池化为 row-major `8x8x1024`；CLS 在缓存前
  被丢弃（`src/nimloth/backbone/dino_grid.py:151-170`）。已有 sidecar 无法恢复 CLS，
  本实验需要从原图重新执行冻结 DINO teacher 并生成有独立 lineage 的新缓存。
- 当前 Stage2 epoch16 state checkpoint 只有 64 个有序 Query slots；projector 要求输入、
  输出和 DINO target 的 token 轴完全一致（`src/nimloth/training/sft/stage2/model.py:61-95`）。
- 已决定新增一个真正的 Qwen CLS Query token，而不是从 K64 state 池化。该 token 放在
  64 个 spatial Query 之后、action start 之前：在 causal decoder 中它可以读取全部
  spatial Query，而已有64个槽位不会读取新增 token；同时它仍属于动作执行前的 state，
  不会看到待执行 action。
- 已决定本次从现有 Stage2 epoch16 增补 CLS、只训练新增 CLS token row，属于评估训练。
  正式训练必须在未来新的 Stage2 run 中从 epoch1 就加入 CLS，并与空间 Query 共同训练；
  不得把本次增补 checkpoint 改名或晋升为正式 Stage2/Stage3 结果。
- 当前 Stage3 residual WM 的 body 使用可训练的逐槽位 `spatial_position` 和
  `temporal_position`；本实验要求空间部分改为非训练的固定二维编码。零初始化 delta
  head 必须继续保证训练前预测逐值等于输入 state。
- 最近可比 Stage3 基线从同一 Stage2 epoch16 fresh start，使用 K64/grid8、H1/T4、
  residual WM、DINO2、Outcome BCE1、SIGReg0；WM/Value/Outcome 梯度在 Qwen hidden
  处停止但允许更新 projector。对照配置和固定数据评估口径沿用该基线。
- 当前 `spatial_grid_v1` CFM 按 row-major 将 K64 恢复为 `8x8`，并注入固定 `(x,y)`
  坐标；它不能把第65个 CLS 伪装成空间格点。本轮新增双分支 CFM：前64槽沿用空间
  conditioning，最后1槽经独立 global conditioning 注入，decoder 从头训练。
- 当前 ValueHead 与 OutcomeHead 都在进入非线性读出前对空间槽位做 mean pooling，无法
  区分均值相同而空间布局不同的 state。用户已明确该问题早于 CLS 接口，但本轮主要关心
  reconstruction，暂不增加 action-conditioned attention pooling 对照；该缺陷不得在本轮
  结论中被解释为已修复。

## Requirements

### R1. DINO target and cache

- 新 DINO target 必须同时保存 `cls`（`1024`）和 pooled spatial grid
  （`64x1024`），保留 teacher source/revision、processor fingerprint、grid size、dtype、
  图像索引和父数据 fingerprint。
- CLS 必须来自冻结 DINOv2 `last_hidden_state[:, 0]`；空间 grid 继续只使用最后
  `patch_count` 个 patch tokens。不得用 patch mean、已有 state mean 或其他 proxy
  冒充 DINO CLS。
- 新缓存使用新 schema/目录，旧 cache 保持只读；train/eval 行、图像身份和 hash 必须
  与当前 Stage3 数据逐项核验。

### R2. Fixed two-dimensional position encoding

- 64 个空间槽位使用确定性的 `8x8` 二维 sine-cosine 编码，shape 为
  `1x64x1024`，作为 buffer 保存，不进入 optimizer。
- 编码按 row-major 对齐 DINO grid；实现必须验证不同槽位编码不同、同一配置跨进程/
  保存恢复逐值一致，且不依赖随机种子。
- CLS 全局槽位使用显式 non-spatial slot metadata 与固定零空间位置向量，不把它伪装成
  某个二维格点；空间 attention 允许 CLS 与全部 spatial slots 双向交互。
- 当前可训练 `spatial_position` 不与固定二维编码叠加，避免“固定+学习位置”成为额外
  混杂变量。H1 的 temporal position 语义保持不变。

### R3. Stage3 training comparison

- 先从现有 Stage2 epoch16 checkpoint 继续一个新的 CLS alignment Stage2，而不是直接
  在 Stage3 同时学习当前观测表示和未来动力学，也不从原始 actor 重训已有 K64 state。
  本次只训练新增 CLS token embedding row，冻结已有64个Query、Qwen backbone和shared
  projector；同时报告 CLS DINO、spatial DINO、LM/格式及策略 success，形成明确标记为
  evaluation-only 的 K64+CLS checkpoint。只有该 checkpoint 通过表示与策略保持验收后
  才进入本次 evaluation-only Stage3。
- Stage3 从上述新 Stage2 checkpoint fresh start，不继承旧 Stage3 WM、optimizer 或
  收敛历史。固定二维位置编码只在 Stage3 WM 内加入，不混入 Stage2 projector。
- 除 CLS state 接口和固定二维位置编码外，数据、sampler、seed、H1/T4、有效 batch、
  学习率、DINO2、Outcome BCE1、Value/LM、SIGReg0、梯度边界、保存和验证口径保持
  与近期 Outcome+BCE Stage3 基线一致。ValueHead 与 OutcomeHead 照常训练，但必须只
  读取前64个空间槽位并保持旧K64 mean-pooling语义；新增CLS不得被朴素纳入均值。两个
  head 的loss/指标只作为训练健康检查，不作为本轮主要比较或成功标准。
- 保留 residual exact-copy 初始化；第一个 optimizer update 前，全部 state 槽位的预测
  必须等于当前 state。
- DINO loss 分别报告 spatial grid MSE 与 CLS MSE，再按已审查的固定权重合成；不得因
  65 个 token 的朴素平均而让 CLS 只占 `1/65`，也不得把两个分项的尺度变化隐藏在总损失。
- 固定数据评估至少报告 observed spatial/CLS 到对应 DINO target、WM predicted 到真实
  future state、WM predicted spatial/CLS 到 future DINO、copy baseline、H1-H4、
  centered variation及 spatial+CLS reconstruction。CFM 必须明确拆分64个空间槽位和1个
  CLS，不得把 K65 reshape 为二维网格。训练只使用 reconstruction train split，验证图像
  不进入拟合。评估使用相同输入样本、噪声与采样设置，分别比较正确 CLS、零 CLS、
  跨样本打乱 CLS；observed reconstruction 与 WM-predicted reconstruction 必须分开。
- 本次评估训练的代码路径必须同时支持未来 fresh Stage2：CLS token/schema 不能作为
  epoch16 resume 的临时补丁，正式训练时可从 Stage2 epoch1 与空间 Query 一起启用。

### R4. Safety and evidence

- 实现先通过 cache schema、shape/order、固定编码、gradient、residual copy、checkpoint
  round-trip 和旧 K64 fail-closed 兼容测试，再做 production-shaped GPU canary。
- 真实实验使用专用提交、远程 clean worktree、唯一输出目录；启动前核验 GPU、磁盘、
  精确 checkpoint/data/cache identity、步数、ETA、checkpoint 策略和硬时限。
- 本次评估训练预计超过10分钟；代码与 canary 通过后，必须给出最终 launch contract
  并取得单独启动批准。提交后监控到完成、失败、取消或暂停并记录终态。

### R5. Frozen-representation WM continuation diagnostic

- 用户追加批准从本轮 Stage3 checkpoint 的现有 WM 权重开始，只继续训练 WM；Qwen、
  vision、Query、projector、ValueHead 与 OutcomeHead 均不加载进 optimizer。该诊断使用
  从同一 Stage3 checkpoint 冻结导出的完整 train/eval state cache，不能退回 Stage2 state
  cache 冒充当前表示。
- continuation 只继承 WM 权重和结构，使用新的 WM-only AdamW optimizer；不得声称忠实
  恢复原多模块 optimizer。运行身份必须固定 Stage3 checkpoint、state projector、WM
  config/weights、cache manifest 和代码 commit。
- 结果同时报告 WM 相对 input-copy 的 H1--H4 改善，以及不随 WM 训练变化的 observed
  state-to-DINO / observed reconstruction ceiling。若 WM 改善但表示 ceiling 仍明显落后
  DINO oracle，应分别归因于动力学学习和 Query/projector 表示瓶颈。

### R6. Evaluation-only Stage2 representation continuation

- 冻结表示 WM 已证明现有 predictor 能超过 input-copy 后，允许从本任务 evaluation-only
  Stage2 epoch8 权重初始化一轮新的表示诊断。该诊断只训练全部65个 Query 的 input
  embedding FP32 master rows 与共享 projector；Qwen backbone、vision、LM head、action/
  format token rows均冻结，不加载 Stage3 WM、ValueHead或OutcomeHead。
- 由于训练参数集合从“仅CLS Query一行”变成“全部Query行+projector”，不得冒充旧
  optimizer的原样续训。使用新的 AdamW，并在运行身份中同时保存直接初始化 checkpoint、
  原始 Stage2 parent、精确行ID、Query/projector LR和冻结集合。
- 本轮沿用 epoch8 的 train/eval JSONL、K65 DINO cache、DINO系数2、有效batch64、seed与
  完整验证口径；Query LR为 `1e-4`，projector LR为 `8e-5`。以验证集 spatial+CLS DINO
  分项之和为收敛指标，至少2轮，连续2轮相对改善不足1%停止；LM、格式与direct-policy
  success作为策略保持门禁，不并入DINO收敛指标。
- 该运行仍标记 `evaluation_only=true`、`formal_stage2=false`，沿用已知发生过上游暴露的
  原划分，只能回答当前表示是否还能继续改善，不能晋升为正式Stage2或未见场景泛化证据。

### R7. Stage2 spatial plateau diagnosis

- 用户要求在继续训练前暂停 R6，并诊断 spatial DINO loss 改善显著慢于 CLS 的原因。
  暂停必须在 optimizer boundary 保存完整 resume checkpoint；计划内退出码75不能被解释为
  训练失败。
- 诊断必须绑定同一个暂停 checkpoint、同一 train/eval JSONL 与 DINO cache，固定 Qwen、
  Query hidden states 和 targets。先测 shared projector 上 spatial 与 CLS 梯度的范数、内积和
  cosine，再从当前 projector 权重只用 K64 spatial target 拟合到收敛。
- spatial-only probe 不得更新源 checkpoint、Qwen、Query 或 CLS target；必须保留原 joint
  projector baseline、固定表示 lineage、逐轮验证 spatial MSE 与 best epoch。旧 K64 probe
  只能作为背景证据，不能代替当前 K65 checkpoint 的匹配对照。
- 若 spatial-only projector 明显优于 joint baseline，结论限于 shared-projector/联合目标的
  优化取舍；若仍停在相近水平，才支持冻结 hidden representation 是主要瓶颈。仅凭 loss
  权重、参数 cosine 或不同 checkpoint 的绝对 MSE 不得下因果结论。

## Acceptance Criteria

- [ ] AC1: 新缓存逐图提供真实 DINO CLS 与 K64 grid，schema、shape、teacher identity、
  split 和 hash 校验通过；测试证明 CLS 没有被 patch mean 替代。
- [ ] AC2: WM 使用固定、可复现、不可训练的二维 sine-cosine 编码；CLS 不占用二维格点，
  attention 与 checkpoint round-trip 保持正确。
- [ ] AC3: 新 state 构造、DINO spatial/CLS 分项、residual exact-copy、梯度边界以及
  Outcome/Value 接口通过 focused tests 和真实 GPU 单步 canary。
- [ ] AC4: 与近期 K64 baseline 的除实验变量外配置差异审计通过，evaluation-only Stage2
  与 Stage3 产生完整可恢复 checkpoint 和逐轮验证指标，元数据明确禁止当作正式模型。
- [ ] AC5: 固定数据报告分别回答 CLS 对齐、空间对齐、WM-vs-copy skill 及 observed/
  predicted spatial+CLS reconstruction；正确/零/打乱 CLS 使用相同样本、空间 state、
  噪声与采样设置，不以总 loss、单张重建图片或 mean-pooling head 指标宣称策略提升。
- [x] AC6: WM-only continuation 从精确 Stage3 predictor 权重初始化，cache 绑定实际 Stage3
  表示 checkpoint，optimizer 只含 WM；报告 WM-vs-copy 与固定表示 ceiling，不把两者混为
  WM 容量结论。
- [ ] AC7: evaluation-only Stage2 representation continuation 只更新65个 Query input rows与
  projector，使用新的 optimizer 并严格记录 epoch8 初始化 lineage；完整验证分别报告
  spatial/CLS DINO与语言门禁，收敛后不自动进入Stage3或RL。
- [ ] AC8: R6 在完整 optimizer-boundary checkpoint 暂停；同 checkpoint 的固定 hidden
  spatial-only projector probe 与 spatial/CLS projector-gradient 诊断完成，来源、基线、
  best epoch和解释边界可审计。

## Out of Scope

- 不改变 DINO teacher 权重、图像预处理、训练/验证任务划分或环境奖励。
- 不把 CLS 直接加入 CFM 的二维网格，不修改 Stage2/Stage3 参数来优化 reconstruction，
  也不把 post-hoc CFM 指标解释为策略或动力学质量。CFM 可从头训练，但只消费冻结导出的
  state 与对应图像；CLS 的增量作用须通过正确/零/打乱 CLS 的配对评估检验。
- 不在本轮同时修改 history size、prediction horizon、WM depth/width、loss 系数或 RL 策略；
  R5 仅延长现有容量 WM 在冻结表示上的训练。
- 不在本轮实现 K64 action-conditioned attention pooling，也不声称修复 ValueHead 或
  OutcomeHead 的 mean-pooling 缺陷。
- 正式训练完成前不进入 RL；本实验结果不自动授权后续 RL。
- 本任务不执行正式 Stage2/Stage3 重训；正式 Stage2 的具体 Stage1 基座、预算和启动合同
  后续单独规划，但必须从 epoch1 启用 CLS。R6 只是从当前 evaluation-only epoch8 初始化
  的表示诊断，不能替代正式训练。
