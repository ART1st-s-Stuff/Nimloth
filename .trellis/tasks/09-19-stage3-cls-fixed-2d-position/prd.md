# Stage3 CLS 与固定二维位置编码实验

## Goal

为世界模型提供 DINOv2 CLS 全局视觉目标及固定二维空间位置编码，优先检验显式全局
场景信息与二维几何是否能改善未来空间 state/DINO 预测及其 reconstruction。现有 K64
residual WM 与冻结 reconstruction decoder 用于保持评估口径；本任务不增加新的
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
- 当前固定 reconstruction 使用的 `spatial_grid_v1` CFM 已按 row-major 将 K64 恢复为
  `8x8`，并注入固定 `(x,y)` 坐标。它只能直接读取 K64；若新 state 为 K65，评估时应
  明确只把 64 个空间槽位送入既有 decoder，CLS 另作特征指标，不能伪装成同一输入。
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
  centered variation及现有 reconstruction。重建只读取空间64槽位，并与相同冻结
  decoder、相同输入样本、相同噪声的现有 K64 结果比较。observed K64 reconstruction
  与 predicted K64 reconstruction 必须分开，不能把后者的变化归因于 decoder。
- 本次评估训练的代码路径必须同时支持未来 fresh Stage2：CLS token/schema 不能作为
  epoch16 resume 的临时补丁，正式训练时可从 Stage2 epoch1 与空间 Query 一起启用。

### R4. Safety and evidence

- 实现先通过 cache schema、shape/order、固定编码、gradient、residual copy、checkpoint
  round-trip 和旧 K64 fail-closed 兼容测试，再做 production-shaped GPU canary。
- 真实实验使用专用提交、远程 clean worktree、唯一输出目录；启动前核验 GPU、磁盘、
  精确 checkpoint/data/cache identity、步数、ETA、checkpoint 策略和硬时限。
- 本次评估训练预计超过10分钟；代码与 canary 通过后，必须给出最终 launch contract
  并取得单独启动批准。提交后监控到完成、失败、取消或暂停并记录终态。

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
  predicted spatial reconstruction；明确现有 decoder 不读取 CLS，不以总 loss、单张
  重建图片或 mean-pooling head 指标宣称策略提升。

## Out of Scope

- 不改变 DINO teacher 权重、图像预处理、训练/验证任务划分或环境奖励。
- 不把 CLS 直接加入现有 CFM 的二维网格，也不训练新的 CLS-conditioned reconstruction
  decoder；本轮沿用冻结 K64 CFM，以相同 decoder、样本和噪声比较空间 state 的可重建性。
  CLS 的直接可用性通过 DINO CLS 对齐和跨场景变化指标评估。
- 不在本轮同时修改 history size、prediction horizon、WM depth/width、loss 系数或 RL 策略。
- 不在本轮实现 K64 action-conditioned attention pooling，也不声称修复 ValueHead 或
  OutcomeHead 的 mean-pooling 缺陷。
- 正式训练完成前不进入 RL；本实验结果不自动授权后续 RL。
- 本任务不执行正式 Stage2/Stage3 重训；正式 Stage2 的具体 Stage1 基座、预算和启动合同
  后续单独规划，但必须从 epoch1 启用 CLS。本次 Stage2 初始化固定为当前 epoch16。
