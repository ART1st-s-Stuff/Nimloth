# WM视觉—目标state对齐优化计划

状态：人类要求持久化的设计计划；尚未授权实现、新训练或新GPU实验。

## 1. 设计目标

State只保留规划需要的部分语义，重点包括：

- 与当前观察相关的视觉空间信息；
- 与任务目标相关的语义及目标—观察关系；
- 对环境预测无关的语言措辞、CoT表面变化等信息应尽量不进入world state。

DINO是视觉teacher，用于把state拉向视觉相关的空间。WM只负责预测这个受约束state的动作条件变化，不负责还原完整Qwen hidden，也不应预测未知的未来CoT。

State默认保持统一的视觉—目标语义表示，不预先划分visual state与semantic/goal state，也不按维度或token硬切分。任何CoT-conditioned state仍必须使用对应观察下实际生成的CoT；禁止fixed、canonical或placeholder CoT。WM不要求预测与规划无关的未来CoT表面变化，但不得用伪造CoT替代真实behavior-time state。

## 2. 当前证据及修订后的判断

ID56在1,742个有exact actual-next behavior-time state的nonterminal transition上得到：

- `predicted -> actual_next` state RMSE均值：`0.52598`；
- `copy -> actual_next` state RMSE均值：`0.11923`；
- predicted在behavior-state RMSE上优于copy：`0/1742`；
- executed action在8个depth-1 predicted states中的top-1：`46.67%`，随机基线`12.5%`；
- ID56报告的predicted对next-image DINO RMSE：`0.83801`；
- ID56报告的copy对同一DINO RMSE：`0.98837`；
- 但后续代码审计确认ID56先把真实图片bicubic resize到CFM的`128×128`再送入DINO；这两项只能视为legacy decoder-resolution sensitivity，不能冒充WM训练时的original-observation DINO target。修正后的原图只读比较由ID57完成，见`E0144`。

这组结果不应简单解释为“WM没有学到视觉变化”。更准确的现状是：

1. WM已经明确包含动作信号；视觉变化信号在legacy 128×128 DINO路径上存在，但必须由ID57 original-observation teacher路径复核；
2. behavior-time projected state、WM predicted state、DINO teacher和ValueHead目前没有稳定处于同一state空间；
3. MCTS递归时predicted state离ValueHead在真实state上见过的manifold较远，因而即使视觉方向有信号，K4递归仍可能不稳定。

抽查还发现actual behavior-time state标准差约`0.49--0.52`，predicted state约`0.824--0.826`。`wm_predictor.norm.weight`的RMS在ID74为`0.8795`，source20为`0.8451`。在确定canonical state归一化前，不能直接断言predictor最终LayerNorm应删除；这个现象首先说明actual projector output与predictor output的接口尺度不一致。

## 3. 当前训练目标的关键缺口

当前SFT2和RL大致使用：

```python
expected_next_state = Projector(next_hidden).detach()
loss = (
    mse(predicted_state, expected_next_state)
    + lambda_dino * mse(predicted_state, next_dino_grid)
)
```

DINO直接监督`predicted_state`，但没有对每个真实turn的：

```python
Projector(current_hidden)
Projector(next_hidden)
```

施加对称、明确的DINO视觉约束。因此可能出现：

- online projector产生一套behavior state；
- predictor被DINO拉向另一套视觉尺度/坐标；
- detached online projector target继续漂移；
- ValueHead主要在actual projected state上训练，却在MCTS深层读取predicted state。

DINO作为state约束的方向保留；需要修复的是state encoder、WM target和ValueHead之间的统一接口。

## 4. 目标state定义

人类明确要求默认使用统一state：视觉、目标语义及其关系共同编码在同一个K16表示中。

\[
z_t=P(h_t)\in\mathbb{R}^{16\times1024}
\]

禁止在缺乏确凿证据时把它预先拆成`visual state`和`goal/semantic state`，也不按slot或维度人为保留独立区域。DINO是统一state的视觉正则/teacher，并不定义state的全部内容。

### 4.1 统一state监督

同一个`z_t`同时接受：

- DINO视觉结构约束；
- 目标语义和目标—观察关系约束；
- dynamics、ValueHead/Q及必要的不变性约束。

允许为训练和诊断添加低容量readout，例如`A_v(z)`预测DINO、`A_g(z)`预测目标，但这些readout不是独立state分支。它们必须从相同完整K16 state读取，且容量受限，避免head独自吸收任务而state不包含相关信息。

视觉监督优先采用cosine、token/slot relational geometry或明确归一化后的损失，避免raw DINO MSE单独把state尺度拉到teacher尺度并覆盖目标语义。actual current、actual next和WM prediction必须使用同一readout、normalization和slot ordering。

### 4.2 Goal语义

DINO不提供目标语义，因此统一state还必须通过真实任务数据验证：

- 能从state读取episode目标及目标类别；
- 能区分同一或相近观察下的不同真实目标；
- 能表示目标与当前观察的关系，例如可见性、相对位置或任务进度；
- ValueHead/Q对目标变化具有正确敏感性。

现有ID189 archive缺少validated goal labels和matched same-observation/different-goal pairs，因此不能用当前证据判断goal是否已被state保留，也不能据此主张拆分。

### 4.3 CoT边界

- state继续是统一的视觉—目标语义表示；
- CoT-conditioned state必须使用本turn真实CoT；
- 不要求WM拟合与世界/目标无关的未来CoT表面随机性；
- 若做CoT不变性约束，只能使用真实采样、真实记录且属于同一observation/goal的CoT，不得构造fixed thought。

### 4.4 允许重新讨论拆分的证据门槛

只有同时具备下列受控证据，才重新讨论visual/semantic factorization：

1. **统一state强基线失败**：对称visual+goal监督、稳定target、充分容量和合理权重扫描后，统一state仍无法同时通过视觉、goal、dynamics和planning门禁；
2. **可重复的优化冲突**：`L_visual`与`L_goal`在projector上的梯度长期显著负相关，且调权、归一化、容量增加或PCGrad等不改变Pareto冲突；
3. **目标反事实失败**：在真实matched same-observation/different-goal数据上，统一state无法对goal变化敏感，同时保持视觉结构稳定；
4. **匹配预算的factorized ablation获胜**：使用相同数据、参数量、训练算力和评估协议，拆分模型同时改善visual、goal、copy-relative dynamics及heldout planning，而非只改善其中一个指标；
5. **跨seed复现**：上述收益在多个训练seed和Base/Common heldout上稳定存在。

ID57只证明actual/predicted state接口错位，不满足这些拆分证据门槛。

## 5. 建议目标函数

设：

\[
z_t=P(h_t),\qquad d_t=DINO(o_t)
\]

### 5.1 State encoder对齐

\[
L_{repr}=
\lambda_v L_{visual}(A_v(z_t), d_t)
+\lambda_g L_{goal}(A_g(z_t),g_t)
+\lambda_{inv}L_{irrelevant}
\]

`A_v`和`A_g`只是同一统一state上的低容量监督/readout head，不创建独立视觉或语义state。actual current与actual next都应用相同`L_repr`。`L_irrelevant`用于抑制与世界预测无关的语言表面变化；在没有合规真实对照数据时不启用。

### 5.2 稳定target encoder

\[
\bar z_{t+1}=stopgrad(P_{EMA}(h_{t+1}))
\]

第一阶段也可以完全冻结projector并离线生成pre-RL target。不得继续使用无锚定、同步漂移的online projector同时充当输入encoder和detached target encoder。

### 5.3 Residual WM

\[
\hat z_{t+1}=z_t+\Delta_\theta(z_t,a_t)
\]

残差最后一层零初始化，使模型初始行为严格等于copy baseline。内部归一化可以保留；absolute output是否保留最终LayerNorm必须由canonical state分布决定。

### 5.4 Dynamics loss

\[
L_{WM}=L_{state}(\hat z_{t+1},\bar z_{t+1})
+\lambda_v L_{visual}(A_v(\hat z_{t+1}),d_{t+1})
+\lambda_g L_{goal}(A_g(\hat z_{t+1}),g_{t+1})
\]

actual state和predicted state必须共享同一个visual/goal readout、normalization、slot ordering和统一目标空间。WM预测完整统一state，不单独预测visual或goal分支。

## 6. 必须先完成的只读诊断

在现有ID189/ID56产物上补充，不训练、不更新checkpoint：

1. `actual current state <-> current DINO`；
2. `actual next state <-> next DINO`；
3. `predicted next state <-> next DINO`；
4. `predicted next state <-> actual next state`；
5. actual/predicted/DINO逐slot、逐维度的mean/std/RMS/cosine；
6. slot ordering一致性及最优slot permutation对照；
7. goal相关的冻结linear probe或检索指标；
8. ValueHead分别在actual state和depth1--4 predicted state上的校准偏移。

判读规则：

- actual-next比prediction更不接近DINO：优先修state projector；
- actual-next已经DINO-aligned而prediction不匹配：优先修WM；
- 两者都接近DINO但彼此不接近：检查归一化、坐标旋转或slot ordering；
- 视觉指标通过而goal probe失败：增加显式goal state监督；
- actual/predicted都合理但ValueHead漂移：单独校准或重训ValueHead。

ID56没有保存exact terminal K16 state，因此仍只允许审计1,742个nonterminal transition；最后120个transition不得重放或填placeholder。

### ID57只读结果

ID57 Job`528490`已在original-observation DINO teacher路径上完成前六项主要state诊断：

- actual same-image state：RMSE`0.97769`、cosine`0.36629`、token-centered cosine`0.35176`；
- next DINO：copy RMSE/cosine=`0.98095/0.36005`，actual-next=`0.97866/0.36391`，predicted=`0.83053/0.62777`；
- predicted相对copy的canonical-DINO skill=`+0.28099`，相对actual-next=`+0.27750`；
- behavior-state skill仍为`-9.44815`，predicted对copy为`0/1742`；
- actual-next/predicted/DINO std=`0.47804/0.82286/1.04802`，slot-deviation RMS=`0.25585/0.46168/0.76248`；
- fixed slot permutation只降低`0.169%` identity cost，排除slot ordering为主要原因；
- original-observation与legacy128结果接近，视觉信号方向经E0144修正后仍成立。

因此当前优先级已确定为state projector/interface：actual projected state没有被充分锚定到视觉teacher，而WM prediction被直接DINO loss拉向另一分布。先对actual current/next projector state施加对称视觉/目标约束并建立frozen/EMA target，再优化WM深度。

尚未完成的诊断为goal retention及ValueHead在actual/depth1--4 predicted state上的校准；现有archive没有validated goal labels或matched same-observation/different-goal pairs，禁止用启发式标签伪造goal probe。

## 7. 训练课程

训练数据只使用批准的pre-RL数据；ID189/source20只能作为冻结heldout domain-transfer评估，禁止进入训练。

### 阶段A：State projector gate

- 训练或校准统一的视觉—目标state encoder；
- actual/predicted state共享同一visual/goal readout、slot和normalization；统一state本身不要求与raw DINO具有完全相同尺度；
- 同时检查visual retrieval、goal probe和CoT/语言表面不变性，禁止只优化视觉指标；
- gate通过后冻结projector，或创建EMA target projector。

### 阶段B：一步Residual WM

- 只训练T1；
- 使用动作类别、blocked/successful movement和state-change幅度平衡采样；
- 零初始化residual predictor；
- checkpoint按copy-relative skill选择，禁止只按absolute MSE选择。

### 阶段C：T2/T4课程

只有T1通过后，按T1 -> T2 -> T4扩展。每个深度独立比较：

\[
skill_d=1-\frac{MSE(\hat z_{t+d},z_{t+d})}
{MSE(z_t,z_{t+d})}
\]

每个深度都必须同时报告canonical state误差、DINO视觉误差、goal probe和state分布漂移。

### 阶段D：Value/Q和MCTS

- 在统一canonical state上训练或校准ValueHead/Q；
- 同时覆盖actual state和合格的predicted state；
- 先做depth1 Q校准，再做K4递归；
- 最后以heldout环境success/reward验证，而不是只看state MSE。

## 8. 建议门禁

### Representation gate

- actual state经同一受限visual readout后稳定保留DINO视觉结构；
- 同一个完整state上的goal probe显著优于不含目标的基线；
- 同一observation/goal下，state不应主要由CoT措辞决定。

### One-step dynamics gate

- canonical-state overall copy-relative skill `> 0`，正式候选建议`> 0.2`；
- 主要action类别分别`> 0`；
- predicted/actual canonical state标准差比例建议位于`0.9--1.1`；
- next-DINO指标优于copy；
- executed-action识别显著高于8-way chance，且不能只由高频action贡献。

### Multi-step gate

- depth1--4分别超过repeated-copy；
- 每层predicted state保持在ValueHead训练分布内；
- Q/value calibration不随深度单调崩坏；
- 通过后才允许重新接入正式K4 MCTS及joint RL。

## 9. Decoder边界

当前ID45 decoder对ID189/ID74 state域的oracle reconstruction无效，不能用其图像质量决定WM训练方向。

新decoder必须：

1. 使用与canonical state完全一致的checkpoint和pre-RL数据；
2. `D(actual_next_state)`明显优于`D(current_state)`及shuffled-state；
3. correct-vs-shuffled sensitivity和跨noise稳定性通过；
4. oracle gate通过后，才能用重建图解释WM动作效果。

## 10. 推荐执行顺序

1. 扩展ID56只读诊断，补齐actual-state到DINO及goal probe；
2. 根据诊断确定主要故障在projector、WM、slot/normalization还是ValueHead；
3. 建立canonical visual-goal state合同；
4. 对称监督actual current/next state并冻结或EMA target projector；
5. 训练pre-RL-only、零初始化的T1 residual WM；
6. 通过copy-relative和DINO/goal双门禁后扩到T2/T4；
7. 重新校准ValueHead/Q；
8. 最后才恢复K4 MCTS和新的joint RL实验。

在以上诊断和门禁完成前，不建议通过增加epoch、调学习率或扩大predictor来延续当前WM训练；这些操作不能修复state空间接口不一致。
