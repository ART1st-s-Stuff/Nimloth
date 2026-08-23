# WM视觉—目标state对齐优化计划

状态：人类要求持久化的设计计划；尚未授权实现、新训练或新GPU实验。

## 1. 设计目标

State只保留规划需要的部分语义，重点包括：

- 与当前观察相关的视觉空间信息；
- 与任务目标相关的语义及目标—观察关系；
- 对环境预测无关的语言措辞、CoT表面变化等信息应尽量不进入world state。

DINO是视觉teacher，用于把state拉向视觉相关的空间。WM只负责预测这个受约束state的动作条件变化，不负责还原完整Qwen hidden，也不应预测未知的未来CoT。

任何CoT-conditioned policy state仍必须使用对应观察下实际生成的CoT；禁止fixed、canonical或placeholder CoT。可以另建observation/goal-grounded world state，但不得用伪造CoT替代真实policy state。

## 2. 当前证据及修订后的判断

ID56在1,742个有exact actual-next behavior-time state的nonterminal transition上得到：

- `predicted -> actual_next` state RMSE均值：`0.52598`；
- `copy -> actual_next` state RMSE均值：`0.11923`；
- predicted在behavior-state RMSE上优于copy：`0/1742`；
- executed action在8个depth-1 predicted states中的top-1：`46.67%`，随机基线`12.5%`；
- predicted对真实next-image DINO target的RMSE：`0.83801`；
- copy对同一DINO target的RMSE：`0.98837`；
- predicted对DINO target在全部transition上优于copy。

这组结果不应简单解释为“WM没有学到视觉变化”。更准确的现状是：

1. WM已经包含动作和视觉变化信号；
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

建议概念上因子化：

\[
z_t=(z_t^{visual}, z_t^{goal})
\]

### 4.1 Visual部分

- 使用冻结DINO grid作为视觉teacher；
- 保留16-slot空间结构；
- state encoder应在真实current和next observation上直接接受视觉对齐监督；
- 如果使用visual adapter `A_v`，其容量应受限，例如identity或低容量线性层，避免adapter独自吸收全部对齐而state本身不变。

### 4.2 Goal部分

DINO不包含完整任务目标，因此需要显式保留：

- episode内静态目标语义：可直接carry/copy，不要求WM重复预测；
- 目标和当前观察之间的动态关系，例如可见性、相对方向或任务进度：由goal-relation head或WM预测；
- ValueHead/Q同时读取visual和goal部分。

第一版可以保持外部`[16,1024]`合同不变，在内部维度、adapter或head层面因子化；是否改变state外部shape属于后续人类设计决策。

### 4.3 CoT边界

- world state优先从observation-grounded特征和显式goal特征提取；
- actual policy state仍保留本turn真实CoT；
- 不要求WM预测下一turn未知CoT；
- 若做CoT不变性约束，只能使用真实采样、真实记录且属于同一observation/goal的CoT，不得构造fixed thought。

## 5. 建议目标函数

设：

\[
z_t=P(h_t),\qquad d_t=DINO(o_t)
\]

### 5.1 State encoder对齐

\[
L_{repr}=
\lambda_v L_{visual}(A_v(z_t), d_t)
+\lambda_g L_{goal}(z_t,g_t)
+\lambda_{inv}L_{irrelevant}
\]

其中`L_irrelevant`用于抑制与世界预测无关的语言表面变化；在没有合规真实对照数据时不启用。

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
\]

actual state和predicted state必须共享同一个visual adapter、normalization、slot ordering和目标空间。

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

## 7. 训练课程

训练数据只使用批准的pre-RL数据；ID189/source20只能作为冻结heldout domain-transfer评估，禁止进入训练。

### 阶段A：State projector gate

- 训练或校准visual/goal state encoder；
- projector与DINO teacher尺度、slot和normalization一致；
- 检查visual retrieval、goal probe和CoT/语言表面不变性；
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

- actual state与canonical visual teacher的尺度、slot和分布一致；
- goal probe显著优于不含目标的基线；
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
