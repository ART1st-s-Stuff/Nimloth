# WM视觉—目标state对齐优化计划

状态：阶段0只读checkpoint矩阵已完成；尚未授权任何训练、新checkpoint或参数更新。

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

尚未完成的诊断为可靠goal retention gate及ValueHead在actual/depth1--4 predicted state上的校准。ID59已从实际source config asset为archive建立唯一instruction→`targetObjectType`标签并找到少量真实same-image/different-goal pairs；但迁移category与source config大量不一致，使原train/validation task-disjoint语义不再足够可信，且当前无充分规模的matched counterfactual。禁止用启发式标签或把当前kNN结果冒充最终goal probe。

## 7. 执行阶段

训练数据只使用批准的pre-RL数据；ID189/source20只能作为冻结heldout domain-transfer评估，禁止进入训练。

### 阶段0：SFT1/SFT2 checkpoint只读矩阵（已授权）

目的：在任何重训前隔离backbone/vision drift、projector drift及ID74 online/vision-EMA安装差异。

- 数据：ID74使用的pre-RL validation JSONL；Base/Common/Long Horizon各确定性选择32条step0--3 early transition，共96条且每条trajectory至多选择一次；每条都有exact next decision state，使用记录中的真实CoT和original observation。初始preflight发现全部step0动作都是action0，因此改为early-step action round-robin，避免把一步WM/Q诊断退化为单动作审计。
- checkpoint组合：
  1. SFT1 backbone/vision + SFT1 projector；
  2. SFT1 backbone/vision + ID74 projector；
  3. ID74 online backbone/vision + SFT1 projector；
  4. ID74 online backbone/vision + ID74 projector；
  5. ID74 vision EMA + SFT1 projector；
  6. ID74 vision EMA + ID74 projector。
- 只读指标：actual current/next到original-observation DINO、current→next copy距离、ID74 WM一步预测相对actual/copy/DINO的误差、ID74 ValueHead在actual current/next及predicted next上的return calibration，以及关键组合之间的state drift。
- 解释边界：ID74 WM/ValueHead应用到非ID74组合时只表示cross-component compatibility；不得解释为对应SFT1 checkpoint自身训练出的WM/Q质量。
- 当前数据没有validated goal labels或matched same-observation/different-goal pairs，本阶段不得生成启发式伪标签或声称完成goal反事实probe。
- 资源和时限：人类确认normal 1×H800；Slurm硬上限1小时45分，从提交到结果交付总预算不超过2小时。
- 冻结边界：所有backbone、vision、projector、WM、ValueHead和DINO cache只读；无optimizer、无backward、无参数更新、无新checkpoint、不得resume或覆盖旧产物。

#### ID58结果

ID58 retry2 Job`528812`在normal/dgx-27以`COMPLETED 0:0`、`00:02:56`完成。96条样本的action0/2/3/4/5计数为`25/23/23/22/3`。

- 固定SFT1 projector时，SFT1→ID74-online backbone drift仅RMSE`0.020806`。
- 更换projector造成RMSE`0.892000--0.892725`，远大于backbone drift。
- SFT1 projector的current-DINO RMSE/cosine为`0.837607/0.656045`（SFT1 backbone）和`0.837198/0.656466`（ID74 online）；ID74 projector对应为`1.137054/0.381029`和`1.136279/0.382105`。视觉退化主要在SFT2 projector，而非backbone/vision。
- ID74 canonical cell的behavior copy/predicted RMSE为`0.703430/0.640073`，aggregate skill`+0.172027`，但action0/2/3 skill分别为`-0.410761/-0.715589/-0.268277`；正aggregate主要受state-change较大的action4/5贡献，未通过主要action分别大于0的门禁。
- canonical predicted-next对DINO优于actual-next：RMSE`0.962218` vs `1.128445`、cosine`0.516583` vs `0.392854`，再次确认predictor被拉向与actual projected state不同的视觉空间。
- actual-next/predicted-next std=`0.984237/0.866125`，比例约`0.8800`，低于建议的0.9下界。
- ValueHead executed-action return RMSE在actual-next/predicted-next为`0.460140/0.472018`，depth1 prediction略差。
- online和vision-EMA在本样本上的projected state完全相同。权重侧审计显示390个BF16 vision tensor中282个不同，但parameter RMSE仅`1.3149e-6`、max abs`2.1267e-4`，在当前BF16 forward中量化为相同state；当前EMA sidecar没有形成可测的state target差异。

因此：SFT1 projector可作为**视觉**anchor，暂不重训SFT1；但在有效goal probe前不能宣称其完整统一visual-goal state已经健康。优先修SFT2 projector/interface，并用每个主要action的copy-relative skill门禁T1 canary，禁止只看aggregate skill。

#### ID59部署actor与grounded-goal结果

ID59 Job`528906`在normal/dgx-10以`COMPLETED 0:0`、`00:13:23`完成。它使用全部3,211条pre-RL train首状态作为gallery、355条validation首状态作为query，所有CoT均为对应归档observation下的真实assistant response；没有生成、optimizer、更新或checkpoint。

- 固定SFT1 projector时，SFT1→部署ID176 actor validation state RMSE仅`0.018796`。
- ID176+SFT1 projector的validation DINO RMSE/cosine为`0.846426/0.642177`，SFT1+SFT1为`0.846767/0.641844`；部署路径视觉非劣门禁通过。
- 同一ID176 hidden换为ID74 projector时state drift RMSE=`0.885311`，DINO RMSE/cosine退化为`1.119524/0.402734`；ID58的projector/interface定位在部署actor上复现。
- 目标标签来自exact instruction到全局一致`targetObjectType`的匹配，不使用启发式标签。后续ID60 preflight确认source `config_id`/seed/UID也可能错绑trajectory row，因此ID59未恢复actual asset/task identity；train `587/3211`、validation `67/355`仅是迁移category与source config的分歧计数。
- ID176+SFT1 flattened-K16目标检索top1/top5/MRR=`0.086455/0.288184/0.196890`，低于冻结DINO的`0.109510/0.371758/0.242340`；goal-above-DINO门禁失败。majority top1=`0.070423`。
- 检索覆盖347条、20个gallery中已有的validation目标；8条`Footstool`因gallery无该目标而显式排除；exact-image candidate排除1个。
- train自然same-image/different-goal pair的state RMSE均值`0.020433`，高于same-goal pair`0.003009`，但计数仅21对与4对且真实CoT同步变化，只能证明少量目标敏感性，不能覆盖检索失败。

因此：SFT1 projector保留为视觉anchor候选，但完整visual-goal gate未通过。ID60使用冻结state、匹配图像/DINO和多数类基线训练低容量goal readout；由于row级task identity不可恢复，人类批准按exact image分组、去重和跨split排除的保守诊断，结果不得称为正式task-generalization。任何projector校准或后续WM训练仍需独立实验身份和授权。

#### ID60 goal probe与ID75 Residual-T1实际结果

- ID60 Job`528931`完成冻结goal probe：state micro/macro top1=`0.057803/0.040432`，低于DINO=`0.106936/0.067128`与majority=`0.072254`；paired-bootstrap state-minus-DINO 95%区间=`[-0.080925,-0.017341]`。goal gate所有条款失败。
- ID75 retry1 Job`529411`使用上述immutable cache并保持exact-copy初始化与raw-DINO-loss=0。overall skill=`+0.365064`、macro primary skill=`+0.210312`、std ratio=`0.981491`且next-DINO优于copy；但action0/2/3/4 skill=`+0.354206/+0.053946/-0.132927/+0.566024`，action3失败。
- ID61进一步用exact archived environment feedback分层：early `move_left`失败率train/external=`22.68%/38.86%`；成功子集skill=`+0.16330`，失败/no-op子集skill=`-9.91418`，且failed image `100%`不变。actual/predicted state-change outcome AUC=`0.99977/0.61650`。因此action3失败来自弱outcome判别与`+16.18pp` outcome shift，而非多数训练样本失败；同类blocked hallucination也见于move_forward/right。
- ID71 matched full-K16 linear probe的state/DINO/ID75 outcome AUC：forward=`0.87276/0.87375/0.73801`、right=`0.61707/0.71802/0.53983`、left=`0.67356/0.76169/0.61650`。left state-minus-DINO CI=`[-0.16659,-0.01324]`，确认SFT1 state相对raw visual evidence丢失侧向碰撞信息；ID75又低于state readout，说明WM同时未充分使用残余信息。
- ID191进一步隔离projector前的same-generation hidden。hidden经冻结SFT1 projector严格重现ID60 state（RMSE/max=`0/0`），但goal micro state/hidden/candidate/DINO=`0.06936/0.05491/0.07803/0.10983`；outcome AUC state/hidden/candidate/DINO分别为forward=`0.89074/0.86537/0.86995/0.87067`、right=`0.72328/0.69015/0.72636/0.71099`、left=`0.69514/0.62475/0.62192/0.77842`。bounded hidden-only adapter只保住视觉，goal与lateral gate均失败。
- ID192把信息位置进一步明确：exact instruction embedding/final的goal micro/macro=`0.99711/0.99000`与`0.99133/0.98266`，而K16仅`0.05491/0.03575`；目标语义在prompt中明确存在、被K16压缩丢失。Outcome AUC pre-LLM/fused-image-final/K16/DINO为forward=`0.83497/0.86799/0.86537/0.87294`、right=`0.52535/0.73952/0.76432/0.71889`、left=`0.71831/0.73119/0.59910/0.73831`。fused-image-final point estimate最接近DINO，但N=`142/193`使严格non-inferiority CI仍过宽。
- 结论：aggregate T1学习信号真实存在，但不能覆盖goal接口失败、lateral collision representation loss与WM outcome退化。SFT1 projector不是已证实瓶颈；现有ID176 K16 hidden不能靠hidden-only projector/adapter修复。未来统一state应显式融合exact instruction embedding，并优先验证same-forward fused current-image final tokens；在其larger grouped确认前，不直接引入部署DINO。ID75 predictor与ID191 adapter均仅为诊断checkpoint。

### 阶段A：Unified state-interface gate

- ID191已否决只读取现有ID176 K16 hidden的projector/adapter校准；ID192确认exact instruction embedding是可靠goal source，并把same-forward fused current-image final tokens定位为首选visual候选；
- 新encoder必须显式读取exact instruction表示和通过更大grouped诊断的current-observation visual/geometry证据；DINO优先作teacher，不默认成为第二个部署视觉模型；
- 输出仍是一个统一K16视觉—目标state，不创建独立部署visual/goal分支；
- actual/predicted state共享同一visual/goal readout、slot和normalization；统一state本身不要求与raw DINO具有完全相同尺度；
- 同时检查visual retrieval、goal probe、movement outcome与CoT/语言表面不变性，禁止只优化视觉指标；
- gate通过后冻结state interface，或创建FP32 EMA target。

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
- movement outcome readout不得显著落后于matched DINO，且需在train/external outcome shift下保持可校准；
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

1. 阶段0的SFT1/SFT2 checkpoint只读矩阵已完成：主要故障定位到SFT2 projector/interface，backbone drift很小，当前EMA没有可测state差异；
2. ID59与ID60已完成部署视觉、grounded kNN和低容量goal probe；视觉anchor通过，但state goal probe显著低于DINO与majority，representation gate失败；
3. ID75零初始化Residual-T1已完成；aggregate、std与next-DINO检查通过，但action3 skill为负，one-step dynamics gate失败；
4. ID61已完成action3分层：WM在成功movement上改善、在blocked/no-op上严重幻觉移动，且train/external outcome分布漂移。停止T2/T4、fresh ValueHead、MCTS和RL；
5. ID71已完成frozen-state outcome probe：state保留部分outcome信号，但侧向动作显著弱于DINO，且ID75进一步低于state readout。因此representation与WM两侧都需修正；
6. ID191已完成hidden隔离与bounded adapter canary：hidden对goal和两类lateral outcome的point estimate均低于projected state，adapter又使left恶化；hidden-only projector校准方向被否决，禁止靠增加rank/epoch延续；
7. ID192已定位exact instruction source并找到fused-image-final visual候选；goal证据充分，但strict visual non-inferiority受小external N产生的wide CI阻塞。先复用immutable 14,261-state cache做exact-image-grouped out-of-fold确认，archive external仍是primary heldout，禁止把OOF冒充新heldout；
8. 后续新数据必须持久化逐步`last_action_success`，并按每个movement action的successful/blocked子集分别报告；该标签只用于监督/诊断，禁止作为部署时已知输入泄露未来；
9. 若grouped确认通过，下一state canary在人类另行授权后，把same-forward fused current-image final grid和exact instruction embedding融合回同一个K16 state；可用低容量`A_v/A_g`监督或蒸馏，但不得解释为独立visual/goal state分支。若确认失败，再评估DINO-teacher distillation或visual encoder repair；
10. 新接口通过visual、goal、outcome与CoT不变性门禁后，建立新的canonical visual-goal state合同；
11. 仅在人类另行授权后，在新canonical state上重新训练fresh、零初始化、outcome-aware T1；比较success-probability/mixture successor与保持copy-safe的门禁，不得跨坐标系复用ID75；
12. 新T1通过每个主要action及其successful/blocked子集的copy-relative和DINO/goal门禁后才扩到T2/T4；
13. 重新训练并校准ValueHead/Q；
14. 最后才恢复K4 MCTS和新的joint RL实验。

在以上诊断和门禁完成前，不建议通过增加epoch、调学习率或扩大predictor来延续当前WM训练；这些操作不能修复state空间接口不一致。
