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
Qwen 多模态编码必须保留完整 prefix。先按 image grids 展开图像占位符，再在同一展开后的字符坐标中定位回答监督 span；在线和缓存路径必须产生相同 input IDs 与 labels。图像 token 数须与对应 grid 的特征数一致，图像 token 不得成为回答 LM 监督。超过显式 max_length 时在预处理阶段报错，不允许截断文本后仍保留全部图像、补造 token 或静默删样本。改变该编码/标签合同必须更新缓存版本并重建文本缓存，旧版本不得视作兼容输入。

仅重建文本缓存时，可显式复用完成且通过身份校验的图像分片。必须核验预处理器、像素参数、dtype、有序图像来源及 grid/像素布局；新目录只链接图像分片，重新生成全部文本与标签，并保留复用审计。不得以图像可复用为由接受旧 CE 缓存版本，也不得修改源分片。

新stage2不是历史`sft2`。旧`training.sft1`、`training.sft2` Python 包已移除，活动调用使用`training.sft.stage1`或`stage3`。历史checkpoint字段仍指WM/value阶段；不改写已保存的历史产物。历史目标字段仍保持原阶段含义；新的整轨迹更新使用独立训练身份，不接受旧窗口优化器续训。

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
正确：保留WM目标时刻及阶段含义；对新的轨迹更新显式版本化，拒绝混用旧优化器恢复状态。

## 模块边界补充
`stage3.algorithm`拥有损失和反传顺序，`stage3.sigreg`拥有跨rank有效状态汇聚及同步随机投影，提取时不改变collective顺序或梯度。canary、动作头专项修复、packed/KV原型及依赖它们的特征审计归`experiments/training/sft/diagnosis`，不得由生产训练导入。Python旧路径不再兼容；保存的模型tensor/state_dict与目标标识保持阶段含义，训练采样和恢复身份须显式版本化。

`stage1.cli.parse_args(argv=None, *, stage="format")`负责入口参数校验；`stage1.checkpoint`负责保存及恢复阶段校验；`stage1.trainer`负责模型构建与训练生命周期，不再动态转发数据模块中的任意属性。数据调用者直接依赖`stage1.data`。

## 经审查的选择性 LM 监督
Stage2 使用全部轨迹的回答对齐 DINO，只对成功轨迹的回答计算 LM；Stage3 使用全部窗口的 WM/value，只对成功轨迹起点回答计算 LM；全部有效轨迹的真实在线 state 对齐 DINO，每个观测仅计一次（含终点，排除补齐）。success 必须来自完整轨迹的显式布尔字段，缺失拒绝。LM 分母为成功回答/窗口数；Stage3 DINO 分母为去重真实观测数，WM/value 分母为有效窗口数，跨累积组和 rank 分别归约；全失败组 LM 为图连接的零。恢复身份须拒绝旧全部轨迹 LM 目标的优化器状态。

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
控制组等价、选行冻结及两步优化保存恢复。LM、WM和SIGReg共用在线图并联合反传；
static DDP须有真实多rank联合反传测试，CPU测试不放行未经验证的Qwen/FA2八卡训练。

## Stage3 full-shard execution and original VAGEN key partition

### Scope and entrypoints
`--distributed-strategy fsdp` shards the full joint Qwen branch; WM/projector/value/outcome
remain replicated DDP. Default DDP remains available for existing compatible scopes.
Stage3 `--fsdp-wrap-granularity linear|block` defaults to linear. Block removes inner linear
handles only inside decoder/vision blocks; vocabulary handles and the complete visual owner
remain explicit. Record the layout in launch arguments. Both layouts use the same FULL_SHARD,
master/reduction/forward dtypes and losses. Layout conversion is permitted only with canonical
full named model/optimizer/vision-EMA state, not raw local shards. Verify exact restored state,
asymmetric-rank supervised/zero-LM backward, nonreentrant checkpoint recomputation and real
image forward before resuming with another layout. Invalid granularity is rejected. Floating
point reduction grouping may change next-update rounding; do not promise bitwise trajectories.
`python -m nimloth.rollout.split_by_eval_keys --source ... --eval-keys-manifest ... --output-root ...`
partitions immutable converted trajectories using explicitly verified original evaluation keys.

### Contracts
Qwen uses FULL_SHARD/original parameters, FP32 trainable masters and gradient reduction, BF16
forward. Frozen vocabulary tables have explicit replicated ownership. Do not accumulate full
FSDP gradients through no_sync. Composite clipping counts each Qwen shard and one WM replica.
Vision EMA keeps the same decay and target semantics; shard swaps must restore online parameters
before backward. Evaluation retains FSDP wrappers and pads every rank to equal forward counts;
padded examples have zero metric/loss weight and are excluded from exports.
All ranks participate in full model/optimizer/EMA gathering; rank0 writes complete CPU artifacts.
Optimizer transformation uses the common Agent parent to include both Qwen and WM groups.
`configure_qwen_tuning(model, args)` resolves language/vision ownership through actual decoder,
visual and vocabulary modules before wrapping; full language tuning must include dense decoder
parameters. Log trainable counts before sharding; selected vocabulary-row freezing follows this
step. Module-name spelling in one Transformers release is not an ownership contract.
`train_sft2(args=None)` releases its heavy training frame and cyclic references before CUDA
synchronization, allocator cache release and ordinary process-group cleanup. CUDA and teardown
errors still propagate. Collected checkpoint tensors must all be CPU, including ignored tensors
not covered by FSDP's automatic offload.
For weighted Stage3 LM windows, one FSDP-owned head invocation jointly checkpoints projection
and CE in chunks of 128 positions whose next-token label is not `-100`. Keep the full vocabulary,
per-window token mean, binary row weights and complete query hidden states. Its private per-window loss
head result keeps backward unsharding ahead of recomputation; read current parameter views.
Zero-weight rows skip answer-token projection and CE, but retain a differentiable single-token head path when all rows have zero weight. Preserve one FSDP-owned head invocation on every rank and explicit zero gradients for trainable head/selected rows. Validate labels for excluded rows too; do not silently accept an empty answer. Validate the mathematical objective against
dense FP32 gradients and validate checkpoint/FSDP behavior against independent chunk references;
report BF16 accumulation-order rounding separately. Do not retain token-by-vocabulary activations,
truncate labels, sample vocabulary or use per-rank varying head-call counts.

### Validation and errors
Stage3 `train.activation_offload` defaults to false, including the joint outcome experiment.
Wrap the joint online forward in `torch.autograd.graph.save_on_cpu(pin_memory=True)`
when enabled. Saved tensor copies retain their values and dtypes; live parameters, optimizer,
backward computation and reduction remain on their original devices. Record enabled state in
resume invariants. Verify CUDA FSDP gradients against the same path without offload; CPU
checks alone cannot validate pinned transfers or memory savings. Account for transfer overhead.
Set online trainable modules recursively to training mode at every training epoch entry;
`from_pretrained` defaults to evaluation mode and enabling checkpointing alone is insufficient.
Teacher encodings run without gradients in temporary evaluation mode. Restore every descendant's
previous mode after teacher/validation contexts, including mixed subtree modes. Verify actual
language and visual checkpoint function calls after loading a saved pretrained model; a newly
constructed tiny model already in training mode cannot detect this initialization regression.

Reject mixed trainable dtypes, missing visual shard ownership, mismatched strategy on resume,
unequal distributed eval call counts, incomplete artifacts and conflicting selected-row identities.
Reject unrecognized model ownership or full mode with no dense language parameters. A checkpoint
from an accidentally frozen language run is not a valid full-training continuation boundary.
Partition rejects input hash drift, duplicate IDs/keys, missing identity and invalid returns.
Only top-level split and explicit split provenance change; original rows and prior source identity
remain auditable. Missing original evaluation keys are reported, never fabricated.

### Cases
Valid: a source trajectory whose exact(eval_set,seed) occurs in the pinned test key manifest
moves wholly to eval with all precomputed finite-horizon targets unchanged.
Invalid: selecting by generic example seed range, replacing a forbidden action, or claiming
a checkpoint has never seen examples merely because its continuation split was corrected.

### Required tests
Check equal-rank zero padding, selected rows and dense export roundtrip, mixed sharded/replicated
optimizer state, EMA swap/restore/save/load, clipping and joint supervised/SIGReg backward. CPU and
synthetic process-group tests do not replace real multi-rank Qwen GPU save/resume validation.
Use a real tiny Qwen to check full-mode dense gradients, selected-row masks, ownership layouts
and trainable counts. Check that training cycles are released before cleanup and that failures
at training, CUDA synchronization, cache release and process-group destruction remain visible.

### Wrong versus correct
Wrong: rank0 saves its local shard as a full model/EMA, or evaluation unwraps a sharded model.
Correct: collective gathering with explicit complete export and synchronized wrapped evaluation.

## Stage3 sampled profiling

- Scope: performance measurement without changing optimizer, sample ownership, RNG or losses.
- Signature: `--step-timing --step-timing-sample-interval N --step-timing-interval M`; N defaults to1.
- Contract: profile complete local optimizer updates1,1+N,... including every accumulated microbatch. Unsampled updates perform no timer CUDA synchronization. M counts sampled updates; report cumulative sampled means, sampled counts and total observed updates. Resume starts a new local profiling sequence.
- Validation: N<1 is rejected; disabled profiling and unsampled updates emit no phase report. Per-section averages divide by the section sample count, not all optimizer updates.
- Cases: N=1 preserves full profiling; N=10/M=1 reports updates1,11,21. A short run may contain only its first profiled update.
- Tests: count synchronization calls across accumulated microbatches, check unsampled silence, mean denominators, defaults and CLI/config validation.
- Wrong versus correct: sampled phase means are not end-to-end throughput. Measure wall time over matched batches separately and account for save/evaluation stalls. Equal effective batch alone does not preserve nonlinear SIGReg microbatch statistics or sampler/resume identity.

## Trajectory-native Stage3 training

- Scope: complete trajectories are the sole Stage3 data/encoding unit, with H=1.
  `--batch-size` counts trajectories per rank/microbatch; `--grad-accum` combines
  complete microbatches. A trajectory never crosses an optimizer update boundary.
- Interface: `Stage3TrajectoryBatch` contains full transcripts, ordered query
  positions `[N,K,2]`, original per-window LM labels, trajectory/state/window
  offsets and current/successor/action/return/outcome/DINO indices.
- Contract: every eligible T-step start contributes one window. Short trajectories
  without such windows are counted and excluded. Successful-window LM means and
  WM/value/DINO/outcome denominators retain their distinct global counts.
  Encode targets first with EMA/eval/no-grad, then online states once. Predict all
  windows from their current state using predicted-state recursion. Aggregate
  differentiable losses and perform ordinary backward; no detached leaf/VJP bridge,
  no window-prefix reconstruction and no cross-update online history cache.
- SIGReg: include all real adjacent transitions of each eligible trajectory once,
  keyed by trajectory/time, irrespective of overlapping WM windows. Exclude
  distributed padding. Gather valid pairs across ranks for each microbatch;
  current states are detached and successor states remain differentiable.
  Preserve the existing formula and synchronized projections. Average microbatch
  regularizers over the actual accumulation-group length (including tails), not
  window counts. Transition weighting is intentional; longer trajectories
  contribute more unique transitions. This is a new statistical grouping, not
  numerical equivalence to the old window batches. Test unequal rank lengths,
  zero-valid ranks, no duplicate overlap, and gradient recipients.
  `lambda_sigreg` defaults to 0.1 and accepts finite nonnegative values; zero
  disables the regularizer for an explicitly configured ablation. It is not a
  fixed DINO-grid interface invariant. Resume still requires the same coefficient.
- Evaluation: use the same trajectory/window path; online policy uses current
  weights in eval mode, target branch uses EMA. Identity is
  `online_policy_eval_target_visual_ema_v1`; old EMA-online metrics are not treated
  as identical. Exports retain every valid window/horizon row and exclude padding.
- Validation: reject noncontiguous trajectories, wrong query/image/label mapping,
  H!=1 and incomplete targets. Resume requires
  `training_unit=complete_trajectory_v1`, trajectory batch units and matching
  sampling/RNG/objective metadata before optimizer restoration. Legacy window
  optimizer checkpoints and missing native identity are rejected.
- Cases/tests: unequal trajectory lengths, globally padded ranks, no-success LM,
  zero-label outcome, complete window coverage, true terminal CoT, causal queries,
  direct gradient recipients, mixed FSDP/DDP and native checkpoint round trip.
  Batch-size metrics are arithmetic per-microbatch means including zero-valid
  padding batches; they must not be weighted again by window counts.
- Wrong: retain both old window and native pipelines, carry learned states between
  updates, silently reinterpret batch units or reuse old optimizer histories.
  Correct: one trajectory path for train/eval, explicit masks/counts and a new
  training identity. Sharing representation does not remove WM supervision.

### Latest resumable checkpoint retention

`--checkpoint-latest-only` is opt-in and spans periodic, stopped, and epoch
checkpoints within one run. Publish and verify a new complete resumable checkpoint
before removing older complete checkpoints. Failed/incomplete saves must preserve
the prior recovery point. Keep best scalar metrics, but do not retain older best
weights in this mode. Final may hardlink the last epoch without duplicating tensors.
Do not follow symlinks or delete data, logs, or evaluation exports. Test epoch1 to
periodic to epoch2, failed saves, and final alias integrity.

The outcome A/B launcher accepts positive `--epochs`; only formal phases use it.
Canaries remain one epoch with explicit step caps. Verify all requested epoch
exports and require the final checkpoint to match the requested completed epoch.


## Stage3 observed-state DINO objective

- WM predicts future states and fits detached encoded successor targets. DINO
  supervises projected online states of real observations, never WM predictions.
  The frozen DINO teacher and EMA target branch receive no gradients.
- Each real observation of an eligible trajectory contributes once, including
  the initial and terminal observations. Window overlap does not repeat this
  loss; distributed padding contributes zero. Normalize by observed-state counts
  across the complete optimizer accumulation group and ranks, independently
  from WM window counts and successful LM counts. Evaluation uses the same rule.
- `dino_grid_mse` reports observed-state training alignment. Predicted-state DINO
  remains a separately named diagnostic and does not enter the objective. Old
  predictive-DINO loss histories are not comparable under the same metric name.
- Resume must reject the old predicted-state DINO objective and normalization
  identity before restoring optimizer state. Do not silently continue it.
- Tests must verify online encoder/projector DINO gradients, absence of WM and
  teacher DINO gradients, detached WM targets, unique observations, terminal
  inclusion, padding exclusion and unequal global population normalization.
- Wrong: fit WM predictions directly to DINO while real states receive no DINO
  anchor. Correct: WM fits real successor states; real online states fit DINO.

## Stage3 optional residual predictor

Stage3 may explicitly select the existing residual grid predictor for joint training.
The direct predictor remains the default. Every autoregressive step predicts
`next_state = current_state + delta_head(body(current_state, action))`; the delta
head starts at zero, but the copy branch remains differentiable with respect to
the input state. Zero delta initialization therefore does not freeze the encoder.
Training, evaluation and checkpoint restore must agree on predictor kind and full
grid configuration; cross-kind restore fails closed. Residual identity is explicit
without changing historical direct checkpoint identity. DINO continues to supervise
real observed states, and WM fits detached encoded future states.

## Stage3 backbone gradient boundary

An explicit opt-in mode blocks WM/value gradients at the backbone hidden output,
before the projector. Projector parameters remain trainable by WM/value; backbone
language/vision and selected vocabulary rows still receive LM/DINO gradients.
Project attached observed hidden and detached current-window hidden in a single
wrapped projector forward, then split the results. This preserves projector DDP
ownership without re-encoding Qwen or bypassing the wrapper. Both initial value
and autoregressive residual-copy paths must use the isolated current states;
do not detach future predictions, which must retain WM/value parameter gradients.
Default connected behavior stays unchanged. Gradient boundary is a strict resume
invariant. Validate isolated loss gradients, identical forward values, and multiple
distributed optimizer updates before GPU deployment.

## Frozen-WM residual diagnostic

1. Scope: run-owned fixed-feature experiments only; formal Stage3 defaults do not change.
2. Entry: `frozen_wm_diagnostic.py train --predictor-kind direct|residual` supports
   both `--mode stage2_state` and `--mode dino`; default remains direct.
3. Contract: residual uses `ResidualTemporalSpatialGridPredictor`, a production
   body plus zero-initialized linear delta head. Every autoregressive output is
   the latest state plus its predicted delta. MSE compares reconstructed full
   values to fixed future targets. Only WM parameters update.
4. Validation: residual kind and module are part of run/resume identity; reject
   mismatched kinds. Missing kind in historical direct metadata means direct,
   and existing direct resume identities remain unchanged.
5. Cases: fresh residual starts at exact copy for every horizon; historical direct
   renders as direct; mixed-kind rendering or cross-kind resume is rejected.
6. Tests: exact T4 copy initialization, delta update, deterministic resume for each
   kind, mismatch rejection and renderer metadata dispatch.
7. Wrong: label transformer internal skip connections as residual prediction.
   Correct: explicitly reconstruct `next = current + delta` at every rollout step.

### Convergence continuation

- `train --continue-from <completed checkpoint> --output <new directory>` preserves
  source weights, optimizer, RNG and global data position. Original `--steps` and
  configuration identify the source and warmup; they do not cap continuation.
- Source files are hashed and pinned in continuation identity. Same-run `--resume`
  also supplies the original `--continue-from`; incompatible identity is rejected.
- Evaluate every complete trajectory epoch. Stop when adjacent-epoch relative
  improvement in mean horizon target-space model MSE is below 1% twice in a row.
  Regressions count as insufficient improvement; source evaluation is the baseline.
- Save metric history and best/last checkpoint pointers. Runtime or disk pauses
  are resumable and must not set convergence completion. Existing sources stay intact.
- Tests cover source identity, optimizer/RNG/global schedule continuity, original
  warmup, patience reset, two insufficient epochs, resumable pause and render guards.
- Wrong: increase the old step budget and restart its warmup or call timeout
  convergence. Correct: continue the same optimization and record the actual stop reason.
