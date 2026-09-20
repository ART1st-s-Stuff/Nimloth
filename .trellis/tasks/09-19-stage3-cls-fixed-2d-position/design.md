# Stage3 CLS、固定二维位置与 reconstruction 设计

## 0. 2026-09-20 起点与 projector 修订

本节覆盖后文以 Stage2/K65 fresh-start 为起点的旧设计。

执行分为两个有证据边界的阶段。阶段A从仍完整的旧 K64 Stage3 epoch2/step46 恢复，使用
旧代码语义重放至epoch5/step115。它不引入CLS或新projector，只恢复已经被清理的续训终点；
保存新的完整checkpoint及与旧 `train_step_log.csv`、固定诊断的比较结果。

阶段B把阶段A的K64 checkpoint迁移到K65：

1. 先执行显式、可审计的model/token转换：以阶段A的K64 Stage3 HF目录为唯一Qwen来源，
   注册一个CLS Query token，按本任务已经验收的全局Query初始化合同生成其embedding row，
   并逐值证明原64个Query与action/boundary rows未改变。转换产物写入独立目录，保存父checkpoint
   与新CLS cache identity；不得借用当前K65 Stage2 checkpoint中的Qwen或token rows。
2. `SplitSpatialGlobalProjector` 对外保持 `(B,65,H)->(B,65,1024)`，内部包含
   `spatial`（64槽共享MLP）和`global`（1槽MLP）两个互不共享的模块。
3. `spatial`严格加载K64 `state_proj.pt`；`global`从同一权重复制初始化，使迁移时新增分支
   与旧映射处于相同坐标系，同时在第一次更新后允许独立适配DINO CLS。
4. checkpoint metadata显式记录 `projector_layout=split_spatial_global_v1`、slot ordering、两个
   分支的维度和初始化来源。旧单projector checkpoint仅允许通过显式K64→K65 migration入口
   加载；普通resume必须结构严格一致，禁止自动宽松加载。
5. optimizer为两个projector分支使用独立命名参数组；两组默认沿用原projector LR。保存/
   恢复必须覆盖两组optimizer state，且参数审计分别证明 spatial/global 均更新、未授权参数
   保持不变。
6. WM迁移只加载shape-compatible的body、action conditioning和residual delta head。
   `spatial_position`不加载；新predictor由fixed `8x8` sin-cos buffer和global zero buffer构造。
   迁移必须fail closed报告missing/unexpected keys白名单，其他差异均拒绝。

阶段B的训练loss仍分别报告`DINO spatial`、`DINO CLS`、`WM spatial`、`WM CLS`、LM、value、
outcome和总loss。总DINO沿用两项相加的既有K65口径，但拆分projector确保CLS的大梯度不会
直接写入spatial projector。Value/Outcome现有读出保持本任务已经批准的K65行为；本轮主要
验收reconstruction，不用head结果证明CLS表征质量。

## 1. 实验边界

本任务从包含近期 Outcome+BCE Stage3 与评估修复的
`codex/stage3-residual-rl` 分支建立专用 worktree。实验分成两个连续阶段：

1. 从既有 Stage2 epoch16 增补一个真正的 Qwen 全局 Query token，只训练该 token
   的 embedding row，使其拟合真实 DINOv2 CLS；
2. 从上述 evaluation-only Stage2 checkpoint fresh-start Stage3，用 K64 spatial + K1
   global state、固定二维空间编码和 residual WM 正常训练。

这是一组联合架构评估，不能把结果单独归因于 CLS 或二维位置。ValueHead 与
OutcomeHead 继续训练并记录健康指标，但不属于主要比较目标。正式模型仍需以后从
Stage2 epoch1 就启用 CLS；本任务不会把 epoch16 增补结果升级为正式模型或 RL 起点。

## 2. State 与 token 合同

统一 state 轴为：

```text
[spatial_00, ..., spatial_63, global_cls]
```

- spatial slots 固定为 row-major `8x8`，索引 `[0, 64)`；
- global slot 固定为最后一个，索引 `64`；
- 配置分别保存 `spatial_grid_size=8`、`spatial_tokens=64`、
  `global_tokens=1`、`state_tokens=65`，不得继续用一个 `grid_tokens` 同时表示两种语义；
- 所有切片通过一个显式 layout 对象完成，不在调用点散落 `[:-1]`、`-1` 或 `64`；
- Qwen prompt 中全局 Query 位于64个 spatial Query之后、`action_start`之前。因果注意力
  保证旧 spatial hidden 不读取新增 token，而 global hidden 可汇总全部 spatial Query，
  且不能读取尚未执行的 action。

旧 K64 checkpoint 没有 global slot 或新 schema 时必须 fail closed；不自动复制、补零、
池化或猜测 CLS。

## 3. DINO target 与缓存

冻结 teacher 的一次前向同时提取：

- `last_hidden_state[:, 0]` 作为真实 CLS，shape `1024`；
- 最后 `patch_count` 个 patch tokens 按当前规则池化为 row-major `64x1024`。

缓存使用新版本 schema，并记录 teacher source/revision、processor fingerprint、图像身份、
split、dtype、grid size、父数据 fingerprint、构建 commit 和逐文件 hash。旧缓存保持只读，
不得从 patch mean 或已有 state 推导伪 CLS。训练与验证加载时同时核对 spatial 与 CLS；
任一部分缺失、顺序不符或 lineage 不一致即拒绝。

## 4. Evaluation-only Stage2 CLS alignment

从远端既有 Stage2 epoch16 完整加载 Qwen、tokenizer 与 shared projector：

- 新 CLS Query row 以64个现有 Query embedding 的均值初始化，保持初始尺度且不借用
  action/format token；
- 冻结 Qwen backbone、视觉模块、旧64个 Query rows、action/format rows与 projector；
- 只有新 CLS row 可训练；LM 目标继续约束它不能破坏后续语言/动作输出；
- CLS 使用真实 DINO CLS MSE，沿用当前 DINO 系数2；spatial DINO 仅作为不变性诊断；
- observed spatial state 必须与 epoch16 在数值容差内一致。由于它位于新增 token 之前且
  其权重与 projector 均冻结，这是一项硬性回归；不满足时停止 Stage3。

输出 checkpoint 明确标记 `evaluation_only=true`、`formal_stage2=false`、父 epoch16 identity、
K64+CLS schema 与新增 token id。训练至少两轮，并在连续两轮 CLS 验证 MSE 相对改善均
不足1%时停止；同时检查 LM、格式与固定 rollout success 未发生不可接受退化。

## 5. Stage3 WM 与固定位置编码

Stage3 fresh-start 新 residual predictor，不继承旧 WM、ValueHead、OutcomeHead、optimizer
或收敛历史。WM 输入输出均为 `(B,T,65,1024)`。

空间位置编码使用标准二维 sine-cosine：x/y 各占一半通道，每个轴内部使用成对 sin/cos
频率，按 row-major 生成 `(1,64,1024)` persistent buffer。它不进入 optimizer，跨 rank、
保存/恢复和随机种子逐值一致。global slot 使用零 positional vector 并由 layout/type metadata
明确标记为 non-spatial；它是唯一没有二维坐标的槽位。现有 learned
`spatial_position` 被替换，不与固定编码叠加；H1 temporal position 保持当前语义。

Transformer 在65个槽位之间正常双向 attention，因此 global state 可影响 predicted
spatial state。residual delta head 继续零初始化，首个 optimizer update 前所有65个预测槽位
逐值等于输入 state。

## 6. Loss、梯度与两个旧 head

空间和 CLS 分开归一化，避免 global token 被 `1/65` 稀释：

```text
L_wm   = L_wm_spatial + L_wm_cls
L_dino = 2 * L_dino_spatial + 2 * L_dino_cls
```

每一项先对自身有效样本、token和通道取 mean，再进入总目标。空间项保持旧权重，CLS 是
新增的同权重目标；日志独立记录四个分项和加权总量。其余 LM、Value、Outcome BCE、
WM warmup、SIGReg0 和 stop-gradient 合同沿用 Outcome epoch5 基线的实际 resolved config。

ValueHead 继续从 decision state 计算 outgoing `Q(s_t,a)`，OutcomeHead 继续从 action-
conditioned predicted successor 计算 BCE。为保持旧 K64 读出语义，两者都只接收 layout
切出的64个 spatial slots，再执行当前 mean pooling；global slot 不直接进入两个 head。
它们正常训练并保存，但只作为健康指标。这不会修复其汇聚缺陷，也不能据此宣称价值或
结果预测变好。PlannerPolicyHead 与 RL 不在本任务中加载或训练。

## 7. Reconstruction 与固定评估

从冻结导出的 K65 state 与对应图像从头训练 `spatial_cls_grid_v1` CFM。它显式拆分
`[spatial_00...spatial_63, global_cls]`：空间分支保持 `8x8` row-major 和固定坐标的逐尺度
注入；CLS 经独立 LayerNorm/MLP 形成 global condition，与空间分支的 global summary
合成后进入时间/残差条件，不占据二维格点。整个 CFM 可训练，但 Qwen、projector、WM、
Value/Outcome 均冻结且不进入 reconstruction optimizer。

decoder fitting 只使用 reconstruction train split；validation 图像完全隔离。固定评估使用
相同样本身份、future horizon、noise seed、采样步数和渲染尺度，分别报告：

1. CLS-aligned Stage2 与 Stage3 observed K65 reconstruction；
2. Stage3 的真实 future K65 state reconstruction，作为 decoder/目标参照；
3. Stage3 WM-predicted K65 state reconstruction，按 H1--H4 与真实 future image比较；
4. copy-current K65 baseline；
5. 相同空间 state 下正确 CLS、零 CLS、跨样本打乱 CLS 的配对指标与图片；
6. observed/predicted spatial 到 DINO grid，以及 global 到 DINO CLS 的 MSE、cosine、
   centered variation 与错误配对优势。

展示图只保留原图、真实 future、copy baseline、真实 future state reconstruction 与模型
predicted reconstruction，并写清 horizon 和列名。实验用的均值图、误差热图等诊断列不
混入最终展示；完整诊断仍保存在机器可读 artifact 中。

正确 CLS 相对零/打乱 CLS 的配对改善，只说明这个 post-hoc decoder 能利用 CLS 中与图像
有关的信息；它不证明 WM 动力学或策略改善。Value/Outcome 指标不参与 reconstruction
结论。

## 8. Checkpoint、兼容与回滚

checkpoint 保存 tokenizer/token id、state layout、DINO cache identity、position encoding
版本、predictor schema、各 loss 权重、train/freeze 参数集及 evaluation-only 标记。加载器
必须验证 total/spatial/global token 数与 head input contract；旧 K64、旧 learned-position
WM、缺失 CLS cache 或 mean-over-K65 的配置均拒绝混用。

实验使用唯一输出目录并保留最后一个可续训 checkpoint、best checkpoint、逐步日志、
resolved config、固定评估 artifact 和结果摘要。原 Stage2 epoch16、旧 Stage3 baseline 与
冻结 CFM 均只读。失败时删除或保留输出遵循单独授权；代码回滚仅需移除本任务 worktree，
不会修改既有模型产物。

## 9. 启动门禁

实现先通过 schema、shape/order、固定编码、grad reachability、head spatial slicing、residual
copy、checkpoint round-trip、K64 fail-closed、CFM spatial/CLS routing 与消融 identity测试；随后运行
production-shaped 单步 GPU canary。正式训练预计超过10分钟，canary 后需提交包含精确
commit、输入、资源、预算、保存策略、停止规则和输出路径的最终 launch contract，并取得
单独启动批准。

## 10. Frozen-representation WM continuation

追加诊断不再执行 Qwen 前向训练。先从选定 Stage3 checkpoint 对完整 train/eval split
分别导出一次不可变的 K65 state、真实 DINO target 和动作序列；cache identity 同时绑定
实际生成表示的 Stage3 `training_state.pt`、`state_proj.pt` 与 WM config/weights，避免只
记录底层 Stage2 初始化而误认来源。

离线训练只实例化 production residual fixed-2D WM，并从 Stage3 `wm_predictor` 权重加载。
optimizer 是新的 AdamW，参数集合必须逐项等于 WM 参数集合；Qwen、vision、Query、
projector、ValueHead、OutcomeHead 不进入该进程。训练采用原 WM LR `3e-4`、有效 batch64、
H1/T4 和相同轨迹划分，以每个完整 epoch 的 validation WM MSE 观察趋势；保存点按 WM-only
update 重新从1计数，不能与原 Stage3 global step 混用。

表示质量与动力学质量分开报告。cache 上 observed state 对 observed DINO，以及冻结 CFM
对 observed state 的 reconstruction 是固定 ceiling；WM-only 更新后只允许 predicted
state 指标变化。最终至少比较 H1--H4 的 model、input-copy、DINO-space readout和
reconstruction，若仍未超过 copy baseline，才把扩大容量作为后续候选实验。

## 11. Stage2 Query/projector-only representation diagnostic

增加独立 `query_projector_only` tuning mode。它复用 Stage2 一次整轨迹前向与既有K65
spatial/CLS目标，不增加另一套loss实现。加载 epoch8 的普通HF权重与 projector 后，冻结
完整模型，再为65个 Query input embedding rows建立FP32 master，并只重新开放 projector。
projector保留FP32 master，前向时由现有 `SharedSlotProjector` 显式转换BF16 hidden。

epoch8 的 `selected_token_rows.pt` 只包含CLS精确FP32行；安装65行master后按token ID做
严格子集恢复，使CLS不经BF16 round-trip，空间行则忠实使用epoch8实际保存的dense权重。
该子集恢复只接受同一input table、ID子集、空protocol行及匹配dtype/维度；任何不一致均
拒绝。保存时继续物化普通HF dense权重并附带全部65行的精确sidecar。

optimizer固定为两个互斥参数组：projector `8e-5` 与Query rows `1e-4`，不包含Qwen、LM
head、vision或protocol rows。参数集合变化使旧单行optimizer不可恢复，因此从epoch8仅
初始化权重并重置optimizer、scheduler与收敛历史。收敛监控使用
`validation_dino_loss = validation_dino_spatial_loss + validation_dino_cls_loss`；语言loss、
格式及后续direct success仅作为保持门禁。运行metadata保留直接初始化checkpoint和原始
Stage2 parent，明确evaluation-only及旧split暴露边界。

## 12. Stage2 spatial plateau diagnosis

R6 使用边界暂停后，不再用 joint validation loss 的继续下降推断 spatial 表示仍在改善。
以暂停 checkpoint 导出 train/eval 每个 answer 的65个 Query final hidden states与对应
DINO targets；cache 绑定 checkpoint safetensors、projector、grid metadata、JSONL hash、
DINO fingerprint、精度与 extraction world size，源 checkpoint 保持只读。

诊断只消费冻结 cache。当前 shared projector 先作为零更新 baseline，分别报告 K64 spatial
与 K1 CLS。随后在确定性的固定 train batches 上分别对 spatial MSE 与 CLS MSE求 projector
梯度，累积 sample-weighted gradient 后报告 norm、dot、cosine及逐batch cosine摘要。该梯度
只回答 shared projector 参数上的局部关系，不外推到已冻结的 Query 或 Qwen 梯度。

spatial-only probe 从当前 shared projector 权重初始化，仍输出完整 K65，但 optimizer loss与
验证指标只切取 layout 声明的 K64 spatial tokens；CLS输出不参与更新。projector FP32、
AdamW、LR `2e-5`、batch64、至少2轮、连续2轮相对改善不足1%停止，最多20轮。保留 baseline、
逐轮metrics、best/last projector和完成摘要；不得把probe权重写回训练 checkpoint或直接用于
Stage3。若 spatial-only 显著优于 joint baseline，再设计 separate head/loss schedule；否则
下一步才考虑以低LR开放Qwen表示。
