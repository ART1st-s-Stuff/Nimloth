# SFT3：世界模型与价值头训练

本目录迁入原 `training/sft2` 的完整实现；公共 `SFT2*` 类型、函数和
checkpoint 标识保留历史名称，避免目录迁移改变恢复及下游加载合同。
这里的 SFT3 不等于新 SFT2 的 query/DINO 对齐阶段。

Stage3 的 WM 预测拟合停止梯度的真实后继编码 state；DINO MSE 只约束
在线编码器经 projector 输出的真实观测 state。每条有效完整轨迹的所有
观测（包括末观测）各计算一次，不按重叠窗口重复，不包含分布式 padding。
`dino_grid_mse` 是此观测对齐训练损失，独立按整个 optimizer accumulation
group、所有 rank 的有效观测数归一化。`predicted_dino_grid_mse` 是未来
预测对冻结 DINO 的诊断，不进入训练损失；逐窗口导出的 `dino_mse` 仍是
这个预测诊断，方便与 copy baseline 和可视化比较。新 checkpoint 使用
`unique_observed_online_state_mse_v1`，拒绝恢复旧预测 DINO 目标的优化器状态。

训练入口为 `python -m nimloth.training.sft.stage3 --config <原 WM/value 配置>`，
调用 `trainer.main()` / `train_sft2()`，配置类型仍为 `nimloth.config.sft2`。
当前入口要求 H=1；历史 H4 配置不属于原生轨迹训练入口，加载时明确拒绝，不能直接复用旧命令。
模型结构与损失系数沿用已有配置；当前训练单位改为完整轨迹，旧窗口模式的优化器状态不可续训。

## 阅读顺序

1. `trainer.py` 构建 Agent、模型包装、EMA、优化器和数据，再启动 `loop.py`。
2. `data/factory.py` 选择 sampler；`data/samplers.py` 定义真实轨迹的采样单位。
3. `batch.py` 对齐起点、执行动作、完整 episode 的 MC return 和后继观测；
   `dino_grid.py` 装配冻结 teacher cache 的空间 target。
4. `algorithm.py` 组织共享在线计算图上的联合损失与反传；`sigreg.py` 负责跨 rank 的有效样本汇聚、可微通信和同步随机投影；`runtime.py` 管理目标编码、反传和更新。
5. `loop.py` 驱动 microbatch、恢复游标和验证；`reporting.py` 汇总指标；
   `checkpoint.py` 保存模型、优化器、EMA 与恢复游标。

## 完整轨迹与多步窗口

一个数据样本是一条完整轨迹，H=1；微批和梯度累积均以完整轨迹为单位。
每条轨迹仅加载一次完整图像历史，目标 EMA/eval 分支先无梯度编码全部状态，
在线分支随后一次编码全部状态与有效窗口起点的回答 LM loss。
长度 L 的轨迹提供 L−T+1 个有效窗口，各含 T 个动作和 T+1 个真实观测；
短轨迹排除，不补造观测。所有窗口统一递推，后续 WM 输入是预测状态。
value 监督使用完整 episode 上计算的 MC return，不在窗口末尾截断。

主损失直接在共享计算图上反传，不再使用 detached leaf、手工 VJP 或历史 state 缓存。
SIGReg 使用全部真实相邻 transition，按轨迹和时间位置去重，排除分布式补齐；
每个微批跨卡形成统计组。起点 detach，梯度进入在线后继状态；与主损失联合反传。

DINO grid 的 `grid.size` 与 `latent.token_count` 由配置显式给出，并要求
普通空间模式下 `latent.token_count == grid.size ** 2`。训练启动时还会读取初始化 checkpoint 的
`grid_state_config.json`，核对 grid 大小、slot 数、DINO identity、state 维度和
row-major 顺序；因此 Stage 3 可以消费相符的 4×4/K16 或 8×8/K64 Stage 2
checkpoint，但不会在两种 state 接口之间静默转换。

CLS/固定二维位置的 evaluation-only 路径使用显式
`spatial_grid_size=8, global_tokens=1, state_tokens=65` 合同，state 顺序固定为
K64 row-major spatial 后接真实 DINO CLS。此时 `latent.token_count` 必须等于
`grid.size ** 2 + grid.global_tokens`，初始化 checkpoint 必须来自带
evaluation-only lineage 的 Stage2 CLS alignment。若 Stage3 的 v2 DINO cache 覆盖
更多 future observations，启动时必须用 `--stage2-aligned-dino-cache` 提供 Stage2
实际使用的 cache：其 corpus fingerprint 必须等于 checkpoint 记录，两个 cache 的
teacher、processor、K64+CLS layout、dtype/dimension 和 ordering 必须逐项一致；两边
corpus fingerprint 与 feature-space identity 都写入 checkpoint 审计元数据。该检查在
distributed setup 和模型加载前执行。`fixed_2d_sincos_v1` 给 K64 使用确定性的二维 sine-cosine persistent
buffer，CLS 使用零空间位置；它不叠加旧的可训练 spatial position。Residual
delta head 仍为零初始化，所以首个更新前 K65 逐值复制输入。

K65 训练把 WM spatial/CLS MSE 和 observed-state DINO spatial/CLS MSE 分开
归一化并分别记录，`lambda_dino` 同时乘到两个 DINO 分项，不做 65-token
平均。`train_step_log.csv` 同时持久化兼容总量 `wm_mse`、`dino_grid_mse`、
`predicted_dino_grid_mse` 及其 `*_spatial_mse` / `*_cls_mse` 分项；总量仍是两个
分别归一化分项的和，不是对 K65 直接取平均。ValueHead、OutcomeHead 与 SIGReg 通过 `GridStateLayout` 只读取 K64，
保持既有 spatial mean-pooling 语义；这不修复两个 head 已知的汇聚限制。
frozen CFM reconstruction 同样只接收 K64，CLS 只通过独立特征指标评估。
配置样例为
`configs/training/sft2/action_outcome_k64_cls_fixed2d_h1_t4_eval.yaml`。

K64 到 K65 会改变 Query 数量和 tokenizer，因此 transition token cache 必须重建。
`--preprocess-cache-reuse-image-root <root>` 可从 `<root>/train` 与 `<root>/val`
分别复用经过完整身份校验的 image shards，同时在新的
`--preprocess-cache-dir` 重建两组 transition shards。该入口不能与
`--require-prebuilt-cache` 同用；配置默认要求 prebuilt 时必须显式传入
`--no-require-prebuilt-cache`，source/destination 目录不得重叠。对于尚未保存独立
image-processor identity 的旧 K64 cache，还必须用
`--preprocess-cache-reuse-processor-source` 指向当时构建 cache 的精确 processor
checkpoint；该路径只验证旧 source，destination transition 始终使用当前 `--model`
processor 构建。既有 `--preprocess-cache-processor-source` 仍只描述 required-prebuilt
destination cache，不能替代 reuse source。加载器会重算旧 base fingerprint，并比较旧
processor 与 K65 destination processor 的视觉配置。任何图像路径/指纹、pixel bounds、
dtype、分片、grid/offset、文件 hash 或视觉 processor 不匹配都会拒绝复用。

## 验证、诊断和兼容

`--eval-only --feature-export-dir <new-directory>` loads the configured weights
and calls the production validation forward without optimizer updates or checkpoint
writes. `--max-val-batches` bounds the number of trajectory batches per rank.
The feature export preserves full spatial grids and distinct online/EMA direct
states; it is consumed by `render_dino_feature_comparison.py` for matched target-only
PCA figures. Stage2 direct reconstruction sees the target observation, while Stage3
WM predicts its features from an earlier state and actions; these tasks differ.

`--eval-only --frozen-wm-cache-dir` exports ordered state/DINO/action sequences for
the offline WM-only diagnostic. When evaluation resumes a Stage3 checkpoint, the
cache identity hashes that checkpoint's `training_state.pt`, `state_proj.pt`, and
WM config/weights; the underlying Stage2 initialization alone is not accepted as
the representation identity.

`evaluate.py` 负责离线 loss 验证，`mcts_evaluation.py` 提供 MCTS checkpoint 及 value 语义校验；真实环境评估统一由 [`../evaluation/`](../evaluation/README.md) 提供。

专项 canary、动作头修复、特征定位审计和 packed/KV 研究原型位于 `experiments/training/sft/diagnosis/`。这些工具可调用训练组件，训练组件不依赖实验工具。

旧 `nimloth.training.sft2` Python 包已移除，代码调用者应使用 `nimloth.training.sft.stage3`。既有 `SFT2*` 类型名、配置字段和 `decision_state_executed_action_mc_v3` 仍描述相同 WM/value 目标，不会被新 Query 对齐阶段静默接受。历史产物不改写；通过实际 checkpoint 加载与恢复测试校验迁移。

K64 Stage3 到 evaluation-only K65 的转换必须显式传入
`--k64-stage3-migration-checkpoint`，并让 `--model` 指向同一个完整 K64 checkpoint。
该入口只作一次性初始化：保留已有64个 Query 与协议 token 行，以64个 Query 行均值初始化
新增 global Query；spatial projector 原样继承，global projector 从它逐值复制；residual WM
只继承 shape-compatible 权重并重建 fixed-2D position。转换后使用新的 optimizer。普通
`--resume` 只接受相同的 `split_spatial_global_v1` schema、layout 与 optimizer 参数组，不会
自动执行 K64→K65 转换。

## 成功轨迹的 LM 监督

训练数据必须包含成功和失败轨迹，不接受 `success_only` 过滤。`batch.py` 将完整轨迹的显式布尔 `success` 传入起点的 `lm_row_weights`；缺失标记拒绝。成功窗口的起点回答先独立计算 token CE 均值，再按成功窗口求平均。失败窗口保留全部真实输入、动作、回报及 WM/value/DINO 监督。全失败组 LM 为图连接的零。

`loop.py` 在一个更新组内统计全部有效窗口数和成功窗口数，跨 rank 归约后分别缩放主损失及 LM。SIGReg 每个微批计算一次，按当前累积组的实际微批数取平均。验证在线状态使用当前 policy 的 eval 权重，目标状态独立使用 EMA；训练 LM 指标按成功窗口数汇总。恢复身份新增监督范围与归一化版本，不能复用旧 LM 目标的优化器状态。

## 可选动作执行结果监督

`--outcome-head` 为 DINO grid 模式建立 FP32 LayerNorm、slot mean pooling 和
binary linear readout。输入直接复用 predictor 输出的动作条件 grid，不增加
predictor 或 Qwen 前向；T 步 logits 与 T 个 outgoing action 一一对应。
`--lambda-outcome` 默认为 0；大于 0 时加入普通 BCEWithLogitsLoss，不做类别
加权或初始 loss 归一化。有效标签按整个 optimizer 累积组、全部 rank 的动作数
归一化，padding 与缺失标签不参与。实验数据审计另行要求完整 outcome 标签。

系数为 0 时仍可建立同形状的冻结 head，但不加入 optimizer 或声明预测能力。
checkpoint 的 `outcome_head.pt` 保存 schema、维度、可用标记和参数；恢复拒绝
不匹配的 head 或 loss 身份。此 head 尚未用于 RL/MCTS。

`query_tune=selected_rows` requires dense full language tuning and freezes all dense
input/output vocabulary rows, replacing only Query and eight action/two action-boundary
rows with FP32 masters. `query_lr` and `protocol_lr` are separate zero-weight-decay
optimizer groups; language and visual parameters share the Qwen LR schedule. Trainable
backbone parameters retain FP32 masters and CUDA forwards use BF16 autocast. This mode
never installs the additive Query adapter. HF export materializes ordinary weights;
`selected_token_rows.pt` preserves exact FP32 masters and token IDs for resume alongside
optimizer state. Legacy modes remain unchanged.


`configs/training/sft2/action_outcome_k64_h1_t4.yaml` supplies the shared one-epoch
A/B hyperparameters. Initialization, train/eval JSONL, DINO/cache and output paths must be
provided through CLI; no historical path is silently reused. Explicitly pass
`--lambda-outcome 0` for control or `--lambda-outcome 1` for treatment, with `--seed 42`
and eight ranks. Both arms construct the head; only treatment optimizes its ordinary BCE.
The config retains the latest two complete step checkpoints; epoch/best/final checkpoints
are outside rolling step retention. Incomplete step directories do not count toward the limit
and are preserved for diagnosis.

`--outcome-eval-dir <new-directory>` enables production-forward export: step-zero
validation and every completed epoch's validation/training windows are written to
separate `epoch_NNN_{eval,train}_rank_NNN.jsonl` files. Existing files are refused.
Rows include pooled WM features and fixed-teacher current/future image hashes;
`copy_mse` persists the initial frozen DINO feature, while `encoded_copy_mse` is
only the moving online-state diagnostic. Training exports fit offline probes;
validation exports never select probe weights or epochs.

After both arms finish, compare their rank exports with:

```bash
python -m nimloth.eval.stage3_outcome \
  --control-train 'control/epoch_001_train_rank_*.jsonl' \
  --treatment-train 'treatment/epoch_001_train_rank_*.jsonl' \
  --control-eval 'control/epoch_001_eval_rank_*.jsonl' \
  --treatment-eval 'treatment/epoch_001_eval_rank_*.jsonl' \
  --fit-probes --output comparison.json
```

The matched frozen linear probes reuse the existing probe utility's LR 3e-3,
weight decay 1e-2 and seed 42071, with the same fixed 300-epoch budget for both
arms. Standardization uses training features only; weights remain in a separate
`.probe_weights.npz`, never in production checkpoints. These are real evaluator
fits and must run remotely under the experiment budget; CPU unit tests use
isolated synthetic inputs only.


For a bounded production canary, `--stop-after-steps 1` executes the ordinary sampler,
losses and accumulation schedule until absolute optimizer step 1. It atomically publishes
`stop_step_000001/` with `STOPPED`, exact consumed microbatch cursor, optimizer and all
rank history caches; `epoch_complete` remains false and neither epoch validation nor a
`final` checkpoint runs. Resume in a fresh process with `--resume --resume-from` that
checkpoint and `--stop-after-steps 2` to verify the next update. The limit is disabled by
0, must exceed the restored step, and never changes the full schedule/epoch length.

`--diagnose-outcome-gradients` performs an additional no-update first-microbatch forward
on unwrapped modules, with an isolated history cache and restored Python/NumPy/Torch RNG.
It reports each rank's predictor gradient norm for weighted BCE versus weighted WM+DINO,
without modifying `.grad`, synchronizing diagnostic gradients or imposing a ratio cutoff.
`outcome_gradients_rank_NNN.json` identifies the rank-local scope and token-input hash;
it is not the global gradient ratio of the complete accumulation group. Non-finite losses
or gradients fail. This option requires a positive outcome coefficient.

## 全量 Qwen 的分片训练

`--distributed-strategy fsdp` 将完整 Qwen 主干与选定 FP32 token rows 交给
FULL_SHARD/use_orig_params，参数前向使用 BF16、梯度归约使用 FP32；冻结 BF16
词表作为 ignored parameters 保持复制，避免同一 handle 混合 dtype。
完整 visual 是嵌套 FSDP owner，视觉子模块不转换内部 FP32 rotary 输入。
WM/projector/value/outcome 仍采用 DDP，学习率和目标不变。优化器保留一个 AdamW，
梯度裁剪只对 Qwen 分片范数跨 rank 求和，再加入一份 WM 梯度范数。
不使用 FSDP no_sync 累积，以免保留完整未分片梯度。

FSDP 是新的恢复身份。CPU policy/归一化测试不证明 CUDA all-gather、EMA交换、
联合损失反传或完整 checkpoint 恢复；启动前必须通过实际多卡 canary。

## 恢复身份

新训练使用 `training_unit=complete_trajectory_v1`，batch 单位为轨迹。
旧窗口运行的优化器状态在加载前拒绝；初始化模型与原有 Stage2 基座仍可使用。
原 `--trajectory-shared-forward` 入口及兼容路径已移除，完整轨迹是唯一训练路径。
CPU 检查不能证明真实多卡训练正确；当前重构须通过集成测试与实际多卡验证。
