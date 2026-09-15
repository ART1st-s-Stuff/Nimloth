# SFT3：世界模型与价值头训练

本目录迁入原 `training/sft2` 的完整实现；公共 `SFT2*` 类型、函数和
checkpoint 标识保留历史名称，避免目录迁移改变恢复及下游加载合同。
这里的 SFT3 不等于新 SFT2 的 query/DINO 对齐阶段。

训练入口为 `python -m nimloth.training.sft.stage3 --config <原 WM/value 配置>`，
调用 `trainer.main()` / `train_sft2()`。继续使用 `nimloth.config.sft2` 的已验证配置。
迁移未改变默认配置、模型结构、损失权重、优化器或 checkpoint schema。

## 阅读顺序

1. `trainer.py` 构建 Agent、模型包装、EMA、优化器和数据，再启动 `loop.py`。
2. `data/factory.py` 选择 sampler；`data/samplers.py` 定义真实轨迹的采样单位。
3. `batch.py` 对齐起点、执行动作、完整 episode 的 MC return 和后继观测；
   `dino_grid.py` 装配冻结 teacher cache 的空间 target。
4. `algorithm.py` 组织主损失及独立 SIGReg 的计算/反传顺序；`sigreg.py` 负责跨 rank 的有效样本汇聚、可微通信和同步随机投影；`runtime.py` 管理目标编码、反传和更新。
5. `loop.py` 驱动 microbatch、恢复游标和验证；`reporting.py` 汇总指标；
   `checkpoint.py` 保存模型、优化器、EMA 与各 rank 的历史缓存。

## 多步窗口与旧单步模式

`prediction_horizon=T>1` 使用 `FutureRolloutBatchSampler`，要求 `history_size=1`。
每个窗口含同一 episode 的 T 个连续执行动作及 T+1 个真实观测，只编码起点
作为在线递推输入。`SFT2Algorithm._rollout_step` 调用 Agent 的真实多步前向：
在动作前状态评分 outgoing Q，再预测后继状态。WM/DINO 对齐全部 T 个后继状态，
value 对齐 T 个实际执行动作的 MC return；这些 return 在完整 episode 上计算，
不在窗口尾截断。CE 只来自窗口起点。采样采用固定长度滑窗，短于 T 的片段不产生
窗口，不补造后继状态，也不跨越 episode 或缺失后继观测的位置。

`prediction_horizon=1` 保留 `OnlineHistoryBatchSampler` 及 H 步历史兼容模式。
此模式的统计单位是当前 transition，历史仅提供因果上下文；旧历史 state 从
rank-local `history_cache.py` 读取 detached tensor。这与 T 个未来预测步不同。

目标状态编码沿用冻结 backbone（可使用已有 backbone EMA）和共享 projector 的
无梯度分支。主损失反传完成后才编码相邻在线状态做 SIGReg；起点 detach，梯度只进入
新状态侧。跨 rank 的有效样本统计、padding 零权重及随机投影同步保持原实现。

DINO grid 的 `grid.size` 与 `latent.token_count` 由配置显式给出，并要求
`latent.token_count == grid.size ** 2`。训练启动时还会读取初始化 checkpoint 的
`grid_state_config.json`，核对 grid 大小、slot 数、DINO identity、state 维度和
row-major 顺序；因此 Stage 3 可以消费相符的 4×4/K16 或 8×8/K64 Stage 2
checkpoint，但不会在两种 state 接口之间静默转换。

## 验证、诊断和兼容

`evaluate.py` 负责离线 loss 验证，`mcts_evaluation.py` 提供 MCTS checkpoint 及 value 语义校验；真实环境评估统一由 [`../evaluation/`](../evaluation/README.md) 提供。

专项 canary、动作头修复、特征定位审计和 packed/KV 研究原型位于 `experiments/training/sft/diagnosis/`。这些工具可调用训练组件，训练组件不依赖实验工具。

旧 `nimloth.training.sft2` Python 包已移除，代码调用者应使用 `nimloth.training.sft.stage3`。既有 `SFT2*` 类型名、配置字段和 `decision_state_executed_action_mc_v3` 仍描述相同 WM/value 目标，不会被新 Query 对齐阶段静默接受。历史产物不改写；通过实际 checkpoint 加载与恢复测试校验迁移。

## 成功轨迹的 LM 监督

训练数据必须包含成功和失败轨迹，不接受 `success_only` 过滤。`batch.py` 将完整轨迹的显式布尔 `success` 传入起点的 `lm_row_weights`；缺失标记拒绝。成功窗口的起点回答先独立计算 token CE 均值，再按成功窗口求平均。失败窗口保留全部真实输入、动作、回报及 WM/value/DINO 监督。全失败组 LM 为图连接的零。

`loop.py` 在一个更新组内统计全部有效窗口数和成功窗口数，跨 rank 归约后分别缩放主损失及 LM。SIGReg 保留原独立反传协议。验证仍沿用状态预测指标；训练 LM 指标按成功窗口数汇总。恢复身份新增监督范围与归一化版本，不能复用旧 LM 目标的优化器状态。

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
主损失加 SIGReg 两次反传或完整 checkpoint 恢复；启动前必须通过实际多卡 canary。

## 同一次更新内共享轨迹编码

`--trajectory-shared-forward` 启用 `trajectory.py`：校验 token、图像及 mRoPE
前缀后，按轨迹合并同一参数更新中的输入。目标 EMA/eval 分支先执行，在线
分支再一次返回各步 Query states 与逐窗口 LM loss。窗口与 SIGReg 仍按原
微批执行，先累加对共享输出的梯度，再统一反传在线编码器和 projector。
当前只支持 H=1、T>1、K>1 且无 encoder/projector dropout 的配置；默认关闭。
参数更新后不保留共享 states。详见三阶段 SFT spec 的梯度和归约合同。

`target_encode`、`online_encode`、`shared_backward` 分别计时；主阶段计时
此时只含窗口 heads。规划与前缀校验开销不在这些分项内，总加速按墙钟时间
比较。`encoder_*` 指标是本地更新组数量的汇总均值，不是全局累计 token 数。
