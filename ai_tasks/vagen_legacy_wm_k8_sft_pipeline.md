# vagen_legacy_wm k=8 rollout → SFT1 → SFT2 任务

日期：2026-07-09
状态：实验前核查与 compact cache 优化已完成；代码已提交，尚未启动 rollout、正式 cache 或训练

## 目标

使用 `vagen_legacy_wm_entropy01_kl001_60step_2env4train` 这一次训练得到的 checkpoint 作为来源，按顺序完成：

1. rollout 数据采集；
2. SFT1 格式微调；
3. SFT2 latent WM + value 微调；

本任务统一采用 `latent_token_count = 8`，即每一步使用 8 个 latent query tokens 承载 Qwen 导出的原始状态。

## 固定要求

- 源 checkpoint：`vagen_legacy_wm_entropy01_kl001_60step_2env4train`。
- latent query tokens 数量：`k = 8`。
- token 方案：slot0 为 `<|latent_state|>`，额外 slot 为 `<|latent_state_1|>` ... `<|latent_state_7|>`。
- latent query tokens 应作为状态查询槽处理，默认不作为普通 CE 文本目标训练。
- 实验启动前必须按实验规则执行 on-experiment-start；实验结束/失败/取消后执行 on-experiment-end。

## 前置检查

1. 定位源 checkpoint 的准确路径与类型：
   - 是否已经是 HuggingFace actor/export；
   - 若仍是 VAGEN/verl checkpoint，先转换或导出到 SFT1/rollout 可直接加载的 HF 格式；
   - 记录 checkpoint step、commit、wandb run、配置摘要。
2. 确认 rollout 使用的 policy 初始化自该 checkpoint，而不是旧的 `global_step_79` 或其他默认 checkpoint。
3. 确认 SFT1 是否已支持 `k=8`：
   - tokenizer 注册 8 个 latent query tokens；
   - chat template 渲染后能把旧单 `<|latent_state|>` block 规范为 8-token block；
   - SFT1 labels mask latent query tokens；
   - preprocess/cache fingerprint 包含 `latent_token_count` 与 label mask 配置。
4. 确认 SFT2 使用包含多 latent query token 支持的代码版本，并在训练参数中显式设置：
   - `--latent-token-count 8`
   - `--mask-latent-query-labels`
5. 为本任务单独设定输出目录和 run name，禁止覆盖既有 rollout / SFT1 / SFT2 输出。

## 阶段 1：rollout

### 输入

- policy checkpoint：`vagen_legacy_wm_entropy01_kl001_60step_2env4train` 对应 HF/actor checkpoint。
- 环境：VAGEN navigation / Nimloth 格式。
- split：至少采集 train / val；若资源允许，同时采 test 供最终对比。

### 输出

- raw rollout JSONL：保留完整轨迹、截图路径、动作、reward/success。
- converted Nimloth SFT records：供 SFT1/SFT2 复用。
- rollout summary：每个 split 的 record 数、transition 数、success rate、失败原因统计。

### 验收

- 确认 train/val/test 不混用。
- 抽样检查记录中每一步包含图像、chat history、action block、success/reward。
- 记录源 checkpoint 与 rollout 输出路径。

## 阶段 2：SFT1（k=8）

### 输入

- 初始模型：使用阶段 1 同一个源 checkpoint 的 HF/actor 版本，除非人类另行指定。
- 数据：阶段 1 rollout 转换后的 SFT records。
- 训练集：train split 中的成功 rollout。
- 验证：val split。

### 关键参数

- `latent_token_count = 8`。
- 训练时注册 8 个 latent query tokens。
- labels 默认 mask latent query tokens。
- 保存 LoRA checkpoint，并导出/merge 最佳 checkpoint 为 HF 格式，供 SFT2 初始化。

### 验收

- SFT1 train/val loss 正常下降或达到收敛判据。
- 每个 checkpoint 的 greedy val/test eval 结果可追踪。
- 生成供 SFT2 使用的 `hf_merged` 或等价完整 HF checkpoint。

## 阶段 3：SFT2（k=8）

### 输入

- 初始模型：阶段 2 最佳 SFT1 HF checkpoint。
- 数据：阶段 1 rollout 转换后的 train/val records；SFT2 允许失败 rollout 进入训练。

### 关键参数

- 显式设置 `latent_token_count = 8`。
- `mask_latent_query_labels = true`。
- checkpoint metadata 必须记录并匹配：
  - `latent_token_count = 8`
  - `qwen_hidden_dim`
  - `state_proj_input_dim = 8 * qwen_hidden_dim`
- 默认保持 SFT2 当前主路径：latent WM loss + value loss + CE loss；是否启用 full-trajectory batching 按当时默认配置执行。

### 验收

- SFT2 使用 CPU-only compact preprocess cache build；cache fingerprint 包含 `latent_token_count=8`、processor source、dtype 与数据源信息，GPU job 强制使用 prebuilt cache。
- `StateProjector` 输入维度为 `8 * qwen_hidden_dim`。
- 训练日志包含 WM MSE、value loss、CE loss、SIGReg、val success rate。
- 保存 best/latest checkpoint，并能用 metadata 检查 k=8 配置。

## 建议输出命名

为避免与旧实验混淆，建议统一使用包含源 run 与 k 的名字，例如：

- rollout：`rollout_vagen_legacy_wm_entropy01_kl001_60step_2env4train_k8`
- SFT1：`sft1_vagen_legacy_wm_entropy01_kl001_60step_2env4train_k8`
- SFT2：`sft2_vagen_legacy_wm_entropy01_kl001_60step_2env4train_k8`

## 已完成准备

- 新增 SFT1 k 配置能力：`experiments/training/sft1/train.py` 支持 `--latent-token-count`、`--[no-]mask-latent-query-labels`，并在渲染/tokenize 阶段规范 latent block。
- SFT1 cache fingerprint / manifest / checkpoint metadata 已包含 latent token 配置。
- SFT1 Slurm wrapper 支持 `LATENT_TOKEN_COUNT`、`MASK_LATENT_QUERY_LABELS`、`TRAIN_JSONL` 与 `VAL_JSONL` 环境变量。
- SFT2 Slurm wrapper 支持 `LATENT_TOKEN_COUNT` 与 `MASK_LATENT_QUERY_LABELS`，并传入 preprocess cache build 与训练入口。
- 新增 k=8 参考配置：
  - `configs/training/sft1/qwen25vl_lora_k8.yaml`
  - `configs/training/sft2/latent_wm_value_k8.yaml`
- 新增 runbook：`experiments/training/vagen_legacy_wm_k8/README.md`。
- VAGEN Nimloth prompt helper 已支持通过 `NIMLOTH_LATENT_TOKEN_COUNT` / `LATENT_TOKEN_COUNT` 生成多 latent query token 格式；parser 可保留 action block 前的 extra latent query tokens。
- 已确认源 checkpoint：`/project/peilab/atst/nimloth/outputs/experiments/training/baseline/2026-06-24/vagen_legacy_wm_entropy01_kl001_60step_2env4train/checkpoints/global_step_60/actor/huggingface`，是完整四分片 HF actor export。
- 源 tokenizer 没有 Nimloth latent/action tokens；rollout 必须沿用源模型训练时的 legacy `eval_mode`，转换阶段再生成 k=8 Nimloth block。
- compact cache 优化提交 `0ffcf1e`：唯一图像 BF16 mmap shards + transition token/index shards，保持独立 per-prefix 语义；SFT1/SFT2 cache 均可在 CPU Slurm job 预建，再以 `afterok` 启动 GPU job。
- 远程真实 processor smoke 已确认 compact 与在线编码的首/末 prefix input IDs、labels、grid 完全一致，pixels 在 BF16 后完全一致；按现有同规模 60,170 unique images 外推，完整 SFT2 train+val compact cache 约 45.67 GiB。
- 验证：py_compile、bash -n、compileall 与相关 SFT2 pytest（31 passed）均通过；详细命令见 `ai_tasks/ai_progress/2026-07-09_vagen_legacy_wm_k8_prep.md`。

## 待确认问题

1. rollout 是否包含 test split；当前执行方案建议同时采 test。
2. SFT1/SFT2 的具体资源配置、训练轮数和 early-stop 标准。
3. GPU 资源仍待确认；cache 存储阻塞已解除：旧 1.3 TiB cache 已经人类批准删除，新 compact cache 当前外推约 45.67 GiB，正式大小以 CPU build manifest 为准。
