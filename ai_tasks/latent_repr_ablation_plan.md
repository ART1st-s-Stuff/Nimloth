# Latent Representation Ablation 实验计划

日期：2026-07-05  
分支：`exp/latent-repr-ablation`  
状态：计划已记录，并补充分 Phase 与配置驱动要求；代码改动与实验启动均待人类确认后执行  
当前基线 checkpoint：`/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_base_8gpu_paper_repro_20260621/global_step_300`

---

## 目标

当前 WM predictor 与 value head 没有按预期工作，尤其是 WM latent 重建出的图像几乎没有有用信息。本实验组用于验证两个候选原因：

1. **单个 latent token 容量不足**：当前 `<|latent_state|>` 单 token 难以承载导航所需的视觉与状态信息，需要更多 latent tokens。
2. **当前 latent 表达包含太多无效/不适合动力学预测的信息**：WM predictor 也许应直接使用 ViT/vision tokens，或使用压缩后的 vision token set，而不是 Qwen 的单点 latent state。

核心问题不是只看某个模型 loss，而是区分以下瓶颈：

- representation 本身是否保留足够状态信息；
- predictor 是否能在该 representation 空间里做下一步/多步预测；
- value head 是否能从该 representation 中读出可决策信息；
- predictor + value head fast path 是否真的提升或至少保持 task success rate。

---

## 实验评价标准

### 1. 图像重建，肉眼观察为主

每种 representation 都需要分别看两类重建：

1. **true feature → current/next image reconstruction**
   - 用真实 Qwen/vision 编码得到的 state feature 重建图像。
   - 目的：判断 representation 本身是否保留视觉/空间状态信息。
2. **predicted feature → next image reconstruction**
   - 用 `predictor(s_t, a_t)` 得到的 predicted next feature 重建下一帧。
   - 目的：判断 predictor 输出是否仍在有效状态流形上。

解释规则：

- 如果 true feature 重建也很差，主要怀疑 representation 或 decoder 条件信息不足。
- 如果 true feature 重建可用但 predicted feature 重建差，主要怀疑 predictor 的动力学预测质量。
- 如果短步预测还可以、多步迅速崩坏，主要怀疑 compounding error。

### 2. Value head 离线指标

value head 不只看单一 top-1 accuracy。建议记录：

- expert/VAGEN action top-1 accuracy；
- expert/VAGEN action top-2 或 top-k accuracy；
- chosen action 的 value regression error；
- ranking accuracy：成功轨迹/高回报 action 是否排在失败轨迹/低回报 action 前；
- calibration：高 value bin 的实际 success rate 是否更高。

如果多个动作在导航中都可行，top-1 只能作为参考，不能单独决定 representation 好坏。

### 3. Fast path task success rate

最终指标是使用 predictor + value head 多步 fast path 后的整体 task success rate。需要分层评估：

| 模式 | 目的 |
|---|---|
| VAGEN slow path baseline | 上限/teacher，对齐给定 baseline checkpoint |
| value head on true current feature | 测 value head 是否能从真实 feature 中读出决策信息 |
| value head on true next feature | 测 target representation 与标签是否合理 |
| value head on predicted next feature | 测 predictor 误差对 value head 的影响 |
| predictor + value head multi-step beam/greedy fast path | 测完整 fast path 的 task success rate |

多步 fast path 至少记录 depth 曲线，例如 depth 1 / 2 / 4 / 8。若 depth 增大后 success rate 或 value accuracy 快速下降，说明 predictor rollout 误差是主要瓶颈。

---

## 消融矩阵

优先做最小但有判别力的矩阵：

| 实验 | State representation | Predictor 输入/输出 | Value head 输入 | 目的 |
|---|---|---|---|---|
| A baseline | 当前单个 `<|latent_state|>` token，经 `StateProjector` | latent → latent | latent | 复现当前问题，作为对照 |
| B multi latent tokens | N 个 latent tokens，例如 4 / 8 / 16 | latent token set → latent token set | pooled/cross-attn latent tokens | 验证是否只是 latent token 数量不够 |
| C raw ViT/vision tokens | Qwen/VAGEN vision patch tokens | vision tokens → vision tokens | vision tokens + semantic embedding | 验证直接使用视觉 token 是否更适合动力学预测 |
| D compressed vision tokens | vision tokens 经 Perceiver/Transformer 压缩成 K tokens | compressed tokens → compressed tokens | compressed tokens + semantic embedding | 区分“必须用原始 ViT token”和“只需要更大 token set” |

D 组很重要，因为它能判断性能提升来自视觉 token 本身，还是来自 token 数/容量增加。

---

## 关于 ViT token value head 的语义条件

如果 WM predictor 使用 ViT/vision tokens，value head 需要额外语义输入，否则它只能看到局部视觉 patch，可能不知道任务目标、指令、历史语境。

计划中使用更明确的输入形式：

```text
value_head(visual_or_compressed_state_tokens_t, static_task_embedding)
```

其中 `static_task_embedding` 应尽量表示目标/指令/稳定上下文，而不是混入当前 observation 的动态视觉信息。候选来源：

- Qwen 对 instruction / goal / history text 的 pooled hidden state；
- 当前 slow path 中与目标语义相关、但不会随预测步变化的 context embedding；
- 若必须使用 Qwen 当前 embedding，需要在实验记录中说明它是否包含当前 observation 信息，避免把动态视觉信息泄漏给多步 fast path。

---

## 可执行性判断

当前计划的实验目标是合理的，但**不适合一次性把 A/B/C/D 全部实现完再开始实验**。原因：

- A 组 baseline 评估链路本身就能暴露现有瓶颈，应先打通并校准指标。
- B 组 multi latent token 会影响 prompt/format、latent extraction、predictor/value head shape 和 checkpoint metadata，改动面较大。
- C/D 组 vision token 路线会引入新的 token-set representation、semantic embedding 和 token-set reconstruction/value head，和 B 组共享基础设施，但不应和 B 组一起无边界实现。
- fast-path task success rate 依赖环境评估，成本高，应放在离线指标和 smoke 通过之后。

因此执行上必须分 Phase，但代码架构应一次性按“配置驱动”设计：Phase 1 完成通用接口和 baseline 配置，后续 Phase 只新增配置或少量实现模块，不为每个实验手改主训练/评估代码。

---

## 配置驱动要求

目标状态：代码改完后，实验通过不同 YAML 配置启动，不需要再修改 Python 主路径。

### 统一入口

建议新增统一入口，至少包含：

```text
python -m nimloth.training.representation_ablation.train --config <yaml>
python -m nimloth.eval.representation_ablation --config <yaml>
python -m nimloth.eval.representation_fastpath --config <yaml>
```

也可以拆成更小入口，但每个入口必须只从 config/CLI 读取实验差异，不允许在代码里切换实验组。

### Config schema 草案

每个训练/评估配置至少应包含：

```yaml
experiment:
  group: representation_ablation
  name: latent_k8_predictor_value
  output_dir: null
  seed: 0

init:
  vagen_checkpoint: /project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_base_8gpu_paper_repro_20260621/global_step_300
  qwen_checkpoint: null
  state_proj_checkpoint: null
  wm_predictor_checkpoint: null
  value_head_checkpoint: null

data:
  train_jsonl: null
  val_jsonl: null
  split_policy: explicit_train_val
  include_failed_rollouts: true

representation:
  type: qwen_latent        # qwen_latent | qwen_multi_latent | qwen_vision_tokens | compressed_vision_tokens
  num_tokens: 1
  dim: 1024
  source: qwen             # qwen | vagen
  projector: linear        # none | linear | mlp
  compressor: null         # null | perceiver | transformer_pool
  semantic_embedding:
    enabled: false
    source: instruction    # instruction | goal | qwen_context
    pooling: mean

predictor:
  type: lewm_ar            # lewm_ar | token_transformer
  train: true
  history_size: 4
  depth: 6
  heads: 16
  hidden_dim: 1024

value_head:
  type: mlp                # mlp | pooled_mlp | cross_attention
  train: true
  use_semantic_embedding: false

reconstruction:
  enabled: true
  type: rcdm               # rcdm | simple_decoder
  condition_source: true_and_predicted
  image_size: 128
  upload_wandb_images: true

train:
  target: predictor_value  # predictor | value | predictor_value | reconstruction
  epochs: 1
  batch_size: 4
  lr: 1.0e-4
  resume: false
  save_interval: 500

eval:
  metrics:
    - reconstruction_strips
    - value_topk
    - value_ranking
    - predictor_multistep
    - fastpath_success
  rollout_depths: [1, 2, 4, 8]
  planner: beam_search

wandb:
  enabled: true
  project: nimloth
  run_name: null
```

### Registry/factory 约束

为避免每个实验改代码，需实现 registry/factory：

- `RepresentationExtractorFactory`：根据 `representation.type` 创建 extractor/cache builder。
- `PredictorFactory`：根据 `predictor.type` 和 representation shape 创建 predictor。
- `ValueHeadFactory`：根据 `value_head.type` 创建 value head。
- `ReconstructionFactory`：根据 `reconstruction.type` 创建 decoder/evaluator。
- `MetricFactory`：根据 `eval.metrics` 选择评估项。

主训练/评估入口只调用 factory，不写 `if experiment == "B"` 这种实验专用逻辑。

### 配置文件命名建议

```text
configs/training/representation_ablation/
  a_qwen_latent_baseline.yaml
  b_qwen_multi_latent_k4.yaml
  b_qwen_multi_latent_k8.yaml
  b_qwen_multi_latent_k16.yaml
  d_compressed_vision_k8.yaml
  d_compressed_vision_k16.yaml
  c_raw_vision_tokens.yaml

configs/eval/representation_ablation/
  a_qwen_latent_reconstruction.yaml
  a_qwen_latent_value_fastpath.yaml
  b_qwen_multi_latent_k8_reconstruction.yaml
  b_qwen_multi_latent_k8_value_fastpath.yaml
  d_compressed_vision_k8_reconstruction.yaml
  d_compressed_vision_k8_value_fastpath.yaml
```

---

## Phase 计划

### Phase 0：代码阅读与接口冻结

目标：确认现有 SFT2、WM predictor、value head、RCDM reconstruction、VAGEN baseline 的真实接口，冻结 config schema。

产出：

- 最终 YAML schema；
- 需要复用/替换的现有模块列表；
- A/B/D/C 的最小配置文件模板；
- smoke 数据规模和输出目录规则。

不启动昂贵实验。

### Phase 1：配置驱动 baseline A 链路

目标：先让当前单 latent baseline 完整可评估，作为所有后续消融的对照。

实现范围：

- representation config/factory 的 `qwen_latent`；
- predictor/value head/reconstruction/eval 统一配置入口；
- true feature vs predicted feature 的重建 strip；
- value top-k/ranking/calibration；
- predictor depth 1/2/4/8 离线曲线；
- fast-path evaluator 的 smoke 入口。

验收：

- 通过 config 启动 baseline A 的离线评估和 reconstruction smoke；
- 不改代码即可切换 reconstruction-only / value-only / fastpath smoke 配置。

### Phase 2：multi latent token B

目标：验证 latent token 数量是否是主要瓶颈。

实现范围：

- prompt/format 支持多个 latent tokens；
- extraction/cache 支持 `(K, D)` latent token set；
- token-set predictor/value head；
- K=4/8/16 配置文件。

验收：

- 不改代码，仅切换 config 即可跑 K=4/8/16；
- true reconstruction 和 value-on-true-feature 先通过 smoke；
- 再启动 predictor/value/fastpath 训练或评估。

### Phase 3：compressed vision tokens D

目标：验证更大、更视觉化的 token set 是否优于 Qwen latent，同时控制 token 数和算力。

实现范围：

- vision token extraction；
- compressor（Perceiver 或 Transformer pooling）配置化；
- semantic embedding 输入；
- compressed token predictor/value/reconstruction 配置。

验收：

- 不改代码，仅切换 config 即可跑 K=8/16 等 compressed vision 实验；
- 与 B 组共享 token-set predictor/value head/evaluator。

### Phase 4：raw vision tokens C（可选）

目标：只在 D 组仍不足或需要验证上限时执行。

实现范围：

- raw vision tokens 的 memory/compute 优化；
- 必要时使用更小 batch 或 token subsampling。

验收：

- 作为上限/诊断实验，不作为第一批主线。

### Phase 5：正式环境 fast-path success rate

目标：在离线指标和 reconstruction 已显示候选 representation 有希望后，再跑昂贵环境评估。

实现范围：

- VAGEN baseline 对齐评估；
- predictor + value head greedy/beam fast path；
- depth 1/2/4/8 success rate 曲线；
- W&B 和输出 README 自动记录。

启动前必须再次向人类确认资源、输出目录、checkpoint 初始化、冻结/训练模块和 resume 策略。

---

## 需要新增或修改的代码范围（暂不实施）

以下是为了完成消融可能需要的代码改动清单。本分支当前只提交计划，**不直接修改代码**。

### 1. Representation extraction / cache

需要新增或扩展统一的 feature extraction pipeline：

- 支持当前单 latent：`<|latent_state|>` hidden → `StateProjector`。
- 支持 multi latent tokens：在 prompt/format 中插入多个 latent token，并提取对应 hidden states。
- 支持 vision tokens：从 Qwen/VAGEN vision encoder 输出中取 patch/token features。
- 支持 compressed vision tokens：新增轻量 tokenizer/compressor，把 vision tokens 压成固定 K 个 tokens。
- 为每种 representation 保存 cache metadata：representation 类型、token 数、feature dim、checkpoint、split、数据源、是否包含语义 embedding。

注意：split 语义必须从实际数据/config/metadata 核实，不能凭文件名推断。

### 2. Predictor variants

当前 `LatentWMPredictor` 接口主要是 `(B, emb_dim)` 单向量输入输出。消融需要扩展或新增 sequence/token-set predictor：

- 输入输出形状支持 `(B, K, D)` token set；
- action embedding 与 token set 融合方式配置化；
- 保留 baseline 单 token 路径，避免破坏旧实验；
- 支持多步 `rollout_states` 返回 token set 序列；
- checkpoint/config 中明确记录 representation 类型与 token 数。

### 3. Value head variants

需要为不同 representation 准备不同 value head：

- 单 latent：保留现有 `ValueHead(state_emb) -> action values`。
- multi latent / vision tokens：新增 pooling 或 cross-attention value head。
- vision token 路线：value head 额外接收 `static_task_embedding`。
- 所有 value head 输出仍为每个 navigation action 的 value，便于统一评估。

### 4. Reconstruction decoder / evaluator

现有 reconstruction 主要围绕单 latent condition。需要扩展为：

- true feature reconstruction；
- predicted next feature reconstruction；
- token-set condition reconstruction；
- 输出固定 strip 格式，例如：
  `current_gt | true_current_recon | next_gt | true_next_recon | pred_next_recon`；
- W&B 上传图像/table，避免只记录 scalar loss。

### 5. Offline metrics and fast-path evaluation

需要新增或扩展评估入口：

- value head top-1/top-k/ranking/calibration 指标；
- predictor one-step MSE/cosine/token-wise error；
- predictor multi-step rollout depth 曲线；
- fast path greedy / beam search task success rate；
- VAGEN baseline checkpoint 对齐评估。

### 6. Config 与实验记录

需要新增配置文件，建议放在：

- `configs/training/representation_ablation/`：训练/特征 cache 配置；
- `configs/eval/representation_ablation/`：重建与 fast-path eval 配置。

每个实验输出目录必须包含 README/metadata，记录：

- git commit；
- checkpoint 初始化来源；
- 数据 split 与来源；
- trainable/frozen 模块；
- representation 类型和 token 数；
- predictor/value head/reconstruction decoder 配置；
- resume/checkpoint 策略；
- W&B run id。

### 7. Tests / smoke checks

正式跑实验前至少补充：

- feature extraction shape test；
- token-set predictor forward/rollout shape test；
- value head with semantic embedding shape test；
- reconstruction evaluator smoke；
- fast-path planner/evaluator smoke。

---

## 实验控制项

- 使用同一 VAGEN baseline checkpoint：
  `/project/peilab/hligb/vagen-navigation/checkpoints/vagen_navigation_repro/navigation_base_8gpu_paper_repro_20260621/global_step_300`
- 同一数据源、同一 split、同一评估环境。
- true feature 与 predicted feature 分开评估。
- 多步 rollout depth 曲线必须记录。
- 训练类实验不得复用已有失败 run 输出目录。
- 启动任何预计超过 3 分钟的训练/评估前，按实验规则向人类确认模块冻结/训练、初始化、输出目录、resume 机制和资源估计。

---

## 建议执行顺序

1. 只实现/复核 baseline A 的完整评估链路：true reconstruction、pred reconstruction、value metrics、fast path success rate。
2. 做 multi latent B 的最小版本，例如 K=4 与 K=8，先看 true reconstruction 和 value on true feature。
3. 如果 B 明显改善，再做 predictor 多步和 fast path。
4. 并行或之后做 compressed vision D，优先于 raw vision C，降低 token 数和算力压力。
5. 最后视 D 的结果决定是否需要 raw ViT/vision tokens C。

---

## 当前待人类确认的问题

1. multi latent token 的 K 值优先选择：是否先用 4/8/16？
2. vision token 来源优先使用 Qwen vision encoder 还是 VAGEN baseline 内部 vision features？
3. `static_task_embedding` 的来源是否限定为 instruction/goal text，还是允许使用当前 Qwen hidden 中的语义 token？
4. 第一轮是否只做离线评估与重建，不直接启动环境 task success rate 大规模评估？
