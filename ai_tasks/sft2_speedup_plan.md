--------
本文为 AI 起草、供人类审阅。
--------

# SFT2 速度优化计划（保持训练语义不变）

目标：降低 SFT2 wall-clock 训练时间，同时**不改变训练目标、样本集合、loss 定义、梯度语义、checkpoint 语义和评估 split 语义**。

当前背景：SFT2 比 SFT1 明显更慢，不主要是模型规模问题，而是 per-step transition 展开、重复 prefix 计算、current/next Qwen forward 串行、SFT2 缺少 SFT1 等级 preprocess cache，以及默认 vision full fine-tune 带来的额外 backward 成本。

## 0. 不允许改变的训练语义

任何优化必须保持以下语义不变：

1. **数据语义不变**
   - train 仍使用 `train_all.jsonl`，包含失败 rollout。
   - val 仍使用 `val_all.jsonl`，不得与 train 混用。
   - transition 展开规则不变：每个 step 的 current prefix / next prefix 与现有 `wm/dataset.py` 对齐。

2. **loss 语义不变**
   - `L = λ_wm * L_wm + λ_value * L_value + λ_ce * L_lm_ce` 不变。
   - CE 只监督当前 transition 的最后 assistant span，不恢复早期 assistant turn 重复监督。
   - WM target 仍是 next prefix 的 Qwen `<|latent_state|>` hidden，经 `state_proj` 后 stop-grad。
   - Value target 仍使用轨迹级 reward 的折扣回报，排序损失定义不变。

3. **梯度语义不变**
   - current prefix Qwen forward 仍参与梯度。
   - next prefix target forward 仍 no-grad，且默认使用 EMA vision 权重。
   - `state_proj`、`LatentWMPredictor`、`ValueHead` 的梯度路径保持不变。
   - DDP / grad accumulation 优化只能改变同步时机，不能改变 effective batch 或 loss scaling。

4. **训练/冻结模块不变**
   - 默认仍是 LLM freeze、vision full tune + EMA。
   - 若做 `vision_tune=freeze/lora` profiling，只作为诊断对照，不得冒充默认实验结果。

5. **checkpoint / resume / metric 不变**
   - checkpoint 内容仍包含 Qwen、state_proj、wm_predictor、value_head、vision_ema、optimizer、training state。
   - best checkpoint 选择规则不变。
   - 监控仍记录 train step log、WM/value/CE 曲线、val success rate。

## 1. 当前慢因判断

### 1.1 Transition 展开导致样本量放大

SFT2 按 per-step transition 训练：

- rollout record 约 3240 条；
- 展开后 transition 约 54702 条；
- 8 卡、`batch_size=1`、`grad_accum=8` 时约 855 optimizer steps / epoch。

这不是 bug，但意味着 SFT2 的 epoch 定义天然比 SFT1 更重。

### 1.2 每个 transition 串行做两次 Qwen forward

每个 transition 当前至少包含：

1. current prefix：带梯度 Qwen forward，提取当前 latent 并计算 CE；
2. next prefix：no-grad Qwen forward，提取 WM target latent。

两个 forward 在同一 rank 内串行执行。优化时必须保持 current/next 的梯度语义不变。

### 1.3 Prefix 重复计算严重

同一 trajectory 的第 t 个 transition 使用从第 0 步到第 t 步的完整 prefix。长轨迹会重复编码大量早期图片和文本。当前 cache 只减轻 CPU/template/image decode；GPU 上的 vision/text forward 仍重复。

### 1.4 SFT2 preprocessing 还不如 SFT1 完整

SFT1 有 preprocess cache、DataLoader worker、prefetch 和 cached tensor collate。SFT2 当前是在线构造 Qwen processor batch，`DataLoader(num_workers=0)`，虽然已有 LRU cache，但还不是磁盘级/worker 级 preprocess pipeline。

### 1.5 默认 vision full tune 比 SFT1 LoRA 重

SFT1 主要是 LoRA；SFT2 默认 full fine-tune vision encoder + EMA。vision backward 本身会显著增加 step cost。这个是规格要求，不应为了提速改变默认语义。

## 2. 优化路线（按优先级）

## P0：加测量，先定位瓶颈比例

目的：确保每个优化都可验证，不把语义变化误认为速度优化。

实现建议：

- 在 SFT2 trainer 中增加可选 profiling 日志（默认关闭或低频）：
  - dataloader / collate / processor 时间；
  - current Qwen forward 时间；
  - next Qwen target forward 时间；
  - WM/value loss 时间；
  - backward 时间；
  - optimizer + EMA update 时间；
  - DDP wait / all-reduce 可用粗略 step 差异估计。
- 只记录 timing，不改训练逻辑。

语义风险：无，只读计时。

验收：能输出 step-level 或 rolling-average timing，且 loss/step 数不变。

## P1：SFT2 preprocess cache 与 DataLoader workers

目标：复用 SFT1 成熟思路，把 CPU-heavy 的 chat template、tokenization、image preprocessing 尽量移出训练 critical path。

实现方案：

1. 为每个 transition prefix 构建 cache item：
   - current prefix processor 输出；
   - next prefix processor 输出（若存在）；
   - CE labels（只对最后 assistant span）；
   - action index / value target / success metadata。
2. cache fingerprint 必须包含：
   - jsonl path + mtime/size 或内容 hash；
   - tokenizer vocab size / special tokens；
   - `max_length`、`min_pixels`、`max_pixels`；
   - CE mask 版本（last assistant span）；
   - transition expansion version。
3. DataLoader 使用 workers + prefetch 读取 cached tensors，collate 只做 pad。
4. current / next enc 的字段与在线 `build_qwen_batch` 输出保持一致。

保持语义的要求：

- cache 结果必须与在线 processor 结果 byte/tensor 等价（允许 padding 在 batch collate 阶段变化，但 unpadded token/image tensors 必须一致）。
- CE labels 必须与当前 last-span 逻辑一致。
- next prefix 不存在时的 terminal dummy 逻辑不变。
- cache miss / rebuild 不能静默使用旧 cache。

验证：

- 新增小样本测试：在线构造 vs cache 构造的 `input_ids`、`labels`、image grid / pixel fields 一致。
- 用 `max_train_records` 小跑，确认 loss 数值在固定 seed 下与在线路径接近/一致。

预期收益：减少 CPU 和 host-side preprocessing 等待；若当前 GPU 已满载，收益有限但可稳定 step time。

## P2：提高 per-rank batch size，降低 grad_accum

目标：减少 batch size=1 带来的 kernel granularity 差和 DDP 同步频率，同时保持 global batch 不变。

当前默认：

```text
world_size=8, batch_size=1, grad_accum=8 => effective batch = 64 transitions
```

可尝试：

```text
batch_size=2, grad_accum=4 => effective batch = 64 transitions
```

保持语义的要求：

- effective batch size 不变；
- loss 仍按 micro loss / grad_accum 缩放；
- tail batch scaling 需正确，不因最后 accumulation 不满而改变梯度尺度；
- DDP no_sync 仍只在非同步 micro-step 使用。

风险：显存 OOM。当前约 47GB/80GB，但长 prefix batch=2 可能因 padding 到最长样本而大幅增加显存。

验证：

- 先用 `max_train_records` 或短跑测试峰值显存。
- 比较 batch=1/grad_accum=8 与 batch=2/grad_accum=4 的前若干 optimizer step loss scale 是否合理。

## P3：减少 next target forward 的重复计算（只缓存 no-grad target）

目标：next prefix target 是 stop-grad，可缓存其 latent，避免每个 epoch 重复 no-grad Qwen forward。

可行方案：

- 对每个 transition 的 next prefix，在当前模型/EMA 权重下计算 target latent。
- 但注意：target latent 依赖 Qwen/vision EMA 权重，训练过程中会变化。

因此不能简单做跨 epoch 固定 cache，否则会改变训练语义。

保持语义的安全方案：

1. **同一 optimizer step 内去重**：如果 batch 内多个 transition 共享同一 next prefix，复用 forward 结果。语义完全不变。
2. **同一 epoch 内缓存需谨慎**：由于 vision EMA 每 step 更新，跨 step 复用会使用旧 target，改变语义。默认不做，除非人类批准 target staleness。
3. **如果 vision_tune=freeze 且无 EMA 变化**，next target 可以跨 step/epoch cache；但这只适用于非默认配置。

优先实现同-step 去重，不实现跨-step stale cache。

验证：

- 对同一 batch 内重复 next prefix，去重前后 target tensor 一致。
- 默认无重复时结果不变。

预期收益：默认 batch=1 时收益很小；配合 batch_size>1 或 trajectory grouping 后收益更大。

## P4：Trajectory-level / packed forward（最大潜在收益，最高复杂度）

目标：避免同一 trajectory 的 prefix 被重复从头 forward。理想情况下，一条 trajectory 只做少数长序列 forward，然后抽取多个 `<|latent_state|>` hidden。

关键难点：

- 当前每个 transition 的 current prefix 都是完整对话前缀；causal LM 对整条 conversation 一次 forward 可以同时得到多个 assistant turn 的 `<|latent_state|>` hidden。
- CE 只监督每个 assistant turn 自己的 span；整条 trajectory forward 可以自然覆盖所有 step 的 CE。
- WM loss 需要 `s_t` 和 `target s_{t+1}`。整条 trajectory forward 可同时提供相邻 latent。
- 但要确认 full conversation forward 的 tokenization、image ordering、mask 与逐 prefix forward 得到的 hidden 是否完全等价。

保持语义的要求：

- 对每个 step 抽取的 current latent 必须等价于逐 prefix current forward 的 latent。
- next latent 必须等价于逐 prefix next forward 的 latent，且 target stop-grad / EMA 权重语义保持。
- CE loss 的 token mask 总和应等价于每个 transition last-span CE 的集合，不能多监督/少监督。
- Value loss 和 WM loss 仍按 transition 计算，action/value target 对齐不变。

潜在问题：

- 如果一次 full trajectory forward 使用包含未来 user observations/actions 的 tokens，虽然 causal mask 防止当前 token看未来 token，但视觉 tokens 和 position/cache 行为必须确认不会泄漏。理论上 causal mask 应保证不泄漏，但需要实证测试。
- Qwen-VL 多图位置编码和 image placeholder 顺序必须与 prefix forward 对齐。

验证必须非常严格：

1. 用一条短 trajectory，分别跑：
   - 逐 prefix current/next forward；
   - full trajectory forward；
   比较每个 `<|latent_state|>` hidden 的数值误差。
2. 比较 CE labels 覆盖 token set 是否等价。
3. 比较 WM/value/CE 总 loss 是否一致。
4. 只在验证通过后替换默认路径；否则保留为实验分支。

预期收益：最大。可把一条 T-step 轨迹从 O(T²) prefix 重复计算降到接近 O(T) full sequence 计算。

## P5：Vision feature / image embedding cache（需谨慎）

目标：缓存图片经过 vision encoder 的特征，避免同一图片在多个 prefix 中重复 vision forward。

语义风险：默认 SFT2 vision encoder 是 trainable + EMA，vision features 会随训练改变。跨 step/epoch 缓存会改变梯度和 target 语义，不允许作为默认优化。

安全范围：

- `vision_tune=freeze` 时可缓存 vision features，语义不变。
- 默认 `vision_tune=full` 时不能缓存带梯度 current forward 的 vision features。
- 对 no-grad EMA target forward，也不能跨 step cache，因为 EMA 权重每 step 更新。

结论：不作为默认优化；只用于诊断或 freeze-vision 对照实验。

## P6：诊断性配置对照（不得冒充默认实验）

为了量化慢因，可以短跑以下配置：

1. `vision_tune=freeze`：估计 vision full backward 成本。
2. `vision_tune=lora`：估计低秩 vision tuning 成本。
3. `--no-gradient-checkpointing`：估计 recompute 成本与显存 tradeoff。
4. `batch_size=2, grad_accum=4`：估计 batch 化收益。
5. `attn_implementation=sdpa` vs `flash_attention_2`：确认实际收益。

这些只作为 profiling，不改变默认实验结论。

## 3. 推荐执行顺序

1. **P0 timing instrumentation**：先确认耗时比例。
2. **P2 batch_size=2/grad_accum=4 短跑**：最小代码改动，可能快速见效。
3. **P1 SFT2 preprocess cache**：复用 SFT1 经验，低语义风险。
4. **P6 诊断性配置对照**：量化 vision full / checkpointing 成本。
5. **P4 trajectory-level forward 原型**：收益最大，但必须以等价性测试为前提。
6. **P3 同-step next target 去重**：配合 batch/trajectory grouping 再做。
7. **P5 vision feature cache**：仅 freeze-vision 配置考虑，默认不做。

## 4. 验收标准

每一项优化进入默认训练路径前必须满足：

- 单元测试覆盖语义等价；
- 小样本固定 seed smoke test 可跑通；
- step log 正常记录 loss 和 metrics；
- checkpoint/resume 不破坏；
- 对同一配置，effective batch、数据样本数、loss 定义不变；
- 若有任何 target staleness、sampling 改变、mask 改变或模块冻结变化，必须标记为新实验配置，并先请求人类确认。
