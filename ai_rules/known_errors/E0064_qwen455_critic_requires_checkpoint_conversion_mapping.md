# E0064 — Transformers4.55 Qwen critic必须复用官方checkpoint key转换

## 已发生的错误

ID41日志揭示4.55-native token critic虽然结构正确，但`from_pretrained`把`model.language_model.*`和`model.visual.*`全部报告为newly initialized。源checkpoint仍保存旧flat keys（如`model.layers.*`）；自定义critic没有`Qwen2_5_VLForConditionalGeneration._checkpoint_conversion_mapping`，因此ID34–ID40的critic backbone实际是确定性随机初始化。此前只检查结构/finite forward/fingerprint变化，没有检查load coverage，相关实验仍只可作为mechanics。

## 正确做法

- 自定义critic复用Transformers4.55 causal Qwen类的官方`_checkpoint_conversion_mapping`。
- `from_pretrained(..., output_loading_info=True)`必须fail closed检查coverage：只允许新scalar `score.weight/bias`缺失和源`lm_head.weight`unexpected。
- 不能用“每次随机fingerprint一致”推断checkpoint已加载；固定seed也会产生一致随机权重。

## 证据

- `src/nimloth/training/rl/verl_critic_455.py`
- `external/VAGEN/verl/verl/workers/fsdp_workers.py`
- ID41 worker log的全backbone newly-initialized列表。
