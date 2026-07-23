# E0042：独立 lm_head 合并后禁止再次 resize

## 错误

在 PEFT `merge_and_unload()` 已生成独立 input embedding 与 `lm_head` 后，再调用
Transformers `resize_token_embeddings()` 同步词表 metadata。

## 后果

- Qwen `resize_token_embeddings()` 会进入 `tie_weights()`；若 nested text config 仍
  声明 tied，训练后的 `lm_head` 会被 input embedding 覆盖。
- safetensors 会把重新共享 storage 的 `lm_head.weight` 当作重复权重省略。
- Transformers 可能静默重建 tied head 而正常训练，vLLM 按 untied config 加载时才
  报缺少 `lm_head.weight`，导致损坏长期不被发现。

## 已发生证据

- 旧 SFT1 epoch5 `hf_merged` 缺少 `lm_head.weight`，adapter 则保存了彼此不同的
  embedding 与 head；SFT2 从该损坏 export 初始化后也会使用错误 policy head。
- `fix/sft1-merge-untied-head` 的真实 k=1 epoch5 重合并导出包含独立 head，精确等于
  adapter head，并由 Transformers 与 vLLM 完整加载。

## 正确做法

1. 只在加载 adapter 之前 resize base model。
2. merge 后只更新 vocab metadata，不再改变 embedding module。
3. 保存前校验 input/output vocab 尺寸、storage 独立性和两层
   `tie_word_embeddings=false`。
4. 导出后校验 safetensors index 含 `lm_head.weight`、其值匹配 adapter，并分别用
   Transformers 与生产配置的 vLLM 加载。
