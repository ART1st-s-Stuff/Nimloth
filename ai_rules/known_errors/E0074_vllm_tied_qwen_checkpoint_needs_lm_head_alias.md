# E0074 — vLLM加载tied Qwen checkpoint必须补LM-head alias

## 已发生的错误

ID55以TP4通过vLLM partition并加载actor/critic。SFT merged checkpoint只存`model.embed_tokens.weight`，Transformers actor通过`tie_word_embeddings=true`正确共享输出头；但vLLM Qwen实现保留独立`language_model.lm_head` load target，FSDP→vLLM weight sync将其判为未初始化并fail closed。尚未W&B/rollout/update。

## 正确做法

- Worker external-lib runtime patch在且仅在config要求tied embeddings且输入没有lm_head时，把同一embedding tensor以`lm_head.weight` alias交给vLLM官方loader。
- 必须恰有一个embedding weight；0个或多个都拒绝，禁止随机head。
- TP generation后仍须比较FSDP actor与vLLM sampled-token behavior/replay log-prob sanity。

## 证据

- `src/nimloth/training/rl/vllm_tied_head.py`
- ID55 vLLM load gate。
