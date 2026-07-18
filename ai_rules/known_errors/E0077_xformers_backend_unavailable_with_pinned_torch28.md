# E0077 — pinned Torch2.8环境中的xFormers backend不可用

## 已发生的错误

ID59禁用FlashInfer sampler JIT后到达KV-cache初始化；vLLM因`VLLM_ATTENTION_BACKEND=XFORMERS`选择xFormers，随后`assert XFORMERS_AVAILABLE`失败。该环境安装的xFormers 0.0.32.post1明确跳过C++ extension，因为它要求Torch>=2.11而当前固定Torch2.8。

## 正确做法

- 不得把“Python可以import xformers”当成vLLM attention backend可用证据。
- 当前固定环境改用已安装且二进制可加载的FlashAttention 2.8.3 backend：`VLLM_ATTENTION_BACKEND=FLASH_ATTN`。
- `VLLM_USE_FLASHINFER_SAMPLER=0`只关闭FlashInfer sampler JIT，不代表必须使用xFormers attention。

## 证据

- ID59 `trainer.log`：`xformers.py: assert XFORMERS_AVAILABLE`。
- `experiments/training/rl/run_verl_online_world8_smoke.sh`。
