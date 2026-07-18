# E0078 — Qwen2.5-VL vision head-dim80要求独立backend选择

## 已发生的错误

ID60全局设置`VLLM_ATTENTION_BACKEND=FLASH_ATTN`后，text backend可加载，但vLLM model profile的dummy video经过Qwen2.5-VL vision attention时报：

`RuntimeError: This flash attention build does not support headdim not being a multiple of 32.`

vision配置的head-dim为80；全局override绕过了模型原有的text/vision分别选择逻辑。

## 正确做法

- 不设置全局`VLLM_ATTENTION_BACKEND`。
- 让vLLM为text head-dim128选择其FlashAttention backend，并让Qwen2.5-VL vision代码在head-dim80时选择支持该维度的upstream FlashAttention。
- 不能因为一种backend可import，就把它强制用于text和vision两条不同attention路径。
- 仍保留`VLLM_USE_FLASHINFER_SAMPLER=0`；sampling backend与attention backend是两个独立选择。

## 证据

- ID60 `trainer.log`的`Qwen2_5_VisionAttention.forward -> vllm_flash_attn`栈。
- `experiments/training/rl/run_verl_online_world8_smoke.sh`。
