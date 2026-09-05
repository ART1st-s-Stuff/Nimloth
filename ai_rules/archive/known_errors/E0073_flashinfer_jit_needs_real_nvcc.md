# E0073：FlashInfer JIT不能使用superpod默认nvcc包装器

## 已确认现象

VAGEN validation完成4卡FSDP与vLLM权重加载后，FlashInfer首次采样会JIT编译sampling
kernel。superpod默认`PATH`把`nvcc`解析为`/usr/bin/nvcc`；该文件是依赖`colorama`的
Python tutorial包装器，不是CUDA编译器，因此4个Ray worker都以
`ModuleNotFoundError: No module named 'colorama'`和`Ninja build failed`退出。

## 正确做法

- 纯greedy评估显式设置`VLLM_USE_FLASHINFER_SAMPLER=0`，使用vLLM的PyTorch sampler，
  避免不必要的运行时CUDA JIT。
- 必须在Ray head启动前设置，使raylet及全部worker继承一致的sampler配置。
- 若任务必须使用FlashInfer JIT，应显式加载并验证真实CUDA工具链；不得把
  `/usr/bin/nvcc`存在误当成CUDA compiler可用。
