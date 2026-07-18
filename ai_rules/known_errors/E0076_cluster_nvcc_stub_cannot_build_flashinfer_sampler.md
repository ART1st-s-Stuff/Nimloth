# E0076 — 集群`/usr/bin/nvcc`是说明stub，不能编译FlashInfer sampler

## 已发生的错误

ID58把venv ninja加入PATH后，FlashInfer sampling JIT实际调用`/usr/bin/nvcc`。该文件不是CUDA compiler，而是依赖colorama的教学说明Python脚本；所有CUDA object编译失败。vLLM模型/tied head已加载，但尚未W&B/rollout/update。

## 正确做法

- 当前4.55 mechanics环境设置`VLLM_USE_FLASHINFER_SAMPLER=0`，使用vLLM原生Torch top-k/top-p sampler，避免运行时CUDA JIT。
- 不得仅安装colorama让教学stub继续运行；它不会生成CUDA object。
- 若未来要求FlashInfer性能，应单独建立与PyTorch cu128匹配的真实CUDA toolkit/cache gate，不在训练任务中临时编译。

## 证据

- `experiments/training/rl/run_verl_online_world8_smoke.sh`
- ID58 ninja output与`/usr/bin/nvcc`内容。
