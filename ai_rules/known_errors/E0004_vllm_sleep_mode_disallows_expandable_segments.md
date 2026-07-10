# E0004 — vLLM sleep memory pool 禁止 expandable segments

## 错误

VAGEN rollout worker 继承了 SFT common env 的：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

vLLM 0.11 sleep-mode `CuMemAllocator` 因此在完整加载 source checkpoint 后 assert 失败。

## 正确做法

需要 vLLM sleep/memory-pool 的 rollout/eval wrapper 必须在启动 Ray 和 worker 前局部执行：

```bash
unset PYTORCH_CUDA_ALLOC_CONF
```

不要全局删除 SFT training 的 allocator 配置；训练和 vLLM rollout 使用各自明确的环境。
