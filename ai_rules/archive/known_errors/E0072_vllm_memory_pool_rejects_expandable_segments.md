# E0072：VERL vLLM memory pool禁止PyTorch expandable segments

## 已确认现象

VAGEN validation的4卡FSDP权重加载完成后，vLLM `CuMemAllocator`在初始化memory pool时
明确断言`PYTORCH_CUDA_ALLOC_CONF`不能包含`expandable_segments:True`。这是allocator
合同冲突，不是显存不足。

## 正确做法

- 使用VERL vLLM sharding manager前unset `PYTORCH_CUDA_ALLOC_CONF`中的expandable segments。
- 必须在Ray head启动前unset，使raylet及其worker继承正确环境。
- 不得通过关闭memory pool或改模型权重掩盖该启动配置错误。
