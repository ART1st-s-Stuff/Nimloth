# E0026：从 GPU checkpoint 恢复 CUDA RNG 前必须把 state 移回 CPU

## 已发生的错误

CFM resume smoke job `474762` 使用 `torch.load(..., map_location=cuda)` 加载训练状态后，直接把其中的 `cuda_rng_state_all` 传给 `torch.cuda.set_rng_state_all`，在恢复训练前报错：`TypeError: RNG state must be a torch.ByteTensor`。

## 原因

`map_location=cuda` 将序列化的 RNG ByteTensor 也移动到了 CUDA，但 PyTorch 的 CUDA RNG restore API 接受的是 CPU ByteTensor。

## 正确做法

模型和 optimizer 可以按目标 device 加载；调用 `torch.cuda.set_rng_state_all` 前，必须对每个 RNG state 执行 `.cpu()`。恢复门禁必须实际跑一次 GPU checkpoint resume，不能只验证 checkpoint 文件可读取。

## 证据

- 失败 job：`474762`
- 修复位置：`src/nimloth/training/reconstruction/cfm_sft2.py::_load_checkpoint`
- 失败输出：`.../cfm/0_smoke_cfm128_sft2e2_k8_n64_steps2_b4_lr1e4/logs/resume-474762.err`
