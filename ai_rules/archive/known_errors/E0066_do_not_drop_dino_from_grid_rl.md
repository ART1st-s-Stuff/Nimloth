# E0066：不要从 grid RL 的 WM 目标中删除 DINO loss

## 已发生的错误

在移除双层 projector、WM EMA 和 DINO decoder 时，错误沿用了“DINO只属于SFT2”的
旧结论，导致RL只保留predicted state到Qwen state target的MSE。

## 正确规则

- SFT2与RL训练grid WM时都计算state MSE和predicted-state DINO-grid MSE；
- 两阶段调用`training/common/world_model.py`中的同一目标函数；
- SFT2可从离线cache读取target，RL必须使用trajectory中真实next image对应的frozen
  DINO target，不能用Qwen state或缺失cache的替代值冒充；
- target生产方式可以因阶段性能需求不同，但loss公式和图像时间对齐必须一致。

CPU单元测试只能验证公式、路径对齐和梯度，不能替代真实DINO checkpoint与GPU显存门禁。
