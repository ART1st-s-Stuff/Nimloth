# E0045 — 原目录覆盖的 latest 不是 atomic checkpoint

## 已发生的错误

进度记录曾把SFT2每20分钟写入的`latest/`称为“atomic latest”。实际`save_checkpoint()`直接在既有`latest/`中依次覆盖模型、aux和`training_state.pt`，没有临时目录写完后的原子rename。

## 风险

若任务在覆盖过程中终止，目录可能同时包含不同保存时刻的文件；仅看到`training_state.pt`不能证明整套checkpoint来自同一步。

## 正确做法

- 未实现temp-dir + fsync/完整性验证 + rename前，不得称为atomic。
- 暂停/恢复时优先使用已完整结束的`epoch_*`或经完整加载验证的checkpoint。
- 如需滚动latest原子性，应先实现并测试目录级事务替换，同时明确旧latest的保留/清理策略。
