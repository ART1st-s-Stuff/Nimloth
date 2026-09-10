# 执行计划
- [x] 用户授权纠正spec漂移后重启，以验证LM loss收敛为目标及每10步保存/成功后清理。
- [x] 实现stage1无query语义与相关回归测试；明确数据/缓存/checkpoint阶段身份。
- [x] 更新本次launch/cleanup与必要调用方参数，旧K配置不再进入stage1。
- [x] 接入已确认的收敛状态、checkpoint恢复与时限暂停；停止条件不是固定epoch。
- [x] 独立审查实际代码路径与stage2不回归，完成相关测试。
- [ ] 提交本次专用branch修正，同步远程专用worktree；CPU数据/模型/真实CLI检查。
- [ ] 构建并全量核验新format-only缓存，刷新资源后启动一次8GPU训练。
- [ ] 核验optimizer step/loss，记录进程、commit、run/监控与交接。
- [ ] epoch成功及final核验后清理当前run step checkpoints；记录最终结果。
