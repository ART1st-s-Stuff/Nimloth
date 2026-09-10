# 执行检查表

- [x] 用户批准任务创建和a100-1执行位置。
- [x] 只读核验GPU、旧失败状态、数据hash、验证报告、模型config和FlashAttention导入。
- [x] 对照当前stage1及近期旧数据实验参数，记录K1/K16差异。
- [ ] 用户审查本次最终参数方案后task.py start。
- [ ] 分派实现agent准备运行脚本；核验本地修改、commit并通过Git同步到专用远程worktree。
- [ ] CPU preflight：干净commit一致、真实CLI、checkpoint shards/keys、全量图像可读、划分无交叠、K1预处理无截断与cache fingerprint。
- [ ] 检查agent审查脚本/恢复/超时/唯一目录；核验实际可训练模块。
- [ ] 刷新GPU资源与端口，单次启动8GPU SFT1；记录真实PID与完整contract。
- [ ] 监控至有限loss与optimizer steps；有异常则留存日志并停止自动重试。
- [ ] 记录运行位置与交接；完成后记录checkpoint与全量offline val（不误报rollout质量）。
