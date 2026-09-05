# E0021：CPU partition 作业 walltime 不得超过 12 小时

错误：SFT2 compact-cache batch script 请求 `24:00:00`，但 superpod `cpu` partition 的 `MaxTime=12:00:00`，作业会一直处于 `PENDING (PartitionTimeLimit)`，依赖训练也不会启动。

正确做法：CPU cache 作业最多请求 `12:00:00`；提交后立即检查 `squeue` reason。长 cache 必须依赖 builder 的 shard/build-state 恢复在后续作业继续，不能通过超出 partition 上限延长单个 job。
