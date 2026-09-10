# 运行方案

本地隔离worktree：/workspace/remote2/nimloth/.worktree/sft1-rollout2000；分支codex/sft1-rollout2000；训练源码基点5425780e78cb11ec0008d36b836a8286f1e1b94d。
远程目标worktree：/mnt/nimloth/.worktree/sft1-rollout2000；通过Git同步本地已提交源码，不能沿用远程旧源码1e3b81a。解释器/mnt/nimloth/venv/bin/python3。

## 参数建议
1 node、8GPU、DDP world8、rank0..7；batch1、GA8、effective batch64；1epoch（预计27 optimizer steps，以实际trainer记录为准）。LoRA r64/alpha128，LR1e-6、embedding/head LR5e-6；K1 generate、回答CE；max_pixels100352，max_length20000（比参考12000保守保留完整轨迹），BF16、flash_attention_2、gradient checkpointing、seed42。每10步保存恢复点，epoch末完整离线验证与checkpoint；格式诊断32条不替代193条val。

远程FlashAttention2版本2.7.4.post1已能导入。当前源码模块支持K1与FA2；实际训练模块列表须在模型加载后记录，不凭LoRA名称声称视觉模块冻结。新的K1缓存使用新目录；fingerprint含路径/mtime/processor/K等，旧K16缓存不兼容。

## 预算及生命周期
单次运行，上限6小时/48GPU小时，实际时长待首次训练速度确认。以独立controller日志和唯一UTC运行目录输出；不得复用失败RUN_OUT。前置CPU数据/模型/入口检查完成后再激活GPU，启动前重新核验空闲资源。失败或超时停止并保留产物，不自动重提。无需Slurm/hold/Ray/vLLM。记录PID与进程组，超时仅作用于本次进程组。不自动执行HF导出（近期导出器存在歧义失败记录），训练checkpoint为本次产物。

## 风险
首次FA2/DDP训练尚未验证；旧验证报告需要当前输入/图像复核；模型shard key需要preflight复核。checkpoint恢复能力以当前实现为准，不能从无checkpoint的失败运行恢复，也不承诺逐位复现。

## 中间checkpoint清理
用户已明确批准：每10步保存，在本次1epoch成功结束且epoch/final完整可读后，仅删除本次run内由此训练创建的中间step checkpoint；先记录精确目录和清单，拒绝符号链接或越界路径。失败时保留全部checkpoint。

## 已核验模块范围
CPU meta模型匹配当前默认LoRA suffix共348个模块：252个语言模块和96个视觉MLP模块。adapter及完整embedding/head可训练，其他base参数由PEFT冻结；不声称视觉模块全部冻结。实际训练参数计数仍记录到训练日志。W&B关闭，与最近旧数据复现实验一致，逐步CSV与本地controller日志保留。
