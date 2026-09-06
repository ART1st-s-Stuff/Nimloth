# 指定step60数据集的测试验证

## 人类授权
2026-09-05：人类批准测试和验证，指定「基于 VAGEN step 60 rollout 生成 SFT1/SFT2 数据集」任务的数据。不替换为其他旧数据集或mock数据，不由本授权推导启动新的数据收集或取消其他任务。

## 2026-09-05T18:43:24Z 远程只读核验
连接使用当前.local/SERVER.md记录的superpod-csejzhang。来源作业548155的squeue与sacct均为PENDING (Priority)，Elapsed=00:00:00，未分配节点。

已核验目录：
`/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60`

递归查询结果均为0：conversion_manifest.json、sft1_train_all.jsonl、sft1_heldout_all.jsonl、sft2_train_all.jsonl、sft2_heldout_all.jsonl、COMPLETE。不能把已有actor merge、partition parquet或兼容性smoke当成转换完成的数据集。

相关「配置远程环境生成 SFT 数据集」任务的近期记录仍是新主机环境准备/离线包转移，未给出完成的数据集证据。本次没有在新主机提交任务，也未改动已有收集任务。

## 前置条件与后续顺序
1. 指定数据任务产出有效conversion manifest和完整shard；核验hash、image引用、terminal CoT及转换版本。
2. 核对batch1的train/held-out分区和样本统计，不以source test代替held-out（来源任务已记录其与train重叠）。确认train_all/train_success最终选择。
3. 使用真实数据和兼容初始化模型确定SFT1、SFT2 query、SFT3迁移以及direct/WM的有界验证配置；SFT2还需要同图像的真实DINO target cache。不要把旧数据名SFT2混淆成新的query stage。
4. 在本任务分支固定已提交源码，记录运行资源、次数、时限、checkpoint、输出/恢复、监控；仅在明确的启动范围内提交远程验证。

当前状态：数据前置条件未满足；未提交测试job、未开始真实训练或rollout，未产生新模型指标。现有360项CPU回归结果仍只是前一阶段代码验证，不计作本次指定数据集验证。

## 548155终态核验（2026-09-06T05:40:59Z）
只读刷新squeue/sacct/scontrol及stdout、rollout和env-server日志：job548155（vagen-k8-full-8g，preempt，dgx-33，8GPU）已FAILED/NonZeroExitCode，ExitCode=1:0，elapsed=00:07:43。日志启动时间2026-09-06T07:32:09+08:00；Slurm start/end为07:32:08/07:39:51，折算UTC为2026-09-05T23:32:08Z/23:39:51Z。源码commit69a0cef6be9b81837f296eadc4775df94c9bc19d。

产物根：/project/peilab/atst/nimloth/outputs/experiments/training/sft1-vagen-step60/20260905T043000Z_existing_rollout_step60_smoke_8gpu_69a0cef6。AI2-THOR smoke通过，四环境端口健康，vLLM模型构建完成；首次validation reset POST /environments返回HTTP500。env_server_548155_dgx-33_0.log明确报错：TypeError: NavigationEnvConfig.__init__() got an unexpected keyword argument 'example_count'。客户端日志路径来自重建VAGEN runtime，服务端trace路径来自主仓external/VAGEN；需进一步核验双方配置/运行版本，尚未实施修复。

该run递归未发现jsonl、summary/manifest或COMPLETE；没有可用于本任务测试的完整数据集或可证明的轨迹恢复状态。未重启、重提或修改远程产物。下一步需在来源任务检查环境配置与服务端版本，修复后按启动合同决定后续运行。本次仅在当前Trellis记录终态；未改写其他任务或远程受保护实验产物。
