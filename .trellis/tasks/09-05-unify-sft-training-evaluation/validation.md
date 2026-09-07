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

## 2026-09-06 normal 三卡测试 allocation

人类明确批准先占用 normal 分区的 3 GPU 节点进行测试。本次先建立一份独立 allocation，不复用正在运行的 step60 数据生成作业 550857，也不取消或修改它。

- 问题与验收：验证统一 SFT 分支在实际三卡 CUDA/NCCL 环境可导入并执行跨 rank SIGReg/DDP 基础路径；随后仅在输入和 checkpoint 合同齐备时执行有界训练 smoke。allocation 本身不证明训练、rollout 或模型质量。
- 源码：Nimloth `ed7a1f1630f200730f1345602a3e95f62044c57a`；VAGEN `b4066c56c727c19a88b593e7207ca5f6c0744a9b`；远程独立 worktree `/project/peilab/atst/nimloth/.worktree/unify-sft-training-evaluation`，启动前要求干净且精确匹配。
- 解释器：`/project/peilab/atst/nimloth/.venv/bin/python3`；入口先使用现有 NCCL SIGReg 测试及三 rank CUDA/NCCL smoke，不加载替代模型。
- 数据与模型：初始资源及通信测试不读取训练数据或 checkpoint。step60 转换数据在 manifest/COMPLETE 和 split 证据齐备前不得进入训练测试。
- 资源：normal / `normal_debug_qos`，1 节点、3 GPU、84 CPU、240G 内存、2 小时，排除 dgx-51；当前资源快照显示 dgx-35 恰有 3 GPU 和 84 CPU 未分配。单次 allocation，不自动重复申请。
- 输出：`/project/peilab/atst/nimloth/outputs/experiments/training/sft-unified-gpu-validation/20260906T070000Z_3gpu_hold_ed7a1f16`，新目录；记录提交命令、job ID、源码、测试日志和结束状态。不启用 W&B。
- 监控：提交后核对 allocation 节点、GRES/CUDA 可见性、三 rank 映射及日志。测试完成后取消精确 allocation 并确认终态；若保留以继续同一测试，记录剩余时限和接手命令。

## 2026-09-06T08:58:29Z 三卡验证结果

- allocation `550875` 于08:56:15Z在`dgx-06`启动，实际可见3张NVIDIA H800，端口30875可用。
- 固定源码与启动合同未变：Nimloth `ed7a1f1630f200730f1345602a3e95f62044c57a`、VAGEN `b4066c56c727c19a88b593e7207ca5f6c0744a9b`，解释器 `/project/peilab/atst/nimloth/.venv/bin/python3`。
- `550875.0`用3个local rank完成NCCL初始化、跨rank SIGReg/DDP汇聚、梯度与同步随机投影检查。rank 0/1/2均输出`RANK_OK`，loss均为`1.33355629`；step为`COMPLETED`、elapsed `00:01:21`、ExitCode `0:0`。
- 控制器记录`test_rc=0`后取消精确allocation；最终allocation为`CANCELLED by 3738`、elapsed `00:01:29`，这是按计划释放资源。batch的`0:9`来自allocation取消，测试step本身成功。
- 08:58:29Z复核：squeue中已无550875，控制器PID 3193884已退出，GPU已释放。输出证据位于 `/project/peilab/atst/nimloth/outputs/experiments/training/sft-unified-gpu-validation/20260906T070000Z_3gpu_hold_ed7a1f16`。
- 有效性边界：通过真实三卡CUDA/NCCL基础验证，但未读取step60数据或checkpoint，未执行真实训练、vLLM或环境rollout，也不构成模型质量证据。指定数据集验证仍受数据产物未完成阻塞。
