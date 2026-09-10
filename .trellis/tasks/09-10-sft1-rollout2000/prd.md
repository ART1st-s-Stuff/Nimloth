# 从2000条rollout启动SFT stage 1

## 目标
在用户已指定的a100-1上，用现有rollout有效训练数据启动一次SFT1格式监督训练，参考最近旧数据stage1实验。

## 背景与证据
- 用户已批准创建任务，且明确在a100-1训练；主目录SERVER.md已更新。
- 实时核验主机n30191，8张A100-SXM4-40GB空闲；/mnt有778GiB可用空间。
- 数据2000条来源：1709 train-all、193 internal heldout、98格式排除；远程已有VALID验证报告，且本轮train/heldout SHA256与报告对应启动记录一致。
- 数据根：/mnt/nimloth/outputs/datasets/sft1-vagen-step60/20260910T093222Z_batch1_original_validation_k16。
- 旧a100运行20260910T094500Z_step60_batch1_sft1_k16_all_lora已FAILED，首步前OOM，无epoch checkpoint，GPU与进程已退出。其缓存为K16，不复用于K1。
- 当前源码近期step79实验脚本使用K1 generate、1epoch、LR1e-6、embedding LR5e-6、LoRA r64/alpha128、batch1/GA8。较老train_8gpu.slurm为20epoch/K16兼容模式/LR2e-4；二者不混同。

## 要求
- 拟以近期K1格式训练为基准，源码将输入latent块规范化为K1；原始JSONL与图像不修改。
- 仅SFT1回答CE与离线验证，不启动stage2、DINO/WM训练、数据收集或环境rollout评估。
- 初始化/mnt/nimloth/checkpoint/hf_actor（模型配置记录global_step_60来源），不使用失败运行恢复状态。
- 一次独立运行；失败不自动重提，保留旧产物。

## 验收
1. 训练真实产生有限loss与optimizer step，或明确记录失败阶段；提交不等于完成。
2. 记录完整命令、源码commit、输入SHA256、训练参数、进程身份、输出与可恢复位置。
3. 输出epoch checkpoint、全量193条离线val结果；未完成须如实交接。
4. 保存监控与接手信息；离线val不称环境成功率。

## 已批准的执行方案
用户在最终参数方案后明确要求每10步保存，1epoch结束后删除中间ckpt；主机会话与既定参数保持不变。仅删除本次新运行的step checkpoint，必须训练成功且epoch/final完整可读；保留epoch/best/final、数据、旧失败产物及日志。此授权不涉及其他运行。
