# 2026-09-10 规划核验

已按更新SERVER.md连接a100-1；旧运行FAILED.json ended_at=2026-09-10T10:12:18Z、exit1，首步前OOM，无epoch checkpoint。当前8GPU均0MiB，无stage1进程。旧输出完整保留。
train SHA256 af1f8d11a52279051d0deea96db81f3522de7bf556b089f921943d832e1224e6；heldout SHA256 1632d9aebbe499076fe1d9477fa83c2dbfbadea593189ece640603bdad0115dd。
当前未启动新训练，等待K1/1epoch方案审查。恢复时先刷新实际进程/GPU，不依据此快照启动。

## 启动前实时核验（用户已批准每10步保存及成功后清理）
- a100-1 n30191：8GPU空闲，torch2.6.0+cu124/transformers4.49.0/peft0.20.0/FA2 2.7.4.post1。
- 已从本地Git bundle建立远程专用worktree /mnt/nimloth/.worktree/sft1-rollout2000；当前基础源码5425780e，stage1真实CLI导入成功，旧sources/nimloth未修改。
- 已重新运行只读原始验证器，全部轨迹/图片hash/划分检查VALID：1709train、193heldout、98排除，闭合2000；manifest payload 602f0e28bf6af7aa513b02d0334c276b6bb1b8e3e9effb094ab8280bc736f678。
- 初始化825权重key与index一致，embedding/head维度151936x2048；两个shard SHA256分别3a31e9a3373684ae16b08faa5bb826e16f7c06876bcce83523c4f636510db243及f57da84efd929fcf4cf5a8c861ccf5eae97e772ecf5d8ed2431d5e50f5d53199。
- 使用本地专用分支向a100-1专用同名分支正常Git同步；不push公共origin，不合并dev。

独立审查通过：8个清理安全测试、shell与embedded Python语法、py_compile、diff检查通过；本地ruff/mypy未安装。验证sampler在world8会将193条unique记录补齐至200采样项，因此val loss为既有sampler聚合，不称193项无padding均值。

## 2026-09-10 spec修正与收敛授权
人类指出SFT1不应存在K，已授权修正代码再启动；随后明确要求先预处理再训练至收敛，撤销固定1epoch。已选择验证LM loss，相邻epoch连续两轮改善不足1%，至少2epoch。时限只暂停并从完整状态继续，不能作为收敛。此前K1/K16新run方案均不启动。

修正：stage1拒绝query CLI/env/YAML，全部角色文本移除历史latent标记；保留真实CoT/action/image。新cache v7+format_answer_ce_v2身份；旧cache拒绝。每10步保存，epoch完整核验后删已覆盖中间ckpt；保留epoch/best/final。

独立审查修复相邻epoch比较及全rank一致暂停；收敛状态、优化器、scheduler、数据游标、RNG进checkpoint。无限epoch采用首epoch5%步数预热后constantLR。

检查：combined233passed+8subtests；1个旧rollout-resolution parquet测试缺pyarrow/fastparquet无法执行（不涉及本次stage1）。源代码/新测试Ruff、compileall、diffcheck通过；共享latent helper既有B008告警不作无关修改。尚未启动新预处理或GPU训练。

最终独立审查通过：最新控制器/启动/清理/监控24passed、10subtests；计划pause必须全部报告exit75且无OOM，有新的完整恢复点才自动续训；清理检查epoch收敛游标与8rank RNG。真实8GPU及暂停恢复仍待运行。
