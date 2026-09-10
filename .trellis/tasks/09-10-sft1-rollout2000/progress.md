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
