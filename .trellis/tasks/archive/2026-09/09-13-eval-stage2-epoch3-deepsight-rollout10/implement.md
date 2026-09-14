# Implementation Plan

- [ ] 记录 a100-1 epoch3 清单、大小、SHA256、`COMMITTED` 内容、训练 commit 和 grid 配置。
- [ ] 刷新 a100-2 GPU、磁盘、进程、端口与目标目录状态；验证任务 worktree、Python、VAGEN、Vulkan 和 DINO teacher。
- [ ] 使用用户配置的现有 SSH 认证，通过 a100-2 直接从 a100-1 拉取，把完整 checkpoint 写入 a100-2 临时目录；核对全部 hash 后原子提交目标目录。
- [ ] 在隔离 worktree 中复用并补齐 epoch3 HF export 和 rollout controller；若需改代码，先做聚焦单元测试和 CPU preflight，再提交同步。
- [ ] 导出 epoch3 rollout model，验证 tokenizer/query token IDs、8x8 grid、slot projector 和模型 shard 完整。
- [ ] 运行环境 render gate 和单 episode 启动 gate；确认真实 AI2-THOR 帧、模型输出及 action parser 正常。
- [ ] 启动 Base 5 + Common Sense 5 的确定性 test rollout，并监控到精确终态；保存逐 episode/turn 记录和汇总。
- [ ] 对实际 rollout turns 执行 HF feature replay，保存 projected/target 8x8x1024 张量及逐 slot/turn cosine、MSE。
- [ ] 仅以全部 target 特征拟合共享 PCA，输出每个 episode 的 DeepSight 风格时间图、误差图、PCA 与色标元数据。
- [ ] 核验 episode 数、split、turn 对齐、tensor shape、finite、汇总重算一致性和图片可读性。
- [ ] 记录实验终态与证据边界；若图不可解释，只报告原因和已保存的 CFM 输入，不自动启动 CFM。

## Validation gates

- checkpoint 全文件 SHA256 source/destination 一致。
- export smoke 能加载模型、64 query tokens 和 slot projector。
- render gate 返回真实环境图像；单 episode gate 完整终止且 controller 清理进程。
- feature replay 对每个 turn 断言 episode ID、turn index、image hash、query count=64、feature dim=1024、finite。
- summary 从逐 turn/episode 文件独立重算一致；PNG 使用同一 PCA/scaling metadata。

## Rollback points

- 复制完成前不发布目标 checkpoint 目录。
- 导出失败时保留 checkpoint，停止 rollout。
- rollout gate 失败时不扩大到 10 episodes。
- feature extraction 失败时保留 rollout 原始记录，禁止把 success rate 误报为 DINO 恢复评估完成。


- [x] 使用既有认证完成传输，未修改 SSH 授权。

2026-09-13 执行更新：用户已设置服务器间认证；实际采用 a100-2 直接 rsync 拉取，无本机中转，无临时密钥。全部15文件SHA256已匹配。导出/特征提取使用99207d35，rollout复用已成功验证的96f5972e，两者分别记录。
