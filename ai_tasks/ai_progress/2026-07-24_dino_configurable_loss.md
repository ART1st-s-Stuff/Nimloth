# 2026-07-24 DINO 可配置附加 loss 修复

## 目标

保持唯一一套 SFT2 核心训练算法。DINO 只作为由配置启用的附加 loss：提供 cached target、decoder loss 和权重，不复制 current/target/CE/WM/value/SIGReg 流程。

## 纠错

- 已用 revert commit `5e086e6` 完整撤回错误的 variant/trainer 复制方案；tree 与错误修改前 `87fc69a` 一致并已推送 `dev`。
- 人类明确指出：模块化目标是把 DINO 作为 configurable loss 加入 SFT2 核心，而不是复制核心训练路径。

## 计划

1. RED：把 DINO 测试改为要求 `SFT2Algorithm(auxiliary_losses=...)`，并禁止独立 `DINOGridSFT2Algorithm`。
2. GREEN：给公共 `SFT2Batch` 增加通用 auxiliary targets；给核心 algorithm 增加通用附加 loss 契约。
3. GREEN：DINO 模块只保留 target assembler 与 `DINOGridLoss`；core SIGReg 通过 WorldModel 公共表示接口兼容 grid state。
4. 清理 trainer 的 algorithm 分叉，验证 loss 权重、梯度、checkpoint/resume 和 latent baseline 不变。
5. 更新 README、known error、进度；运行定向和扩展回归后提交推送 `dev`。

## 验证

- 待执行。
