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

## 已完成

- RED 提交 `9e11523` 要求 DINO 通过 `SFT2Algorithm(auxiliary_losses=...)` 接入，并禁止独立 DINO algorithm。
- `SFT2Batch` 新增通用 `auxiliary_targets`；核心 algorithm 新增通用附加 loss 输出契约。
- `DINOGridLoss` 只执行 decoded prediction 与 cached target 的 MSE；原 `DINOGridSFT2Algorithm` 和 `DINOGridSFT2Batch` 已删除。
- Grid/latent 都通过 `WorldModel.sigreg_state()` 向同一个核心 SIGReg 阶段提供 `(B,D)` 表示。
- trainer 始终构造唯一 `SFT2Algorithm`；`lambda_dino` 只控制附加 loss 权重并写入 checkpoint invariant。
- `d770a53` 推送后首轮服务器定向回归 `42 passed`，覆盖 DINO、核心 SFT2 loss/loop/resume、RL algorithm 与 grid WM。
- 新增同一核心 `training_sigreg_step` 对 grid slots 做 mean-pool 的回归，并用非默认 `lambda_dino=0.25` 证明权重不是硬编码。

## 验证

- `compileall`、`git diff --check` 通过。
- 静态扫描确认生产代码不再存在独立 DINO SFT2 algorithm/batch。
- 首轮服务器定向回归：`42 passed`。
- 新增 SIGReg/configurability 测试后待复跑，并需执行扩展回归。
