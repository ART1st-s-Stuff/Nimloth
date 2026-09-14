# 评估 Stage2 epoch3 的 rollout DINO 特征恢复

## Goal

在 a100-2 上让当前 Stage2 epoch3 checkpoint 自主完成 10 个 held-out navigation rollout，并直接比较每个实际访问状态中 checkpoint 投影的 8x8 DINO 特征与冻结 DINO teacher 特征，判断 state 是否恢复了足量的空间和语义信息。

## Background

- 来源 checkpoint 是 a100-1 上 `/mnt/nimloth/outputs/experiments/sft2-dino-information-test/20260912_epoch2_selective_rows_fresh_r2/train/epoch_003`。
- 该目录在 2026-09-13 核验为 8.7 GB，包含 `COMMITTED`、adapter、`slot_projector.pt`、`grid_state_config.json` 和 `training_state.pt`。
- 训练进程绑定 commit `99207d3518ddba896efd803031d8954141197291`，配置为 K=64、8x8 grid。
- a100-2 在前置检查时八张 A100 40 GB 均空闲，`/mnt` 可用约 1.3 TB，并有已经跑通的 Stage2 epoch2 rollout controller 可复用。
- DeepSight Figure 7 采用按时间排列的单通道伪彩色 DINO 特征图，但论文和公开代码没有披露 1024 维到单通道的精确降维方法。本任务明确记录自己的可复现降维方法，不把它冒充作者未公开的方法。

## Requirements

1. 禁止 checkpoint 数据经过本机中转；必须在两台服务器间直接传输。完整复制 epoch3 checkpoint 到 a100-2 的唯一实验目录，复制前后保存文件清单、大小和 SHA256；不删除或覆盖已有 checkpoint、rollout 或训练输出。
2. 从复制后的 epoch3 导出可供现有 rollout runtime 加载的模型；严格保留 tokenizer、64 个 query token、8x8 grid 配置和 slot projector，并记录导出来源。
3. checkpoint 自主执行总计 10 个 test rollout：Base 5 个、Common Sense 5 个；复用已验证的环境、prompt、parser、动作、成功阈值和确定性生成设置。
4. 保存每个 episode 的逐 turn 观测、实际上下文、模型原始输出、解析动作、奖励、终止原因和 success；rollout success rate 只作为这 10 个 episode 的小样本诊断，不称为标准 120-episode 评估。
5. 对 rollout 实际访问的每个 state，使用相同 checkpoint 和实际上下文提取 projected 8x8x1024 特征，并用训练时一致的冻结 DINO teacher 从对应观测提取 target 8x8x1024 特征。保存原始张量和逐 slot、逐 turn 指标。
6. 生成 DeepSight 风格时间序列图：同一 episode 的时间沿横轴排列；至少包含观测、projected 特征、target DINO 特征和误差四行；预测与 target 共用固定投影基和色标。
7. 单通道图只用本次 10 个 rollout 的所有 target DINO slot 拟合一个 PCA 基，取 PC1；将同一基不变地应用于 projected 特征；色标由全体 target PC1 的固定分位数确定并在所有图中共享。另保存 `1 - cosine_similarity` 的 8x8 误差图。
8. 汇总整体、按 eval set、按 rollout、按 turn 的 cosine similarity 和 MSE，以及 10-rollout success rate。结果必须区分视觉证据、特征相似度和策略成功率。
9. 保存足以在后续运行 CFM 的原始 observation、projected feature、target feature、episode/turn/action 对齐信息。本次不自动启动 CFM；只有 DeepSight 风格图无法解释恢复质量时再单独决定。

## Acceptance Criteria

- [ ] a100-2 目标 checkpoint 有完整复制证明，关键文件 hash 与 a100-1 一致。
- [ ] 恰好 10 个新的 held-out episode 具有完整 terminal record；Base/Common Sense 各 5 个，无旧结果混入。
- [ ] 每个可评估 turn 都有 shape 为 8x8x1024 的 projected/target 成对原始特征，episode、turn、观测和动作身份一致。
- [ ] 汇总文件给出样本计数、cosine、MSE、success 数和 success rate，并能追溯到逐 turn 记录。
- [ ] 每个 episode 产生一张共享 PCA/共享色标的 DeepSight 风格图，同时保存 PCA 参数和色标参数，结果可复现。
- [ ] 控制日志记录 checkpoint、commit、命令、环境、端口、PID、起止时间、阶段终态和产物路径；失败不会自动无限重试。

## Out of Scope

- 不训练或续训模型。
- 不运行标准 120-episode success-rate 评估。
- 不以 CFM reconstruction 代替本次直接 feature comparison，也不在本任务中自动启动 CFM。
- 不删除现有 checkpoint、训练输出或历史评估结果。
