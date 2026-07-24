# 2026-07-24 DINO variant 模块化修复

## 目标

在 `dev` 完整解耦 DINO-grid variant：SFT2 trainer/checkpoint 与 RL trainer/checkpoint 不再识别 DINO、Grid 具体类型或具体 artifact 文件名；新增 variant 只修改自身实现与注册表。

## 计划

1. RED：添加静态依赖边界、SFT2 variant、world-model loader/checkpoint 契约测试。
2. GREEN：建立 SFT2 variant registry，将模型、batch、algorithm、指标、invariants 组装移出 trainer。
3. GREEN：建立 world-model checkpoint loader registry，将 Grid 恢复、冻结、broadcast 和完整性判断移出 RL/SFT2 公共代码。
4. REFACTOR：用 WorldModel 多态组件接口统一 DDP 包装、optimizer groups 和 checkpoint predictor 恢复。
5. 更新 README、进度和 known error；运行静态、定向与扩展回归后提交并推送 `dev`。

## 已完成

- 人类确认修复范围为“完整解耦”。
- 确认当前耦合点散布于 SFT2 trainer 的模型/batch/optimizer/invariants/metrics/algorithm，以及 RL trainer/checkpoint 和 SFT2 checkpoint。
- RED 提交 `6af3c99` 新增公共编排禁止出现具体 DINO/Grid 类型或 artifact 名的静态边界，以及统一 loader 恢复测试。
- 新增 SFT2 variant registry；latent 与 DINO-grid 分别拥有模型、batch、algorithm、runtime metadata、metric 和 checkpoint invariant。
- 新增 WorldModel loader registry；grid loader 自己识别 predictor config、重建组件、声明旧 checkpoint 必需 artifact 并应用 RL 冻结语义。
- WorldModel 现在多态声明 optimization/broadcast components，并可原位包装 DDP 组件；公共 trainer 不再用 `isinstance` 重建具体模型。

## 文件修改

- `src/nimloth/training/sft2/variant.py`、`dino_grid.py`：variant registry 与具体实现。
- `src/nimloth/wm/factory.py`、`grid_factory.py`：checkpoint loader registry。
- `src/nimloth/wm/model.py`、`grid.py`：组件和阶段策略多态接口。
- SFT2/RL trainer/checkpoint：删除 DINO/Grid 具体依赖。
- `tests/training/sft2/test_variant_boundary.py`、`tests/training/rl/test_grid_checkpoint.py`：架构与恢复契约。

## 验证

- `compileall`、`git diff --check` 通过。
- 静态扫描确认四个公共 trainer/checkpoint 不含 DINO/Grid 具体类型、模块或 artifact 名。
- 服务器 Pytest 待提交同步后执行。

## 风险

- 活跃 ID46 DINO SFT2 checkpoint 仍需可恢复；loader registry 必须支持现有无通用 manifest 的 grid artifact。
- 不改变 DINO、WM、SIGReg、value、PPO 数学与冻结语义。
