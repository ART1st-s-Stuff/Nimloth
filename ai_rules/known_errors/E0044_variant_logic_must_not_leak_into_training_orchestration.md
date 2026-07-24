# E0044：训练 variant 细节不得泄漏到公共编排

## 错误

为接入 DINO-grid 监督，在 SFT2/RL trainer 和公共 checkpoint 中直接加入
`args.objective == "dino_grid"`、`isinstance(GridWorldModel)`、具体模块构造和具体
artifact 文件名。

## 后果

- 增加一种监督目标需要同时修改 trainer、optimizer、DDP、日志、checkpoint 和 RL loader。
- 公共训练生命周期与具体数学目标耦合，README 中“独立 objective”与实际依赖不一致。
- variant checkpoint 的恢复和冻结语义散落在不同阶段，容易出现只修一条路径的错误。

## 已发生证据

- 合并后的 `src/nimloth/training/sft2/trainer.py` 曾包含多处 DINO/Grid 分支；
  `training/rl/trainer.py` 和两阶段 checkpoint 也直接识别 Grid 类型与文件名。
- 人类在合并后明确指出该实现破坏模块化；修复提交建立 SFT2 variant registry 和
  WorldModel loader registry，并用静态边界测试防止回归。

## 正确做法

1. 公共 trainer 只编排设备、分布式、optimizer 和 loop。
2. objective variant 自己拥有模型、batch、algorithm、指标和 checkpoint invariants。
3. WorldModel loader 自己识别 config、声明 artifact、重建组件并应用阶段冻结语义。
4. 公共 checkpoint 只调用多态保存/恢复与完整性接口，禁止硬编码 variant 文件名。
5. 新增 variant 时只修改其实现和注册表，并通过静态依赖边界测试。
