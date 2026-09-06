# 实施计划（待审查，未启动）

1. [x] 实施前准备：核验Git/worktree合同、实际基点与其他任务现场；在专用codex分支工作。保留当前未跟踪spec的准确内容。补齐调用/数据/梯度/checkpoint清单。
2. [x] 建立sft模块与入口映射：迁入可复用SFT1/SFT2生命周期、配置、数据和恢复逻辑，先通过导入/兼容检查，避免重命名时同时改所有算法。
3. [x] 实现SFT1与SFT2：分别限定CE、CE+query/DINO监督；统一teacher forcing/state抽取与冻结策略，更新阶段配置及有意义的损失/梯度测试。
4. [x] 对齐SFT3窗口：复用旧多步递推、MC和SIGReg；明确T步均值、terminal/尾窗口规则和H=1约束。检查旧单步兼容路径。
5. [x] 整理eval_direct/eval_wm：复用真实环境和现有planner，统一评估协议及结果；补真实反馈驱动重新规划的接口测试。
6. [x] 完成迁移：更新活动调用者、CLI、配置、加载器与README；旧入口按审查确定的策略薄转发。核验无第二套算法残留。
7. [x] 全范围检查与审查（完成代码审核及检查；提交/合并仍未执行）：记录检查结果、未验证GPU/rollout项和完整diff。独立check审查通过后再按项目规则请求提交审查；不自动提交/推送/合并。

## 验证
- 对新sft Python做语法/导入检查，`git diff --check`。
- `python -m pytest tests/training/sft`（实施中新建镜像测试路径）。
- 迁移期间运行仍保留的 `tests/training/sft1`、`tests/training/sft2` 和 `tests/training/common`。
- 受影响 `tests/training/rl`、`tests/recon`、`tests/eval/rollout_browser` 及 `tests/wm` 的接口/加载检查。
- 对比迁移前后受控输入的原有SFT3 loss和梯度；新增SFT2验证projector/query梯度、DINO冻结、回答mask、空间对应。
- checkpoint往返、旧格式识别、恢复位置与阶段不匹配拒绝测试。
- eval使用隔离环境替身仅验证调用顺序/反馈/计数；不作为真实rollout成功证据。GPU/DDP、vLLM、真实环境评估留到另行授权的实验。
- 项目根目录未发现统一pyproject/pytest/lint配置；实施时按环境与拥有模块确认可用检查工具，不宣称未运行的lint/type检查通过。

## 审查/回退点
目录迁移、算法改动、评估接线分别审查；以专用分支的普通diff/提交分组回退，不修改权重和实验输出，不force操作。实现/check使用Trellis子agent时按模块分配所有权并传入本任务上下文。

## 2026-09-05 实施授权与范围澄清
人类已批准实施并要求新branch。SFT3（旧SFT2）仅迁移及可读性优化，保留现有算法和生命周期，不完全重写。SFT1/2按新设计可做更多必要改写。主要工作目录为当前任务隔离worktree，控制目录原计划保留供定位，后续计划/验收状态以此工作目录为准。

## 2026-09-06 补充整改
- [x] 删除旧训练包并迁移所有活动调用，校验无残留导入。
- [x] 将canary及实验原型移出核心，明确模块职责，改善局部可读性。
- [x] 中文化SFT所有README，更新入口与模块索引。
- [x] 实施后独立审核并运行受影响回归，记录未验证项。
