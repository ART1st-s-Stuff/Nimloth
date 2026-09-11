# 标准化 B 训练并重构 SFT 三阶段评估

## 目标
以 B 为唯一标准 Stage 1 流程；通过正式统一入口反复运行 VAGEN、SFT1、SFT2、SFT3 和 RL 的真实环境 success rate 评估，无需临时拼接 Python 脚本。

## 已核验背景
- B 实现在旧任务 research/run_b_pilot.py、resume_b.py、prepare_b_prompt_data.py、initialize_action_tokens.py，存在机器路径绑定。
- 正式训练已有 stage1 CLI 和 convergence；不能另造训练器。
- training/sft/evaluation 已有 direct/wm 接线，但没有完整的阶段与模型来源区分。
- stage3/mcts_evaluation.py 已有 checkpoint 合同；目录迁移后部分名称仍保留旧 SFT2 语义。
- 新任务基点 d85732c8 包含 B 及后续 Query 实现，领先 dev 17 个提交。不得回退这些已存在行为。

## 要求
R1. B 成为标准：语义动作词初始化、修正后的 rollout 一致 prompt、action token 权重默认 8、先处理缓存再训练、验证 LM loss 收敛（至少两轮，连续两轮相对改善不足 1%）、每 10 步保存和已约定的完整 epoch 后中间 checkpoint 清理。以上是可显式覆盖的配置，不是机器相关代码常量；保留恢复训练能力。
R2. 删除被 B 替代的旧 Stage 1 可执行脚本、旧实现、失效配置和调用，不留兼容转发、第二套实现或任务 research 运行依赖。保留历史实验输出和 checkpoint，不改写 archive。
R3. 所有活动 success rate 运行与统计统一入口、配置合同和结果 schema。入口显式区分 VAGEN / stage1 / stage2 / stage3 / RL；禁止临时运行自编脚本绕过入口。
R4. VAGEN 使用其语义动作与原 prompt；Nimloth 使用各阶段训练匹配的 prompt/action 协议。原始生成文本和 token 必须可追溯，不吞 action special token、不修补无效输出为有效动作。
R5. 每个 SFT stage 有独立拥有的 eval 接口，由统一入口调用。stage1 无 query/latent/WM；stage2 按其 query 合同启用模块，但直接执行模型动作，不调用外部 WM/value/MCTS。
R6. stage3 与 RL 复用同一个真实 WM、value head、MCTS 执行路径，真实观测后重新规划；不得改变现有状态时刻、价值目标、搜索评分语义。
R7. 可复用逻辑属于 src/nimloth，配置属于 configs，experiments 只保留薄启动层，测试镜像所属模块。机器路径、解释器、依赖目录、GPU 和端口允许正式参数/本地配置指定。
R8. 评估记录输入 checkpoint 身份、阶段、协议、完整参数、episode 身份、真实环境 success 和终止原因。拒绝阶段错配、缺组件、恢复合同变化及不完整结果冒充全量指标。

## 验收
- 正式命令完成 B 的处理、初始化、训练/续训，生产代码不再导入 research。
- 一个评估入口覆盖五种来源，三个 stage 的 eval 可分别验证；stage1/2 不加载 planner，stage3/RL 使用同一 planner。
- 同一模型/种子/参数的 prompt、动作解码和结果统计可追溯；完整 held-out 默认 base/common_sense 各 60，样本 smoke 单独标识。
- 清理清单覆盖旧入口、调用方、配置、文档、测试；没有活动重复实现和失效导入。
- 聚焦测试与受影响包检查通过；真实 GPU/环境 smoke 使用正式入口，报告实际完成范围，不以 CPU 测试宣称 success rate 验证。

## 范围边界
不重训全部模型，不改变 stage2/3/RL 目标，不修改历史权重/数据/输出。此次先完成代码规划；GPU 验收启动前核验可用 checkpoint、资源与预算，缺少某阶段产物明确报告未验证。

## 2026-09-11 人类批准与本轮范围
人类已批准规划并要求先提交已有更改再实施。此次只正式化 B，然后实现 Stage 1/Stage 2 的正式统一 success rate 评估（保留 VAGEN 协议分支）。Stage 3/RL 的新增评估、入口迁移和重构全部延期，现有行为保持；以上涉及五种来源的完整验收是后续目标，不是本轮实施范围。先 B 后评估，不启动长期训练或全量评估。
