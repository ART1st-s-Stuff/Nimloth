# 执行计划

- [ ] 规划审核后激活任务。基点 d85732c8，分支 codex/sft-stage-eval-refactor，worktree /workspace/remote2/nimloth/.worktree/sft-stage-eval-refactor。
- [ ] 完整审计 stage1 B、所有活动 success rate 入口和调用方，形成逐文件保留/迁移/删除清单；读取各 owning README、配置/接口及 CoT/state 规范。
- [ ] 将 B 配置、prompt、初始化、缓存和训练/恢复移入正式归属，删除旧 Stage1 流程和 task research 依赖。
- [ ] 实现统一 typed 评估配置和阶段校验；分别实现 VAGEN、stage1、stage2 协议；接入各 stage eval。
- [ ] 将 stage3/RL 接到共享 MCTS 评估，保持 checkpoint/状态/Q 合同。
- [ ] 合并服务启动、checkpoint 导出消费、输出/恢复/统计，清理所有旧 success rate 运行入口与重复代码。
- [ ] 更新正式配置、CLI 示例、README 和项目 spec；spec 沿原格式描述算法/接口，不写成实现草案。
- [ ] Trellis check 审查全部差异。验证 CLI 参数覆盖、B prompt 等价、语义初始化与权重、query 协议、动作 token/EOS、阶段拒绝、MCTS 调用、结果计数/恢复和子进程清理。
- [ ] 检索无遗留生产调用：rg 定位原路径/旧入口/research 导入；运行受影响 tests/training、tests/agent、tests/rollout、tests/environment 及仓库配置的 lint/type 检查，git diff --check。
- [ ] 按实验合同核验远程 checkpoint/运行依赖/资源，通过唯一正式入口进行真实 smoke；无合适产物的阶段明确标记未验证，不伪造兼容产物。
- [ ] 保存最终删除清单、验收证据和局限，提交前展示范围及检查结果。此任务不自动 push/merge，不恢复旧失败评估。

## 回退
只回退本任务源码改动，保留原 checkpoint 与实验输出。不得通过保留旧实现来实现运行时 fallback。

## 2026-09-11 人类批准与本轮范围
人类已批准规划并要求先提交已有更改再实施。此次只正式化 B，然后实现 Stage 1/Stage 2 的正式统一 success rate 评估（保留 VAGEN 协议分支）。Stage 3/RL 的新增评估、入口迁移和重构全部延期，现有行为保持；以上涉及五种来源的完整验收是后续目标，不是本轮实施范围。先 B 后评估，不启动长期训练或全量评估。
