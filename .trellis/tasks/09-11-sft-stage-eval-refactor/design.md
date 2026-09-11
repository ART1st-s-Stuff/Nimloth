# 设计

## 入口与归属
扩展现有 nimloth.training.sft.evaluation 为唯一 success rate 调度入口，以显式模型来源/阶段替代用户手工组合 direct/wm。复用现有 EvaluationConfig，并由边界层构建和校验 typed config；运行时不解析配置文件。每个 stage 自己拥有环境 eval 适配接口，避免与现有离线 evaluate 混淆；它们调用共享 rollout 引擎，不复制 episode 循环。VAGEN 和 RL 同样经统一入口路由。

## 协议与组件
VAGEN 与 Nimloth 分离 prompt/动作协议，共享环境执行、持久化和统计。训练与评估共用 B prompt 定义，保留图片位置和动作语义，不在推理时临时替换文字。解码保留 action token，区分 EOS/padding，记录原始响应。Stage2 按 checkpoint 的 query 生成/注入合同加载所需组件；不用 WM 决策。Stage3/RL 的 WM、value 与 MCTS 继续复用现有 AgentRuntime/PlanningPolicy 和 checkpoint 校验。

## B 正式化
把 B 数据修正、语义初始化、预处理与续训组织逻辑迁入所属正式模块；复用 stage1 trainer/convergence/checkpoint。标准配置保存已确认超参数，路径和运行资源由 CLI/本地配置提供。删除替代实现及旧启动入口，修改所有活动调用；不保留旧入口的兼容壳。

## 结果与恢复
统一 episode 身份和 success 统计；保存 checkpoint/协议/参数合同，resume 逐项匹配。部分运行单独标识 completed/requested，不能当完整评估。环境反馈是 success 唯一来源；格式指标另存。服务生命周期由正式启动层管理，异常退出释放自己创建的子进程，禁止全局杀进程。

## 迁移与风险
先审计活动调用再删除，旧源码通过 Git 历史追溯，不复制到 legacy 目录。历史实验及 archive 原样保留。重点风险为特殊 token 丢失、图片位置变化、stage2 query 错接、旧 SFT2 名称误识别和 RL 价值语义漂移。优先用端到端合同测试覆盖，真实 smoke 再确认运行链；不重新搭建独立评估器。

## 2026-09-11 人类批准与本轮范围
人类已批准规划并要求先提交已有更改再实施。此次只正式化 B，然后实现 Stage 1/Stage 2 的正式统一 success rate 评估（保留 VAGEN 协议分支）。Stage 3/RL 的新增评估、入口迁移和重构全部延期，现有行为保持；以上涉及五种来源的完整验收是后续目标，不是本轮实施范围。先 B 后评估，不启动长期训练或全量评估。
