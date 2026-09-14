# Design — SFT debug-to-training workflow audit

## Evidence model

审计以可复核的事件表为核心。每个关键事件记录：时间、动作/决定、来源、所属工作类别、是否阻塞训练、阻塞类型、证据强度。

证据优先级：

1. 当前 task artifacts、实验记录、Git commit/diff 与实际配置/日志；
2. Trellis workspace journal 与明确时间戳的本地运行记录；
3. `trellis mem` / Pi 原始会话，用于定位审批等待和 Agent 行为，再由文件证据交叉核验；
4. 用户主观体验，作为必须解释的观测，不擅自改写为精确时长。

## Delay taxonomy

- `required-technical-work`：训练前确有必要的语义修复、数据/配置确认或实验准备。
- `required-safety-gate`：实施批准、精确实验 launch approval、commit 等硬门禁。
- `avoidable-approval-latency`：本可提前合并询问、设定授权包络或延后至真正阻塞点的等待。
- `verification-cost`：测试执行时间、重复验证、验证范围膨胀或失败后重跑。
- `rework`：不充分规划、上下文丢失、误解或工具故障导致的返工。
- `context-switch`：SFT 关键路径之外的 storage/worktree/治理工作。
- `platform-visibility`：用户无法判断 Agent 当前步骤、命令、阻塞原因、预计下一门禁。
- `evidence-gap`：无法从本地资料可靠判定。

## Analysis method

1. 先从 8 月 26 日父任务和 8 月 29 日训练子任务抽取正式状态与关键节点。
2. 用 Git、journal、实验记录补足文件层时间线。
3. 用 `trellis mem` 搜索 SFT1/SFT2、approval/批准、test/验证、training/launch 等会话，识别等待区间和 Agent 活动。
4. 将所有事件映射到延迟分类，检查哪些位于“开始新一轮训练”的关键路径。
5. 对 Pi/pi-app 可见性问题，读取当前本地 Pi 文档/设置及实际会话表现，区分现有能力、项目 adapter 行为和 GUI 缺口。
6. 形成根因树与分层改进建议；不把相关性冒充因果，不用总日历时间冒充 Agent 实际运行时间。

## Output

`research/workflow-audit-2026-08-31.md` 包含：

- executive summary；
- 时间线；
- 延迟分解；
- 三个用户体验问题的专项分析；
- 必要门禁与可避免摩擦；
- 改进建议与优先级；
- 证据目录、局限与未决问题。

## Safety and rollback

本任务只读项目与平台证据；唯一新增/修改内容限于本任务目录。若误触其他文件，停止并报告，不自动覆盖现有 dirty change。删除本任务新增报告即可回滚交付物；未经批准不执行删除。
