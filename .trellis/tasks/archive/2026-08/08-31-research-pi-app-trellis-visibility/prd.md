# 调研 pi-app Trellis 实时可见性

## Goal

只读追踪 Trellis task、项目 Pi extension、Pi worker/session events、pi-app adapter与Trellis侧栏之间的真实数据流，解释为何用户当前只能看到“进行中”，并形成一个可实施、可验证且不复制TaskTree权威的端到端实时可见性方案。

## User-observed problem

pi-app当前Trellis界面只暴露任务处于“进行中”，无法回答：

- 主Agent此刻是否真的在工作；
- 正在执行哪个阶段、工具或命令；
- 是等待人类、Slurm、VPN、子代理还是证据；
- 已等待多久、最近一次有效进展何时发生；
- 下一步动作或审批门禁是什么；
- 当前活动是否属于任务关键路径；
- Agent正在完成Trellis父/子任务树中的哪个任务节点；
- Agent正在完成该任务`implement.md`计划清单中的哪一项，以及该项是working、blocked、waiting还是done。

## Requirements

- 只读检查Nimloth项目内的Trellis/Pi adapter、相关spec与runtime约定，以及`/workspace/pi-app`中的Trellis adapter、workspace task reader、side panel、worker events、timeline/tool card和通知实现。
- 完整区分以下数据面：
  - Trellis持久任务生命周期状态；
  - Trellis parent/children任务树；
  - `implement.md`有序计划清单与checkbox状态；
  - 当前session/runtime execution状态；
  - 主Agent tool execution事件；
  - subagent run状态；
  - 远程job/外部等待状态；
  - 需要人类响应的approval状态。
- 设计一个明确的“当前工作项游标”，把实际session/run绑定到唯一Trellis task和唯一plan item；不得仅从tool名称或assistant文本猜测。
- 每个可见plan item至少需要：稳定identity、所属task、层级/顺序、文本、计划状态、实时执行状态、执行者、开始/更新时间、阻塞原因和最近证据。
- 兼容现有没有显式item ID的`implement.md`，并为新任务定义稳定ID约定；不得把行号作为长期identity。
- UI必须同时展示task tree路径、计划项进度和当前执行者，并允许展开父/子任务及任务内计划清单。
- 用源码证明“只能看到进行中”的直接原因，不能仅依据界面文案或旧审计结论。
- 评估至少三类方案：
  1. 仅扩展现有task静态字段；
  2. 增加Trellis runtime execution状态，由pi-app读取；
  3. 直接由pi-app聚合Pi worker/tool/session事件；
  4. 如证据支持，可推荐混合方案。
- 对每个方案评估：所有权、更新来源、实时性、重启恢复、stale检测、跨project/session隔离、兼容性、通知、测试成本和是否会形成第二任务权威。
- 推荐明确的MVP字段、生产者/消费者边界、存储或事件传输方式、失效策略、UI位置、测试矩阵和分阶段实施顺序。
- 明确哪些改动属于Nimloth项目，哪些属于pi-app仓库；不得用一个仓库中的临时替代冒充端到端完成。

## Authorization and exclusions

- 人类已批准创建并执行“先做只读调研”的Trellis任务。
- 在最终计划再次获得开始批准前，只允许编写本任务的规划与context清单。
- 执行阶段仍为只读；唯一允许写入本任务目录中的研究报告和任务进度。
- 不修改Nimloth源码、`.pi/`、`.trellis/spec/`、Pi安装、pi-app源码/配置、TaskTree、memory、实验或远程系统。
- 不启动实验、GPU、Slurm或远程job；不commit、push、merge或清理既有dirty changes。
- 不把execution状态写入`.pi/task-tree/`；Trellis仍是唯一任务权威。

## Deliverable

`research/pi-app-trellis-visibility-design-2026-08-31.md`，至少包含：

1. 当前端到端数据流图；
2. “只显示进行中”的源码级根因；
3. 已有能力与缺口矩阵；
4. 方案比较和推荐架构；
5. task tree + plan item + execution cursor的MVP schema/UI/notification行为；
6. 现有Markdown计划的稳定ID与迁移策略；
7. 分仓库实施计划、测试计划和迁移/回滚策略；
8. 事实、推断、未决决定和证据缺口。

## Acceptance Criteria

- [x] 用当前源码定位任务状态从Trellis到pi-app侧栏的完整读取与渲染路径。
- [x] 证明哪些实时信息已经存在于Pi事件或subagent adapter中，哪些当前没有生产者。
- [x] 清楚解释为何`in_progress`无法代表主Agent实时活动或阻塞原因。
- [x] 比较至少三种方案，并给出唯一推荐的MVP架构及选择理由。
- [x] 推荐方案不复制TaskTree，不把静态task状态冒充实时execution状态，并满足`ctx.cwd`与root/session cache隔离合同。
- [x] 给出Nimloth与pi-app各自的文件级实施范围、兼容策略和验证矩阵。
- [x] 报告区分已验证事实、设计建议和仍需人类决定的问题。
- [x] `task.py validate`、task范围空白检查和限定目录审查通过。
- [x] 用当前Trellis实现证明task parent/children与`implement.md`计划项的实际所有权和数据缺口。
- [x] 定义稳定plan item identity、状态机和session/run绑定合同。
- [x] 给出能展示“任务树节点 → 当前计划项 → 执行者/状态/证据”的具体UI模型。
- [x] 说明现有无ID Markdown计划的兼容/迁移方案，不制造第二任务权威。
- [x] 修订推荐架构：第一版必须包含显式current work item，不能只显示tool activity。

## Resolved decisions

- 第一版采用混合传输：Trellis runtime assignment提供显式work-item语义，pi-app worker event提供观测到的tool/run事实。
- 新plan item使用task-local显式`[W-xxx]`；旧Markdown使用标记为unstable的derived ID并渐进迁移，不建立sidecar清单。
- runtime状态由Agent显式声明，tool lifecycle自动补充heartbeat/observed activity；禁止从tool或文本猜work item。
- working/verifying/delegated依赖heartbeat并可stale；waiting/blocked跨turn保留，直到显式release、task/item变化或引用失效。
- assignment支持main与subagent、多session聚合和同item并发警告。
- approval与remote wait进入第一版通用状态/typed evidence，但不在可见性层直接控制审批或Slurm。
