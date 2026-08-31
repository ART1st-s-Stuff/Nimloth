# Design — pi-app Trellis实时可见性调研

## Requirement correction

人类已拒绝只展示session/tool activity的MVP。第一版必须回答：主Agent或子代理正在执行Trellis任务树中的哪个task，以及该task计划清单中的哪一个work item、当前状态和证据。因此本任务返回规划阶段；此前的event-only推荐只作为被否决方案保留。

本设计中的“任务树”仅指Trellis `task.json.parent/children`。Pi TaskTree继续保持空，不作为状态源或展示后端。

## Research ownership

本任务只产出设计证据，不实现产品代码。

- Nimloth侧证据所有者：`.trellis/`任务/runtime合同、`.pi/extensions/trellis/`、项目spec与skills。
- Pi core侧证据所有者：extension API、tool/session lifecycle和TUI/UI原语。
- pi-app侧证据所有者：worker session events、extension compatibility adapters、workspace task reader、Trellis side panel、timeline/tool cards和desktop notifications。
- 最终实施必须按仓库拆分；本任务只描述兼容接口和依赖顺序。

## Evidence method

1. 从pi-app侧栏展示的“进行中”文本反向追踪到reader、adapter和`task.json.status`。
2. 从Pi主Agent与subagent工具执行正向追踪`tool_execution_start/update/end`及现有adapter payload。
3. 对比静态task生命周期、session执行活动和外部等待状态，识别缺失的生产者或桥接层。
4. 对每项结论记录当前源码路径、symbol和行为，不用README名称替代实现证据。
5. 使用已有tests/config验证边界；本任务不运行GUI自动化或修改配置。

## Candidate architectures

### A. Static task enrichment

扩展task reader/UI展示现有task artifact中的phase、checklist、最近progress。

- 优点：实现简单、跨重启稳定。
- 限制：不能可靠表示主Agent当前工具、等待原因和elapsed；写入频率过高会污染任务历史。

### B. Trellis runtime execution state

由项目Pi extension或轻量工具产生project/session-scoped ephemeral execution状态，pi-app消费。

- 优点：语义字段明确，可跨Pi CLI/Desktop统一。
- 风险：需定义stale、session隔离、owner和异常退出行为；不能演变为第二任务系统。

### C. pi-app worker event aggregation

pi-app直接从现有session/tool事件推断working、tool、elapsed和last activity。

- 优点：实时，不要求项目频繁写文件。
- 限制：只能看到“发生了什么工具调用”，通常不知道为什么等待、下一门禁和远程job语义。

### D. Hybrid

worker事件负责事实性实时活动；Trellis runtime语义负责`blocked_on`、`next_gate`、关键路径和外部job。人类新要求使混合方案成为第一版必要条件，但仍需确定work-item identity、cursor owner和迁移合同。

## Required work-item model

推荐设计必须把三层信息合并而不复制权威：

1. **Task tree authority**：`task.json.parent/children/status`；
2. **Plan authority**：`implement.md`中的有序work items和checkbox；
3. **Runtime cursor**：session/run当前指向哪个task/work item、由谁执行、处于何种实时状态。

runtime cursor只能引用task与plan item，不能复制其标题、优先级、完整清单或acceptance criteria作为第二份权威。它可承载ephemeral字段：executor、working/blocked/waiting、since、last evidence和stale。

需要研究并固定：

- 新plan item的稳定ID写法；
- 旧Markdown无ID item的derived identity与迁移失败行为；
- 主Agent把item委派给subagent时cursor如何形成父子run关系；
- checkbox与runtime状态冲突时UI如何fail closed；
- session异常退出后cursor何时标记stale而不是继续显示working。

## Required design properties

- Trellis task仍是目标、状态、优先级、层级和acceptance criteria的唯一权威。
- execution状态必须明确为session-scoped runtime projection，不写TaskTree，不改变task lifecycle语义。
- project root以`ctx.cwd`为准；所有cache/status键至少包含resolved root与session/context identity。
- 缺失、malformed、过期或来自另一root的runtime状态必须安全降级到静态task视图。
- UI不得把“最后一次tool事件”误报为当前仍在运行。
- 自动推断与Agent声明字段必须在UI中有可辨别的证据边界。

## Expected recommendation shape

报告应选择一个MVP，而不是只列选项。第一版必须展示`task tree path → current plan item → executor/runtime status`，不能退化为tool activity卡。推荐内容至少固定：

- task、plan item和runtime cursor schema字段及枚举；
- producer/consumer；
- transport/storage；
- heartbeat/stale规则；
- main Agent、subagent、plan item ownership transfer、approval和remote job的覆盖边界；
- pi-app展示位置与空状态；
- Nimloth/pi-app分仓库实施顺序；
- unit/integration/manual validation；
- backward compatibility和rollback。

## Safety and rollback

执行阶段只读。若后续实施：

- runtime桥接必须可feature-flag或通过缺失schema自然降级；
- pi-app必须兼容没有新extension/runtime状态的普通Trellis项目；
- 删除runtime projection不能损坏task artifacts；
- 任何跨仓库schema在实施前必须有version字段与fixture。
