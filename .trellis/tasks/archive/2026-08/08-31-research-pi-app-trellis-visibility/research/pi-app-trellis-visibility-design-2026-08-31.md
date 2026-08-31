# pi-app Trellis任务树与当前工作项可见性设计

日期：2026-08-31

范围：只读检查Nimloth Trellis/Pi extension、Pi SDK 0.83.0与pi-app 0.5.7。没有修改产品代码、配置、TaskTree或远程系统。

## 1. 最终结论

用户需要的可见性不是“Agent是否正在调用工具”，而是：

```text
Trellis任务树中的哪个task
→ 该task计划清单中的哪个work item
→ 由主Agent还是哪个子代理执行
→ 当前是working、verifying、waiting、blocked还是done
→ 最近证据与下一动作是什么
```

当前系统无法回答，因为Trellis只持久化：

- `task.json`中的任务生命周期和parent/children；
- `implement.md`中的自由格式Markdown checkbox；
- per-session active task pointer。

它没有稳定的plan-item identity、current work-item cursor、executor/run绑定或item级runtime状态。pi-app虽已拥有run/tool/subagent事件，但tool名称不能可靠推出它属于哪一条计划项。

### 推荐架构

第一版必须采用混合架构：

1. **Task authority**：`task.json.parent/children/status`；
2. **Plan authority**：`implement.md`标题、work-item文本和checkbox；
3. **Runtime assignment projection**：`.trellis/.runtime/execution/<context-key>.json`只引用task/work-item，并记录executor和实时状态；
4. **Observed activity**：pi-app现有AppEvent提供run/tool/elapsed/subagent事实；
5. **UI**：pi-app把四层合并成可展开的Trellis任务树。

Pi TaskTree继续保持空，不作为数据源、兼容层或后端。

## 2. 当前数据面与缺口

### 2.1 Task tree

Trellis任务关系由`task.json.parent/children`持久化。当前项目有一个真实父子链：

```text
08-26-state-interface-v2-sft1-canary-exp
└── 08-29-train-sft1-query-state
```

`.trellis/scripts/common/tasks.py:children_progress()`只能按child task status计算`[n/m done]`。它不解析task内部计划项，也不表示哪个child正在被某个session执行。

### 2.2 Plan list

`implement.md`是人类可读的执行计划权威。当前5个active task共有376条checkbox item：

- 显式稳定work-item ID：0；
- 嵌套checkbox：0；
- 计划项结构主要来自Markdown heading + flat checkbox。

当前`.trellis/scripts/`和`.pi/extensions/trellis/`都没有：

- `implement.md`结构化parser；
- item ID校验；
- current item命令；
- item assignment/runtime cursor。

`implement.jsonl`只是context manifest，不是任务清单，不能复用为work-item状态。

### 2.3 Active task

当前task pointer是per-session runtime：

```text
.trellis/.runtime/sessions/<context-key>.json
```

它只引用task，不引用`implement.md`item。不同session可以指向不同task，这是正确行为。

pi-app当前reader在Electron main process中执行`task.py current`，但没有传selected Pi session identity，因此通常不能解析正确pointer。

### 2.4 Agent activity

pi-app已经从Pi SDK获得：

- Agent running/idle/failed；
- 当前tool、tool update和错误；
- elapsed；
- queue；
- `trellis_subagent`与`subagent`tree details；
- background session live cache；
- completion通知。

主要路径：

- `/workspace/pi-app/src/worker/worker-session-events.ts`；
- `packages/shared/app-events.ts`；
- `src/renderer/src/stores/apply-app-event-{run,tool}.ts`；
- `src/renderer/src/lib/live-session-timeline-cache.ts`；
- `src/renderer/src/features/run/run-panel.tsx`；
- `src/renderer/src/features/composer/composer-agent-activity.tsx`。

这些事件没有work-item reference，所以只能证明“发生了bash/read/subagent”，不能证明“正在完成W-042”。

### 2.5 Trellis side panel

当前panel只读取task/PRD/journal静态字段：

```text
workspace-task-panel-reader.ts
→ side-panel-registry.ts
→ adapter.sidePanel.getState
→ WorkspaceTasksSidePanel
```

`task.json.status=in_progress`被固定映射为“进行中”。面板没有event订阅、work-item parser、runtime assignment或stale语义。

## 3. 为什么不能自动猜当前计划项

以下启发式都不可靠：

- “第一个未勾选checkbox”可能只是未来任务，不一定正在做；
- 当前tool `bash`或`read`可属于任意计划项；
- assistant文本可能只是在解释、回顾或处理插入问题；
- Git changed file无法识别审批、等待、调研或remote job；
- subagent prompt可能同时覆盖多项；
- 并行subagent可能在不同item上工作。

因此current work item必须由Agent/workflow显式声明，tool事件只能作为观测证据补充。

## 4. Plan-item identity

### 4.1 新任务：显式ID

推荐在`implement.md`中使用task-local、稳定、可见的ID：

```markdown
## 3. Formal training

- [x] [W-031] 生成并验证Formal32 launch lock。
- [ ] [W-032] 提交并监控Formal32训练。
- [ ] [W-033] 审查terminal state evidence。
```

完整引用为：

```text
08-29-train-sft1-query-state#W-032
```

规则：

- `W-<三位以上数字>`，在一个task内唯一；
- ID不随heading、顺序或文本变化；
- heading形成UI section，不参与identity；
- checkbox是plan完成状态权威；
- duplicate ID或cursor引用不存在的ID必须报错，不能按文本模糊匹配。

显式ID比HTML comment更容易审查，比sidecar mapping更不易漂移。

### 4.2 现有任务：legacy derived ID

不能一次性改写当前376项并声称无风险。兼容parser应为无ID项生成：

```text
legacy-<hash(task-id + heading-path + normalized-text)>
```

行为：

- UI可立即展示旧计划；
- legacy ID明确标记为`unstable`；
- 文本/heading变化后旧cursor不能重新模糊绑定，必须显示`orphaned/stale`；
- 当某个legacy item成为active时，可通过单独、可审查的plan migration为其加入显式`W-xxx`；
- completed历史item可继续使用derived ID，不要求全量迁移。

### 4.3 拒绝sidecar作为plan identity权威

独立`work-items.json`若复制item文本、顺序或done状态，会与`implement.md`形成第二份任务清单。sidecar最多只能保存runtime引用，不能拥有plan内容。

## 5. 状态模型

### 5.1 Plan状态

只来自`implement.md`：

- `[ ]` → `pending`；
- `[x]`/`[X]` → `done`。

第一版不扩展Markdown checkbox字符表示blocked/waiting，避免渲染器和parser不一致。

### 5.2 Runtime assignment状态

仅对active assignment存在：

- `working`：正在推进该item；
- `verifying`：正在验证该item的完成条件；
- `delegated`：主Agent已将该item交给一个或多个子代理；
- `waiting_human`：等待用户决定或批准；
- `waiting_external`：等待Slurm、VPN、服务或其他外部状态；
- `blocked`：缺少证据、依赖或发生不可自主解决的问题；
- `failed`：本次assignment失败，但plan item仍未完成。

`stale`和`conflict`是有效性标记，不是正常业务状态：

- `stale`：working assignment失去heartbeat、session退出或task pointer改变；
- `conflict`：checkbox已done但仍有active assignment，或runtime引用不存在的task/item。

### 5.3 Effective UI状态

| Plan | Runtime | UI |
|---|---|---|
| done | none | done |
| pending | none | pending |
| pending | working/verifying/... | 对应runtime状态 |
| done | active | conflict，不冒充正常working |
| missing item | any | orphaned/stale |

Agent不能仅在runtime中把item标记done；完成必须更新`implement.md`checkbox并经过正常diff审查。

## 6. Runtime assignment projection

建议独立于active-task pointer：

```text
.trellis/.runtime/execution/<context-key>.json
```

`.trellis/.runtime/`已被项目gitignore，projection不会进入Git或取代task artifacts。

### 6.1 Schema v1

```json
{
  "schemaVersion": 1,
  "rootFingerprint": "sha256(realpath)",
  "contextKey": "pi_<session-id>",
  "updatedAt": "2026-08-31T08:00:00Z",
  "primary": {
    "assignmentId": "a-...",
    "taskRef": "08-29-train-sft1-query-state",
    "workItemRef": "W-032",
    "declaredState": "waiting_external",
    "executor": {
      "kind": "main",
      "sessionId": "...",
      "agent": "main"
    },
    "since": "...",
    "blockedOn": "slurm",
    "nextAction": "allocation后检查update-0",
    "lastEvidence": [
      { "kind": "job", "ref": "slurm:539107", "summary": "PENDING Priority", "at": "..." }
    ]
  },
  "delegated": [
    {
      "assignmentId": "a-child-...",
      "taskRef": "08-29-train-sft1-query-state",
      "workItemRef": "W-032",
      "declaredState": "working",
      "executor": {
        "kind": "subagent",
        "agent": "scout",
        "runId": "...",
        "toolCallId": "..."
      },
      "since": "...",
      "updatedAt": "..."
    }
  ]
}
```

runtime只保存引用、assignment和简短证据指针，不复制task标题、优先级、完整item文本、顺序或checkbox。

### 6.2 Writer

推荐由Nimloth Pi Trellis extension提供显式本地工具，例如：

```text
trellis_work_item select <task#item>
trellis_work_item update --state verifying
trellis_work_item block --on human|slurm|vpn|evidence
trellis_work_item evidence <typed-ref>
trellis_work_item release
```

实际工具名和参数在实施task中锁定。该工具：

- 通过`ctx.cwd`解析root；
- 通过session manager解析context key；
- 解析并验证task和implement item；
- atomic write runtime projection；
- 不编辑checkbox；
- 不需要人类Git approval，因为只写gitignored runtime状态。

workflow/agent规则要求：

- 开始一个实质plan item前显式select；
- item切换、委派、阻塞、等待、验证时更新；
- 完成时先更新`implement.md`checkbox，再release或select下一项；
- 若用户插入临时查询，可保留原primary assignment并标记短暂interruption，而不是伪造新plan item。

### 6.3 自动观测

Pi extension和pi-app可把tool lifecycle附着到已声明assignment：

- `tool_execution_start/update/end`更新active tool、heartbeat和错误；
- `ask_user_question`自动提示可能进入`waiting_human`，但最终语义仍需显式声明；
- `trellis_subagent`新增可选`workItemRef`，自动建立delegated assignment；
- generic `subagent`由pi-app根据当前assignment和tree tool details关联，若目标item不同则必须显式handoff；
- 不保存完整tool args，避免把secret或大command复制到runtime。

## 7. Stale、恢复与冲突

### 7.1 Working heartbeat

`working/verifying/delegated`必须有heartbeat。若worker/session不再live且heartbeat超过实施时确定的阈值，UI显示stale，不继续显示“正在工作”。

### 7.2 Waiting状态

`waiting_human/waiting_external/blocked`允许Agent turn结束后持续存在；UI显示`since`和最后更新时间。它们在以下情况失效：

- task归档或item变done；
- 当前session切换到另一task/item并显式supersede；
- runtime引用校验失败；
- 人类/Agent显式release。

### 7.3 Session shutdown

extension在正常shutdown时把working/verifying assignment标记paused/stale；waiting状态保留。异常退出由heartbeat过期处理。

### 7.4 多session/多agent

- 每个session单独runtime文件；
- pi-app按resolved root聚合fresh assignments；
- 同一item可显示多个executor；
- 两个main session同时声明独占item时显示concurrency warning，不静默选择一个；
- cache key必须包含root + context key，同session ID在另一worktree不得串状态。

## 8. pi-app目标UI

```text
Trellis

当前执行
● W-032 提交并监控Formal32训练
  waiting-slurm · 2h 14m
  main agent · session 01a053...
  Job 539107 · PENDING Priority · 3m前更新
  下一动作：allocation后检查update-0

任务树
▾ 建立DeepSight-aligned K16 state                  进行中
  └─ ▾ 正式训练SFT1 Query-State                    进行中
       Formal training                             1/3 done
       ├─ ✓ W-031 生成并验证launch lock             done
       ├─ ● W-032 提交并监控训练                    waiting-slurm
       │    └─ ◐ scout / run ...                   working
       └─ ○ W-033 审查terminal state evidence      pending
```

### 8.1 信息层级

- task badge明确标为“任务生命周期”；
- work-item badge是当前执行状态；
- executor可展开查看main/subagent run、active tool和elapsed；
- evidence显示typed reference，不粘贴完整日志；
- current session高亮，其他fresh session仍可见；
- legacy ID显示迁移警告；
- stale/conflict用显著但非破坏性状态展示。

### 8.2 面板刷新

- task/plan文件变化：重读task tree与plan；
- runtime projection变化：watch/poll并更新assignment；
- 当前selected session的AppEvent：实时更新tool/elapsed；
- background session：使用existing live cache；
- 保留手动refresh；
- malformed runtime、duplicate ID或cycle显示可见错误，不吞掉。

## 9. 分仓库实施范围

### 9.1 Nimloth/Trellis本地层

预计涉及：

- `.trellis/spec/governance/tasks-progress-and-memory.md`：work-item authority、ID和runtime合同；
- `.trellis/workflow.md`：item select/update/release门禁；
- `.trellis/scripts/`：Markdown parser、ID validation和runtime读写；
- `.pi/extensions/trellis/index.ts`：work-item工具、tool heartbeat、subagent assignment；
- `.agents/skills/on-progress/`及相关start/continue/implement规则：何时更新item；
- 当前/后续task plan：逐步采用显式`W-xxx`，不批量改写历史任务。

这些属于项目规则与平台integration变更，实施前必须有独立Trellis任务和明确approval。

### 9.2 pi-app

预计涉及：

- `packages/shared/ipc-contract.ts`：selected session与Trellis execution state contract；
- `src/main/workspace-task-panel-reader.ts`：task tree、plan parser结果、runtime assignments；
- `src/main/side-panel-registry.ts`和IPC handler：session-aware async state；
- worker/AppEvent bridge：assignment与observed tool关联；
- `workspace-tasks-side-panel.tsx`：树形task/plan/executor UI；
- reader/parser/runtime/panel/session-switch tests。

pi-app当前没有自己的Trellis，且`main`已有无关dirty changes。实施时必须使用经批准的独立pi-app branch/worktree或等待该状态处理，不能直接覆盖。

## 10. 验证矩阵

### Plan parser

- heading + flat checkbox解析；
- explicit ID唯一性；
- duplicate/malformed ID fail closed；
- legacy derived ID稳定性与文本变化后orphan行为；
- checkbox状态只来自Markdown；
- 376条现有item可无修改读取。

### Task tree

- parent/children展开；
- archived child仍计入done progress；
- missing child、cycle和重复引用可见报错；
- task lifecycle与item runtime不混淆。

### Runtime

- root/session隔离；
- atomic write与malformed文件；
- working heartbeat/stale；
- waiting跨turn保留；
- task/item变更后的orphan/conflict；
- 同item多executor；
- session shutdown和异常退出。

### Pi extension

- `ctx.cwd`优先于Desktop process cwd；
- select/update/block/evidence/release；
- tool start/update/end heartbeat；
- Trellis subagent delegation与generic subagent关联；
- `/reload`恢复；
- 无active task/item时明确失败，不猜测。

### pi-app

- selected session显示正确task/item；
- background session不串到当前session；
- task tree、plan progress、executor、evidence渲染；
- working/waiting/stale/conflict状态；
- runtime producer缺失时退回静态task/plan，不冒充live；
- 不读取或写入`.pi/task-tree/`；
- unit、typecheck、lint、build和开发模式人工验证。

## 11. 实施顺序

1. **合同与parser**：锁定work-item ID、plan parser、runtime schema和fixtures；
2. **Nimloth Pi producer**：显式cursor工具、heartbeat和subagent handoff；
3. **pi-app consumer**：session-aware reader、任务树和当前item UI；
4. **集成验证**：长bash、审批等待、Slurm等待、并行subagent、session切换、reload；
5. **渐进迁移**：新task使用显式ID，现有active item按需迁移；
6. **可选上游化**：本地合同稳定后再决定是否贡献Trellis/pi-app上游。

建议以一个Nimloth parent task管理端到端验收，并拆成两个可独立审查的children：

- Trellis plan-item/runtime producer；
- pi-app task-tree/runtime consumer。

## 12. 风险与边界

### 必须避免

- 把runtime assignment变成第二份任务清单；
- 通过first-unchecked或LLM文本猜当前item；
- item runtime声称done但checkbox未更新；
- 工作session消失后继续显示working；
- 同session ID跨root复用；
- 为可见性启用Pi TaskTree；
- 一次性重写当前376条历史/活动计划项。

### 已验证事实

- 当前5个active task、376条checkbox、0条稳定item ID；
- Trellis没有plan parser/current-item cursor；
- pi-app拥有实时run/tool/subagent facts但没有work-item reference；
- Trellis side panel只显示静态task lifecycle；
- project `.trellis/.runtime/`已gitignored，可承载非权威projection。

### 尚未验证

- 没有实现parser/schema或运行测试；
- 没有决定heartbeat具体阈值；
- 没有对376条计划进行migration；
- 没有修改pi-app dirty main；
- 没有验证Claude/Codex producer；第一版范围可先锁定Pi，但runtime schema应保持平台中立。

## 13. 最短结论

第一版不应只增加“Agent正在跑bash”的卡片。它必须增加一个显式、可验证的链路：

```text
Pi session/run
→ runtime assignment
→ task#work-item ID
→ implement.md计划项
→ task.json parent/children任务树
```

pi-app再把observed tool events叠加到这条链路上。这样用户看到的不是泛化的“进行中”，而是“哪个Agent正在完成哪条计划、状态如何、证据是什么”。
