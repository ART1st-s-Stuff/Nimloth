# Design — Trellis工作项级可见性

## Architecture

```text
implement.md + task.json
        │
        ▼
Trellis plan/dashboard parser ──────┐
        ▲                           │ versioned JSON
        │ runtime assignment        ▼
Pi Trellis extension          pi-app main reader
        │ tool/run events            │
        └───────────────► pi-app renderer merge
                                      │
                                      ▼
                         task tree + current work item UI
```

## Authority boundaries

- Task identity/tree/lifecycle：`task.json`。
- Plan content/order/done：`implement.md`。
- Current executor/state/evidence：gitignored runtime assignment。
- Actual tool/run activity：Pi/pi-app AppEvent。
- Dashboard JSON：读取时生成的projection，不持久化第二份清单。

## Cross-child contract

Producer child先锁定`trellis-work-item-dashboard-v1`fixture。Consumer只能依赖该versioned JSON，不自行发明Markdown parser或runtime状态。

建议dashboard envelope：

```json
{
  "schemaVersion": 1,
  "root": { "fingerprint": "..." },
  "selectedContext": "pi_...",
  "taskTree": [],
  "assignments": [],
  "issues": []
}
```

Task node包含task lifecycle、parent/children和plan sections；plan item包含explicit/legacy identity、text、checkbox status和stability。Assignment只引用`taskRef/workItemRef`并包含executor、declared state、timestamps、blocker、next action和typed evidence。

## Task review and approval model

Dashboard同时提供非持久化review projection：

- raw `prd.md`、`design.md`、`implement.md`及各自SHA-256；
- task metadata、parent/children、目标仓库/路径、validation、风险、排除项和未决问题；
- 相对上一次review request hashes的artifact diff摘要；
- 一个可选typed approval request。

Approval request至少包含：

```json
{
  "requestId": "...",
  "taskRef": "08-31-...",
  "kind": "implementation",
  "artifactHashes": { "prd.md": "...", "design.md": "...", "implement.md": "..." },
  "scope": ["..."],
  "exclusions": ["commit", "push", "merge"],
  "sessionId": "...",
  "createdAt": "..."
}
```

pi-app必须显示完整请求并把`approve | decline | comment`返回同一session/request。Approval receipt只证明该exact request；artifact hash变化后立即失效。UI不直接扩大scope，也不把“批准规划”解释为“批准实施”或实验启动。

Agent收到有效implementation receipt后才运行`task.py start`；experiment launch、commit、push/merge继续使用独立request。

### 人类审批流程

1. Agent完成/修订规划后发布hash-bound review request，task仍保持`planning`。
2. pi-app在workspace“待审批”列表显示task、parent/children、请求类型和source session。
3. 人类打开review页，依次查看：范围摘要 → PRD → design → plan/checklist → validation/风险/排除项 → 自上次请求后的diff。
4. 底部只显示当前request允许的动作：
   - `评论并继续规划`：返回comment，task保持planning；
   - `拒绝本次实施`：明确decline该request，不删除task；
   - `批准实施`：仅批准当前artifact hashes与列出的scope/exclusions；
   - `稍后处理`：关闭view但request继续pending。
5. 有效批准返回source session；Agent复核hash未变化后运行`task.py start`。若artifact变化，UI和Agent都将receipt标为expired并重新请求。
6. 实验启动、commit、push/merge不会因implementation approval自动出现为已授权，必须各自发布新request。

Review状态是独立projection，例如`draft | awaiting_review | changes_requested | approved_exact | expired`；它不能覆盖`task.json.status`。因此task可同时显示`planning · awaiting implementation review`，避免把“已创建”误解为“可实施”。

## `ask_user_question` correlation

只读源码调查已确认：response按`request.id → source Worker`路由，暂无答案串session证据；但background request与session switch会确定性自动cancel，cancel在RPC层变成`undefined`，足以被外部工具呈现为user declined。Renderer还丢失`toolCallId/session`并只维护一个active/suspended slot。

第一版采用workspace-scoped pending-attention registry：

- identity：root/workspace + sessionFile + request ID + toolCallId；
- background ask进入全局待处理列表，不自动cancel；
- session切换只suspend UI，不完成request；
- “稍后处理”不回传terminal result；
- 只有用户明确拒绝才产生user-declined；background/session-reset/worker-abort保留独立system reason；
- 并发request存入map/queue并逐项恢复；
- response仍由Main现有source map返回原Worker，unknown/late response fail closed。

证据见`research/ask-user-question-cross-session-correlation-2026-08-31.md`。实施必须先添加RED tests，再改变background/session-reset策略和Renderer store模型。

## Plan grammar

- Explicit：`- [ ] [W-001] text`。
- Legacy：无IDcheckbox生成`legacy-<hash(task + heading path + normalized text)>`并标记unstable。
- ID在task内唯一；duplicate/malformed为dashboard issue。
- Heading形成section path；第一版现有flat items全部支持。
- Checkbox是唯一done来源。

## Runtime writer

- 文件：`.trellis/.runtime/execution/<context-key>.json`。
- atomic temp-write + rename。
- root/context/task/item全部验证。
- working/verifying/delegated由heartbeat证明freshness。
- waiting/blocked跨turn保留，直到release/supersede/reference invalidation。
- session shutdown标记live work paused/stale；异常退出靠heartbeat。
- 多executor使用assignments，不覆盖单一全局cursor。

## Pi integration

Pi extension增加显式work-item工具，并把已选item与tool lifecycle、Trellis subagent run关联。`ctx.cwd`是root权威；cache/runtime key包含root+context。

Agent必须显式select item；禁止根据first-unchecked、tool name、assistant text或changed files猜测。

## pi-app integration

- renderer将selected session identity传给state provider；
- task panel提供“执行”与“审查/审批”两个清晰分区；review分区可查看raw artifacts、hash/diff、门禁范围和approval response目标session；
- main以异步child process调用Trellis dashboard JSON，避免阻塞Electron main；
- renderer订阅现有AppEvent，以session/run/toolCallId叠加observed activity；
- dashboard assignment负责语义状态；AppEvent只负责事实性activity；
- runtime缺失时仍显示静态task tree/plan，但标为无live assignment；
- approval channel缺失或correlation不确定时只允许复制/查看，不显示可执行approve按钮。

## Compatibility

- 没有新dashboard subcommand的Trellis项目回退当前静态reader。
- schema version未知时不部分解析。
- legacy items可见但显示unstable。
- 不要求一次性迁移旧task。
- 删除runtime文件不会损坏task/plan。

## Repository isolation

Nimloth直接在canonical `dev`按项目合同规划；pi-app当前`main` dirty且ahead 3，实施应在经批准的独立worktree/branch中进行。两边各自保留一名writer，使用fixture而非共享未提交文件耦合。

## Rollback

- Producer：移除extension tool/runtime writer后，task/plan不受影响。
- Consumer：feature/fallback检测不到dashboard即恢复静态Trellis panel。
- Runtime projection可安全删除；不得自动修改checkbox或TaskTree。
