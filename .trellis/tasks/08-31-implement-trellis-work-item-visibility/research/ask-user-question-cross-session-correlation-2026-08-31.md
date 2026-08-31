# `ask_user_question`跨session误报declined调查

日期：2026-08-31

范围：只读源码调查`/workspace/pi-app`与安装的Pi SDK；未启动应用、未修改pi-app或运行写入型测试。

Subagent原始输出：`/home/user/.pi/agent/sessions/--workspace-remote2-nimloth--/subagent-artifacts/outputs/2852eb34-4ca8-4b59-ad6b-1e44e046ff01/context.md`

## 1. 结论

不能证明“另一个session的答案被路由给当前request”。Main process当前以随机`request.id`和记录的source Worker路由response/cancel，这层已有隔离。

但存在两条源码确定的自动取消路径，足以解释用户看到另一个session中的问题变成`User declined to answer questions`：

1. 切换session会对旧session已展示/挂起的问卷发送`session-reset` cancel；
2. background Worker发起非notification UI request时，Main不展示而直接发送`background-session` cancel。

所有cancel在questionnaire RPC层都变成`undefined`。当前可读源码中没有精确字符串`User declined to answer questions`，因此只能判断：外部`ask_user_question`工具很可能把`undefined`格式化为declined，但精确字符串生产者尚未定位，不能宣称已完全确认。

此外存在两个独立缺口：

- Renderer只有一个全局`activePending`和一个`suspended`，并发前台问卷会覆盖可见状态，产生liveness风险；
- Renderer转换request时丢弃`toolCallId`和session identity，timeline绑定退化为“最近一个同名live tool”的启发式，可能错绑卡片，但提交response仍按`request.id`路由。

## 2. 已证实数据流

1. `src/worker/worker-runtime.ts:165-175`装饰`ask_user_question`扩展工具。
2. `questionnaire-tool-decorator.ts:38-52`使用原始`toolCallId`创建RPC UI并执行真实工具。
3. `questionnaire-rpc-ui.ts:39-67`缓存问卷Promise；cancelled或缺少答案使`ui.select/input`返回`undefined`。
4. `desktop-ui-bridge.ts:299-312`创建随机`request.id`并携带可选`toolCallId`。
5. `worker-manager-pool.ts:198-217`只转发foreground session请求；background request立即cancel。
6. `worker-manager.ts:897-920`以`Map<request.id, WorkerSlot>`把answer/cancel路由给原Worker，不fallback到当前foreground。
7. `extension-ui-channel.ts:25-30`把request转为Renderer pending时丢弃`toolCallId`和session字段。
8. `extension-ui-store.ts:23-31,59-96`只维护一个`activePending`和一个`suspended`。
9. `extension-ui-host.tsx:84-97`按request id提交；Main IPC再路由给Worker bridge的`pending.get(id)`。
10. `worker-session-events.ts:305-323`和`apply-app-event-tool.ts:7-14,75-102`仍以原始`toolCallId`更新timeline。

## 3. Declined候选来源

### 用户显式取消

`ExtensionUIHost.cancelWorker`返回`{ cancelled: true }`。

### Session切换

`open-session.ts:23-25`切换前调用`resetForSessionContext()`；`extension-ui-store.ts:91-95`向旧request发送`session-reset` cancel。

这是确定行为，不是偶发竞态。若工具把`undefined`呈现为declined，用户仅切换会话也会看到“declined”。

### Background session

`worker-manager-pool.ts:203-210`向background request发送`background-session` cancel。提交`9f794e8 fix(extension-ui): cancel background dialogs`明确引入/维护了这一策略。

这是用户所述“别的会话里显示declined”的最直接源码解释：后台问题从未显示，却被系统取消并可能被工具文案解释为用户拒绝。

### Abort/worker lifecycle

`desktop-ui-bridge.ts:77-106,299-312`在AbortSignal已经或随后abort时resolve `{ cancelled: true, answers: [] }`，同样进入`undefined`路径。

## 4. 已有隔离与残余风险

### 已有保护

- response按随机`request.id`返回source Worker；
- stale/unknown source不fallback到当前session；
- worker stop/compaction只清理该slot请求；
- tool AppEvent带`sessionFile`并按session cache；
- stale A response不会被发送到B。

### 高风险

- session switch自动取消旧问卷；
- background ask自动取消且用户根本看不到问题。

### 中风险

- 并发request覆盖全局single-slot UI；被覆盖request可能一直等待到abort；
- request到Renderer后丢失`toolCallId/session`，问卷卡片可能绑定错误timeline tool。

### 低风险/未验证

- submit与session reset/cancel竞态使用first-arrival语义，但缺少用户可见诊断和测试。

## 5. 对审批UI的设计影响

Task implementation approval不能建立在“后台问题自动cancel”的transport上，否则Agent新建task后只要用户切换session，审批请求就可能变成declined。

第一版应采用workspace-scoped pending-attention registry：

- key：`root/workspace + sessionFile + request.id + toolCallId`；
- background question进入全局“待处理问题/审批”列表，不自动cancel；
- session switch只隐藏/挂起dialog，不完成request；
- 用户可“稍后处理”而不向Worker返回cancel；
- 只有明确的“拒绝/取消请求”才生成user-declined；
- worker terminate/abort/background policy等系统取消必须保留reason，不能伪装成user decline；
- 并发request使用map/queue，不是single active slot；
- 点击pending item可切到source session或原地审查，但response始终返回source request；
- task approval额外绑定taskRef、approval kind和artifact hashes。

## 6. 建议RED tests

1. **Session switch**：A显示问卷后切B；A request保持pending/suspended，不产生user-declined，B不会收到response。
2. **Background ask**：B后台发问；全局attention list出现B request，不自动cancel；用户回答后只回B。
3. **Concurrent asks**：同一或不同session连续两个request；两者均可见/可恢复，reset不会遗留隐藏Promise。
4. **Full correlation**：Renderer pending保留`requestId/toolCallId/sessionFile/root`，并发同名工具精确绑定对应timeline row。
5. **Explicit decline vs system cancel**：用户拒绝、session switch、worker abort、background和dialog hide产生不同terminal/nonterminal语义。
6. **Submit/cancel race**：一个request只有一个终态，late response可诊断且不会发送给其他session。
7. **Task approval**：response绑定exact artifact hashes；plan变化后旧approval失效。
8. **Cross-session isolation**：A stale response永不发送B，并验证A的用户可见reason准确。

## 7. 现有测试与缺口

已有：

- `src/main/__tests__/worker-manager-extension-ui.test.ts:63-173`：source routing、stale source、cancel、background auto-cancel、worker exit；
- `src/renderer/src/stores/__tests__/extension-ui-store.test.ts:25-45`：session reset取消一个suspended dialog；
- `src/renderer/src/lib/extension-ui-channel.test.ts:24-51`：按request id dismiss suspended dialog。

缺少：

- `questionnaire-rpc-ui`与`desktop-ui-bridge`单元测试；
- 真`ask_user_question`的`undefined → final tool output`测试；
- background/session-switch的最终用户文案测试；
- 并发queue/map、完整correlation和submit/cancel race测试。

## 8. 事实、判断与证据缺口

### 事实

- background和session switch会自动cancel问卷；
- cancel变为RPC UI的`undefined`；
- response本身按source request id隔离；
- Renderer丢失toolCallId/session且使用single-slot store。

### 判断

用户看到的declined更可能是系统取消被外部工具呈现为用户拒绝，而不是跨session答案串线。

### 证据缺口

- 尚未定位精确字符串生产者；
- 尚未启动pi-app做端到端复现；
- 尚未运行建议RED tests；
- 因而当前只能确认取消机制问题与correlation设计缺口，不能声称完整bug已复现。
