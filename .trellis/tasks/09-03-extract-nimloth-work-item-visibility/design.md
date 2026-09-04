# 技术设计 — 独立Nimloth work-item可见性

## 1. 组件

```text
Nimloth .pi/extensions/nimloth-work-items/
├─ core.mjs          # parser、identity、runtime、projection纯逻辑
├─ index.ts          # Pi动态tool与lifecycle adapter
└─ projection.mjs    # 只读CLI

pi-app Main projection reader + IPC
pi-app stable Trellis Activity view
```

上游Trellis extension/scripts保持不变。

## 2. Parser与identity

Parser只读取active task的`implement.md`：维护ATX heading stack，识别普通`- [ ]`/`- [x]`checkbox。Identity输入为：

```text
schema-version + canonical task ref + normalized heading path + normalized checkbox text
```

使用完整SHA-256作为内部ref。Line number只用于展示/issue，不进入identity。相同heading path和normalized text重复时两项均标记ambiguous，不可选择。Plan fingerprint由有序heading/item结构生成；文字变化产生新identity/fingerprint。

## 3. 激活状态机

`index.ts`启动时注册`nimloth_work_item`，随后立即从active set移除。`session_start`与`before_agent_start`调用`reconcileTool(ctx)`：

```text
enabled = active task exists
       && task.status == in_progress
       && design.md exists
       && implement.md exists
       && parser valid
       && pending items > 0
```

启用时在`pi.getActiveTools()`上只添加自身；停用时只过滤自身，保留其他工具。Tool不设置`promptSnippet/promptGuidelines`，避免额外常驻prompt。状态改变通过`pi.setActiveTools()`在下一model request生效。

不创建auto-discovered project skill或伪动态skill。`select/update/evidence/release`的简短指导只写在动态active tool的description/result中，因此ineligible session没有该能力的skill metadata。

## 4. Runtime schema

路径：

```text
<resolved .local>/nimloth-work-items/v1/<worktree-root-sha256>/<context-key>.json
```

Schema包含：`schemaVersion/rootFingerprint/contextKey/sessionId/taskRef/planFingerprint/assignment`。Assignment包含itemRef、executor、state、since/updatedAt、nullable blocker/nextAction、最多8条evidence。字符串和文件总bytes有上限。

写入使用同目录temp + fsync + rename；读取执行schema、root/context/task/plan/item校验。Plan变化、checked item、task切换或completed使assignment stale并从projection中标记；不改task artifact。

## 5. Tool transitions

- `select`：无current assignment或已release/stale；item必须pending且唯一。
- `update`：要求current valid assignment；state变化遵守显式transition table。
- `evidence`：要求valid assignment；kind allowlist，ref/summary bounded，FIFO最多8条。
- `release`：写入releasedAt并清空active assignment语义。

`blocked`要求blocker；其他state不得自动编造blocker。任何非法输入返回结构化错误，不写文件。

## 6. Projection CLI

`projection.mjs`从自身cwd解析project root和context key，读取上游active-task pointer、task.json、implement.md及extension runtime，输出：

```json
{
  "schemaVersion": 1,
  "root": {"path": "...", "fingerprint": "..."},
  "task": {"ref": "...", "status": "...", "planFingerprint": "..."},
  "items": [{"ref": "...", "text": "...", "checked": false, "headingPath": []}],
  "current": {"itemRef": "...", "text": "...", "executor": {}, "state": "working"},
  "issues": []
}
```

CLI只读、stdout有size limit、错误也输出versioned issue JSON。Runtime不保存item text；projection即时join。task status不是`in_progress`时强制`current=null`并输出`inactive-task` issue。

Projection v1在producer侧执行consumer同值边界：item text 2048、heading segment 512、heading depth 32、items 2000、issues 100、duplicate lines 100、task ref 512、status 64、blocker/next/evidence ref与summary 512、executor 128、session 256。超限plan字段截成schema合法投影并增加bounded issue；不得把consumer-invalid JSON交给stdout size fallback。

## 7. pi-app consumer

Main新增独立reader：复用registered worktree验证，选择目标worktree后以该目录为cwd运行project-owned projection CLI，设置context key，限制timeout/stdout，验证schema/root fingerprint。Renderer不能提交CLI路径。

Stable `TrellisSidePanel`的Activity child逐步切换为：

- source selector；
- selected session/context identity；
- current task/item、executor/state/elapsed；
- blocker、next action、bounded evidence；
- issues和静态task browser链接。

Projection失败显示“实时步骤不可用”，Documents/task browser及Trellis workflow保持可用。Consumer不import旧`TrellisDashboardV1`。

## 8. Tests

### Nimloth

- Markdown headings/checkbox、normalization、duplicate ambiguity、text-change invalidation、checked state；
- activation matrix与active-tool preservation；
- select/update/evidence/release transition、bounds、atomic write、corrupt/stale/root/context/task rejection；
- projection missing/stale/orphan/checked conflict、planning/completed inactive suppression、全部projection-v1边界及无CoT/plan持久化；
- upstream 0.6.16 manifest保持GREEN。

### pi-app

- command timeout/overflow/stderr/schema/root/source校验；
- registered worktree source与WSL路径边界；
- Activity rendering、source/session切换、stale response和降级；
- absence test证明不依赖dashboard/approval/TaskTree types。

## 9. 分支与回滚

Nimloth和pi-app分别从当前parent integration HEAD创建同名task branch/worktree，一次只允许每个cwd一个writer。先完成Nimloth producer/projection，再完成pi-app consumer。回滚删除独立extension和consumer；`.local/nimloth-work-items/`是非权威ignored状态，可停止读取而无需迁移。
