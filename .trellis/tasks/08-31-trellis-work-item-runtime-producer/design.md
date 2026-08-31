# Design — Trellis work-item runtime producer

## Components

1. `work_items` parser：task tree + implement.md grammar。
2. Dashboard read model：versioned JSON projection。
3. Review/approval projection：raw artifacts、hash/diff和typed request/receipt validation。
4. Runtime assignment store：per-root/per-context atomic files。
5. Pi extension tool/events：cursor writer与heartbeat。
6. Workflow/spec/skills：声明更新时机。

## Grammar

```markdown
- [ ] [W-001] Pending item
- [x] [W-002] Done item
```

ID task-local唯一。无ID项生成`legacy-<hash(task+heading+normalized-text)>`并标`stable=false`。不自动写回旧Markdown。

## Dashboard

由Trellis本地CLI一次性返回：

- selected context/current task；
- root task trees与active/archived child status；
- per-task sections/items；
- fresh/stale assignments；
- parse/runtime issues。

Consumer不解析Markdown。最终subcommand名称在implementation approval前锁定。

## Review/approval projection

- Read raw planning artifacts without synthesizing replacement content.
- Compute SHA-256 per artifact and a deterministic review-set hash.
- Return artifact presence, sections, validation commands, scope/exclusions and changes since the request's bound hashes.
- Validate typed request/receipt identity using root + task + session + request ID.
- Receipt is evidence for one exact gate only; it never directly editscheckbox or broadens authorization.
- Approval state is not inferred fromtask lifecycle `planning/in_progress`.

## Runtime

- path：`.trellis/.runtime/execution/<context-key>.json`；
- temp file + fsync/rename按平台能力实现；
- schema version拒绝未知major；
- assignment引用`taskRef/workItemRef`；
- evidence只允许typed reference和短summary；
- 不保存完整tool args/output/CoT。

## State machine

```text
working ↔ verifying
working → delegated | waiting_human | waiting_external | blocked | failed
delegated/waiting/blocked/failed → working
working/verifying/delegated → stale (heartbeat/session failure)
```

Plan done不属于runtime transition：先更新checkbox，再release assignment。

## Extension ownership

显式work-item tool负责语义声明；tool lifecycle只更新observed activity/heartbeat。Subagent dispatch需要明确work-item reference，不从prompt文本猜测。

## Compatibility

- active pointer schema保持不变；
- no implement.md返回empty plan而非错误；
- legacy item可展示但cursor失配后orphan；
- runtime文件删除即回退静态状态；
- first release只支持Pi producer，dashboard schema平台中立。

## Protected boundaries

`.trellis/.template-hashes.json`、runtime session pointers和memory JSONL不得手改。对Trellis-managed本地文件的intentional divergence必须记录并运行`trellis update --dry-run`。
