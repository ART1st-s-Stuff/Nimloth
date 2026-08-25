# Platform Integration Contract

## 1. Scope / Trigger

Apply this contract when changing Trellis adapters under `.pi/`, `.claude/`, `.codex/`, shared `.agents/skills/`, or context-loading behavior. Project rules live in shared specs and project-local skills; platform files are thin adapters and must resolve the same active worktree and Trellis task.

Source evidence: the [Pi Trellis adapter](../../../.pi/extensions/trellis/index.ts), [failure investigation](../../tasks/00-bootstrap-guidelines/research/pi-desktop-context-root.md), and [workflow platform boundary](../../workflow.md#platform-consistency-and-upgrade-boundary).

## 2. Signatures

Pi event/tool callbacks receive a context with the session working directory:

```ts
interface PiExtensionContext {
  cwd?: string;
  // model, sessionManager, ui, ...
}

const resolveContextRoot = (ctx?: PiExtensionContext) =>
  findRoot(ctx?.cwd ?? bootstrapRoot);
```

The Pi tool and context-producing events must call `resolveContextRoot(ctx)` before reading:

- `.pi/agents/<agent>.md`;
- `.trellis/workflow.md`;
- `.trellis/tasks/` and session active-task pointers;
- task JSONL/spec/research context.

## 3. Contracts

- `ctx.cwd` is authoritative when present. Desktop NodeService `process.cwd()` may be the desktop application directory and must not select project files.
- `process.cwd()` is a bootstrap fallback only for contexts where Pi supplies no callback context.
- Every cache containing workflow/task content is isolated by both resolved project root and session/context key. In the current adapter this covers `turnCache`, `startupCtxCache`, `taskCtxSnapshot`, `lastSentTaskCtx`, and `lastSentRuntimeCtx`; the same session ID in another project cannot reuse any of them.
- Agent discovery and prompt construction use the same resolved root. Discovery fails visibly when `.pi/agents/<name>.md` is absent there; it must not search or build a prompt from another worktree.
- Claude Code and Codex adapters load the same `.trellis/workflow.md`, task artifacts, task JSONL, and repository-owned `.agents/skills/` semantics.
- Codex repository configuration does not silently modify the user's global hook setting; agents report the `features.hooks = true` and `/hooks` requirements.

## 4. Validation and Error Matrix

| Condition | Required behavior |
|---|---|
| `ctx.cwd` points to the active Nimloth worktree | Resolve that worktree and find its `.pi/agents/` and `.trellis/` |
| Desktop process cwd is `/workspace/pi-app` | Ignore it when callback `ctx.cwd` exists |
| callback context has no cwd | Use the factory-time bootstrap root; missing project files fail visibly |
| project/session changes after reload | Recompute from the new callback context; cache key changes with root |
| requested agent definition is absent in the active project | Return `No definition found`; do not fall back to another project |
| active-task pointer is missing, unreadable, or escapes the resolved root | Inject `No active Trellis task found`; do not invent or follow the task |
| task JSONL is missing, malformed, or references a missing file | Runtime omits unreadable curated blocks; `task.py validate` must report malformed rows/missing references before release, and no adapter may synthesize replacement context |

## 5. Good / Base / Bad Cases

- **Good:** Pi Desktop callback has the worktree `ctx.cwd`; sub-agent discovery and task injection use that worktree even though the NodeService cwd is the desktop app.
- **Base:** Pi CLI starts inside the worktree; `ctx.cwd` and process cwd agree and resolve the same root.
- **Bad:** an extension captures `findRoot(process.cwd())` once and uses it for every project/session, causing missing agents or cross-project context leakage.

## 6. Tests Required

When the Pi adapter changes:

1. Parse/bundle `.pi/extensions/trellis/index.ts` as TypeScript.
2. Run a harness with foreign `process.cwd()` and worktree callback `ctx.cwd`; assert agent lookup and context use the worktree.
3. Use two different roots with the same context key; assert cached workflow/task content does not cross roots.
4. After `/reload`, dispatch a safe `trellis-implement` or `trellis-check` probe and confirm it resolves `.pi/agents/` in the active worktree.
5. Run `trellis platforms` and `trellis update --dry-run`; record intentional adapter divergence.

## 7. Wrong vs Correct

### Wrong

```ts
const root = findRoot(process.cwd());
// Every callback and tool permanently reads from root.
```

### Correct

```ts
const bootstrapRoot = findRoot(process.cwd());
const resolveContextRoot = (ctx?: PiExtensionContext) =>
  findRoot(ctx?.cwd ?? bootstrapRoot);

async function execute(..., ctx?: PiExtensionContext) {
  const root = resolveContextRoot(ctx);
  // Discover agents and task context under this root.
}
```

The active callback context chooses the project; the bootstrap value only keeps non-callback startup behavior defined.
