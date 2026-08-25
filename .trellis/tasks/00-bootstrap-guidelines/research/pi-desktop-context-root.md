# Pi Desktop Trellis Context-Root Compatibility

## Confirmed failure

The generated project agent definitions exist:

- `.pi/agents/trellis-implement.md`
- `.pi/agents/trellis-check.md`
- `.pi/agents/trellis-research.md`

However, `trellis_subagent` reported `No definition found for: trellis-implement` after the desktop app was restarted.

## Root cause

The Pi Desktop NodeService processes run with `/workspace/pi-app` as their OS process working directory, while the active Pi session uses `/workspace/remote2/nimloth-chore-trellis-init`.

The generated Trellis extension captured its root at factory load with:

```ts
const root = findRoot(process.cwd());
```

It therefore searched `/workspace/pi-app/.pi/agents/` instead of the session project's `.pi/agents/`.

Pi's extension API documents `ctx.cwd` as the current working directory and uses it in project-local extension examples. Tool execution and session/agent event handlers receive this context.

Evidence:

- Pi docs: `/home/user/.local/share/npm/lib/node_modules/@earendil-works/pi-coding-agent/docs/extensions.md`, `ctx.cwd` section.
- Generated extension: `.pi/extensions/trellis/index.ts`, extension factory and `isTrellisAgent` call.
- Runtime process inspection: Pi Desktop NodeService cwd `/workspace/pi-app`; tool/session cwd is the Nimloth worktree.

## Local compatibility design

- Keep `process.cwd()` only as a bootstrap fallback.
- Resolve the active root from `ctx.cwd` for each tool call and context-producing event.
- Include the active root in context cache keys so switching projects/sessions cannot reuse another project's cached Trellis context.
- After editing, run `/reload`; verify `trellis_subagent` can resolve and launch `trellis-implement` from the current worktree.

This is an intentional local edit to an upstream-managed generated file and must be reviewed during future `trellis update` conflicts.
