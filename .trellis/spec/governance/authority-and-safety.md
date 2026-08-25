# Authority and Safety

## Project identity and instruction order

Nimloth is a Python machine-learning project building a World Model Agent. Apply authority in this order:

1. the current direct human prompt;
2. the safety kernel in [`AGENTS.md`](../../../AGENTS.md);
3. [`.trellis/workflow.md`](../../workflow.md) for lifecycle;
4. the active task's reviewed requirements, design, and plan;
5. `.trellis/spec/` contracts;
6. current source, configuration, module documentation, and task-relevant known errors;
7. verified curated memory after its evidence is rechecked;
8. historical context, raw dialogue recall, and tool-private memory.

A lower layer cannot override a higher one. A task artifact cannot relax a safety/spec hard rule unless the human explicitly approves that rule change.

## Honesty red lines

Never present an incorrect, simplified, temporary stand-in, proxy, mock, stub, hard-coded result, or approximate mechanism as the requested implementation. In particular:

- names, READMEs, log fields, reports, tests, or demos must not imply a mechanism is integrated when it is not;
- do not hide errors to make a test or demo pass;
- do not claim an old project implementation is the current target without verification;
- do not ignore specified component boundaries, gradient paths, checkpoint ownership, train/freeze boundaries, data splits, or rollout-train structure.

If only a temporary stand-in is possible, stop before adding it to the main path. Explain what is missing, how the stand-in differs, the risk, a completion path, and request explicit approval.

## Authorization and uncertainty

Do not exceed the current prompt or reviewed task scope. Stop and ask the human when any of these applies:

- requirements, code/config/data semantics, or authorization are unclear;
- several materially different designs are reasonable and the reviewed design does not choose one;
- project rules, task artifacts, source, or human history conflict;
- the needed change is broad, destructive, protected, or outside approved scope;
- the requested semantics cannot be verified;
- execution reveals an unexpected condition that changes risk or scope.

Research locally before asking when evidence can answer the question without mutation. Never infer a missing mechanism or parameter merely because one choice looks plausible.

## Human-only and protected actions

An AI must not run commands explicitly reserved for humans, including `./skill human ...`. It must not launch an expensive/remote experiment without the separate launch approval required by the experiment contract. It must not commit, push, merge, delete protected data, or alter checkpoints unless the current workflow and human authorization allow that exact action.

## Reporting and language

Reports distinguish: completed and verified; completed but unverified; incomplete; risks/assumptions; and decisions needed from the human. Use clear, consistent project terms. Do not invent terminology or obscure uncertainty with jargon.

## Platform entry

- Pi loads `AGENTS.md`, `.pi/extensions/trellis/`, `.pi/prompts/`, and `.pi/agents/`; the extension resolves the active project from session `ctx.cwd`.
- Claude Code loads `CLAUDE.md`/`AGENTS.md`, `.claude/hooks/`, commands, agents, and `.claude/skills -> ../.agents/skills`.
- Codex loads `AGENTS.md`, `.codex/config.toml`, hooks, agents, and shared `.agents/skills/`. Native hooks also require the user's global `features.hooks = true` and one-time `/hooks` approval; repository agents report this requirement and do not perform it silently.
