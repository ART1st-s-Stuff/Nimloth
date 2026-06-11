---
name: memory
description: Lightweight human-approved project memory management. Use when creating, searching, inspecting, correcting, or upvoting durable project memory.
---

# memory skill

Use this skill when you need to create, search, inspect, correct, or upvote durable project memory.

## Purpose

The memory system is a lightweight, human-approved, searchable index of durable project knowledge.

Memory should be:

- short;
- useful for future AI agents;
- mostly an index to existing code/docs;
- backed by file-segment evidence;
- not a task log;
- not a long explanation.

## Commands

Use the repository skill wrapper commands:

```bash
./skill memory add <title> <content>
./skill memory set <id> <field=value> [field=value ...]
./skill memory search <keyword-regex> [--field all|title|content|evidence.filename|tags] [--tag TAG] [--level LEVEL] [--include-archived]
./skill memory get <id>
./skill memory upvote <id>
./skill memory human-verify <id>
```

Human-only approval command:

```bash
./skill human memory-approve
```

AI agents must never run `./skill human ...` commands.

## Data model

Each memory has `id`, `title`, `content`, `evidence`, `tags`, `level`, timestamps, and optional `human_suggestions`.

- `evidence`: JSON list of file segment references: `[{"filename":"...","line_start":1,"total_lines":10}]`
- `tags`: JSON list of strings
- `level`: `pending-human-verification`, `verified`, or `archived`

## Rules for AI agents

1. Do not manually edit `.memory/memories.jsonl`.
2. Do not create long memories. Prefer concise index-like entries.
3. Do not store transient progress, TODOs, or task logs in memory.
4. Evidence must be a file-segment reference, not free text.
5. AI-created memories start as `pending-human-verification`.
6. AI must not claim a memory is human-approved unless its level is `verified`.
7. If a pending memory contains `human_suggestions`, the AI must follow those suggestions by editing the memory with `./skill memory set ...` before asking for approval again.
8. Before relying on a memory, run `./skill memory get <id>`, inspect the evidence file segment, and verify that the memory still matches the referenced file.
9. Only after verification and confirming it helped the current task, run `./skill memory upvote <id>`.
10. If a memory is wrong, correct it with `./skill memory set ...`; if obsolete, let stale archive rules handle it or ask the human.
11. Human approval is done with `./skill human memory-approve`, not by AI.

## Human approval flow

AI may submit a pending memory:

```bash
./skill memory add "Rules live in ai_rules" "Detailed AI rules are under ai_rules; AGENTS.md is the short entry."
./skill memory set M0001 'evidence=[{"filename":"AGENTS.md","line_start":1,"total_lines":20}]' 'tags=["rules","startup"]'
./skill memory human-verify M0001
```

Human reviews pending memories:

```bash
./skill human memory-approve
```

Approved memories become `verified`; rejected memories are deleted. If the human types anything other than `a/r/s/q`, that text is stored as `human_suggestions`, the memory remains pending, and the AI must revise the memory according to the suggestion. Suggestions are removed automatically when the memory is approved.

## Stale/archive policy

The CLI performs lazy stale cleanup. Verified memories are archived when either condition is true:

- not trigger-verified for 7 days;
- not upvoted/used for 14 days.

`upvote` means: the agent first verified the evidence, then found the memory useful for the current task.
