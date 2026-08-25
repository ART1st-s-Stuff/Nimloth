# Git, Worktrees, and Protected Files

## Worktree boundary

Before mutation, run `git status --short --branch` and verify the actual branch; never infer it from a directory name. Local work belongs in `../nimloth-<branch-name>`, replacing `/` in the branch with `-`. Do not modify the worktree that holds `main` unless the current human prompt explicitly permits it.

Every repository-mutating shell command must be bound to the intended worktree in that same invocation (explicit tool cwd or `cd "$WT_DIR" && ...`). See known error [`E0094`](../../../ai_rules/known_errors/E0094_bind_repo_mutations_to_the_target_worktree.md).

Use the repository-owned [`git-worktree` skill](../../../.agents/skills/git-worktree/SKILL.md) for creation and setup. `.local/` is shared, ignored, machine-specific state; project-local portable skills are versioned in `.agents/skills/` and must not be replaced by absolute symlinks.

## Change discipline

- Understand adjacent source, tests, config, and module README before editing.
- Prefer small, verifiable, reversible changes; do not refactor unrelated code.
- Check for an existing implementation before adding new code. Reuse only when it keeps the design readable.
- New `src/` modules require a README index; update the owning module README when boundaries change.
- Keep Python clear and modular, add useful type hints, prefer configuration to hard-coding, and use concise Chinese comments for the reason behind complex logic.
- Do not hide errors or weaken verification.

## Protected content

Do not modify without explicit human approval:

- `ai_notes/archive/`;
- `qc_*.md`;
- files marked human-authored/read-only unless the approved scope names them;
- large data, model weights, checkpoints, and experiment output;
- memory JSONL files directly;
- `.trellis/.template-hashes.json` or runtime session pointers manually.

If a protected or unrecognized file appears necessary, stop, explain why, and ask before changing it.

## Git and review

- Preserve unrelated user/concurrent dirty changes.
- Do not create, switch, merge, or rewrite important branches when strategy is unclear.
- Use semi-linear merge policy.
- Before a work commit, present the full scope, validation evidence, logical commit groups, and unrecognized dirty files for one-shot human approval.
- Do not amend or push through the Trellis work flow. Automatic task-archive/workspace bookkeeping commits may run only after work commits and finish-work review.
- Never leave completion claims unsupported by the current working-tree diff and executed checks.
