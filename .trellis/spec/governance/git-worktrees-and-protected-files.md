# Git, Worktrees, and Protected Files

## Canonical root and worktree boundary

The sole daily local development root is `/workspace/remote2/nimloth`, and its approved daily branch is `dev`. Work directly there by default; a task does not by itself justify a worktree. During migration, stop if the canonical root has not actually cut over to `dev`. Never infer branch, repository, or cutover state from a directory name, and do not modify a worktree that actually holds `main` unless the current human prompt explicitly permits it.

A child worktree is justified only by approved concurrent modifications, experiment exact-source isolation, risky integration/regression isolation, or an explicit human request. Its path is:

```text
/workspace/remote2/nimloth/.worktree/<branch-name-with-slashes-replaced-by-hyphens>
```

Before every repository mutation, bind the command to the intended worktree in the same invocation (explicit tool cwd or `cd "$WT_DIR" && ...`) and verify the command cwd, `git rev-parse --show-toplevel`, actual branch, and `git status --short --branch`. See known error [`E0094`](../../../ai_rules/known_errors/E0094_bind_repo_mutations_to_the_target_worktree.md).

Use the repository-owned [`git-worktree` skill](../../../.agents/skills/git-worktree/SKILL.md) for creation, setup, verification, and cleanup. The canonical root owns the real ignored, machine-specific `.local/` directory; each child uses a verified symlink to `/workspace/remote2/nimloth/.local`. Project-local portable skills remain tracked entities in `.agents/skills/` and must not be replaced by absolute symlinks.

## Child cleanup boundary

Before cleanup, inspect the exact child path's tracked, untracked, and recursively enumerated ignored payload, plus every populated recursive submodule's tracked, untracked, and ignored state. A clean parent Git status does not mean ignored or nested-submodule payload is absent. Stop for any unapproved payload or mismatch. Verify that `.local` is a symlink resolving to the canonical owner, unlink only that symlink, then run ordinary `git worktree remove` for the exact path and verify that both path and registration disappeared.

Git may reject ordinary removal when a worktree contains submodules even after the tree is clean and submodules are deinitialized. That refusal is a stop condition, not permission to retry with `--force`. Without explicit human approval naming the exact verified path, never use `--force`, an automatic force fallback, manual edits under `.git/worktrees`, recursive filesystem deletion, or a repository-wide prune as a substitute for exact cleanup.

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
