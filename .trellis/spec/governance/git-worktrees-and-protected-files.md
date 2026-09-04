# Git, Worktrees, and Protected Files

## Task branches and the canonical slot

`dev` is Nimloth's integration branch and the base for new task branches; it is not the default implementation surface. Every task that changes a repository records and uses its own `task/*`, `feat/*`, `fix/*`, or `exp/*` branch. Different tasks must not share one implementation branch, and a task is not complete while its accepted changes exist only as uncommitted state on `dev`.

The canonical root `/workspace/remote2/nimloth` provides one primary task slot. When no other task owns that slot, one selected task may work there on its dedicated branch. Switching the canonical root from `dev` requires a clean tracked/untracked state and the exact approved task branch/base; if migration-era or concurrent dirty state is present, stop rather than stash, reset, clean, or overwrite it.

Additional parallel tasks use independent child worktrees. Experiment exact-source runs, risky integration/regression work, or explicit human review targets use the same isolation when needed. Their path is:

```text
/workspace/remote2/nimloth/.worktree/<branch-name-with-slashes-replaced-by-hyphens>
```

A worktree is an isolated execution directory, not a second task authority. Task artifacts remain authoritative, while review tools may read another registered worktree without switching the active workspace, session, worker cwd, or Git checkout.

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
