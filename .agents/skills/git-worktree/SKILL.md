---
name: git-worktree
description: >-
  Creates and validates Nimloth git worktrees, enforces branch-to-path naming,
  and shares machine-specific .local state. Use before local repository changes,
  when creating a branch worktree, or when a worktree lacks .local.
---

# Git Worktree

## Trigger

Use before local repository mutation, worktree creation, or repair of missing shared `.local/` state.

## Authority

Read:

- [`AGENTS.md`](../../../AGENTS.md) for the direct safety kernel;
- [Git/worktree/protected-file spec](../../../.trellis/spec/governance/git-worktrees-and-protected-files.md);
- [authority and safety](../../../.trellis/spec/governance/authority-and-safety.md).

Rules:

- local path is `../nimloth-<branch-name>` with `/` replaced by `-`;
- never edit the worktree holding `main` unless the human prompt explicitly permits it;
- verify actual path and branch; do not infer branch from the directory name;
- every repository-mutating command uses explicit tool cwd or `cd "$WT_DIR" && ...` in that same invocation;
- portable project skills under `.agents/skills/` are versioned entities, not symlinks to another clone/worktree;
- only machine-specific state remains under ignored `.local/`.

## Create

Run from the repository/worktree that owns shared local state:

```bash
BRANCH="feat/my-feature"
WT_DIR="../nimloth-$(printf '%s' "$BRANCH" | tr '/' '-')"

pwd
git status --short --branch
git worktree add -b "$BRANCH" "$WT_DIR" <approved-start-point>
```

If the branch already exists, omit `-b` and pass the existing branch. Do not guess the start point or switch important branches when strategy is unclear.

## Set up shared local state

```bash
MAIN="../nimloth"  # replace with the actual shared-local-state worktree
cd "$WT_DIR" && \
  ln -sfn "$MAIN/.local" .local
```

Do not symlink `git-worktree`, `slurm`, or other portable repository skills: they arrive through Git. If a future skill is intentionally machine-specific, place it under `.local/` rather than adding an absolute `.agents/skills/` symlink.

Verify in the same target-bound command:

```bash
cd "$WT_DIR" && \
  pwd && \
  git branch --show-current && \
  git status --short --branch && \
  test -f .local/SERVER.md && \
  test -f .agents/skills/git-worktree/SKILL.md && \
  test -f .agents/skills/slurm/SKILL.md
```

## Remote worktrees

Read `.local/SERVER.md` and the [`slurm` skill](../slurm/SKILL.md). Keep remote code at the approved commit and never edit production code directly on the server.

## Cleanup

Before removal, verify the target has no uncommitted or unpushed important work. Then run from the controlling worktree:

```bash
git worktree remove "$WT_DIR"
git worktree prune
```

Never use `--force` without explicit approval for the exact verified paths.
