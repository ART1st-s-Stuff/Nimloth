# E0136 — preflight tests must not dirty the production worktree

## Error

ID189 Job `525905` failed before formal output because server-side pytest preflight imported `external/le-wm` in the production worktree and created `external/le-wm/__pycache__/module.cpython-312.pyc`. The runner's clean-tree gate then correctly rejected the dirty submodule.

## Cause

Tests were run directly in the production worktree without `PYTHONDONTWRITEBYTECODE=1` and without isolating pytest caches.

## Correct practice

Run tests in a separate test worktree. If a production-worktree import probe is unavoidable, set `PYTHONDONTWRITEBYTECODE=1`, disable/cache pytest outside the repo, and re-check every parent/submodule with `git status --porcelain --untracked-files=all` after the probe and immediately before submission.

## Evidence

- Job control artifact: `outputs/experiments/training/rl/slurm/id185-ray-525905-bc120/progress.md`.
- Gate: `experiments/training/rl/run_vagen_k4_id189_source20_base_common120.sh` clean-tree loop.
