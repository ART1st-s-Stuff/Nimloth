# Integration inventory — 2026-08-25

## Root histories

- local dev: `44af540e76e81222591b0b320240a41864369d83`
- origin/dev: `0111d0de609e2cf91023638de53f7951bf0109e7`
- Trellis baseline: `d92b76a48da3f227ca4fdd5ca56c805d17e32f04`
- ID185 source: `7d87a14e97b35cd5cb21f7fc1e53b80afbb1877f`
- ID185 merge base with dev: `0111d0de609e2cf91023638de53f7951bf0109e7`
- ID185 source commits after base: 358
- ID185 merge commits after base: 0
- path overlap between Trellis and ID185: only `AI_branch_progress.md`

## Submodules

Trellis/dev baseline:

- RCDM `71daaf10a73bb2012864f0827c68d209fc92b0a5`
- VAGEN legacy `192c35a91f3941b72d5e1272af6603ef7a7d93e0`
- le-wm `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`

ID185:

- RCDM unchanged
- VAGEN upstream `9f1e89eb8c9839a406b6e62aa75703494a79e5b5`
- le-wm unchanged
- nested VERL at VAGEN tip: `494f264494b2525f2c13595f63ac4912963e6d2f`

Remote checks:

- VAGEN remote `refs/heads/nimloth/upstream-joint-policy-scaffold` resolves to `9f1e89e`.
- VERL remote `refs/heads/nimloth/upstream-joint-policy-scaffold` resolves to `494f264`.
- VAGEN `.gitmodules` branch hint for VERL still names `nimloth/vagen-lite-async-policy-state-capture`, but the gitlink is exact and fetchable; integration preserves the source commit rather than silently editing this contract.

## Dirty state to preserve

Original dev worktree:

- `external/le-wm/__pycache__/` untracked inside submodule;
- `.pi/task-tree/history.jsonl`, `tasks.json`, `tasks.json.bak` untracked.

No source integration worktree has uncommitted production changes.

## Local validation capability

The available local Python installations do not provide pytest; static Python and shell validation remain available. Historical feature-branch experiment/test records are evidence of the source branch, not post-Trellis integration test evidence.
