# E0138 — do not deflate full float state browser archives

## Error

ID189 normal 4×2 retry2 used serial `np.savez_compressed` for every full per-turn state archive. After the first 40 rollouts completed, the single TaskRunner spent more than 2h21m CPU, reached about 111 GB RSS, and still could not atomically commit the first batch before the 5-hour job deadline became impossible.

## Cause

The archived float32 latent/current/predicted states are effectively incompressible. ZIP deflate consumed one CPU for roughly 800 archives per batch while providing little storage reduction; artifacts were retained in memory until atomic batch commit.

## Correct practice

Use ZIP-stored `np.savez` for full float state archives. Preserve the same `.npz` keys, float32 dtypes, shapes, finite checks, and SHA256 gates. Test the ZIP members are `ZIP_STORED` so compression cannot silently return.

## Evidence

- VAGEN `vagen/ray_trainer.py::_pack_k4_state_trace`.
- `tests/test_evaluation_rollout_browser.py`.
- ID189 retry2 output `progress.md`, Job `527287`.
