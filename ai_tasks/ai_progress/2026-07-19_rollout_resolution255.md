# Rollout image resolution correction and paired probe

## Goal

1. Make navigation rollout retain RGB 255×255 PIL images instead of forcing them to 512×512.
2. Build a non-destructive 255×255 derivative of the converted rollout dataset.
3. Compare the source step-60 checkpoint on the same 120 train tasks under the old and corrected resolution paths.

## Confirmed mismatch

- Effective production rollout used Nimloth VAGEN `e7cc2d0` with nested verl `65316156`.
- AI2-THOR emitted 255×255 frames, but both Qwen rollout managers called verl `process_image()` whose default `min_pixels=512*512`; this changed each image to 512×512.
- Qwen2.5-VL then smart-resized 512 to 504, giving `image_grid_thw=[1,36,36]` and 18×18=324 merged vision tokens.
- Historical source VAGEN `f7aefd3` passed 255×255 images directly; standard Qwen processing maps this to 252, `image_grid_thw=[1,18,18]`, and 9×9=81 merged vision tokens.
- Existing 512×512 rollout dumps and the recorded native 504/grid36 SFT diagnostics corroborate the production path.

## Human decisions

- Synchronize all active experiment branches while preserving each branch's VAGEN lineage.
- Preserve existing images and create a derived dataset with rewritten JSONL paths.
- Probe 120 train tasks as base_train seeds 1–60 plus common_sense_train seeds 1–60.
- Run a paired A/B comparison: old 512→504 path versus corrected 255→252 path, with identical checkpoint/tasks/eval settings.

## Completed

### VAGEN fix

- Canonical VAGEN fix: `a01f7af fix(rollout): preserve 255px navigation images` on `nimloth/fix-rollout-image255`.
- Added `vagen.rollout.image_utils.prepare_rollout_image()` and used it in both local and service Qwen rollout managers.
- RED test failed before implementation; GREEN validation: `2 passed`, compileall and diff-check passed.
- Ported and pushed equivalent fixes for every distinct active VAGEN baseline:
  - `7908040 -> 55c7b7f`
  - `bb26c0d -> d3e82f1`
  - `154c537 -> 178a6dc`
  - `93c1124 -> d28a41e`
  - `3003c2e -> d30d6d9`

### Nimloth branch synchronization

- dev: `8ce2512`
- exp/k8-preprojection-recon: `56fa44a`
- exp/latent-repr-ablation: `150face`
- exp/vagen-1action: `463d000`
- feat/dinov3-query-alignment: `378fd3b`
- feat/fsdp-dynamic-rollout: `54cac57`
- feat/reconstruct: `2ecfe59`
- feat/rl: `8d9cf2c`
- feat/sft1-dino16-grid-wm: `630fa4f`
- feat/sft1-hligb-step10-rollout: `037b519`
- nimloth-lewm-repro: `6edab8d`
- recon-compare-qwen: `c6451fe`

All root commits modify only the VAGEN gitlink. Existing unrelated `.memory/memories.jsonl` changes in feat/reconstruct and recon-compare-qwen were left untouched.

### Dataset/probe tooling prepared on dev

- `derive_rollout_images_255.py`: hashes unique source paths, writes RGB 255×255 BICUBIC PNGs into a separate tree, rewrites all top-level JSONL files, preserves sources, supports resume, and writes a manifest.
- `derive_rollout_images_255.slurm`: CPU conversion job wrapper.
- `rollouts_greedy_parallel.slurm`: prepared `ROLLOUT_TRAIN120=1`, alternate `VAGEN_DIR`, exact source eval kwargs, W&B logging, and expected PNG-size gate.
- `compare_rollout_resolution_probe.py`: reads both converted `traj_success` and raw `metrics.success`, and now rejects visible runtime-config/metadata inconsistency instead of silently producing false all-failure or false seed-paired results.
- `recover_rollout_resolution_pairs.py`: diagnostic recovery for E0030 dumps using control-batch membership, actual runtime config, instruction, and minimum initial-frame RMSE.
- Current server validation: comparator/recovery `4 passed`; complete resolution contract suite `5 passed` plus VAGEN image tests `2 passed`. Python compileall, shell `bash -n`, and diff-check passed before launch.

## Completed experiments (2026-07-20 server time)

- CPU image derivation job `481070`: `COMPLETED 0:0` in `00:08:55` on `intel-01` with 8 CPUs/64 GiB. The initial 32-CPU request was rejected before job creation by `QOSMaxCpuPerNode`, so the approved conversion used 8 workers. Exhaustive validation passed: all four JSONL counts preserved; 81,570 references map to 73,648 existing unique RGB 255×255 images; all 73,648 source images remain RGB 512×512 and aggregate source bytes remain `11,161,340,020`. Derived logical bytes=`4,510,928,566` (4.201 GiB), while NFS `du` reports 16 GiB because of allocation units. Output: `converted_strict_k8_b6c811c_images255`.
- Old-resolution A job `481071`: `COMPLETED 0:0` in `00:20:22`, old VAGEN `e7cc2d0`, W&B `9l4vjc1j`, 2,370 RGB512 references passed. Trusted runtime-config success: base 11/60, common 8/60, overall 19/120=15.83%; action validity=1.0.
- Corrected-resolution B job `481072`: `COMPLETED 0:0` in `00:23:21`, fixed VAGEN `a01f7af`, W&B `8lct7arz`, 2,364 RGB255 references passed. Trusted runtime-config success: base 13/60, common 9/60, overall 22/120=18.33%; action validity=1.0.
- Both GPU arms used exact step60, the same control parquet contents and greedy parameters, and separate normal 6-GPU allocations (2 env + 4 policy). Exact server code was Nimloth `f7ea3da`, both nested verl `65316156`.

## Discovered metadata error and trustworthy comparison

- Raw dumps are affected by E0030: trainer `zip(micro_validation_rst, env_configs, uids)` assumes async recorder order equals input order. A has 16/120 and B has 14/120 visible cross-config metadata mismatches; seed permutations within one config can be invisible. Direct `(data_source, env_seed)` pairing is therefore invalid. Earlier direct comparison outputs were renamed `*.invalid_metadata.json` and README claims were corrected.
- Aggregate totals and grouping by actual runtime `config_id` remain trustworthy.
- Diagnostic task identities were recovered using each reported key only for its control-batch membership, then actual runtime config, exact instruction, and minimum initial-frame RMSE. All 120 tasks paired across 104 groups; 13 repeated-instruction groups used frame assignment. Assigned RMSE max=5.106; smallest rejected alternative=43.170, giving clear separation. Exact seed labels were not recovered.
- Recovered outcomes: both success=18, old-only=1, new-only=4, both failure=97. Success changed 19/120→22/120, +3 tasks / +2.5 percentage points; exact two-sided McNemar `p=0.375`. Per runtime source: base +2/60 (+3.33 pp), common +1/60 (+1.67 pp).
- Conclusion: on these train120 tasks the 252 path is slightly higher, but the effect is not statistically distinguishable and resolution alone does not recover the historical 86/120=71.67%. That historical result used a different held-out task set and older source runtime, so its absolute rate is not directly comparable.
- Evidence: group `progress.md`, `paired_comparison.runtime_identity.json`, per-arm `validation_summary.runtime_config.json`, and corrected READMEs.

## Stable-identity fix and approved rerun preparation

- Human selected “fix and rerun”; repo memory M0012 remains pending at the human's request.
- Fixed lineage: VAGEN `192c35a` on `nimloth/fix-rollout-image255`. Old-resolution lineage: VAGEN `ef851af` on `nimloth/fix-validation-identity-e7cc2d0`.
- Both managers now save reset-time `env_id -> input index`; trainer attaches metadata through a lightweight fail-fast stable-identity helper instead of positional zip.
- Tests cover shuffled returns, service environment reuse, missing identity, local/service manager mapping contract and trainer use. Fixed lineage: 7 passed including image contracts; old lineage: 8 passed including source-eval contract; compileall and diff checks passed.
- Added `validate_rollout_train120_dump.py` as an automatic post-rollout gate: exactly 120 expected keys, stable UID, matching runtime config/eval set, `metrics.success`, existing RGB images and exact size. Root rerun tooling now passes 10 tests plus shell/compile/diff checks.

## Exact-seed rerun completed

- Final clean server root: Nimloth `b27d0e3`, fixed VAGEN `192c35a`, old VAGEN `ef851af`, both verl `65316156`. Root/VAGEN tests: 10/7/8 passed respectively.
- Old504 A job `481089`: `COMPLETED 0:0` in `00:16:56` on dgx-13; W&B `aj3cfv27` finished under exact ID12 name. Stable gate passed: 120/120 exact keys, zero metadata mismatches, 2,361 RGB512 references. Success base/common/overall=`14/60`, `8/60`, `22/120=18.33%`; action validity=1.0.
- New252 B job `481090`: `COMPLETED 0:0` in `00:15:56` on dgx-12; W&B `pc45edc4` finished under exact ID13 name. Stable gate passed: 120/120 exact keys, zero metadata mismatches, 2,365 RGB255 references. Success base/common/overall=`12/60`, `9/60`, `21/120=17.50%`; action validity=1.0.
- Both used independent normal 6-GPU allocations, frozen step60, identical control tasks and greedy parameters. No resume/checkpoint is needed.

## Authoritative strict comparison

- Exact seed pairing: both success=20, old-only=2, new-only=1, both failure=97.
- Corrected B minus old A: `-1/120=-0.83` percentage points; exact two-sided McNemar `p=1.0`.
- Per source: base `14/60→12/60` (-3.33 pp); common `8/60→9/60` (+1.67 pp).
- Conclusion: changing only `512→504/grid36` to `255→252/grid18` does not improve success on train120. The strict effect is negligible and statistically indistinguishable, so resolution alone cannot explain the historical 71.67% versus current low train120 rates.
- Historical `86/120` used a different held-out task set and older source runtime; its absolute difference remains non-causal. Other runtime/task-distribution differences must be investigated separately if reproducing 71.67% is still required.
- Authoritative evidence: `paired_comparison.strict_seed_identity.json`, strict arm READMEs and `stable_identity_validation.json`. It supersedes the diagnostic runtime-identity pairing for the primary conclusion.
