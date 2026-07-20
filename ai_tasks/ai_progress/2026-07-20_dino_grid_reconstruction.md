# 2026-07-20 DINO-supervised SFT1 grid reconstruction

## Human request and protocol

- Reconstruct from the latest completed DINO-supervised SFT1 ID18 epoch5/final result.
- SFT2 is incomplete: do not load, render, or compare WM-predicted state.
- Human selected only the directly DINO-supervised projected grid, not preprojection query hidden.
- Train the recon CFM on all strict train transitions and evaluate on disjoint strict val.
- Preserve matched noise, Euler50, CFG2 and diverse40/200-frame comparison against the proven Qwen ViT-token CFM.
- Human requested one 8GPU allocation: parallel cache first, then CFM/eval in the same allocation.

## Representation and implementation

- SFT1 source: `.../vagen_legacy_wm_k16_grid/2026-07-19/sft1/18_retry1_dino2l_grid4_k16_prefix_success7309_l1_ep5_b1_ga8_ws8_px602112/final/hf_merged`.
- State is the shared token-wise projector output `[16,1024]`, row-major 4x4, directly aligned to DINOv2-large pooled patch-grid targets during SFT1.
- Commit `80aaac5` added an explicit `dino_grid_state` cache projector, CFM semantics and no-WM evaluator.
- Commit `f0d4204` added strict source-model/cache lineage, row-major metadata validation, BF16 projector execution matching SFT1, CFM/cache eval fingerprint validation, and known error E0049.
- Server targeted tests: `14 passed, 1 Pillow deprecation warning`; compile and diff checks passed.

## Data and training

- Strict train: 3,217 training-split trajectories / 59,389 transitions, successes and failures.
- Strict val: 355 held-out validation trajectories / 6,054 transitions; validation only.
- CFM-only training: 128px TokenConditionedFlowUNet, 30 epochs/max55,680 steps, b32, lr3e-5→1e-5 at37,120, condition dropout0.15, shape-compatible initialization from proven Qwen ViT CFM.
- Frozen: SFT1 Qwen/query adapter/shared slot projector. DINO teacher and WM are absent.

## Final experiment result

- Human first requested preempt `dgx-40`; agent failed to hold it before preparation, and another user's job took all GPUs. Human corrected this and specified then-IDLE `dgx-03`; error recorded as `E0049`.
- Holder Slurm `482045` ran on preempt `dgx-03` with one node / 8 GPUs / 128 CPUs / 768G. After all output gates, it was cancelled at elapsed `03:04:45` to release GPUs.
- W&B identities:
  - ID45 CFM: [`f9wj5gza`](https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth-recon/runs/f9wj5gza), `45_sft1e5_dinogrid16x1024_cfm_ep30_b32_drop015`.
  - ID46 eval: [`ek552cqe`](https://wandb.ai/art2nd-hong-kong-university-of-science-and-technology/nimloth-recon/runs/ek552cqe), `46_sft1e5_dinogridcfm_qwenvit_diverse40`.
- Output root: `/project/peilab/atst/nimloth/outputs/experiments/vagen_legacy_wm_k16_grid/2026-07-19/reconstruction`; final marker `PIPELINE_COMPLETED`, gate `final_gate.json`.

### Cache evidence

- Query cache train/val: `59,389/6,054`, fingerprints `2f3825837d4b8e07/4dca28dedb62f0fb`, shape `[16,2048]`.
- DINO-grid cache train/val: same row counts, fingerprints `a1eceb48063f4138/fee377fa57374b9a`, shape `[16,1024]`, explicit source-query fingerprints.
- Full gate scanned all `120+16` query and `120+16` grid shards: every shard exists, total counts match and every tensor is finite.

### CFM evidence

- Completed 30 epochs / 55,680 sampled steps. Last train flow MSE=`0.02488226`; best fixed-subset correct MSE=`0.03185286` at step29,000.
- Final full-val correct/shuffled MSE=`0.03929895/0.04263037`, difference=`0.00333143`, ratio=`1.08477135` over all6,054 val items. This is materially stronger condition sensitivity than prior query-latent k8/k1 ratios near`1.012/1.005`, while not by itself proving exact reconstruction.
- Best and final checkpoints each have 180 finite model tensors and strict-reload as CFM config `16×1024`, base64, condition256, time512.

### Three-column evaluation

- Diverse40 / 200 frames, matched noise, Euler50, CFG2; exactly `GT | Qwen ViT-token CFM | DINO-grid CFM`, no WM predictor.
- Image L1: Qwen=`0.27630016`, DINO-grid=`0.23360273`; DINO/Qwen=`0.84546723`; DINO lower on75% of frames.
- Four contact sheets and all240 per-frame/run images were generated; independent gate found244 PNGs total, all nonempty and PIL-decodable.
- Visual inspection: DINO-grid usually preserves room color, corridor/window placement and coarse geometry better, but still occasionally generates a plausible mismatched room and blurred/warped boundaries. Do not describe it as pixel-perfect.

### Recoverable failures and fixes

- Step`482045.1` completed both caches then failed before CFM because `--resume` was used only for the reserved W&B ID but the trainer correctly required a checkpoint. Commit`0f73260` added independent `--wandb-resume`; caches were reused unchanged.
- Step`482045.10` completed the entire CFM, then evaluator lineage looked for noncanonical `condition_token_count/condition_token_dim` instead of checkpoint `token_count/token_dim`. Commit`9d8b1b4` fixed the gate; only eval was rerun.
- Final eval retry omitted explicit `srun --ntasks=1`, so eight deterministic copies ran against the same output/W&B ID. Metrics were identical, but W&B contains duplicate uploads and the Slurm step ended nonzero from the concurrent race. Final metadata, PNGs and metrics passed an independent post-run gate. Future same-allocation single-process commands must state `--ntasks=1`.
- Runtime only reported the known C++ extension warning for torch`2.8.0+cu128 < 2.11.0`; no OOM/NCCL/NaN affected the valid outputs.
