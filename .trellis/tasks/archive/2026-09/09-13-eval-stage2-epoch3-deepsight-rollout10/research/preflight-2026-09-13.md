# Preflight evidence — 2026-09-13

- a100-1 source checkpoint: `/mnt/nimloth/outputs/experiments/sft2-dino-information-test/20260912_epoch2_selective_rows_fresh_r2/train/epoch_003`.
- Size observed: 8.7 GB. Required files observed: `COMMITTED`, `adapter_model.safetensors`, `slot_projector.pt`, `grid_state_config.json`, `training_state.pt`, tokenizer files.
- Training process cwd: `/mnt/nimloth/.worktree/sft2-selective-fresh`; commit `99207d3518ddba896efd803031d8954141197291`.
- Training command confirms `latent-token-count=64`, `grid-size=8`, FP32 embedding masters, selective query/protocol token learning rates, and convergence training.
- a100-2 observed eight idle A100-SXM4-40GB GPUs and about 1.3 TB free under `/mnt`.
- Existing completed epoch2 rollout evidence is rooted at `/mnt/nimloth/outputs/experiments/sft2-dino-information-test/20260912_epoch2_fp32_lr5e5_r2/rollout_epoch002_r3_controller` and uses the project Stage2 evaluation entrypoint.
- Direct a100-1 to alias `a100-2` fails DNS; direct IP reaches the host but authentication is unavailable. SSH agent forwarding was rejected by automatic approval review because it exposes the local signing agent to a100-1. The safe default is a standard local relay transfer.
- DeepSight Figure 7 provides a temporal single-channel pseudocolor layout but neither the paper nor released visualization script specifies the 1024D reduction. This task defines and records its own shared target-fitted PCA method.
