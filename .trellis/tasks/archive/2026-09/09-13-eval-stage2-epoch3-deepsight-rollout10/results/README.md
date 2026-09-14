# Epoch3 rollout DINO comparison

10 test rollouts: Base5 and CommonSense5, seeds1–5. Success4/10 (Base1/5, CommonSense3/5). This is a small diagnostic sample.

134 executed states: mean slot cosine0.6138268816; MSE0.9148713555. Each state has64 slots with1024-dimensional projected and frozen DINOv2-large target features. Raw tensors retained on a100-2 under /mnt/nimloth/outputs/experiments/sft2-dino-information-test/20260913_epoch3_deepsight_rollout10/features_r4.

28 PNG pages in figures/: RGB, projected shared PC1, target shared PC1, 1-cosine. One PCA basis fitted only on all targets and applied unchanged to projected features; shared1–99% target color limits. PC1 explains15.4755% target variance. This is a documented DeepSight-style layout; the paper does not disclose its exact feature reduction.

Visual inspection of base1/common_sense1: projected PC1 preserves coarse vertical distribution but weaker local structure and response to camera motion. This does not establish full-dimensional collapse or sufficient recovery. No CFM was run.

Source checkpoint copied directly server-to-server with15 matching SHA256 files, epoch3 step81. Original checkpoint masters preserved. Replay BF16 parameters match rollout inference precision, FP32 rotary buffers preserved. Six focused CPU tests passed, real GPU replay and CPU rendering completed. Independent summary recomputation passed.
