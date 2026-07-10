# E0011 — Resume 必须复现每个 micro-step 的随机流

## 错误

即使 full model、aux、EMA、optimizer 与 DataLoader 位置都恢复，若训练 loss（例如 SIGReg 随机投影）继续使用进程启动后的全局 RNG，resume 后的随机数与 uninterrupted run 不同，最终参数不会精确一致。

## 正确做法

本项目对每个训练 micro-step 使用 counter-based seed：由固定 `base_seed + epoch + micro_step + rank` 唯一决定，并在 forward 前重置 Python、Torch CPU 和当前 CUDA RNG。这样跳过已消费 DataLoader batches 后，后续 stochastic operations 与 uninterrupted run 使用相同随机流。

Checkpoint 同时记录并在 resume 时严格检查：seed、world size、grad accumulation、每 rank micro-batch 数和 RNG schedule version。任一项变化都拒绝精确 resume。

DDP bucket 顺序也属于数值轨迹的一部分。真实 gate 中 counter RNG 使 resume 后第一个 forward/loss 完全一致，但第一次 resumed optimizer step 后出现极小差异；原因是新 DDP 进程重新 warm up/rebuild buckets。所有每步都会用到的 model/aux DDP wrappers 应从构造时使用 `static_graph=True`，让 uninterrupted 与 resumed reduction buckets 一致。
