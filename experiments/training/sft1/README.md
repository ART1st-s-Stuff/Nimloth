# Stage 1 format training

`train.py` 是正式 Stage 1 workflow 的薄入口；生产逻辑见 [stage1](../../../src/nimloth/training/sft/stage1/README.md)。不再使用旧 Slurm 训练/缓存脚本或 K 配置。

```bash
python experiments/training/sft1/train.py --source-model /path/to/hf_actor --train-jsonl /path/to/train_all.jsonl --val-jsonl /path/to/heldout_all.jsonl --format-eval-jsonl /path/to/heldout_all.jsonl --output-dir /path/to/new-run --config configs/training/sft1/format.yaml --nproc-per-node 8 --success-only
# 相同命令追加 --resume 即从完整边界恢复。
```

调用者选择服务器、解释器和 GPU。标准流程要求显式 `--success-only`，分别从 train/val 源记录选择布尔 `success is True` 的轨迹，记录实际动作分布但不要求成功子集覆盖全部动作，然后准备 Stage 1 prompt 和语义 action token、构建缓存并训练到成功 heldout 的验证 LM loss 收敛。`--format-eval-jsonl` 独立保留完整 heldout prompt，固定用于每轮 32 条严格自由生成检查。额外训练 CLI 参数覆盖配置。

## 数据与诊断工具

| File | Purpose |
|------|---------|
| `convert_rollouts.py` | VAGEN rollout JSONL → Nimloth SFT records |
| `vagen_step60_data.py` | Pinned step60 source partition, overlap, conversion and complete-shard contracts |
| `vagen_step60_checkpoint.py` | Non-overwriting step60 shard audit, legacy FSDP merge plan and HF load validation |
| `extract_vagen_step60_evidence.py` | Non-overwriting W&B prompt/reward extractor that excludes assistant CoT and emits the hash-bound reconstruction fixture |
| `vagen_step60_runtime_contract.py` | Non-overwriting Git-computed reconstruction contract producer; prints the payload hash for approval |
| `hash_vagen_step60_runtime_contract.py` | Independent runtime-contract payload hash recomputation/check CLI |
| `vagen_step60_collect.py` | Evidence-backed reconstructed legacy service client, frozen-policy rollout, EOS/terminal audit and reserved-directory/COMPLETE-last v3 shards; unavailable exact source commit remains provenance only |
| `vagen_step60_convert.py` | Complete batch1 shards → linked K16 SFT1/SFT2 views, v3 rejections and hash manifest; validates reserved-directory publication only after `conversion_manifest.json` appears last |
| `validate_vagen_step60_conversion.py` | Independent published-conversion hash/count/envelope validator |
| `derive_rollout_images_255.py` | Preserve sources and derive RGB 255×255 images with rewritten JSONLs |
| `derive_rollout_images_255.slurm` | CPU wrapper for the non-destructive image derivation |
| `merge_lora_ckpt.py` | LoRA adapter → `hf_merged` for VAGEN eval / SFT2 init |
| `rollouts_greedy_parallel.slurm` | Greedy rollout collection (Slurm array) |
| `env_external_4gpu.slurm` | Shared 4-GPU AI2-THOR env for rollouts/eval |
| `summarize_eval_rollouts.py` | Deferred historical WM/Stage 3 caller only; early-stage evaluation does not use this |
| `summarize_before_after_rollouts.py` | Before/after training comparison |
| `compare_rollout_resolution_probe.py` | Paired comparison for dumps with verified stable metadata; fails on visible runtime/metadata mismatch |
| `recover_rollout_resolution_pairs.py` | Diagnostic recovery for E0030-corrupted dumps via batch/runtime/instruction/initial-frame identity |
| `validate_rollout_train120_dump.py` | Exact 120-key, stable metadata/UID, runtime-config and RGB PNG completion gate |

Stage 1/2 与 VAGEN 的 success rate 只通过 [统一评估入口](../sft/evaluation/README.md) 运行；旧独立评估和 watcher 已移除。数据收集工具不承担新阶段评估。
