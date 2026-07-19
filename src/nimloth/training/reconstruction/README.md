# Reconstruction decoder training

This package trains a **post-hoc** image decoder for WM diagnostics.  It freezes
Qwen, `StateProjector`, and the WM predictor checkpoint, then trains only
`WMImageDecoder` to reconstruct observations from projected WM states.

The decoder is not part of the SFT2/RL objective.  Use it to compare:

- `decoder(s_next)` vs next image: decoder/oracle upper bound.
- `decoder(wm_predictor(s_t, a_t))` vs next image: WM predictive reconstruction.
- `decoder(s_t)` or shuffled actions vs next image: baselines.

## Train

```bash
python -m nimloth.training.reconstruction.cli \
  --model /path/to/export_best_hf \
  --state-proj-checkpoint /path/to/best/state_proj.pt \
  --wm-checkpoint /path/to/best/wm_predictor \
  --train-jsonl /path/to/train.jsonl \
  --val-jsonl /path/to/val.jsonl \
  --output-dir outputs/experiments/training/reconstruction/<date>/<name>
```

## Eval an existing decoder

```bash
python -m nimloth.eval.reconstruction \
  --model /path/to/export_best_hf \
  --state-proj-checkpoint /path/to/best/state_proj.pt \
  --wm-checkpoint /path/to/best/wm_predictor \
  --decoder-checkpoint outputs/.../best \
  --val-jsonl /path/to/val.jsonl \
  --output-dir outputs/experiments/training/reconstruction/<date>/<name>/eval
```

## Train RCDM visualization model

RCDM is a heavier diffusion-based alternative to `WMImageDecoder`. It trains a
UNet from `external/RCDM` conditioned on SFT2 WM states. It is still post-hoc and
does not change SFT2/RL losses.

```bash
python -m nimloth.training.reconstruction.rcdm_sft2 \
  --model /path/to/export_best_hf \
  --state-proj-checkpoint /path/to/best/state_proj.pt \
  --wm-checkpoint /path/to/best/wm_predictor \
  --train-jsonl /path/to/train.jsonl \
  --val-jsonl /path/to/val.jsonl \
  --output-dir outputs/experiments/training/reconstruction/<date>/<rcdm_name> \
  --wandb-run-name <rcdm_name> \
  --state-cache-dir outputs/experiments/training/reconstruction/cache/<cache_name> \
  --build-state-cache
```

`--state-cache-dir` stores compressed shards of `StateProjector(Qwen
<|latent_state|>)` plus image paths.  This is the preferred full-run path: after
cache build, RCDM training no longer loads or runs Qwen.  The cache defaults to
`float16` state embeddings and `gzip` compression.

Resume the latest RCDM checkpoint in the same output directory:

```bash
python -m nimloth.training.reconstruction.rcdm_sft2 \
  ...same flags as above... \
  --resume
```

Use `--resume-checkpoint outputs/.../training_state_000001000.pt` to resume a
specific checkpoint. W&B uses `wandb_run_id.txt` in the output directory when
`--resume` is set.

## Train direct-state conditional flow matching

CFM is a lighter alternative to RCDM. It reuses a completed RCDM state cache,
trains only a token-conditioned UNet velocity field, and reports both validation
flow MSE and shuffled-condition sensitivity. Qwen, `StateProjector`, and the WM
predictor remain frozen.

```bash
python -m nimloth.training.reconstruction.cfm_sft2 \
  --state-cache-dir outputs/.../state_cache \
  --source-checkpoint outputs/.../sft2/epoch_002 \
  --wm-checkpoint outputs/.../sft2/epoch_002/wm_predictor \
  --output-dir outputs/.../cfm/<run_name> \
  --latent-token-count 8 \
  --epochs 10 --batch-size 32 --lr 1e-4 \
  --wandb-project nimloth-recon --wandb-run-name <run_name>
```

Use `--resume` to load the latest `checkpoint_*.pt` in the output directory.
The cache fingerprint, model config, data sizes, k, optimizer hyperparameters,
and seed must match. The final gate saves 5-step and 50-step ODE contact sheets
for current-state and WM-predicted-next reconstruction.

For a true 8-query CFM, point `--state-cache-dir` at a
`qwen_query_hidden` cache. Query-cache extraction requires the canonical merged
Hugging Face SFT2 handoff produced by the same snapshot+merge path used by RL
(`prepare_k8_sft2_init.py` / `.slurm`). It rejects raw PEFT adapter directories:
direct distributed PEFT loading is version-sensitive, and raw
`load_state_dict` does not restore saved adapter keys. The merged handoff has
already verified all adapter tensors, k/query protocol, model shards, and
source epoch completeness. Query-only cache builds do not instantiate
StateProjector/WM modules, which also permits newer projector dimensions
without changing the preprojection probe.

The trainer reads manifest `state_shape=[8,2048]`,
flattens storage only at the model boundary, and keeps eight condition tokens
inside every cross-attention layer. A proven ViT-token CFM can initialize all
shape-compatible UNet weights:

```bash
python -m nimloth.training.reconstruction.cfm_sft2 \
  --state-cache-dir outputs/.../query_hidden_cache \
  --latent-token-count 8 \
  --init-legacy-cfm-checkpoint /path/to/vit_token_cfm_low_lr_best_val.pt \
  --condition-dropout 0.15 --lr 3e-5 \
  --lr-decay-step 37120 --lr-after-decay 1e-5 \
  --positive-cache-dir outputs/.../positive_cache \
  --positive-cfm-checkpoint /path/to/vit_token_cfm_low_lr_best_val.pt \
  ...
```

The resulting contact sheet uses matched noise for
`GT | Qwen positive | Qwen wrong | 8-query CFM | query wrong`.  This direct
CFM remains invalid if correct and wrong Query conditions produce the same
scene, regardless of flow loss.

## Decode WM-predicted State into query latent

`projected_query_decoder.py` trains a post-hoc symmetric MLP
`8192 → 8192 → 8×2048`. It uses only adjacent rows and equal supervision from
(1) the true current projected State and (2) the frozen WM's one-step prediction
from the true previous projected State and previous action. Both targets are the
current Qwen query hidden vectors; Qwen, StateProjector, WM, and CFM stay frozen.

```bash
python -m nimloth.training.reconstruction.projected_query_decoder \
  --projected-cache-dir outputs/.../projected_cache \
  --query-cache-dir outputs/.../query_cache \
  --wm-checkpoint outputs/.../sft2/epoch_002/wm_predictor \
  --output-dir outputs/.../projected_query_decoder \
  --hidden-dim 8192 --epochs 5 --batch-size 128
```

After training a fresh query-latent CFM, the teacher-forced evaluator produces
exactly `GT | old Qwen ViT-token CFM | query-latent CFM |
WM predicted State → Decoder → query-latent CFM` with matched noise. Every row
predicts time `t` from the true projected State at `t-1`; it is not a recursive
rollout.

```bash
python -m nimloth.eval.query_cfm_teacher_forced \
  --query-cache outputs/.../query_cache/val \
  --projected-cache outputs/.../projected_cache/val \
  --qwen-cache outputs/.../positive_cache/val \
  --wm-checkpoint outputs/.../sft2/epoch_002/wm_predictor \
  --decoder-checkpoint outputs/.../projected_query_decoder/best.pt \
  --query-cfm-checkpoint outputs/.../query_cfm/best.pt \
  --qwen-cfm-checkpoint /path/to/vit_token_cfm_low_lr_best_val.pt \
  --selections configs/eval/reconstruction/query_cfm_action_sequences.json \
  --output-dir outputs/.../teacher_forced_eval
```

## Compare projected and preprojection query states

Build a separate cache of the frozen Qwen hidden vectors at all k latent-query
positions by adding `--state-cache-representation qwen_query_hidden
--cache-only` to the RCDM cache command.  Then run the controlled decoder pair:

```bash
python -m nimloth.training.reconstruction.query_state_ablation \
  --projected-cache-dir outputs/.../projected_state_cache \
  --query-cache-dir outputs/.../query_hidden_cache \
  --output-dir outputs/.../query_state_ablation/<run_name> \
  --query-token-count 8 --max-steps 18560 --batch-size 16 \
  --wandb-project nimloth-recon --wandb-run-name <run_name>
```

Both branches receive identical examples and use the same patch-decoder body.
One branch gets the single projected 1024-d state; the other gets all k Qwen
query hidden vectors before `StateProjector`.  This deterministic decoder later
failed its visual-fidelity gate and must not be used to localize information.

The valid retry first builds an aligned positive-control cache using the exact
old Qwen-vision + 16x512 compressor representation that produced
scene-conditioned ViT-token CFM images:

```bash
python -m nimloth.training.reconstruction.qwen_positive_cache \
  --source-cache-dir outputs/.../query_hidden_cache/train \
  --qwen-checkpoint /path/to/old/sft2_step1000 \
  --compressor-checkpoint /path/to/rollout4/compressor \
  --output-dir outputs/.../positive_cache/train
```

Then map query/projected states into that frozen visual-token space and render
all branches through the frozen proven CFM checkpoint:

```bash
python -m nimloth.training.reconstruction.state_to_vision_tokens \
  --projected-cache-dir outputs/.../projected_cache \
  --query-cache-dir outputs/.../query_hidden_cache \
  --positive-cache-dir outputs/.../positive_cache \
  --cfm-checkpoint /path/to/vit_token_cfm_low_lr_best_val.pt \
  --output-dir outputs/.../state_to_vision_tokens
```

The contact sheet is valid only if the true Qwen positive-control column first
recovers scene-conditioned structure.  Otherwise the run is a decoder/pipeline
failure and says nothing about State information.

## Sample from RCDM

Sample from a trained RCDM checkpoint:

```bash
python -m nimloth.eval.rcdm_reconstruction \
  --model /path/to/export_best_hf \
  --state-proj-checkpoint /path/to/best/state_proj.pt \
  --wm-checkpoint /path/to/best/wm_predictor \
  --rcdm-checkpoint outputs/.../ema_0.9999_000100000.pt \
  --metadata outputs/.../metadata.json \
  --val-jsonl /path/to/val.jsonl \
  --output-dir outputs/.../rcdm_samples \
  --timestep-respacing 100
```
