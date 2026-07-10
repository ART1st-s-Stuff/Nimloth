# vagen_legacy_wm_entropy01_kl001_60step_2env4train k=8 pipeline

This directory is a runbook/template for the rollout → SFT1 → SFT2 pipeline using
`vagen_legacy_wm_entropy01_kl001_60step_2env4train` as the source checkpoint and
`LATENT_TOKEN_COUNT=8` for SFT training/eval.

## 0. Required path variables

Set these on the server before launching jobs:

```bash
export REPO=/project/peilab/atst/nimloth
export SFT1_RUNS_ROOT=${REPO}/experiments/navigation_baseline/runs
export SOURCE_RUN_NAME=vagen_legacy_wm_entropy01_kl001_60step_2env4train
export SOURCE_RUN_DIR=${SFT1_RUNS_ROOT}/${SOURCE_RUN_NAME}

# Fill after locating the exact checkpoint/export.
export INIT_HF=${SOURCE_RUN_DIR}/checkpoints/<global_step>/actor/huggingface

export RUN_DATE=$(date +%Y-%m-%d)
export LATENT_TOKEN_COUNT=8
export MASK_LATENT_QUERY_LABELS=1
```

Before launching, verify `INIT_HF` exists and contains a HF model/tokenizer.
If the source checkpoint is still a VAGEN/verl sharded checkpoint, export/convert it to HF first.

## 1. Rollout collection

Use the source checkpoint to collect train/val/test rollouts. For the initial rollout from the
legacy source checkpoint, keep `LATENT_TOKEN_COUNT=1` unless that checkpoint already has the k=8
extra tokens; the SFT scripts can normalize the collected single-token records to k=8 later.

After collection, convert rollouts with `experiments/training/sft1/convert_rollouts.py` and record:

- raw rollout directory
- converted records directory
- train/val/test counts
- success rates

Recommended converted records variable for later phases:

```bash
export RECORDS_ROOT=${SFT1_RUNS_ROOT}/sft_records_${SOURCE_RUN_NAME}_nimloth_format
```

## 2. SFT1 k=8

```bash
export LATENT_TOKEN_COUNT=8
export MASK_LATENT_QUERY_LABELS=1
export BASELINE_RUN_NAME=${SOURCE_RUN_NAME}
export EXPERIMENT_NAME=sft1_${SOURCE_RUN_NAME}_k8
export WANDB_RUN_NAME=${EXPERIMENT_NAME}
export TRAIN_OUT=${REPO}/outputs/experiments/training/sft1/${RUN_DATE}/${EXPERIMENT_NAME}
export TRAIN_JSONL=${RECORDS_ROOT}/train_success.jsonl
export VAL_JSONL=${RECORDS_ROOT}/val_all.jsonl

bash experiments/training/sft1/submit_train_8gpu.sh
```

SFT1 code normalizes rendered latent blocks to k=8 and masks latent query token labels by default.

## 3. SFT2 k=8

Use the best/selected SFT1 checkpoint exported or merged to HF.

```bash
export LATENT_TOKEN_COUNT=8
export MASK_LATENT_QUERY_LABELS=1
export SFT1_RUN=${TRAIN_OUT}
export BASE_HF=${INIT_HF}
export EXPERIMENT_NAME=sft2_${SOURCE_RUN_NAME}_k8
export TRAIN_OUT=${REPO}/outputs/experiments/training/sft2/${RUN_DATE}/${EXPERIMENT_NAME}
export CONFIG=${REPO}/configs/training/sft2/latent_wm_value_k8.yaml

bash experiments/training/sft2/submit_default_8gpu.sh
```

SFT2 checkpoints should record `latent_token_count=8`, `qwen_hidden_dim`, and
`state_proj_input_dim=8*qwen_hidden_dim`.

## Notes

- VAGEN prompt helpers support `NIMLOTH_LATENT_TOKEN_COUNT` / `LATENT_TOKEN_COUNT` at import time.
- Parser accepts extra latent query tokens between `<|latent_state|>` and `<|action_start|>`.
- Do not reuse output directories from older k=1 experiments.
