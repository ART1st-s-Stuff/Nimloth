# vagen_legacy_wm_entropy01_kl001_60step_2env4train k=8 pipeline

This directory is a runbook/template for the rollout → SFT1 → SFT2 pipeline using
`vagen_legacy_wm_entropy01_kl001_60step_2env4train` as the source checkpoint and
`LATENT_TOKEN_COUNT=8` for SFT training/eval.

## 0. Required path variables

Set these on the server before launching jobs:

```bash
# REPO must be the clean server worktree at the committed pipeline revision.
export REPO=/project/peilab/atst/nimloth/.worktree/vagen-legacy-wm-k8-2513d79
export SOURCE_RUN_NAME=vagen_legacy_wm_entropy01_kl001_60step_2env4train
export SOURCE_RUN_DIR=/project/peilab/atst/nimloth/outputs/experiments/training/baseline/2026-06-24/${SOURCE_RUN_NAME}
export SOURCE_CHECKPOINT_STEP=60
export INIT_HF=${SOURCE_RUN_DIR}/checkpoints/global_step_${SOURCE_CHECKPOINT_STEP}/actor/huggingface

export RUN_DATE=$(date +%Y-%m-%d)
export SFT1_OUTPUT_DATE_ROOT=/project/peilab/atst/nimloth/outputs/experiments/training/sft1/${RUN_DATE}
export SFT1_RUNS_ROOT=${SFT1_OUTPUT_DATE_ROOT}
export LATENT_TOKEN_COUNT=8
export MASK_LATENT_QUERY_LABELS=1
```

Verified source: `global_step_60/actor/huggingface` contains a complete four-shard HF model.
Its tokenizer has no Nimloth latent/action tokens, so source-policy rollout must stay on the
legacy `eval_mode` prompt; k=8 tokens are introduced by conversion/SFT, not by source rollout.

## 1. Rollout collection

Use the source checkpoint to collect train/val/test rollouts with the legacy `eval_mode` prompt.
The rollout-only HF load disables VAGEN resume, so generated dumps are named `0.jsonl` even though
the source policy is checkpoint step 60.

```bash
export ROLLOUT_RUN_NAME=rollout_${SOURCE_RUN_NAME}_k8
export ROLLOUT_RUN_DIR=${SFT1_OUTPUT_DATE_ROOT}/${ROLLOUT_RUN_NAME}
export ROLLOUT_DUMP_STEP=0
export BASELINE_RUN_NAME=${SOURCE_RUN_NAME}
export INIT_HF_STEP=${SOURCE_CHECKPOINT_STEP}
export LATENT_TOKEN_COUNT=1

# Optionally set ENV_NODE/NODELIST; the rollout array waits for the env ready flag.
bash ${REPO}/experiments/training/sft1/submit_rollouts_greedy.sh
```

After collection, convert rollouts with `experiments/training/sft1/convert_rollouts.py` and record:

- raw rollout directory
- converted records directory
- train/val/test counts
- success rates

Recommended converted records variable for later phases:

```bash
export RECORDS_ROOT=${SFT1_OUTPUT_DATE_ROOT}/records_${SOURCE_RUN_NAME}_k8
export NIMLOTH_LATENT_TOKEN_COUNT=8
python ${REPO}/experiments/training/sft1/convert_rollouts.py \
  --input-root ${ROLLOUT_RUN_DIR}/validation \
  --output-root ${RECORDS_ROOT} \
  --checkpoint-hf ${INIT_HF} \
  --checkpoint-step ${SOURCE_CHECKPOINT_STEP} \
  --rollout-step ${ROLLOUT_DUMP_STEP}
```

## 2. SFT1 k=8

```bash
export LATENT_TOKEN_COUNT=8
export MASK_LATENT_QUERY_LABELS=1
export BASELINE_RUN_NAME=${SOURCE_RUN_NAME}
export EXPERIMENT_NAME=sft1_${SOURCE_RUN_NAME}_k8
export WANDB_RUN_NAME=${EXPERIMENT_NAME}
export TRAIN_OUT=${SFT1_OUTPUT_DATE_ROOT}/${EXPERIMENT_NAME}
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
export TRAIN_OUT=/project/peilab/atst/nimloth/outputs/experiments/training/sft2/${RUN_DATE}/${EXPERIMENT_NAME}
export CONFIG=${REPO}/configs/training/sft2/latent_wm_value_k8.yaml

bash experiments/training/sft2/submit_default_8gpu.sh
```

SFT2 checkpoints should record `latent_token_count=8`, `qwen_hidden_dim`, and
`state_proj_input_dim=8*qwen_hidden_dim`.

## Notes

- VAGEN prompt helpers support `NIMLOTH_LATENT_TOKEN_COUNT` / `LATENT_TOKEN_COUNT` at import time.
- Parser accepts extra latent query tokens between `<|latent_state|>` and `<|action_start|>`.
- Do not reuse output directories from older k=1 experiments.
