#!/usr/bin/env bash
# Shared Hydra overrides for reproducible env sampling and auditable rollout dumps.
# Requires VAGEN branch with env metadata dump + stable uids (fix/env-reproduction).

: "${REPO:=/project/peilab/atst/nimloth}"
: "${VAGEN_DATA_SEED:=42}"
: "${VAGEN_BASE_SEED:=42}"

VAGEN_CUSTOM_CLS_PATH="${REPO}/external/VAGEN/vagen/gym_agent_dataset.py"

VAGEN_ENV_REPRO_ARGS=(
  "data.custom_cls.path=${VAGEN_CUSTOM_CLS_PATH}"
  "data.seed=${VAGEN_DATA_SEED}"
  "+data.base_seed=${VAGEN_BASE_SEED}"
  "data.validation_shuffle=False"
  "+trainer.assert_val_env_composition=True"
  '+trainer.val_env_composition.navigation_base={count:60,eval_set:base}'
  '+trainer.val_env_composition.navigation_common={count:60,eval_set:common_sense}'
)
