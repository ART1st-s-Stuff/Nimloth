#!/usr/bin/env bash
# Shared Hydra CLI overrides aligned with VAGEN paper Table 23 (arxiv 2510.16907).
# Nimloth keeps longer sequence caps and env settings in train.yaml / val.yaml
# (max_turns=20, max_actions_per_step=1, response_length_per_turn=512).

: "${VAGEN_MAX_PROMPT_LENGTH:=3000}"
: "${VAGEN_MAX_RESPONSE_LENGTH:=20000}"

# Paper Table 23 uses "masked_gae"; current verl/vagen exposes equivalent token masking via gae.
VAGEN_PAPER_PPO_ARGS=(
  "data.max_prompt_length=${VAGEN_MAX_PROMPT_LENGTH}"
  "data.max_response_length=${VAGEN_MAX_RESPONSE_LENGTH}"
  "algorithm.adv_estimator=gae"
  "algorithm.use_kl_in_reward=True"
  "algorithm.kl_ctrl.kl_coef=0.001"
  "algorithm.gamma=1.0"
  "algorithm.lam=1.0"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=0.001"
  "actor_rollout_ref.actor.entropy_coeff=0.05"
  "actor_rollout_ref.rollout.temperature=0.7"
  "actor_rollout_ref.rollout.top_p=0.95"
  "critic.optim.lr=1e-5"
)
