# E0040: Action-only update is not VAGEN PPO

## Error

The first dynamic RL implementation saved generated thoughts but computed PPO ratios, entropy, and clipped policy loss only for the framework-selected action token. It also had no reference-policy KL. Calling this PPO scope compatible with VAGEN was incorrect.

VAGEN applies its response loss mask to every policy-generated response token and supports reference-policy KL. In Nimloth inject mode, sampled thought tokens and the sampled action token are policy decisions; framework-injected latent queries and delimiters are deterministic context and must remain masked.

## Correct practice

Persist exact thought token IDs and behavior log-probs during rollout. Like VAGEN, recompute PPO-old log-probs with the actor over the complete stored response before updates; do not rely only on generation-engine values. Recompute current-policy log-probs for every sampled thought/action token, broadcast each turn advantage across those tokens, and take the clipped PPO/entropy mean over the flattened response mask. Evaluate the immutable merged SFT2 base with RL LoRA adapters disabled as the reference policy and apply VAGEN-compatible sampled-token KL. Keep WM predictor and Value head losses in the same optimizer update. Version this scope in trajectory/checkpoint protocol metadata so action-only checkpoints cannot resume.
