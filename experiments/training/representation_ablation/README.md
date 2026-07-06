# Representation ablation training launchers

This directory contains Slurm launchers for config-driven representation ablation training.

## `train_compressed_vision_predictor.slurm`

Runs the LeWM-style compressed Qwen vision-token predictor diagnostic:

- frozen: Qwen2.5-VL checkpoint / vision encoder;
- trainable: `AttentionTokenCompressor` and `TokenSetWMPredictor`;
- loss: autoregressive predictor MSE plus `lambda_sigreg * SIGReg(compressed_tokens)`;
- no reconstruction loss; RCDM is a later visualization diagnostic.

The launcher writes a run-specific YAML under the output directory so the committed templates can keep path fields null.
