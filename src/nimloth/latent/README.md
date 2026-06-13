# Latent Extraction

This module implements the first Nimloth extraction path described in `DESIGN_DOCS.md`.

## Step Semantics

For each environment step, the prompt/response format must contain:

```text
</think><|latent_state|> ... <|action_start|><|act_...|><|action_end|>
```

The extraction utilities read:

- `latent_state`: final-layer hidden state at the `<|latent_state|>` token.
- `action_prior.logits/probs`: causal LM logits at `<|action_start|>`, restricted to the configured action tokens. These logits predict the token immediately after `<|action_start|>`.

In an autoregressive model, logits at position `i` predict token `i + 1`. Therefore the probability prior over the first action token is produced at `<|action_start|>`.

## Token Setup

Call `add_special_tokens(tokenizer)` before tokenizing Nimloth data. If any tokens are newly added, resize model embeddings:

```python
from nimloth.latent import LatentActionExtractor, add_special_tokens

added = add_special_tokens(tokenizer)
if added:
    model.resize_token_embeddings(len(tokenizer))

extractor = LatentActionExtractor(tokenizer)
```

## Model Extraction

```python
inputs = tokenizer(step_text, return_tensors="pt")
latent_state, action_prior, positions = extractor.extract_from_model(model, **inputs)

state_vector = latent_state
prior_probs = action_prior.probs if action_prior is not None else None
```

This is only the extraction layer. It does not start training, rollout collection, evaluation, or dataset splitting.
