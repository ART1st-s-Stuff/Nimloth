# E0030: Validate prompts against archived runtime output

## What happened

The SFT1 rollout implementation called a new explicit-one-action prompt an exact reproduction of checkpoint eval job `479904`. Its golden test was derived from the source checkout, but the archived job artifact `results.jsonl` later showed that the actual runtime prompt said multiple actions were allowed and contained multi-action examples. The environment still executed only the first action because `max_actions_per_step=1`.

A 120-record gate therefore used a materially different prompt while being documented as source-exact.

## Required prevention

When exact reproduction is required, establish a runtime golden from the original run artifact (serialized prompt/conversation or captured request), not only from a current checkout or reconstructed config. Compare the complete rendered system and turn templates, including examples, contradictory wording, separators, and whitespace.

If archived runtime output conflicts with source code or intended semantics, treat the runtime artifact as evidence of what actually ran. Stop and ask the human which behavior to reproduce; do not silently "fix" contradictory prompt wording.
