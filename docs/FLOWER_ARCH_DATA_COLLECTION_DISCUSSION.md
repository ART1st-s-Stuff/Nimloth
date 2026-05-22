# Flower Architecture And Data Collection Notes

Date: 2026-05-22

## Context

Current goal: improve Flower EB-Nav performance without adding scene-specific or task-specific rules.

Recent findings:

- Direct Qwen planner can be made format-stable with continued SFT, but policy still over-selects forward.
- Forward-heavy action distribution is not itself a bug; many embodied tasks naturally contain many forward steps.
- Manual rules such as "if forward failed" or "if target not visible" are too navigation-specific for the desired design.
- Learned WM/value modules should not replace Qwen by default. They should support conservative reranking/gating.

## Preferred Architecture

Use a two-speed decision system:

- Qwen planner: deep reasoning, belief/state description, candidate action proposal.
- WM predictor: fast transition model. It predicts what will happen after a candidate action.
- Value head: fast task utility model. It ranks candidates using current belief, candidate action, and WM-predicted next belief/effects.
- Gate: conservative decision layer. It keeps Qwen's action unless an alternative has clear advantage and low uncertainty.

Clean responsibility split:

- WM predictor answers: "What will happen if action a is executed?"
- Value head answers: "Given that predicted outcome, is action a useful for the task?"
- Gate answers: "Is the learned alternative reliable enough to override Qwen?"

Avoid letting the value head become a black-box action classifier that directly replaces Qwen.

## Task-Agnostic Fork Collection

Do not trigger forks with hand-written navigation rules.

Use task-agnostic uncertainty and information signals instead:

- Qwen action distribution entropy is high.
- WM ensemble disagreement is high.
- Value ensemble disagreement is high.
- Qwen action confidence and WM predicted effect are inconsistent.
- Predicted next-belief variance is high.
- Current state embedding is far from the training distribution.
- Recent transition prediction error is rising.

The collection objective should become:

- collect counterfactual action outcomes;
- collect high-information states;
- collect model-disagreement states;
- avoid assuming that low rollout success makes the data useless.

Failed rollouts are still useful if they contain reliable transition labels.

## Training Principle

Do not manually downweight forward samples just because they are forward.

Instead, use automatic per-sample effective learning rate:

```text
effective_lr_i = base_lr * learnability_i * reliability_i * novelty_i
```

Skip a sample if:

```text
effective_lr_i < threshold
```

Interpretation:

- `reliability_i`: label/outcome is trustworthy.
- `novelty_i`: sample is not redundant with already-learned data.
- `learnability_i`: model can still learn useful signal from the sample.

Important distinction:

- high target uncertainty -> lower learning rate or skip;
- high model uncertainty with reliable label -> keep or increase learning;
- low model uncertainty plus low loss plus redundant representation -> lower learning rate or skip.

This should automatically reduce the training influence of repeated easy samples without encoding an action-specific penalty.

## Module-Specific Training

Qwen planner:

- train format stability;
- train belief / observation description;
- train action-effect prediction in text;
- train final action token;
- use adaptive sample learning rate rather than action weighting.

WM predictor:

- train current belief + action -> next latent / next belief;
- estimate epistemic uncertainty with ensemble disagreement;
- prefer delta prediction and contrastive/action-sensitive objectives to avoid identity collapse.

Value head:

- train on candidate comparisons and outcome consistency, not only sparse episode success;
- input current belief, candidate action, WM-predicted next belief/effect, and WM uncertainty;
- output task utility score and value uncertainty;
- use clear fork outcome differences as high-reliability labels;
- downscale noisy or indistinguishable candidate comparisons.

Gate:

- keep Qwen by default;
- override only when alternative value margin is clear and WM/value uncertainty is low;
- evaluate non-regression against direct Qwen closed-loop rollout.

## Recommended Next Training Loop

Use iterative offline training with online data collection:

1. Train Qwen/WM/value/gate from expert success data plus existing fork/counterfactual transitions.
2. Run closed-loop evaluation.
3. Collect high-uncertainty and high-disagreement states.
4. Fork candidate actions in those states to get counterfactual outcomes.
5. Update per-sample reliability/novelty/learnability scores.
6. Continue offline training.
7. Only consider online RL after the conservative gate is stable and non-regressive.

