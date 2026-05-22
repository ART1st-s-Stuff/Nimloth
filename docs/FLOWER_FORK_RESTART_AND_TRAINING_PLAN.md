# Flower Fork Restart And Training Plan

Date: 2026-05-23

## Goal

Restart fork-based data collection in a task-agnostic way, then use the collected data to improve the current Flower architecture without prematurely switching to online RL.

The intended decision hierarchy remains:

- Qwen planner: deep reasoning and action proposal.
- WM predictor: fast transition prediction.
- Value head / ranker: fast task-utility estimation over candidate actions.
- Gate: conservative override layer that should not replace Qwen by default.

## Core Principle

The fork pipeline should no longer be treated as a way to collect only successful rollouts.

Instead, it should collect counterfactual transition data:

```text
same state + different candidate actions -> different observed consequences
```

This makes failed episodes still useful, as long as the transition outcomes are real and well-labeled.

## Restarting Fork Collection

Keep the existing fork collector infrastructure, rollout logging, and per-step transition recording.

Replace task-specific trigger rules with task-agnostic fork triggers.

Avoid rules like:

- forward failed -> fork
- target invisible -> rotate
- wall ahead -> try lateral motion

Use generic triggers instead:

- Qwen action distribution entropy is high.
- WM ensemble disagreement is high.
- Value ensemble disagreement is high.
- Qwen proposed action conflicts with WM/value predictions.
- Predicted next-belief variance is high.
- Current state embedding is far from the training distribution.
- Recent prediction error is increasing.

Candidate actions should come from a generic pool:

- Qwen proposed action
- top-k actions by planner prior
- top-k actions by learned value
- high-uncertainty actions
- a small random action component for coverage

Each fork state should record:

- current image / belief / latent
- complete action history
- Qwen proposed action
- Qwen raw reasoning / response / confidence
- candidate actions considered
- actual next observation / next latent for each candidate
- short-horizon continuation outcome for each candidate
- WM ensemble predictions
- value ensemble predictions
- uncertainty metrics

## Training After Collection

Training should proceed in layers rather than mixing all objectives at once.

WM predictor:

```text
current belief + action -> delta next latent / next belief
```

Uncertainty should come from ensemble disagreement or residual behavior, not from many extra heads. The first requirement is action sensitivity: same state plus different action should produce different predicted deltas.

Value head / ranker:

```text
same state:
  if outcome(action_a) better than outcome(action_b),
  then value(a) > value(b)
```

Do not train only on sparse final success. Use pairwise ranking within a fork group, and lower reliability or skip comparisons when fork outcomes are noisy or indistinguishable.

Qwen planner:

```text
effective_lr = base_lr * reliability * novelty * learnability
```

If `effective_lr < threshold`, skip the sample. This should reduce the influence of repeated already-learned samples without introducing action-specific weighting.

Gate:

```text
override only if:
  alternative value margin is clear
  WM uncertainty is low
  value uncertainty is low
otherwise keep Qwen
```

Non-regression against direct Qwen is the hard constraint.

## Online Training

Do not switch to online RL yet.

Current reasons to avoid PPO/GRPO immediately:

- WM/value are not yet well-calibrated.
- Fork rollout success is still low.
- Qwen still carries a strong action prior and only recently became format-stable.
- Online RL would likely reinforce bad behavior too early.
- Reward hacking risk is still high.

The preferred loop is:

```text
online collection
offline training
closed-loop evaluation
repeat
```

Only consider GRPO/PPO after:

- direct Qwen + gate is non-regressive in closed-loop rollout;
- value/ranking quality correlates with rollout success;
- uncertainty predicts failure or OOD behavior;
- reward is generic and hard to exploit;
- rollout success is no longer extremely low;
- checkpoint/resume and automated evaluation are stable.

First consider GRPO over fork groups. Only later consider PPO, and only if the gate/policy stack is already stable.

## Recommended Phases

1. Restart uncertainty-driven fork collection.
2. Train WM delta predictor ensemble.
3. Train pairwise value/ranker on fork outcome comparisons.
4. Train conservative gate and verify non-regression against direct Qwen.
5. Run closed-loop evaluation and collect new hard states.
6. If ranking and gate are stable, try GRPO on fork-group comparisons.
7. Only after the above is stable, consider PPO or broader online RL.

