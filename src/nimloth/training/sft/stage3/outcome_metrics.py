"""Labeled window/horizon outcome counts, with separate unique-transition support."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F


class OutcomeMetrics:
    def __init__(self):
        # Per horizon: failure TP, FP, FN, TN, BCE sum.
        self.rows = {}
        self.transitions = set()

    def update(self, batch, logits):
        labels = batch.outcome_targets.detach().cpu()
        mask = batch.outcome_mask.detach().cpu() & (batch.sample_weights.detach().cpu()[:, None] > 0)
        logits = logits.detach().float().cpu()
        if logits.shape != labels.shape or mask.shape != labels.shape:
            raise ValueError("outcome metric shapes disagree")
        if not torch.isfinite(logits[mask]).all():
            raise ValueError("nonfinite outcome logits")
        if not ((labels[mask] == 0) | (labels[mask] == 1)).all():
            raise ValueError("outcome metric labels must be binary")
        for h in range(logits.shape[1]):
            valid = mask[:, h]
            predicted_failure = logits[valid, h] < 0
            failure = labels[valid, h] == 0
            values = [int((predicted_failure & failure).sum()),
                      int((predicted_failure & ~failure).sum()),
                      int((~predicted_failure & failure).sum()),
                      int((~predicted_failure & ~failure).sum()),
                      float(F.binary_cross_entropy_with_logits(
                          logits[valid, h], labels[valid, h].float(), reduction="sum"))]
            previous = self.rows.setdefault(h + 1, [0.] * 5)
            self.rows[h + 1] = [a + b for a, b in zip(previous, values, strict=True)]
            for index in batch.next_indices.detach().cpu()[valid, h].tolist():
                # A successor key identifies its single incoming executed action.
                self.transitions.add(batch.state_keys[index])

    def metrics(self):
        parts = [(self.rows, self.transitions)]
        if dist.is_available() and dist.is_initialized():
            parts = [None] * dist.get_world_size()
            dist.all_gather_object(parts, (self.rows, self.transitions))
        return merge_outcome_metrics(parts)


def merge_outcome_metrics(parts):
    rows, transitions = {}, set()
    for rank_rows, rank_transitions in parts:
        transitions.update(rank_transitions)
        for h, values in rank_rows.items():
            rows[h] = [a + b for a, b in zip(rows.get(h, [0.] * 5), values, strict=True)]
    if not rows:
        return {}
    total = [sum(row[i] for row in rows.values()) for i in range(5)]
    result = {"outcome_unique_transition_count": float(len(transitions)),
              "outcome_count": float(sum(total[:4]))}
    for name, (tp, fp, fn, tn, bce) in [(f"h{h}", row) for h, row in sorted(rows.items())] + [("all", total)]:
        prefix = f"outcome_{name}_"
        count = tp + fp + fn + tn
        result.update({prefix + key: float(value) for key, value in {
            "count": count, "success_count": fp+tn, "failure_count": tp+fn,
            "failure_tp": tp, "failure_fp": fp, "failure_fn": fn, "failure_tn": tn}.items()})
        for key, numerator, denominator in (
            ("bce", bce, count), ("accuracy", tp+tn, count),
            ("always_success_accuracy", fp+tn, count),
            ("failure_fraction", tp+fn, count),
            ("failure_precision", tp, tp+fp), ("failure_recall", tp, tp+fn),
            ("failure_f1", 2*tp, 2*tp+fp+fn)):
            defined = denominator > 0
            if key in {"failure_precision", "failure_recall", "failure_f1"}:
                # A single-class evaluation cannot establish failure discrimination.
                defined = defined and (tp + fn > 0) and (fp + tn > 0)
            result[prefix + key + "_defined"] = float(defined)
            if defined:
                result[prefix + key] = numerator / denominator
    return result
