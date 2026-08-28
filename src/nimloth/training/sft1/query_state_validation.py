"""Detached direct-view diagnostics for Query-State CPU/tiny validation.

The report contains no pass/fail threshold and cannot select a checkpoint.  It
exists to prove that raw Query hidden, canonical state, LM CE, and same-forward
action logits can be accumulated without introducing another trainable readout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from nimloth.training.sft1.query_state import (
    QUERY_STATE_OBJECTIVE_VERSION,
    QUERY_STATE_SCHEMA,
    QueryStateObjectiveOutput,
    QueryStateTargets,
)


QUERY_STATE_DIAGNOSTIC_SCHEMA = "nimloth_sft1_query_state_diagnostic_v1"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class QueryStateDiagnosticReport:
    schema: str
    training_schema: str
    objective_version: str
    config_identity: str
    source_manifest_identity: str
    checkpoint_identity: str
    checkpoint_step: int
    sample_count: int
    record_ids: tuple[str, ...]
    metrics: Mapping[str, float]
    diagnostic_only: bool = True
    automatic_model_quality_pass: None = None


class QueryStateDiagnosticAccumulator:
    """Accumulate only detached outputs of the unique direct-state root."""

    def __init__(self) -> None:
        self._raw_query_hidden: list[torch.Tensor] = []
        self._state: list[torch.Tensor] = []
        self._dino: list[torch.Tensor] = []
        self._action_logits: list[torch.Tensor] = []
        self._executed_actions: list[torch.Tensor] = []
        self._record_ids: list[str] = []
        self._loss_sums = {"direct_state_mse": 0.0, "lm_ce": 0.0}
        self._loss_counts = {"direct_state_mse": 0, "lm_ce": 0}

    def add(
        self,
        output: QueryStateObjectiveOutput,
        targets: QueryStateTargets,
        *,
        executed_action_indices: torch.Tensor,
        record_ids: tuple[str, ...],
    ) -> None:
        size = int(output.state.shape[0])
        if (
            output.raw_query_hidden.shape != (size, 16, 2048)
            or output.state.shape != (size, 16, 1024)
            or output.action_logits.shape != (size, 8)
            or targets.dino_regions.shape != (size, 16, 1024)
            or targets.sample_valid.shape != (size,)
            or executed_action_indices.shape != (size,)
            or len(record_ids) != size
        ):
            raise ValueError("Query-State diagnostic input shapes do not align")
        if targets.sample_valid.dtype != torch.bool:
            raise ValueError("Query-State diagnostic sample validity must be boolean")
        if executed_action_indices.dtype != torch.long or torch.any(
            (executed_action_indices < 0) | (executed_action_indices >= 8)
        ):
            raise ValueError("Query-State diagnostic executed actions are invalid")
        if any(not isinstance(value, str) or not value for value in record_ids):
            raise ValueError("Query-State diagnostic record identities are invalid")
        if set(output.loss_sums) != {"direct_state_mse", "lm_ce"} or set(
            output.local_valid_counts
        ) != {"direct_state_mse", "lm_ce"}:
            raise ValueError("Query-State diagnostic active-loss contract changed")
        valid = targets.sample_valid
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        expected_state_count = int(indices.numel()) * 16 * 1024
        if int(output.local_valid_counts["direct_state_mse"]) != expected_state_count:
            raise ValueError("Query-State diagnostic state count disagrees with valid rows")
        if indices.numel():
            self._raw_query_hidden.append(
                output.raw_query_hidden.index_select(0, indices.to(output.raw_query_hidden.device))
                .detach().float().cpu()
            )
            self._state.append(
                output.state.index_select(0, indices.to(output.state.device))
                .detach().float().cpu()
            )
            self._dino.append(
                targets.dino_regions.index_select(0, indices.to(targets.dino_regions.device))
                .detach().float().cpu()
            )
            self._action_logits.append(
                output.action_logits.index_select(0, indices.to(output.action_logits.device))
                .detach().float().cpu()
            )
            self._executed_actions.append(
                executed_action_indices.index_select(
                    0, indices.to(executed_action_indices.device)
                ).detach().long().cpu()
            )
            self._record_ids.extend(record_ids[int(index)] for index in indices.tolist())
        for name in self._loss_sums:
            if name not in output.loss_sums or name not in output.local_valid_counts:
                raise ValueError("Query-State diagnostic loss component is missing")
            value = output.loss_sums[name]
            if value.ndim != 0 or not torch.isfinite(value):
                raise ValueError("Query-State diagnostic loss sum is invalid")
            count = int(output.local_valid_counts[name])
            if count < 0:
                raise ValueError("Query-State diagnostic loss count is invalid")
            self._loss_sums[name] += float(value.detach().item())
            self._loss_counts[name] += count

    def metric_cursor(self) -> Mapping[str, float | int]:
        return {
            "sample_count": len(self._record_ids),
            **{f"sum/{name}": value for name, value in self._loss_sums.items()},
            **{f"count/{name}": value for name, value in self._loss_counts.items()},
        }

    def finalize(
        self,
        *,
        config_identity: str,
        source_manifest_identity: str,
        checkpoint_identity: str,
        checkpoint_step: int,
    ) -> QueryStateDiagnosticReport:
        for name, value in (
            ("config_identity", config_identity),
            ("source_manifest_identity", source_manifest_identity),
            ("checkpoint_identity", checkpoint_identity),
        ):
            if len(value) != 64 or any(char not in _HEX for char in value):
                raise ValueError(f"Query-State diagnostic {name} must be SHA256")
        if checkpoint_step < 0 or not self._record_ids:
            raise ValueError("Query-State diagnostic requires samples and a valid step")
        if any(count < 1 for count in self._loss_counts.values()):
            raise ValueError("Query-State diagnostic requires both active loss counts")

        raw = torch.cat(self._raw_query_hidden, dim=0)
        state = torch.cat(self._state, dim=0)
        dino = torch.cat(self._dino, dim=0)
        action = torch.cat(self._action_logits, dim=0)
        executed = torch.cat(self._executed_actions, dim=0)
        if not all(torch.isfinite(value).all() for value in (raw, state, dino, action)):
            raise ValueError("Query-State diagnostic tensors must be finite")
        one_hot = F.one_hot(executed, num_classes=8).bool()
        executed_logits = action.gather(1, executed[:, None]).squeeze(1)
        alternative = action.masked_fill(one_hot, float("-inf")).max(dim=1).values
        direct_state_mse = float((state - dino).square().mean().item())
        training_state_mse = (
            self._loss_sums["direct_state_mse"]
            / self._loss_counts["direct_state_mse"]
        )
        if not torch.isclose(
            torch.tensor(direct_state_mse),
            torch.tensor(training_state_mse),
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError(
                "Query-State diagnostic direct-state tensors disagree with loss sums"
            )
        metrics = {
            "direct_state/mse": direct_state_mse,
            "direct_state/cosine": float(
                F.cosine_similarity(state, dino, dim=-1).mean().item()
            ),
            "raw_query/norm_mean": float(raw.norm(dim=-1).mean().item()),
            "raw_query/slot_variance": float(raw.var(dim=1, unbiased=False).mean().item()),
            "canonical_state/norm_mean": float(state.norm(dim=-1).mean().item()),
            "canonical_state/slot_variance": float(
                state.var(dim=1, unbiased=False).mean().item()
            ),
            "lm/ce": self._loss_sums["lm_ce"] / self._loss_counts["lm_ce"],
            "action/logit_std": float(action.std(unbiased=False).item()),
            "action/executed_margin_mean": float(
                (executed_logits - alternative).mean().item()
            ),
        }
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise ValueError("Query-State diagnostic metric is non-finite")
        return QueryStateDiagnosticReport(
            schema=QUERY_STATE_DIAGNOSTIC_SCHEMA,
            training_schema=QUERY_STATE_SCHEMA,
            objective_version=QUERY_STATE_OBJECTIVE_VERSION,
            config_identity=config_identity,
            source_manifest_identity=source_manifest_identity,
            checkpoint_identity=checkpoint_identity,
            checkpoint_step=checkpoint_step,
            sample_count=len(self._record_ids),
            record_ids=tuple(self._record_ids),
            metrics=metrics,
        )


__all__ = [
    "QUERY_STATE_DIAGNOSTIC_SCHEMA",
    "QueryStateDiagnosticAccumulator",
    "QueryStateDiagnosticReport",
]
