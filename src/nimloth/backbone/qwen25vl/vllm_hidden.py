"""Expose selected Qwen states from the existing vLLM rollout forward.

The extension captures only Nimloth's generated latent-query tokens and the
``action_start`` boundary.  It does not retain prompt activations and does not
run Qwen a second time.  The action logits are computed from the already
captured boundary hidden state with the loaded vLLM model's normal LM head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class VLLMPolicyState:
    """Small rollout result returned by the vLLM tensor-parallel workers."""

    latent_hidden: torch.Tensor
    action_logits: torch.Tensor


@dataclass(frozen=True)
class _PolicyStateCaptureSpec:
    latent_token_ids: tuple[int, ...]
    action_start_token_id: int
    action_token_ids: tuple[int, ...]
    capture_token_ids: frozenset[int]


def _policy_state_capture_spec(
    latent_token_ids: Sequence[int],
    action_start_token_id: int,
    action_token_ids: Sequence[int],
) -> _PolicyStateCaptureSpec:
    latent_ids = tuple(int(value) for value in latent_token_ids)
    action_ids = tuple(int(value) for value in action_token_ids)
    action_start_id = int(action_start_token_id)
    if not latent_ids or len(set(latent_ids)) != len(latent_ids):
        raise ValueError("policy state capture requires unique latent token ids")
    if not action_ids or len(set(action_ids)) != len(action_ids):
        raise ValueError("policy state capture requires unique action token ids")
    if action_start_id in latent_ids:
        raise ValueError("action_start token must differ from latent tokens")
    return _PolicyStateCaptureSpec(
        latent_token_ids=latent_ids,
        action_start_token_id=action_start_id,
        action_token_ids=action_ids,
        capture_token_ids=frozenset((*latent_ids, action_start_id)),
    )


def _hidden_tensor(model_output: Any) -> torch.Tensor:
    """Return the final hidden tensor from a normal vLLM model forward."""

    value = model_output[0] if isinstance(model_output, tuple) else model_output
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise RuntimeError(
            "vLLM hidden capture expected model output with shape (N, D)"
        )
    return value


def _input_ids(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor | None:
    """Read ``input_ids`` from either supported vLLM model call convention."""

    value = kwargs.get("input_ids")
    if value is None and args:
        value = args[0]
    return value if isinstance(value, torch.Tensor) else None


def _runner_input_ids(model_runner: Any, row_count: int) -> torch.Tensor | None:
    """Read the V1 runner token buffer used for multimodal embedding input.

    vLLM 0.11 always calls multimodal models with ``input_ids=None`` and
    ``inputs_embeds=...``.  The aligned token IDs remain in the runner's
    persistent ``input_ids.gpu`` buffer, so state capture must read that same
    buffer instead of trying to recover IDs from embeddings.
    """

    input_buffer = getattr(getattr(model_runner, "input_ids", None), "gpu", None)
    if not isinstance(input_buffer, torch.Tensor) or input_buffer.ndim != 1:
        return None
    if input_buffer.numel() < row_count:
        raise RuntimeError(
            "vLLM runner input ID buffer is shorter than model hidden states: "
            f"{input_buffer.numel()} < {row_count}"
        )
    return input_buffer[:row_count]


def _cpu_offsets(value: Any, count: int) -> tuple[int, ...] | None:
    """Normalize vLLM's Tensor/CpuGpuBuffer query offsets."""

    cpu_value = getattr(value, "cpu", None)
    if isinstance(cpu_value, torch.Tensor):
        value = cpu_value
    elif callable(cpu_value):
        value = cpu_value()
    elif isinstance(getattr(value, "gpu", None), torch.Tensor):
        value = value.gpu.cpu()
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        return None
    if value.numel() < count:
        return None
    return tuple(int(item) for item in value[:count].tolist())


def _request_segments(
    model_runner: Any,
    row_count: int,
) -> tuple[tuple[str, int, int], ...]:
    """Map flattened vLLM forward rows back to request identities."""

    input_batch = getattr(model_runner, "input_batch", None)
    request_ids = getattr(input_batch, "req_ids", None)
    if request_ids is None:
        return (("__serial__", 0, row_count),)
    num_reqs = int(getattr(input_batch, "num_reqs", len(request_ids)))
    request_ids = tuple(str(value) for value in list(request_ids)[:num_reqs])
    if len(request_ids) != num_reqs or len(set(request_ids)) != num_reqs:
        raise RuntimeError("vLLM capture requires unique active request IDs")
    offsets = None
    for candidate in (
        getattr(input_batch, "query_start_loc", None),
        getattr(model_runner, "query_start_loc", None),
    ):
        offsets = _cpu_offsets(candidate, num_reqs + 1)
        if offsets is not None:
            break
    if offsets is None:
        raise RuntimeError("vLLM capture cannot read per-request query offsets")
    if (
        offsets[0] != 0
        or offsets[-1] > row_count
        or any(left > right for left, right in zip(offsets, offsets[1:]))
    ):
        raise RuntimeError(
            "vLLM request offsets do not align with captured hidden rows: "
            f"offsets={offsets}, rows={row_count}"
        )
    return tuple(
        (request_id, offsets[index], offsets[index + 1])
        for index, request_id in enumerate(request_ids)
    )


class PolicyStateCaptureWorkerExtension:
    """vLLM ``worker_extension_cls`` for identity-safe Nimloth state capture.

    V1 flattens active requests into one token tensor.  The hook uses the
    runner's request IDs and query offsets to retain a separate decode stream
    per request; this is required before VAGEN may batch active environments.
    """

    def _nimloth_ensure_capture_hook(self) -> None:
        if getattr(self, "_nimloth_capture_handle", None) is not None:
            return
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None)
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("vLLM worker has no loaded model_runner.model")
        self._nimloth_capture_handle = model.register_forward_hook(
            self._nimloth_capture_forward,
            with_kwargs=True,
        )

    def nimloth_start_policy_state_capture(
        self,
        latent_token_ids: Sequence[int],
        action_start_token_id: int,
        action_token_ids: Sequence[int],
    ) -> bool:
        spec = _policy_state_capture_spec(
            latent_token_ids,
            action_start_token_id,
            action_token_ids,
        )
        request_specs = getattr(self, "_nimloth_request_capture_specs", {})
        if getattr(self, "_nimloth_capture_active", False) or request_specs:
            raise RuntimeError("a vLLM policy state capture is already active")

        self._nimloth_ensure_capture_hook()
        self._nimloth_latent_token_ids = spec.latent_token_ids
        self._nimloth_action_start_token_id = spec.action_start_token_id
        self._nimloth_action_token_ids = spec.action_token_ids
        self._nimloth_capture_token_ids = spec.capture_token_ids
        self._nimloth_capture_entries_by_request: dict[
            str, list[tuple[int, torch.Tensor]]
        ] = {}
        self._nimloth_capture_active = True
        return True

    def nimloth_start_policy_state_capture_for_request(
        self,
        request_id: str,
        latent_token_ids: Sequence[int],
        action_start_token_id: int,
        action_token_ids: Sequence[int],
    ) -> bool:
        """Enable capture for one active async vLLM request."""

        identity = str(request_id)
        if not identity:
            raise ValueError("policy state capture requires a request id")
        if getattr(self, "_nimloth_capture_active", False):
            raise RuntimeError("serial policy state capture is already active")
        specs = getattr(self, "_nimloth_request_capture_specs", None)
        if specs is None:
            specs = {}
            self._nimloth_request_capture_specs = specs
        prepared = getattr(self, "_nimloth_prepared_request_captures", {})
        if identity in specs or identity in prepared:
            raise RuntimeError(
                f"policy state capture for request {identity!r} is already active"
            )
        spec = _policy_state_capture_spec(
            latent_token_ids,
            action_start_token_id,
            action_token_ids,
        )
        self._nimloth_ensure_capture_hook()
        specs[identity] = spec
        entries = getattr(self, "_nimloth_request_capture_entries", None)
        if entries is None:
            entries = {}
            self._nimloth_request_capture_entries = entries
        entries[identity] = []
        return True

    def _nimloth_capture_forward(
        self,
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        serial_active = getattr(self, "_nimloth_capture_active", False)
        request_specs = getattr(self, "_nimloth_request_capture_specs", {})
        if not serial_active and not request_specs:
            return
        hidden = _hidden_tensor(output)
        input_ids = _input_ids(args, kwargs)
        if input_ids is None:
            input_ids = _runner_input_ids(self.model_runner, hidden.shape[0])
        if input_ids is None:
            return
        flat_ids = input_ids.reshape(-1)
        if hidden.shape[0] != flat_ids.numel():
            raise RuntimeError(
                "vLLM input IDs do not align with captured hidden states: "
                f"{flat_ids.numel()} != {hidden.shape[0]}"
            )
        serial_entries = getattr(
            self,
            "_nimloth_capture_entries_by_request",
            {},
        )
        scoped_entries = getattr(
            self,
            "_nimloth_request_capture_entries",
            {},
        )
        for request_id, start, end in _request_segments(
            self.model_runner,
            hidden.shape[0],
        ):
            scoped_spec = request_specs.get(request_id)
            if scoped_spec is not None:
                wanted = scoped_spec.capture_token_ids
                request_entries = scoped_entries[request_id]
            elif serial_active:
                wanted = self._nimloth_capture_token_ids
                request_entries = serial_entries.setdefault(request_id, [])
            else:
                continue
            for index in range(start, end):
                token_id = int(flat_ids[index].item())
                if token_id in wanted:
                    # The selected state is tiny (K latent rows plus one action row).
                    # Keep it on-device until the LM head has consumed action_hidden.
                    request_entries.append(
                        (token_id, hidden[index].detach().clone())
                    )

    def _nimloth_select_latent_state(
        self,
        entries: list[tuple[int, torch.Tensor]],
        *,
        request_id: str,
        spec: _PolicyStateCaptureSpec,
    ) -> torch.Tensor:
        expected = spec.latent_token_ids
        captured_ids = tuple(token_id for token_id, _hidden in entries)
        start = None
        width = len(expected)
        for index in range(len(captured_ids) - width, -1, -1):
            if captured_ids[index : index + width] == expected:
                start = index
                break
        if start is None:
            raise RuntimeError(
                "vLLM did not capture the generated terminal latent sequence: "
                f"request_id={request_id!r}, expected={expected}, "
                f"captured={captured_ids}"
            )
        return torch.stack(
            [hidden for _token_id, hidden in entries[start : start + width]],
            dim=0,
        )

    def _nimloth_select_policy_state(
        self,
        entries: list[tuple[int, torch.Tensor]],
        *,
        request_id: str,
        spec: _PolicyStateCaptureSpec,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (*spec.latent_token_ids, spec.action_start_token_id)
        captured_ids = tuple(token_id for token_id, _hidden in entries)
        start = None
        width = len(expected)
        for index in range(len(captured_ids) - width, -1, -1):
            if captured_ids[index : index + width] == expected:
                start = index
                break
        if start is None:
            raise RuntimeError(
                "vLLM did not capture the generated policy state sequence: "
                f"request_id={request_id!r}, expected={expected}, "
                f"captured={captured_ids}"
            )

        selected = entries[start : start + width]
        return (
            torch.stack(
                [hidden for _token_id, hidden in selected[:-1]],
                dim=0,
            ),
            selected[-1][1].unsqueeze(0),
        )

    def _nimloth_action_logits(
        self,
        action_hidden: torch.Tensor,
        *,
        spec: _PolicyStateCaptureSpec,
    ) -> torch.Tensor:
        model = self.model_runner.model
        logits = model.compute_logits(action_hidden)
        if not isinstance(logits, torch.Tensor) or logits.shape[0] != 1:
            raise RuntimeError("vLLM compute_logits did not return one action row")
        action_indices = torch.tensor(
            spec.action_token_ids,
            dtype=torch.long,
            device=logits.device,
        )
        action_logits = logits[0].index_select(0, action_indices)
        if not torch.isfinite(action_logits).all():
            raise RuntimeError("vLLM returned non-finite raw action logits")
        return action_logits

    def _nimloth_resolve_policy_state(
        self,
        entries: list[tuple[int, torch.Tensor]],
        *,
        request_id: str,
        spec: _PolicyStateCaptureSpec | None = None,
    ) -> dict[str, list[float] | list[list[float]]]:
        if spec is None:
            spec = _PolicyStateCaptureSpec(
                latent_token_ids=self._nimloth_latent_token_ids,
                action_start_token_id=self._nimloth_action_start_token_id,
                action_token_ids=self._nimloth_action_token_ids,
                capture_token_ids=self._nimloth_capture_token_ids,
            )
        latent_hidden, action_hidden = self._nimloth_select_policy_state(
            entries,
            request_id=request_id,
            spec=spec,
        )
        action_logits = self._nimloth_action_logits(action_hidden, spec=spec)
        # vLLM V1 UtilityResult intentionally omits arbitrary nested Python
        # type metadata unless insecure serialization is enabled. Return plain
        # numeric containers and rebuild tensors in the trusted frontend.
        return {
            "latent_hidden": latent_hidden.float().cpu().tolist(),
            "action_logits": action_logits.float().cpu().tolist(),
        }

    def nimloth_pop_policy_state_captures(
        self,
    ) -> dict[str, dict[str, list[float] | list[list[float]]]]:
        self._nimloth_capture_active = False
        entries_by_request = dict(
            getattr(self, "_nimloth_capture_entries_by_request", {})
        )
        self._nimloth_capture_entries_by_request = {}
        if not entries_by_request:
            raise RuntimeError("vLLM captured no policy-state requests")
        return {
            request_id: self._nimloth_resolve_policy_state(
                entries,
                request_id=request_id,
            )
            for request_id, entries in entries_by_request.items()
        }

    def nimloth_pop_policy_state_capture(
        self,
    ) -> dict[str, list[float] | list[list[float]]]:
        results = self.nimloth_pop_policy_state_captures()
        if len(results) != 1:
            raise RuntimeError(
                "serial policy-state capture received multiple requests: "
                f"{sorted(results)}"
            )
        return next(iter(results.values()))

    def nimloth_pop_latent_state_capture_for_request(
        self,
        request_id: str,
    ) -> dict[str, list[list[float]]]:
        """Return K latent rows without requiring or scoring action_start."""

        identity = str(request_id)
        specs = getattr(self, "_nimloth_request_capture_specs", {})
        if identity not in specs:
            raise RuntimeError(
                f"policy state capture for request {identity!r} is not active"
            )
        entries = getattr(self, "_nimloth_request_capture_entries", {})
        try:
            latent_hidden = self._nimloth_select_latent_state(
                entries.get(identity, []),
                request_id=identity,
                spec=specs[identity],
            )
            return {"latent_hidden": latent_hidden.float().cpu().tolist()}
        finally:
            specs.pop(identity, None)
            entries.pop(identity, None)
            getattr(self, "_nimloth_prepared_request_captures", {}).pop(
                identity,
                None,
            )

    def nimloth_prepare_policy_state_capture_for_request(
        self,
        request_id: str,
    ) -> dict[str, list[list[float]]]:
        """Validate one rank before any TP LM-head collective is entered."""

        identity = str(request_id)
        specs = getattr(self, "_nimloth_request_capture_specs", {})
        if identity not in specs:
            raise RuntimeError(
                f"policy state capture for request {identity!r} is not active"
            )
        entries = getattr(self, "_nimloth_request_capture_entries", {})
        spec = specs[identity]
        latent_hidden, action_hidden = self._nimloth_select_policy_state(
            entries.get(identity, []),
            request_id=identity,
            spec=spec,
        )
        prepared = getattr(self, "_nimloth_prepared_request_captures", None)
        if prepared is None:
            prepared = {}
            self._nimloth_prepared_request_captures = prepared
        prepared[identity] = (spec, latent_hidden, action_hidden)
        del specs[identity]
        entries.pop(identity, None)
        return {"latent_hidden": latent_hidden.float().cpu().tolist()}

    def nimloth_finish_policy_state_capture_for_request(
        self,
        request_id: str,
    ) -> dict[str, list[float]]:
        """Compute raw action logits after every TP rank reported readiness."""

        identity = str(request_id)
        prepared = getattr(self, "_nimloth_prepared_request_captures", {})
        if identity not in prepared:
            raise RuntimeError(
                f"prepared policy state for request {identity!r} is not active"
            )
        spec, _latent_hidden, action_hidden = prepared[identity]
        try:
            action_logits = self._nimloth_action_logits(
                action_hidden,
                spec=spec,
            )
            return {"action_logits": action_logits.float().cpu().tolist()}
        finally:
            prepared.pop(identity, None)

    def _nimloth_tensor_parallel_rank(self) -> int:
        """Return the TP-local rank used to own the single planning replica."""

        try:
            from vllm.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
            )

            rank = int(get_tensor_model_parallel_rank())
        except (ImportError, RuntimeError):
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                rank = 0
            else:
                rank = int(torch.distributed.get_rank())
        if rank < 0:
            raise RuntimeError("vLLM tensor-parallel rank must be non-negative")
        return rank

    def nimloth_install_frozen_k4_planner(
        self,
        transport_path: str,
        expected_snapshot_id: str,
        expected_source_step: int,
        expected_contract_id: str,
        expected_activation_version: int,
    ) -> dict[str, Any]:
        """Install one immutable full planning snapshot on TP rank zero only."""

        from pathlib import Path

        if not isinstance(transport_path, str) or not transport_path:
            raise ValueError("frozen K4 planner transport_path must be non-empty")
        if not isinstance(expected_snapshot_id, str) or not expected_snapshot_id:
            raise ValueError("frozen K4 planner snapshot id must be non-empty")
        if not isinstance(expected_contract_id, str) or not expected_contract_id:
            raise ValueError("frozen K4 planner contract id must be non-empty")
        for field, value in (
            ("source_step", expected_source_step),
            ("activation_version", expected_activation_version),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"frozen K4 planner {field} must be non-negative int")
        rank = self._nimloth_tensor_parallel_rank()
        snapshot = None
        if rank == 0:
            from nimloth.training.rl.joint_planner import (
                load_frozen_planning_snapshot_file,
            )

            model = getattr(getattr(self, "model_runner", None), "model", None)
            if not isinstance(model, torch.nn.Module):
                raise RuntimeError("vLLM rank-zero planner has no loaded model")
            parameter = next(model.parameters(), None)
            if parameter is None:
                raise RuntimeError("vLLM rank-zero model has no parameter device")
            snapshot = load_frozen_planning_snapshot_file(
                Path(transport_path),
                device=parameter.device,
            )
            if (
                snapshot.snapshot_id != expected_snapshot_id
                or snapshot.source_step != expected_source_step
                or snapshot.contract_id != expected_contract_id
            ):
                raise ValueError(
                    "frozen K4 planner transport identity does not match install request"
                )
        self._nimloth_frozen_k4_planner = snapshot
        self._nimloth_frozen_k4_planner_identity = {
            "transport_path": str(Path(transport_path).resolve()),
            "snapshot_id": expected_snapshot_id,
            "source_step": expected_source_step,
            "contract_id": expected_contract_id,
            "activation_version": expected_activation_version,
        }
        return {
            **self._nimloth_frozen_k4_planner_identity,
            "tensor_parallel_rank": rank,
            "owns_planner": rank == 0,
        }

    def nimloth_score_frozen_k4_planner(
        self,
        latent_hidden: Sequence[Sequence[float]],
        expected_snapshot_id: str,
        expected_activation_version: int,
    ) -> dict[str, Any]:
        """Run direct Q and K4 MCTS only on the installed TP-rank-zero model."""

        import time

        identity = getattr(self, "_nimloth_frozen_k4_planner_identity", None)
        if not isinstance(identity, dict):
            raise RuntimeError("frozen K4 planner is not installed")
        if (
            identity["snapshot_id"] != expected_snapshot_id
            or identity["activation_version"] != expected_activation_version
        ):
            raise ValueError("frozen K4 planner score identity mismatch")
        rank = self._nimloth_tensor_parallel_rank()
        base = {
            "snapshot_id": identity["snapshot_id"],
            "source_step": identity["source_step"],
            "contract_id": identity["contract_id"],
            "activation_version": identity["activation_version"],
            "tensor_parallel_rank": rank,
        }
        if rank != 0:
            return {**base, "scored": False}
        snapshot = getattr(self, "_nimloth_frozen_k4_planner", None)
        if snapshot is None:
            raise RuntimeError("TP rank zero has no frozen K4 planner module")
        parameter = next(snapshot.parameters(), None)
        if parameter is None:
            raise RuntimeError("frozen K4 planner has no parameters")
        hidden = torch.as_tensor(
            latent_hidden,
            dtype=torch.float32,
            device=parameter.device,
        ).unsqueeze(0)
        if hidden.ndim != 3 or not torch.isfinite(hidden).all():
            raise ValueError("frozen K4 planner latent hidden is invalid")
        started = time.perf_counter()
        score = snapshot.score(hidden)
        latency = time.perf_counter() - started
        return {
            **base,
            "scored": True,
            "score_dtype": snapshot.score_dtype,
            "planning_config": snapshot.planning_config.to_mapping(),
            "direct_all_action_q": score.direct_all_action_q[0].float().cpu().tolist(),
            "planner_root_mean_values": score.planner_root_mean_values[0]
            .float()
            .cpu()
            .tolist(),
            "planner_root_visit_counts": score.root_visit_counts[0].cpu().tolist(),
            "candidate_sequences": score.candidate_sequences[0].cpu().tolist(),
            "candidate_mean_values": score.candidate_mean_values[0]
            .float()
            .cpu()
            .tolist(),
            "candidate_visit_counts": score.candidate_visit_counts[0].cpu().tolist(),
            "planner_latency_seconds": float(latency),
        }

    def nimloth_abort_policy_state_capture(self) -> bool:
        self._nimloth_capture_active = False
        self._nimloth_capture_entries_by_request = {}
        return True

    def nimloth_abort_policy_state_capture_for_request(
        self,
        request_id: str,
    ) -> bool:
        """Discard one async request without disturbing concurrent captures."""

        identity = str(request_id)
        specs = getattr(self, "_nimloth_request_capture_specs", {})
        specs.pop(identity, None)
        getattr(self, "_nimloth_request_capture_entries", {}).pop(identity, None)
        getattr(self, "_nimloth_prepared_request_captures", {}).pop(identity, None)
        # Cleanup is intentionally idempotent so a partially successful TP pop
        # cannot mask the original generation/capture exception.
        return True


async def async_start_policy_state_capture_for_request(
    engine: Any,
    *,
    request_id: str,
    latent_token_ids: Sequence[int],
    action_start_token_id: int,
    action_token_ids: Sequence[int],
) -> None:
    """Enable one request-scoped capture on every async vLLM TP worker."""

    identity = str(request_id)
    if not identity:
        raise ValueError("policy state capture requires a request id")
    try:
        results = await engine.collective_rpc(
            "nimloth_start_policy_state_capture_for_request",
            args=(
                identity,
                tuple(int(value) for value in latent_token_ids),
                int(action_start_token_id),
                tuple(int(value) for value in action_token_ids),
            ),
        )
        if not results or not all(value is True for value in results):
            raise RuntimeError(
                f"not every vLLM worker enabled capture for request {identity!r}"
            )
    except BaseException:
        await engine.collective_rpc(
            "nimloth_abort_policy_state_capture_for_request",
            args=(identity,),
        )
        raise


def start_policy_state_capture(
    engine: Any,
    *,
    latent_token_ids: Sequence[int],
    action_start_token_id: int,
    action_token_ids: Sequence[int],
) -> None:
    """Enable identity-aware capture on every vLLM tensor-parallel worker."""

    results = engine.collective_rpc(
        "nimloth_start_policy_state_capture",
        args=(
            tuple(int(value) for value in latent_token_ids),
            int(action_start_token_id),
            tuple(int(value) for value in action_token_ids),
        ),
    )
    if not results or not all(value is True for value in results):
        raise RuntimeError("not every vLLM worker enabled policy state capture")


def _validated_policy_state(
    results: list[dict[str, Any]],
    *,
    request_id: str,
) -> VLLMPolicyState:
    tensors: list[dict[str, torch.Tensor]] = []
    for rank, result in enumerate(results):
        rank_tensors: dict[str, torch.Tensor] = {}
        for field in ("latent_hidden", "action_logits"):
            value = result.get(field)
            try:
                tensor = torch.as_tensor(value, dtype=torch.float32)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"vLLM TP rank {rank} returned invalid {field} for "
                    f"request {request_id!r}: {type(value).__name__}"
                ) from exc
            if field == "latent_hidden" and tensor.ndim != 2:
                raise RuntimeError(
                    f"vLLM TP rank {rank} latent_hidden must be rank 2"
                )
            if field == "action_logits" and tensor.ndim != 1:
                raise RuntimeError(
                    f"vLLM TP rank {rank} action_logits must be rank 1"
                )
            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"vLLM TP rank {rank} returned non-finite {field}"
                )
            rank_tensors[field] = tensor
        tensors.append(rank_tensors)

    reference = tensors[0]
    for field in ("latent_hidden", "action_logits"):
        value = reference[field]
        for rank, rank_result in enumerate(tensors[1:], start=1):
            rank_value = rank_result[field]
            if rank_value.shape != value.shape or not torch.allclose(
                rank_value,
                value,
                rtol=1e-4,
                atol=1e-5,
            ):
                raise RuntimeError(
                    f"vLLM TP rank {rank} policy state field {field} differs "
                    f"from rank 0 for request {request_id!r}"
                )
    return VLLMPolicyState(
        latent_hidden=reference["latent_hidden"],
        action_logits=reference["action_logits"],
    )


def pop_policy_state_captures(
    engine: Any,
    *,
    request_ids: Sequence[str],
) -> dict[str, VLLMPolicyState]:
    """Return TP-consistent states aligned to explicit vLLM request IDs."""

    expected = tuple(str(value) for value in request_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("policy-state capture requires unique request IDs")
    worker_results = engine.collective_rpc("nimloth_pop_policy_state_captures")
    if not worker_results or not all(
        isinstance(value, dict) for value in worker_results
    ):
        raise RuntimeError("vLLM workers returned incomplete policy states")
    expected_set = set(expected)
    for rank, result in enumerate(worker_results):
        if set(result) != expected_set:
            raise RuntimeError(
                f"vLLM TP rank {rank} request IDs differ from generation: "
                f"expected={sorted(expected_set)}, actual={sorted(result)}"
            )
    return {
        request_id: _validated_policy_state(
            [result[request_id] for result in worker_results],
            request_id=request_id,
        )
        for request_id in expected
    }


def _validated_latent_hidden(
    results: list[dict[str, Any]],
    *,
    request_id: str,
) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for rank, result in enumerate(results):
        if set(result) != {"latent_hidden"}:
            raise RuntimeError(
                f"vLLM TP rank {rank} returned terminal action evidence for "
                f"request {request_id!r}"
            )
        try:
            tensor = torch.as_tensor(result["latent_hidden"], dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"vLLM TP rank {rank} returned invalid terminal latent state"
            ) from exc
        if (
            tensor.ndim != 2
            or tensor.shape[0] < 1
            or tensor.shape[1] < 1
            or not torch.isfinite(tensor).all()
        ):
            raise RuntimeError(
                f"vLLM TP rank {rank} returned invalid terminal latent state"
            )
        tensors.append(tensor)
    reference = tensors[0]
    for rank, value in enumerate(tensors[1:], start=1):
        if value.shape != reference.shape or not torch.allclose(
            value,
            reference,
            rtol=1e-4,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"vLLM TP rank {rank} terminal latent state differs from rank 0 "
                f"for request {request_id!r}"
            )
    return reference


async def async_pop_latent_state_capture_for_request(
    engine: Any,
    *,
    request_id: str,
) -> torch.Tensor:
    """Return TP-consistent K rows for a terminal trace without LM-head work."""

    identity = str(request_id)
    results = await engine.collective_rpc(
        "nimloth_pop_latent_state_capture_for_request",
        args=(identity,),
    )
    if not results or not all(isinstance(value, dict) for value in results):
        raise RuntimeError(
            f"vLLM workers returned incomplete terminal state for request {identity!r}"
        )
    return _validated_latent_hidden(results, request_id=identity)


async def async_pop_policy_state_capture_for_request(
    engine: Any,
    *,
    request_id: str,
) -> VLLMPolicyState:
    """Return one async state with a TP-safe two-phase LM-head protocol."""

    identity = str(request_id)
    prepared = await engine.collective_rpc(
        "nimloth_prepare_policy_state_capture_for_request",
        args=(identity,),
    )
    if not prepared or not all(isinstance(value, dict) for value in prepared):
        raise RuntimeError(
            f"vLLM workers did not prepare state for request {identity!r}"
        )
    logits = await engine.collective_rpc(
        "nimloth_finish_policy_state_capture_for_request",
        args=(identity,),
    )
    if not logits or not all(isinstance(value, dict) for value in logits):
        raise RuntimeError(
            f"vLLM workers returned incomplete logits for request {identity!r}"
        )
    if len(prepared) != len(logits):
        raise RuntimeError(
            f"vLLM TP worker count changed while capturing request {identity!r}"
        )
    results = [
        {
            "latent_hidden": hidden["latent_hidden"],
            "action_logits": action["action_logits"],
        }
        for hidden, action in zip(prepared, logits, strict=True)
    ]
    return _validated_policy_state(results, request_id=identity)


def pop_policy_state_capture(engine: Any) -> VLLMPolicyState:
    """Return one serial TP-consistent latent state and raw action logits."""

    results = engine.collective_rpc("nimloth_pop_policy_state_capture")
    if not results or not all(isinstance(value, dict) for value in results):
        raise RuntimeError("vLLM workers returned incomplete policy states")
    return _validated_policy_state(results, request_id="__serial__")


async def async_install_frozen_k4_planner(
    engine: Any,
    *,
    transport_path: str,
    expected_snapshot_id: str,
    expected_source_step: int,
    expected_contract_id: str,
    expected_activation_version: int,
) -> dict[str, Any]:
    """Install one shared-file transport and prove exactly TP rank zero owns it."""

    results = await engine.collective_rpc(
        "nimloth_install_frozen_k4_planner",
        args=(
            transport_path,
            expected_snapshot_id,
            expected_source_step,
            expected_contract_id,
            expected_activation_version,
        ),
    )
    if not results or not all(isinstance(value, dict) for value in results):
        raise RuntimeError("vLLM workers returned incomplete K4 planner install status")
    owners = [value for value in results if value.get("owns_planner") is True]
    if len(owners) != 1 or owners[0].get("tensor_parallel_rank") != 0:
        raise RuntimeError("exactly TP rank zero must own the frozen K4 planner")
    for value in results:
        if (
            value.get("snapshot_id") != expected_snapshot_id
            or value.get("source_step") != expected_source_step
            or value.get("contract_id") != expected_contract_id
            or value.get("activation_version") != expected_activation_version
        ):
            raise RuntimeError("vLLM workers disagree on frozen K4 planner identity")
    return owners[0]


async def async_score_frozen_k4_planner(
    engine: Any,
    *,
    latent_hidden: torch.Tensor,
    expected_snapshot_id: str,
    expected_activation_version: int,
) -> dict[str, Any]:
    """Score one captured real state and accept only TP-rank-zero output."""

    if (
        not isinstance(latent_hidden, torch.Tensor)
        or latent_hidden.ndim != 2
        or not torch.isfinite(latent_hidden).all()
    ):
        raise ValueError("K4 planner scoring requires finite rank-2 latent hidden")
    results = await engine.collective_rpc(
        "nimloth_score_frozen_k4_planner",
        args=(
            latent_hidden.float().cpu().tolist(),
            expected_snapshot_id,
            expected_activation_version,
        ),
    )
    if not results or not all(isinstance(value, dict) for value in results):
        raise RuntimeError("vLLM workers returned incomplete K4 planner results")
    scored = [value for value in results if value.get("scored") is True]
    if len(scored) != 1 or scored[0].get("tensor_parallel_rank") != 0:
        raise RuntimeError("exactly TP rank zero must return K4 planning scores")
    for value in results:
        if (
            value.get("snapshot_id") != expected_snapshot_id
            or value.get("activation_version") != expected_activation_version
        ):
            raise RuntimeError("vLLM workers scored different K4 snapshot identities")
    return scored[0]


async def async_abort_policy_state_capture_for_request(
    engine: Any,
    *,
    request_id: str,
) -> None:
    """Clear one failed async request while preserving other active captures."""

    identity = str(request_id)
    results = await engine.collective_rpc(
        "nimloth_abort_policy_state_capture_for_request",
        args=(identity,),
    )
    if not results or not all(value is True for value in results):
        raise RuntimeError(
            f"policy state capture for request {identity!r} is not active"
        )


def abort_policy_state_capture(engine: Any) -> None:
    """Clear worker capture state after a failed generation."""

    engine.collective_rpc("nimloth_abort_policy_state_capture")


__all__ = [
    "PolicyStateCaptureWorkerExtension",
    "VLLMPolicyState",
    "abort_policy_state_capture",
    "async_abort_policy_state_capture_for_request",
    "async_install_frozen_k4_planner",
    "async_pop_latent_state_capture_for_request",
    "async_pop_policy_state_capture_for_request",
    "async_score_frozen_k4_planner",
    "async_start_policy_state_capture_for_request",
    "pop_policy_state_capture",
    "pop_policy_state_captures",
    "start_policy_state_capture",
]
