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


class PolicyStateCaptureWorkerExtension:
    """vLLM ``worker_extension_cls`` for one serial Nimloth rollout request.

    The rollout process generates one request at a time.  A forward hook records
    selected rows in decode order.  ``collective_rpc`` then asks every TP worker
    to resolve the final complete latent block and the following action boundary.
    """

    def nimloth_start_policy_state_capture(
        self,
        latent_token_ids: Sequence[int],
        action_start_token_id: int,
        action_token_ids: Sequence[int],
    ) -> bool:
        latent_ids = tuple(int(value) for value in latent_token_ids)
        action_ids = tuple(int(value) for value in action_token_ids)
        action_start_id = int(action_start_token_id)
        if not latent_ids or len(set(latent_ids)) != len(latent_ids):
            raise ValueError("policy state capture requires unique latent token ids")
        if not action_ids or len(set(action_ids)) != len(action_ids):
            raise ValueError("policy state capture requires unique action token ids")
        if action_start_id in latent_ids:
            raise ValueError("action_start token must differ from latent tokens")
        if getattr(self, "_nimloth_capture_active", False):
            raise RuntimeError("a vLLM policy state capture is already active")

        self._nimloth_latent_token_ids = latent_ids
        self._nimloth_action_start_token_id = action_start_id
        self._nimloth_action_token_ids = action_ids
        self._nimloth_capture_token_ids = frozenset((*latent_ids, action_start_id))
        self._nimloth_capture_entries: list[tuple[int, torch.Tensor]] = []
        self._nimloth_capture_active = True
        if getattr(self, "_nimloth_capture_handle", None) is None:
            model_runner = getattr(self, "model_runner", None)
            model = getattr(model_runner, "model", None)
            if not isinstance(model, torch.nn.Module):
                raise RuntimeError("vLLM worker has no loaded model_runner.model")
            self._nimloth_capture_handle = model.register_forward_hook(
                self._nimloth_capture_forward,
                with_kwargs=True,
            )
        return True

    def _nimloth_capture_forward(
        self,
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        if not getattr(self, "_nimloth_capture_active", False):
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
        wanted = self._nimloth_capture_token_ids
        for index in range(flat_ids.numel()):
            token_id = int(flat_ids[index].item())
            if token_id in wanted:
                # The selected state is tiny (K latent rows plus one action row).
                # Keep it on-device until the LM head has consumed action_hidden.
                self._nimloth_capture_entries.append(
                    (token_id, hidden[index].detach().clone())
                )

    def nimloth_pop_policy_state_capture(self) -> dict[str, torch.Tensor]:
        self._nimloth_capture_active = False
        entries = list(getattr(self, "_nimloth_capture_entries", []))
        self._nimloth_capture_entries = []
        latent_ids = self._nimloth_latent_token_ids
        expected = (*latent_ids, self._nimloth_action_start_token_id)
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
                f"expected={expected}, captured={captured_ids}"
            )

        selected = entries[start : start + width]
        latent_hidden = torch.stack(
            [hidden for _token_id, hidden in selected[:-1]],
            dim=0,
        )
        action_hidden = selected[-1][1].unsqueeze(0)
        model = self.model_runner.model
        logits = model.compute_logits(action_hidden)
        if not isinstance(logits, torch.Tensor) or logits.shape[0] != 1:
            raise RuntimeError("vLLM compute_logits did not return one action row")
        action_indices = torch.tensor(
            self._nimloth_action_token_ids,
            dtype=torch.long,
            device=logits.device,
        )
        action_logits = logits[0].index_select(0, action_indices)
        if not torch.isfinite(action_logits).all():
            raise RuntimeError("vLLM returned non-finite raw action logits")
        return {
            "latent_hidden": latent_hidden.float().cpu(),
            "action_logits": action_logits.float().cpu(),
        }

    def nimloth_abort_policy_state_capture(self) -> bool:
        self._nimloth_capture_active = False
        self._nimloth_capture_entries = []
        return True


def start_policy_state_capture(
    engine: Any,
    *,
    latent_token_ids: Sequence[int],
    action_start_token_id: int,
    action_token_ids: Sequence[int],
) -> None:
    """Enable one serial capture on every vLLM tensor-parallel worker."""

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


def pop_policy_state_capture(engine: Any) -> VLLMPolicyState:
    """Return TP-consistent latent hidden states and raw action logits."""

    results = engine.collective_rpc("nimloth_pop_policy_state_capture")
    if not results or not all(isinstance(value, dict) for value in results):
        raise RuntimeError("vLLM workers returned incomplete policy states")
    reference = results[0]
    for field in ("latent_hidden", "action_logits"):
        value = reference.get(field)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"vLLM policy state is missing tensor field {field}")
        for rank, rank_result in enumerate(results[1:], start=1):
            rank_value = rank_result.get(field)
            if (
                not isinstance(rank_value, torch.Tensor)
                or rank_value.shape != value.shape
                or not torch.allclose(rank_value, value, rtol=1e-4, atol=1e-5)
            ):
                raise RuntimeError(
                    f"vLLM TP rank {rank} policy state field {field} differs "
                    "from rank 0"
                )
    return VLLMPolicyState(
        latent_hidden=reference["latent_hidden"],
        action_logits=reference["action_logits"],
    )


def abort_policy_state_capture(engine: Any) -> None:
    """Clear worker capture state after a failed generation."""

    engine.collective_rpc("nimloth_abort_policy_state_capture")


__all__ = [
    "PolicyStateCaptureWorkerExtension",
    "VLLMPolicyState",
    "abort_policy_state_capture",
    "pop_policy_state_capture",
    "start_policy_state_capture",
]
