"""Selected-state capture from one vLLM Qwen rollout forward."""

from __future__ import annotations

import pytest
import torch

from nimloth.backbone.qwen25vl.vllm_hidden import (
    PolicyStateCaptureWorkerExtension,
    pop_policy_state_capture,
    start_policy_state_capture,
)


class _Model(torch.nn.Module):
    def forward(self, input_ids=None, inputs_embeds=None, **_kwargs):
        if input_ids is None:
            assert inputs_embeds is not None
            input_ids = inputs_embeds[:, 0].long()
        return torch.stack(
            (input_ids.float(), input_ids.float() + 0.5),
            dim=-1,
        )

    def compute_logits(self, hidden):
        weights = torch.arange(256, dtype=hidden.dtype, device=hidden.device)
        return hidden[:, :1] * weights.unsqueeze(0)


class _Worker(PolicyStateCaptureWorkerExtension):
    def __init__(self) -> None:
        self.model_runner = type(
            "Runner",
            (),
            {
                "model": _Model(),
                "input_ids": type(
                    "InputBuffer",
                    (),
                    {"gpu": torch.empty(0, dtype=torch.long)},
                )(),
            },
        )()


class _Engine:
    def __init__(self, workers) -> None:
        self.workers = workers

    def collective_rpc(self, method, args=()):
        return [getattr(worker, method)(*args) for worker in self.workers]


def _start(worker) -> None:
    worker.nimloth_start_policy_state_capture((101, 102), 103, (10, 20))


def test_worker_returns_last_complete_policy_state_sequence() -> None:
    worker = _Worker()
    _start(worker)
    # Prompt history may contain an older latent block and action boundary.
    worker.model_runner.model(torch.tensor([4, 101, 102, 103, 5]))
    # Newly generated tokens arrive in separate decode forwards.
    worker.model_runner.model(torch.tensor([101]))
    worker.model_runner.model(torch.tensor([102]))
    worker.model_runner.model(torch.tensor([103]))

    result = worker.nimloth_pop_policy_state_capture()

    assert result["latent_hidden"] == [
        [101.0, 101.5],
        [102.0, 102.5],
    ]
    assert result["action_logits"] == [1030.0, 2060.0]


def test_frontend_requires_tensor_parallel_policy_state_parity() -> None:
    workers = [_Worker(), _Worker()]
    engine = _Engine(workers)
    start_policy_state_capture(
        engine,
        latent_token_ids=(101, 102),
        action_start_token_id=103,
        action_token_ids=(10, 20),
    )
    for worker in workers:
        worker.model_runner.model(torch.tensor([101, 102, 103]))

    state = pop_policy_state_capture(engine)

    assert state.latent_hidden.shape == (2, 2)
    assert state.action_logits.tolist() == [1030.0, 2060.0]


def test_worker_rejects_missing_action_boundary() -> None:
    worker = _Worker()
    _start(worker)
    worker.model_runner.model(torch.tensor([101, 102]))

    with pytest.raises(RuntimeError, match="policy state sequence"):
        worker.nimloth_pop_policy_state_capture()


def test_worker_reads_v1_multimodal_runner_token_buffer() -> None:
    worker = _Worker()
    _start(worker)
    token_ids = torch.tensor([101, 102, 103])
    worker.model_runner.input_ids.gpu = token_ids
    worker.model_runner.model(
        input_ids=None,
        inputs_embeds=token_ids.float().unsqueeze(-1),
    )

    result = worker.nimloth_pop_policy_state_capture()

    assert result["latent_hidden"] == [
        [101.0, 101.5],
        [102.0, 102.5],
    ]
    assert result["action_logits"] == [1030.0, 2060.0]
