"""Selected-state capture from one vLLM Qwen rollout forward."""

from __future__ import annotations

import asyncio

import pytest
import torch

from nimloth.backbone.qwen25vl.vllm_hidden import (
    PolicyStateCaptureWorkerExtension,
    async_abort_policy_state_capture_for_request,
    async_pop_policy_state_capture_for_request,
    async_start_policy_state_capture_for_request,
    pop_policy_state_capture,
    pop_policy_state_captures,
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


def _batched_decode(worker: _Worker, rows: dict[str, list[int]]) -> None:
    request_ids = list(rows)
    flat_ids = torch.tensor(
        [token_id for request_id in request_ids for token_id in rows[request_id]]
    )
    offsets = [0]
    for request_id in request_ids:
        offsets.append(offsets[-1] + len(rows[request_id]))
    worker.model_runner.input_batch = type(
        "InputBatch",
        (),
        {
            "req_ids": request_ids,
            "num_reqs": len(request_ids),
            "query_start_loc": torch.tensor(offsets),
        },
    )()
    worker.model_runner.input_ids.gpu = flat_ids
    worker.model_runner.model(
        input_ids=None,
        inputs_embeds=flat_ids.float().unsqueeze(-1),
    )


def test_worker_separates_interleaved_policy_states_by_vllm_request() -> None:
    worker = _Worker()
    _start(worker)
    _batched_decode(worker, {"request-a": [101], "request-b": [101]})
    _batched_decode(worker, {"request-b": [102], "request-a": [102]})
    _batched_decode(worker, {"request-a": [103], "request-b": [103]})

    results = worker.nimloth_pop_policy_state_captures()

    assert list(results) == ["request-a", "request-b"]
    assert results["request-a"]["latent_hidden"] == [
        [101.0, 101.5],
        [102.0, 102.5],
    ]
    assert results["request-b"]["latent_hidden"] == [
        [101.0, 101.5],
        [102.0, 102.5],
    ]
    assert results["request-a"]["action_logits"] == [1030.0, 2060.0]
    assert results["request-b"]["action_logits"] == [1030.0, 2060.0]


def test_frontend_returns_batched_states_in_requested_identity_order() -> None:
    workers = [_Worker(), _Worker()]
    engine = _Engine(workers)
    start_policy_state_capture(
        engine,
        latent_token_ids=(101, 102),
        action_start_token_id=103,
        action_token_ids=(10, 20),
    )
    for worker in workers:
        _batched_decode(worker, {"request-a": [101], "request-b": [101]})
        _batched_decode(worker, {"request-b": [102], "request-a": [102]})
        _batched_decode(worker, {"request-a": [103], "request-b": [103]})

    states = pop_policy_state_captures(
        engine,
        request_ids=("request-b", "request-a"),
    )

    assert list(states) == ["request-b", "request-a"]
    assert states["request-b"].latent_hidden.shape == (2, 2)
    assert states["request-a"].action_logits.tolist() == [1030.0, 2060.0]


class _AsyncEngine(_Engine):
    async def collective_rpc(self, method, args=()):
        return super().collective_rpc(method, args=args)


def test_worker_request_scoped_capture_survives_out_of_order_pop() -> None:
    worker = _Worker()
    worker.nimloth_start_policy_state_capture_for_request(
        "request-a", (101, 102), 103, (10, 20)
    )
    worker.nimloth_start_policy_state_capture_for_request(
        "request-b", (101, 102), 103, (10, 20)
    )
    _batched_decode(worker, {"request-a": [101], "request-b": [101]})
    _batched_decode(worker, {"request-b": [102], "request-a": [102]})
    _batched_decode(worker, {"request-a": [103], "request-b": [103]})

    prepared_b = worker.nimloth_prepare_policy_state_capture_for_request(
        "request-b"
    )
    result_b = worker.nimloth_finish_policy_state_capture_for_request("request-b")
    prepared_a = worker.nimloth_prepare_policy_state_capture_for_request(
        "request-a"
    )
    result_a = worker.nimloth_finish_policy_state_capture_for_request("request-a")

    assert prepared_b["latent_hidden"] == [[101.0, 101.5], [102.0, 102.5]]
    assert result_b["action_logits"] == [1030.0, 2060.0]
    assert prepared_a["latent_hidden"] == [[101.0, 101.5], [102.0, 102.5]]
    assert result_a["action_logits"] == [1030.0, 2060.0]


def test_worker_request_abort_does_not_clear_another_request() -> None:
    worker = _Worker()
    for request_id in ("request-a", "request-b"):
        worker.nimloth_start_policy_state_capture_for_request(
            request_id, (101, 102), 103, (10, 20)
        )
    _batched_decode(worker, {"request-a": [101], "request-b": [101, 102, 103]})

    assert worker.nimloth_abort_policy_state_capture_for_request("request-a") is True
    worker.nimloth_prepare_policy_state_capture_for_request("request-b")
    result_b = worker.nimloth_finish_policy_state_capture_for_request("request-b")

    assert result_b["action_logits"] == [1030.0, 2060.0]
    with pytest.raises(RuntimeError, match="not active"):
        worker.nimloth_prepare_policy_state_capture_for_request("request-a")


def test_worker_prepare_failure_happens_before_any_rank_computes_logits() -> None:
    workers = [_Worker(), _Worker()]
    engine = _AsyncEngine(workers)
    for worker in workers:
        worker.nimloth_start_policy_state_capture_for_request(
            "request-a", (101, 102), 103, (10, 20)
        )
    _batched_decode(workers[0], {"request-a": [101, 102, 103]})
    _batched_decode(workers[1], {"request-a": [101, 102]})
    calls = [0, 0]
    for rank, worker in enumerate(workers):
        original = worker.model_runner.model.compute_logits

        def tracked(hidden, *, rank=rank, original=original):
            calls[rank] += 1
            return original(hidden)

        worker.model_runner.model.compute_logits = tracked

    with pytest.raises(RuntimeError, match="policy state sequence"):
        asyncio.run(
            async_pop_policy_state_capture_for_request(
                engine,
                request_id="request-a",
            )
        )

    assert calls == [0, 0]


def test_worker_rejects_duplicate_request_capture_identity() -> None:
    worker = _Worker()
    worker.nimloth_start_policy_state_capture_for_request(
        "request-a", (101, 102), 103, (10, 20)
    )

    with pytest.raises(RuntimeError, match="already active"):
        worker.nimloth_start_policy_state_capture_for_request(
            "request-a", (101, 102), 103, (10, 20)
        )


def test_async_frontend_binds_capture_to_exact_request() -> None:
    async def exercise() -> None:
        workers = [_Worker(), _Worker()]
        engine = _AsyncEngine(workers)
        await async_start_policy_state_capture_for_request(
            engine,
            request_id="request-a",
            latent_token_ids=(101, 102),
            action_start_token_id=103,
            action_token_ids=(10, 20),
        )
        for worker in workers:
            _batched_decode(worker, {"request-a": [101, 102, 103]})
        state = await async_pop_policy_state_capture_for_request(
            engine,
            request_id="request-a",
        )
        assert state.latent_hidden.shape == (2, 2)
        assert state.action_logits.tolist() == [1030.0, 2060.0]
        await async_abort_policy_state_capture_for_request(
            engine,
            request_id="request-a",
        )

    asyncio.run(exercise())
