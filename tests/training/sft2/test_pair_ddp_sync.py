from __future__ import annotations

from nimloth.training.sft2.trainer import _ddp_sync_policy


def test_pair_parallel_aux_uses_nonstatic_accumulation_sync() -> None:
    policy = _ddp_sync_policy(world=4, qwen_pair_parallel=True)

    assert policy == {"aux_static_graph": False, "aux_no_sync": True}


def test_single_device_ddp_keeps_static_graph_workaround() -> None:
    policy = _ddp_sync_policy(world=4, qwen_pair_parallel=False)

    assert policy == {"aux_static_graph": True, "aux_no_sync": False}


def test_single_process_needs_no_ddp_sync_policy() -> None:
    policy = _ddp_sync_policy(world=1, qwen_pair_parallel=True)

    assert policy == {"aux_static_graph": False, "aux_no_sync": False}
