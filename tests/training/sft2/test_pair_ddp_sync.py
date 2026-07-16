from __future__ import annotations

import torch

from nimloth.training.sft2.trainer import _ddp_sync_policy, _resolve_pair_aux_device


def test_pair_aux_follows_final_norm_even_when_lm_head_is_cuda_zero() -> None:
    model = type(
        "Model",
        (),
        {"hf_device_map": {"lm_head": 0, "model.language_model.norm": 1}},
    )()

    assert _resolve_pair_aux_device(model, torch.device("cuda:0")) == torch.device("cuda:1")


def test_pair_parallel_aux_uses_nonstatic_accumulation_sync() -> None:
    policy = _ddp_sync_policy(world=4, qwen_pair_parallel=True)

    assert policy == {"aux_static_graph": False, "aux_no_sync": True}


def test_single_device_ddp_keeps_static_graph_workaround() -> None:
    policy = _ddp_sync_policy(world=4, qwen_pair_parallel=False)

    assert policy == {"aux_static_graph": True, "aux_no_sync": False}


def test_single_process_needs_no_ddp_sync_policy() -> None:
    policy = _ddp_sync_policy(world=1, qwen_pair_parallel=True)

    assert policy == {"aux_static_graph": False, "aux_no_sync": False}
