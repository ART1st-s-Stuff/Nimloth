from __future__ import annotations

import torch

from nimloth.training.common import dist as dist_helpers


def test_setup_dist_initializes_nccl_with_explicit_local_device(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "5")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("NIMLOTH_DDP_GPU_STRIDE", raising=False)

    selected: list[int] = []
    initialized: list[dict] = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(
        dist_helpers.dist,
        "init_process_group",
        lambda **kwargs: initialized.append(kwargs),
    )

    rank, world, local, device = dist_helpers.setup_dist()

    assert (rank, world, local) == (3, 5, 1)
    assert device == torch.device("cuda:1")
    assert selected == [1]
    assert initialized == [{"backend": "nccl", "device_id": torch.device("cuda:1")}]
