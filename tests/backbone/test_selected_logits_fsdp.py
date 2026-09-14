"""Two-rank CPU FSDP graph check; GPU allocator behavior is a remote gate."""
import copy

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import functional as F

from nimloth.backbone.selected_token_rows import _install_leaf


def _worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(9)
        head = torch.nn.Linear(4, 16, bias=False)
        _install_leaf(head, (0, 1), (2, 3))
        reference = copy.deepcopy(head)
        wrapped = FSDP(head, ignored_states={head.weight}, use_orig_params=True,
                       device_id=torch.device("cpu"))
        # Different token lengths and a zero-weight rank retain identical reducer
        # participation without a duplicate forward or detached selected rows.
        inputs = [torch.randn(rank + 2, 4), torch.randn(rank + 3, 4)]
        result, expected = [], []
        for hidden in inputs:
            result.append(wrapped(hidden).square().mean())
            logits = F.linear(hidden, reference.weight)
            for ids, rows in ((reference.nimloth_query_ids, reference.nimloth_query_rows),
                              (reference.nimloth_protocol_ids, reference.nimloth_protocol_rows)):
                logits = logits.index_copy(-1, ids, F.linear(hidden, rows))
            expected.append(logits.square().mean())
        (sum(result) * rank).backward()
        (sum(expected) * rank).backward()
        for parameter in reference.parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(2)
        with FSDP.summon_full_params(wrapped, with_grads=True):
            for actual, old in zip(head.parameters(), reference.parameters(), strict=True):
                if actual.requires_grad:
                    assert actual.grad is not None
                    torch.testing.assert_close(actual.grad, old.grad)
    finally:
        dist.destroy_process_group()


def test_selected_logits_two_rank_fsdp(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "selected-fsdp"),),
                               nprocs=2, join=True)
