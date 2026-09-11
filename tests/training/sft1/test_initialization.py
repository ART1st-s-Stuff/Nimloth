"""CPU invariants for B semantic initialization, not model-quality evidence."""


import pytest
import torch
from safetensors.torch import save_file

from nimloth.training.sft.stage1 import initialization as module


def test_semantic_means_and_boundaries_use_each_matrix_and_only_target_rows():
    inp = torch.arange(48).reshape(12, 4).to(torch.bfloat16)
    head = inp.flip(0).clone()
    old_inp, old_head = inp.clone(), head.clone()
    sources = {8: [0], 9: [0], 10: [1, 3], 11: [2]}
    expected, report = module.initialize_rows(inp, head, sources, tied=False)
    for kind, weight, old in (("input", inp, old_inp), ("output", head, old_head)):
        assert torch.equal(weight[:8], old[:8])
        for row, ids in sources.items():
            assert torch.equal(weight[row], old[ids].float().mean(0).bfloat16())
            assert torch.equal(expected[kind][row], weight[row])
            assert report[kind][str(row)]["source_ids"] == ids
    assert not torch.equal(inp[10], head[10])


def test_tied_storage_kept_and_means_come_from_snapshot():
    weight = torch.arange(24).reshape(6, 4).bfloat16()
    old = weight.clone()
    module.initialize_rows(weight, weight, {4: [0, 1], 5: [2, 3]}, tied=True)
    assert torch.equal(weight[4], old[:2].float().mean(0).bfloat16())
    assert torch.equal(weight[:4], old[:4])


@pytest.mark.parametrize("tied,alias", [(True, False), (False, True)])
def test_reject_inconsistent_tie(tied, alias):
    weight = torch.ones(6, 4, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="aliasing"):
        module.initialize_rows(
            weight, weight if alias else weight.clone(), {5: [0]}, tied=tied
        )


@pytest.mark.parametrize("sources", [{5: [5]}, {5: []}, {6: [0]}, {5: [-1]}, {}])
def test_reject_invalid_sources_without_modifying(sources):
    weight = torch.arange(24).reshape(6, 4).bfloat16()
    old = weight.clone()
    with pytest.raises(ValueError):
        module.initialize_rows(weight, weight.clone(), sources, tied=False)
    assert torch.equal(weight, old)


def test_streaming_verification_catches_unapproved_changes(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    weight = torch.arange(32).reshape(8, 4).bfloat16()
    saved = weight[:6].clone()
    expected = {"input": {5: weight[0].clone()}}
    saved[5] = expected["input"][5]
    save_file(
        {"embedding": weight, "other": torch.ones(2)}, str(source / "model.safetensors")
    )
    save_file(
        {"embedding": saved, "other": torch.ones(2)}, str(output / "model.safetensors")
    )
    assert module.verify_saved(source, output, {"embedding": "input"}, expected) == 2
    saved[2, 0] += 1
    save_file(
        {"embedding": saved, "other": torch.ones(2)}, str(output / "model.safetensors")
    )
    with pytest.raises(ValueError, match="Unexpected tensor change"):
        module.verify_saved(source, output, {"embedding": "input"}, expected)


def test_tied_serialization_omission_requires_original_exact_equality(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    weight = torch.arange(24).reshape(6, 4).bfloat16()
    saved = weight.clone()
    saved[5] = weight[0]
    expected = {"input": {5: weight[0]}, "output": {5: weight[0]}}
    names = {"embedding": "input", "head": "output"}
    save_file(
        {"embedding": weight, "head": weight.clone()}, str(source / "model.safetensors")
    )
    save_file({"embedding": saved}, str(output / "model.safetensors"))
    assert module.verify_saved(source, output, names, expected, tied=True) == 2
    # Even a mismatch in a replaced action row must fail, since loading tied it.
    inconsistent = weight.clone()
    inconsistent[5, 0] += 1
    save_file(
        {"embedding": weight, "head": inconsistent}, str(source / "model.safetensors")
    )
    with pytest.raises(ValueError, match="Original tied matrices disagree"):
        module.verify_saved(source, output, names, expected, tied=True)
