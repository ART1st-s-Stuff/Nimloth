"""Serialized tiny checkpoint checks; not a seven-GPU training substitute."""
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from query_control import cleanup_epochs, validate_boundary
from run_query_gate import attempt_paths, validate_checkout
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY
from nimloth.latent import latent_state_tokens
from nimloth.training.sft.stage1.checkpoint import save_checkpoint
from nimloth.training.sft.stage1.convergence import ConvergencePolicy, ConvergenceState
from nimloth.training.sft.stage2.config import QueryAlignmentConfig
from nimloth.training.sft.stage2.model import QueryAlignmentModel
from nimloth.wm.grid import SharedSlotProjector


def test_query_gate_rejects_uninitialized_lewm_before_gpu_launch(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "checkout"
    (checkout / "external" / "le-wm").mkdir(parents=True)
    responses = iter(["source-commit\n", "", "dependency-commit\n"])

    def fake_check_output(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr("run_query_gate.subprocess.check_output", fake_check_output)
    with pytest.raises(RuntimeError, match="initialize the pinned submodule"):
        validate_checkout(checkout, "source-commit")


def test_query_gate_attempt_uses_distinct_preserved_outputs(tmp_path):
    assert attempt_paths(tmp_path, "retry1") == (
        tmp_path / "gate_control_retry1",
        tmp_path / "gate_retry1",
    )
    with pytest.raises(ValueError, match="invalid gate attempt name"):
        attempt_paths(tmp_path, "../retry")


def saved_epoch(tmp_path):
    tokens = latent_state_tokens(1)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"[UNK]": 0, tokens[0]: 1}, unk_token="[UNK]")), unk_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 2))
            self.config = SimpleNamespace(nimloth_training_stage="query")

        def save_pretrained(self, path, **kwargs):
            save_file({"tiny.weight": self.weight.detach()}, str(path / "adapter_model.safetensors"))
            (path / "adapter_config.json").write_text("{}")

    class Processor:
        def save_pretrained(self, path):
            tokenizer.save_pretrained(path)
            (path / "preprocessor_config.json").write_text("{}")

    objective = QueryAlignmentConfig(grid_size=1, projector_hidden_dim=2)
    model = QueryAlignmentModel(TinyLM(), SharedSlotProjector(2, DINOV2_LARGE_IDENTITY.hidden_size, 2, grid_tokens=1), [1], objective)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    convergence = ConvergenceState()
    convergence.observe(epoch=1, loss=1.0, policy=ConvergencePolicy(2, 2, 0.01))
    identity = {"stage": "query", "convergence": {"monitor": "validation_total_loss"},
                "latent_token_count": 1, **asdict(objective)}
    save_checkpoint(model, Processor(), tmp_path, "epoch_001", optimizer, scheduler,
                    10, 1, 1.0, lora=True, latent_token_count=1, world_size=7,
                    identity=identity, convergence_state=convergence.state_dict(),
                    rank_rng_states=[{"python": None, "numpy": None, "torch_cpu": None, "torch_cuda": None}] * 7)
    (tmp_path / "launch.json").write_text(json.dumps({"run_output": str(tmp_path)}))
    return tmp_path / "epoch_001"


def test_actual_saved_epoch_and_final_without_commit_marker(tmp_path):
    import shutil

    epoch = saved_epoch(tmp_path)
    assert validate_boundary(epoch)["step"] == 10
    shutil.copytree(epoch, tmp_path / "final")
    (tmp_path / "final" / "COMMITTED").unlink()
    assert validate_boundary(tmp_path / "final")["epoch"] == 1


@pytest.mark.parametrize("kind", ["adapter", "projector", "grid"])
def test_corrupt_retained_checkpoint_refuses_cleanup(tmp_path, kind):
    epoch = saved_epoch(tmp_path)
    protected = tmp_path / "resume_step_00000010"
    protected.mkdir()
    if kind == "adapter":
        save_file({"tiny.weight": torch.tensor(float("nan"))}, str(epoch / "adapter_model.safetensors"))
    elif kind == "projector":
        torch.save({"bad": torch.tensor(float("inf"))}, epoch / "slot_projector.pt")
    else:
        metadata = json.loads((epoch / "grid_state_config.json").read_text())
        metadata["objective"]["weight_dino"] = 2
        (epoch / "grid_state_config.json").write_text(json.dumps(metadata))
    with pytest.raises(AssertionError):
        cleanup_epochs(tmp_path)
    assert protected.is_dir()
    assert not list(tmp_path.glob("cleanup*complete.json"))
