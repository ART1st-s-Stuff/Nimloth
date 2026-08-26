from __future__ import annotations

import json
from pathlib import Path

import pytest

from nimloth.latent import LatentActionTokens, latent_state_tokens

from nimloth.training.sft1.controller import (
    SFT1V2Phase,
    SFT1V2PreflightEvidence,
    run_controlled_phase,
)
from nimloth.training.sft1.identity import audit_id176_processor_identity


def _evidence(phase: SFT1V2Phase) -> SFT1V2PreflightEvidence:
    return SFT1V2PreflightEvidence(
        phase=phase.value, config_identity="a" * 64, repo="/repo",
        commit="b" * 40, interpreter="/repo/.venv-vagen-main/bin/python3",
        parent_clean=True, submodules_clean=True, launch_locked=True,
        checked_paths=("data", "checkpoint"), cache_identity="c" * 64,
        output_unused=True,
        row_audit={"train_rows": 12841, "external_validation_rows": 1413},
        max_token_upper_bound=4096,
        output_free_bytes=200_000_000_000,
    )


def test_processor_identity_binds_every_processor_and_k16_action_file(
    tmp_path: Path,
) -> None:
    for name in (
        "preprocessor_config.json", "video_preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "chat_template.jinja",
    ):
        (tmp_path / name).write_text(name, encoding="utf-8")
    tokens = LatentActionTokens()
    values = {
        token: 100 + index
        for index, token in enumerate((
            *latent_state_tokens(16, tokens), tokens.action_start,
            tokens.action_end, *tokens.action_tokens,
        ))
    }
    (tmp_path / "added_tokens.json").write_text(json.dumps(values), encoding="utf-8")
    identity = audit_id176_processor_identity(tmp_path)
    assert identity.action_token_ids == tuple(
        values[token] for token in tokens.action_tokens
    )
    assert all(len(value) == 64 for key, value in identity.__dict__.items() if key.endswith("sha256"))


def test_controller_is_sequential_and_failed_smoke_cannot_fall_through(tmp_path: Path) -> None:
    cache = run_controlled_phase(
        phase=SFT1V2Phase.CACHE, controller_dir=tmp_path,
        preflight=_evidence(SFT1V2Phase.CACHE),
        executor=lambda: {"cache_manifest": "d" * 64},
    )
    assert cache.status == "completed"

    with pytest.raises(RuntimeError, match="smoke failed"):
        run_controlled_phase(
            phase=SFT1V2Phase.SMOKE, controller_dir=tmp_path,
            preflight=_evidence(SFT1V2Phase.SMOKE),
            executor=lambda: (_ for _ in ()).throw(RuntimeError("smoke failed")),
        )
    assert (tmp_path / "smoke.failed.attempt_001.json").is_file()
    assert not (tmp_path / "smoke.complete.json").exists()
    with pytest.raises(RuntimeError, match="requires completed smoke"):
        run_controlled_phase(
            phase=SFT1V2Phase.RESUME_SMOKE, controller_dir=tmp_path,
            preflight=_evidence(SFT1V2Phase.RESUME_SMOKE),
            executor=lambda: {"checkpoint": "path"},
        )


def test_slurm_wrapper_executes_only_through_gated_controller_contract() -> None:
    root = Path(__file__).resolve().parents[3]
    source = (root / "experiments/training/sft1/state_interface_v2_canary.slurm").read_text()
    for required in (
        "REPO", "EXPECTED_COMMIT", ".venv-vagen-main/bin/python3",
        "MASTER_PORT", "TMPDIR", "HF_HOME", "TRANSFORMERS_CACHE",
        "PYTHONDONTWRITEBYTECODE", "torch.distributed.run",
        "--controller-dir", "RESUME_CHECKPOINT", "launch-locked",
    ):
        assert required in source
    assert "exit 73" not in source
    assert "wandb init" not in source.lower()
    assert "\nsbatch " not in source.lower()
