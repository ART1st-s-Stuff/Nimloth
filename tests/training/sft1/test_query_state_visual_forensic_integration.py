from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from nimloth.training.sft1.query_state_training_backend import (
    _baseline_payload,
    _load_visual_fork_actor_baseline,
    _safety_requires_forensic_hard_stop,
    _validation_boundary_plan,
    _validation_safety_publication,
    _verify_visual_fork_step0_parity,
    build_query_state_training_updates,
)
from nimloth.training.sft1.query_state_training_config import (
    parse_query_state_training_config,
    query_state_training_run_identity,
)
from nimloth.training.sft1.query_state_training_controller import (
    QueryStateTrainingController,
)
from nimloth.training.sft1.query_state_training_preflight import (
    _reject_visual_fixed_budget_completion_restart,
    _required_output_free_bytes,
)
from nimloth.training.sft1.query_state_visual_forensic_fork import (
    authenticate_visual_fork_ancestor,
)
from tests.training.sft1.test_query_state_training_config import _raw
from tests.training.sft1.test_query_state_visual_forensic_fork import (
    _write_forensic_checkpoint,
)
from nimloth.training.sft1.query_state import DirectSlotProjector
from nimloth.training.sft1.query_state_training_validation import (
    QueryStateActorSafetyVerdict,
)
from torch import nn


def _visual_raw() -> dict:
    return _raw(mode="visual_only_forensic_fork")


def test_visual_fork_is_a_strict_production_training_mode() -> None:
    config = parse_query_state_training_config(_visual_raw())
    assert config.mode == "visual_only_forensic_fork"
    assert config.schedule["schedule_start_update"] == 1605
    assert config.schedule["max_updates"] == 8025
    assert config.forensic_fork["actor_policy"] == "report_only"
    assert config.initialization["resume_mode"] == "fresh"

    for section, field, value in (
        ("initialization", "direct_head_initialization", "fresh_seeded_no_bias"),
        ("forensic_fork", "actor_policy", "hard_stop"),
        ("forensic_fork", "ancestor_protected", False),
        ("early_stopping", "enabled", True),
    ):
        bad = _visual_raw()
        bad[section][field] = value
        with pytest.raises(ValueError):
            parse_query_state_training_config(bad)

    exact_resume = _visual_raw()
    exact_resume["initialization"].update(
        resume_mode="exact_restart",
        resume_checkpoint=(
            "/outputs/visual_only_forensic_fork/checkpoints/update_00003210"
        ),
    )
    exact_resume["tracking"]["resume"] = "must"
    resumed = parse_query_state_training_config(exact_resume)
    assert query_state_training_run_identity(resumed) == query_state_training_run_identity(
        config
    )

    ancestor_resume = _visual_raw()
    ancestor_resume["initialization"].update(
        resume_mode="exact_restart",
        resume_checkpoint=ancestor_resume["forensic_fork"]["ancestor_checkpoint_path"],
    )
    ancestor_resume["tracking"]["resume"] = "must"
    with pytest.raises(ValueError, match="run-owned checkpoint"):
        parse_query_state_training_config(ancestor_resume)


def test_visual_fork_identity_binds_distinct_ancestor_source_and_is_fresh_from_formal38() -> None:
    visual = parse_query_state_training_config(_visual_raw())
    formal = parse_query_state_training_config(_raw(mode="formal"))
    assert visual.forensic_fork["ancestor_source_commit"] != visual.source["commit"]
    assert query_state_training_run_identity(visual) != query_state_training_run_identity(formal)

    changed = _visual_raw()
    changed["forensic_fork"]["ancestor_source_commit"] = "9" * 40
    assert query_state_training_run_identity(
        parse_query_state_training_config(changed)
    ) != query_state_training_run_identity(visual)

    changed_manifest = _visual_raw()
    changed_manifest["forensic_fork"]["ancestor_source_manifest_identity"] = "9" * 64
    assert query_state_training_run_identity(
        parse_query_state_training_config(changed_manifest)
    ) != query_state_training_run_identity(visual)

    equal_source = _visual_raw()
    equal_source["forensic_fork"]["ancestor_source_commit"] = equal_source["source"]["commit"]
    with pytest.raises(ValueError, match="ancestor source commit.*current"):
        parse_query_state_training_config(equal_source)

    equal_manifest = _visual_raw()
    equal_manifest["forensic_fork"]["ancestor_source_manifest_identity"] = (
        equal_manifest["source"]["source_manifest_identity"]
    )
    with pytest.raises(ValueError, match="ancestor source manifest.*current"):
        parse_query_state_training_config(equal_manifest)


@pytest.mark.parametrize("output_field", ["run_root", "controller_root"])
@pytest.mark.parametrize("relationship", ["equal", "descendant", "ancestor"])
def test_visual_fork_output_roots_are_tree_disjoint_from_protected_ancestor(
    output_field: str,
    relationship: str,
) -> None:
    raw = _visual_raw()
    protected_run_root = Path(
        raw["forensic_fork"]["ancestor_checkpoint_path"]
    ).resolve().parents[1]
    conflicting = {
        "equal": protected_run_root,
        "descendant": protected_run_root / "fork-output",
        "ancestor": protected_run_root.parent,
    }[relationship]
    raw["output"][output_field] = str(conflicting)
    with pytest.raises(ValueError, match="protected Formal38 ancestor tree"):
        parse_query_state_training_config(raw)


def test_visual_fork_schedule_uses_original_epoch_offset_and_fixed_validation() -> None:
    config = parse_query_state_training_config(_visual_raw())
    updates = build_query_state_training_updates(
        tuple(range(12836)),
        epochs=4,
        schedule_epoch_offset=1,
        seed=int(config.schedule["seed"]),
        rank=0,
        world_size=8,
        rows_per_rank_update=1,
        expected_updates=6420,
    )
    assert len(updates) == 6420
    assert _validation_boundary_plan(
        config, update=3210, epoch=2, actual_terminal=False
    ) == {
        "calibration": True,
        "holdout": False,
        "generation_format": True,
        "actual_terminal": False,
    }
    assert _validation_boundary_plan(
        config, update=8025, epoch=5, actual_terminal=True
    ) == {
        "calibration": True,
        "holdout": True,
        "generation_format": True,
        "actual_terminal": True,
    }


def test_production_visual_fork_authenticates_formal38_control_and_all_rank_shards(
    tmp_path: Path,
) -> None:
    root = nn.Module()
    root.objective = nn.Module()
    root.objective.projector = DirectSlotProjector()
    checkpoint, failure, identity = _write_forensic_checkpoint(tmp_path, root)
    raw = _visual_raw()
    assert raw["source"]["source_manifest_identity"] != identity.source_manifest_identity
    raw["forensic_fork"].update(
        ancestor_source_commit=identity.source_commit,
        ancestor_source_manifest_identity=identity.source_manifest_identity,
        ancestor_checkpoint_path=str(checkpoint.resolve()),
        ancestor_failure_manifest_path=str(failure.resolve()),
        ancestor_control_sha256=__import__("hashlib").sha256(
            (checkpoint / "control.json").read_bytes()
        ).hexdigest(),
        ancestor_run_identity=identity.run_identity,
        ancestor_source_config_identity=identity.config_identity,
    )
    raw["model"]["initialization_identity"] = (
        "formal38_forensic_model_only:"
        + raw["forensic_fork"]["ancestor_control_sha256"]
    )
    raw["artifacts"]["file_sha256"].update({
        str(checkpoint / "control.json"): raw["forensic_fork"]["ancestor_control_sha256"],
        str(failure): __import__("hashlib").sha256(failure.read_bytes()).hexdigest(),
        **{
            str(checkpoint / f"rank_{rank:05d}_of_00008{suffix}"): __import__("hashlib").sha256(
                (checkpoint / f"rank_{rank:05d}_of_00008{suffix}").read_bytes()
            ).hexdigest()
            for rank in range(8)
            for suffix in (".pt", ".json")
        },
    })
    config = parse_query_state_training_config(raw)
    assert authenticate_visual_fork_ancestor(config) == identity

    mismatch = deepcopy(raw)
    mismatch["forensic_fork"]["ancestor_source_commit"] = "9" * 40
    with pytest.raises(ValueError, match="control/failure identity mismatch"):
        authenticate_visual_fork_ancestor(
            parse_query_state_training_config(mismatch)
        )

    manifest_mismatch = deepcopy(raw)
    manifest_mismatch["forensic_fork"]["ancestor_source_manifest_identity"] = (
        "9" * 64
    )
    with pytest.raises(ValueError, match="control/failure identity mismatch"):
        authenticate_visual_fork_ancestor(
            parse_query_state_training_config(manifest_mismatch)
        )

    (checkpoint / "rank_00007_of_00008.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="rank shard inventory"):
        authenticate_visual_fork_ancestor(config)


_VISUAL_PARITY_METRICS = (
    "raw_query/norm_mean",
    "raw_query/slot_variance",
    "raw_query/offdiag_pairwise_cosine",
    "raw_query/effective_rank",
    "canonical_state/norm_mean",
    "canonical_state/slot_variance",
    "canonical_state/offdiag_pairwise_cosine",
    "canonical_state/effective_rank",
    "canonical_state/collapse",
    "direct_state/dino_mse",
    "direct_state/dino_cosine",
    "direct_state/content_relation",
    "upstream/fused_to_raw_relation",
    "upstream/instruction_to_state_relation",
    "pairs/same_image_multi_instruction_state_distance",
    "pairs/same_instruction_multi_image_state_distance",
    "pairs/same_image_pair_count",
    "pairs/same_instruction_pair_count",
    "pairs/same_image_group_count",
    "pairs/same_instruction_group_count",
)


def _parity_metrics() -> dict[str, float]:
    return {name: float(index + 1) / 10.0 for index, name in enumerate(_VISUAL_PARITY_METRICS)}


def test_visual_fork_actor_baseline_remains_authenticated_id176_and_covers_disjoint_splits(
    tmp_path: Path,
) -> None:
    raw = _visual_raw()
    config = parse_query_state_training_config(raw)
    calibration = tuple(f"cal-{index:04d}" for index in range(80))
    holdout = tuple(f"hold-{index:04d}" for index in range(1333))
    logits = {
        identity: tuple(float(index) for index in range(8))
        for identity in (*calibration, *holdout)
    }
    payload = _baseline_payload(config, logits)
    baseline_path = tmp_path / "formal38-actor-baseline-id176.json"
    baseline_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    raw["forensic_fork"]["id176_actor_baseline_path"] = str(baseline_path.resolve())
    raw["forensic_fork"]["id176_actor_baseline_sha256"] = hashlib.sha256(
        baseline_path.read_bytes()
    ).hexdigest()
    raw["artifacts"]["file_sha256"][str(baseline_path.resolve())] = raw[
        "forensic_fork"
    ]["id176_actor_baseline_sha256"]
    locked = parse_query_state_training_config(raw)

    loaded, identity = _load_visual_fork_actor_baseline(
        locked,
        calibration_row_identities=calibration,
        holdout_row_identities=holdout,
    )
    assert loaded == logits
    assert identity == payload["identity"]
    assert payload["actor_checkpoint_identity"] == locked.initialization[
        "actor_checkpoint_identity"
    ]

    with pytest.raises(ValueError, match="disjoint calibration/holdout"):
        _load_visual_fork_actor_baseline(
            locked,
            calibration_row_identities=calibration,
            holdout_row_identities=(*holdout[:-1], calibration[0]),
        )


def test_visual_fork_step0_parity_is_immutable_and_fail_closed(tmp_path: Path) -> None:
    metrics = _parity_metrics()
    protected_run = tmp_path / "formal38"
    checkpoint = protected_run / "forensics" / "unsafe_update_00001605"
    failure = (
        protected_run
        / "durable"
        / "failures"
        / "unsafe_00001284_00001605.json"
    )
    failure.parent.mkdir(parents=True)
    failure.write_text(
        __import__("json").dumps({
            "validation": {"calibration": {"diagnostics": {"metrics": metrics}}}
        }) + "\n",
        encoding="utf-8",
    )
    raw = _visual_raw()
    raw["forensic_fork"]["ancestor_checkpoint_path"] = str(checkpoint.resolve())
    raw["forensic_fork"]["ancestor_failure_manifest_path"] = str(failure.resolve())
    raw["artifacts"]["file_sha256"].update({
        str(failure.resolve()): hashlib.sha256(failure.read_bytes()).hexdigest(),
        str(checkpoint / "control.json"): "1" * 64,
        **{
            str(checkpoint / f"rank_{rank:05d}_of_00008{suffix}"): "1" * 64
            for rank in range(8)
            for suffix in (".pt", ".json")
        },
    })
    config = parse_query_state_training_config(raw)
    publication = {"diagnostics": {"metrics": dict(metrics)}}
    assert _verify_visual_fork_step0_parity(config, publication)["passed"] is True
    publication["diagnostics"]["metrics"]["direct_state/dino_mse"] += 0.1
    with pytest.raises(RuntimeError, match="parity failed"):
        _verify_visual_fork_step0_parity(config, publication)

    missing = {"diagnostics": {"metrics": dict(metrics)}}
    del missing["diagnostics"]["metrics"]["direct_state/content_relation"]
    with pytest.raises(ValueError, match="incomplete"):
        _verify_visual_fork_step0_parity(config, missing)
    nonfinite = {"diagnostics": {"metrics": dict(metrics)}}
    nonfinite["diagnostics"]["metrics"]["raw_query/norm_mean"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        _verify_visual_fork_step0_parity(config, nonfinite)


def test_visual_fork_known_unsafe_step0_is_report_only_while_formal_still_stops() -> None:
    known_unsafe = QueryStateActorSafetyVerdict(
        passed=False,
        checks={
            "kl": False,  # Formal38 update1605: 1.057509
            "top1": False,  # Formal38 update1605: 0.675
            "logit_rms_ratio": False,  # Formal38 update1605: 1.660988
        },
        tolerances={"kl_max": 0.1, "top1_min": 0.9, "logit_rms_ratio_max": 1.2},
    )
    generation = {"due": True, "passed": False}

    visual = _validation_safety_publication(
        known_unsafe,
        generation_format=generation,
        report_only=True,
    )
    assert _safety_requires_forensic_hard_stop(
        mode="visual_only_forensic_fork", safety=visual
    ) is False
    assert visual == {
        "passed": True,
        "observed_passed": False,
        "observed_actor_passed": False,
        "observed_generation_passed": False,
        "report_only": True,
        "checks": {
            "kl": False,
            "top1": False,
            "logit_rms_ratio": False,
            "generation_format": False,
        },
        "tolerances": known_unsafe.tolerances,
        "generation_format_due": True,
    }
    formal = _validation_safety_publication(
        known_unsafe,
        generation_format=generation,
        report_only=False,
    )
    assert formal["passed"] is False
    assert formal["report_only"] is False
    assert _safety_requires_forensic_hard_stop(mode="formal", safety=formal) is True


def test_visual_fork_storage_budget_is_rolling_not_max_payload_count() -> None:
    config = parse_query_state_training_config(_visual_raw())
    assert _required_output_free_bytes(config, completed_checkpoint_update=1605) == (
        300_000_000_000 + 5 * 23_370_000_000
    )


def test_visual_restart_rejects_fixed_budget_completion_before_checkpoint_authentication(
    tmp_path: Path,
) -> None:
    raw = _visual_raw()
    raw["initialization"].update(
        resume_mode="exact_restart",
        resume_checkpoint=(
            raw["output"]["run_root"] + "/checkpoints/update_00008025"
        ),
    )
    raw["tracking"]["resume"] = "must"
    config = parse_query_state_training_config(raw)
    run_root = tmp_path / "run"
    controller_root = tmp_path / "controller"
    run_root.mkdir()
    controller_root.mkdir()
    _reject_visual_fixed_budget_completion_restart(
        config,
        run_root=run_root,
        controller_root=controller_root,
    )
    (controller_root / "VISUAL_FIXED_BUDGET_COMPLETED.json").write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="fixed-budget.*cannot restart"):
        _reject_visual_fixed_budget_completion_restart(
            config,
            run_root=run_root,
            controller_root=controller_root,
        )


def test_visual_fork_controller_is_nonterminal_diagnostic_owner(tmp_path: Path) -> None:
    controller = QueryStateTrainingController(
        run_root=tmp_path / "fork",
        controller_root=tmp_path / "controller",
        run_identity="a" * 64,
        mode="visual_only_forensic_fork",
    )
    controller.claim(resolved_config={"mode": controller.mode}, command_manifest={"argv": []})
    block = controller.record_visual_retention_block(
        update=1926,
        checkpoint=str(tmp_path / "fork" / "checkpoints" / "update_00001926"),
        reason="W&B mirror is incomplete",
    )
    assert __import__("json").loads(block.read_text(encoding="utf-8"))[
        "checkpoint_remains_authoritative"
    ] is True
    terminal = controller.record_visual_fixed_budget_completion(
        update=8025,
        details={
            "final_update": 8025,
            "checkpoint": str(tmp_path / "fork" / "checkpoints" / "update_00008025"),
            "terminal_primary": False,
            "actor_policy": "report_only",
            "generation_policy": "report_only",
            "observed_actor_safety_passed": False,
            "observed_generation_format_passed": False,
            "observed_actor_generation": {
                "observed_actor_passed": False,
                "observed_generation_passed": False,
                "observed_passed": False,
            },
            "terminal_safety_passed": None,
            "sft1_control_authorization": False,
            "holdout_controls_selection": False,
            "best_checkpoint": None,
        },
    )
    payload = json.loads(terminal.read_text(encoding="utf-8"))
    assert payload["mode"] == "visual_only_forensic_fork"
    assert payload["kind"] == "visual_fixed_budget_diagnostic_complete"
    assert payload["update"] == 8025
    assert payload["actor_policy"] == "report_only"
    assert payload["generation_policy"] == "report_only"
    assert payload["observed_actor_safety_passed"] is False
    assert payload["observed_generation_format_passed"] is False
    assert payload["terminal_safety_passed"] is None
    assert payload["sft1_control_authorization"] is False
    assert payload["automatic_sft2_authorization"] is False

    formal = QueryStateTrainingController(
        run_root=tmp_path / "formal",
        controller_root=tmp_path / "formal-controller",
        run_identity="b" * 64,
        mode="formal",
    )
    formal.claim(resolved_config={"mode": "formal"}, command_manifest={"argv": []})
    with pytest.raises(ValueError, match="visual-fork-only"):
        formal.record_visual_fixed_budget_completion(
            update=8025,
            details={
                "final_update": 8025,
                "checkpoint": "/checkpoint",
                "terminal_primary": False,
                "actor_policy": "report_only",
                "generation_policy": "report_only",
                "holdout_controls_selection": False,
                "best_checkpoint": None,
            },
        )
