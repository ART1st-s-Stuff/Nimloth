from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from nimloth.training.sft1.manifest import (
    parse_sft1_v2_manifest,
    verify_pinned_vagen_verl_source,
)
from tests.training.sft1._state_v2_fixtures import manifest_raw


ROOT = Path(__file__).resolve().parents[3]


def test_manifest_binds_all_identities_and_exact_source_tree() -> None:
    raw = manifest_raw()
    first = parse_sft1_v2_manifest(raw)
    second = parse_sft1_v2_manifest(deepcopy(raw))
    assert first.identity == second.identity
    assert len(first.identity) == 64
    assert first.query_count == 16
    assert first.action_token_ids == tuple(range(100, 108))
    verify_pinned_vagen_verl_source(ROOT)


def test_manifest_rejects_stale_or_legacy_identity() -> None:
    cases = []
    unknown = deepcopy(manifest_raw())
    unknown["legacy_teacher"] = "alias"
    cases.append((unknown, "unknown SFT1-v2 manifest field"))
    stale_verl = deepcopy(manifest_raw())
    stale_verl["verl_commit"] = "0" * 40
    cases.append((stale_verl, "VERL commit"))
    old_objective = deepcopy(manifest_raw())
    old_objective["objective_version"] = "legacy_dino_grid"
    cases.append((old_objective, "old state objective"))
    mixed_actions = deepcopy(manifest_raw())
    mixed_actions["action_token_ids"] = [100] * 8
    cases.append((mixed_actions, "eight distinct"))
    malformed_hash = deepcopy(manifest_raw())
    malformed_hash["teacher_cache_sha256"] = "not-a-digest"
    cases.append((malformed_hash, "SHA256"))

    for raw, message in cases:
        with pytest.raises(ValueError, match=message):
            parse_sft1_v2_manifest(raw)
