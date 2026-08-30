"""Deterministic coverage and connected-component split manifests.

The functions are resolver primitives only.  They never inspect validation
metrics and require all per-stratum counts/ratios as explicit inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Any, Mapping, Sequence

from nimloth.agent.transcript import AgentTranscript
from nimloth.agent.templates.nimloth import NimlothPromptTemplate
from nimloth.backbone.qwen25vl.turn_generation import (
    TURN_RESPONSE_PARSER_PROTOCOL_IDENTITY,
    TURN_RESPONSE_PROMPT_PROTOCOL_IDENTITY,
    build_turn_response_policy_prompt,
    response_policy_prompt_identity,
)
from nimloth.environment.navigation.action_space import NAVIGATION_ACTION_SPACE
from nimloth.training.sft1.real_rows import SFT1V2Early4Row, SFT1V2RowAudit


QUERY_STATE_TRAINING_MANIFEST_SCHEMA = "nimloth_sft1_query_state_training_manifest_v1"
QUERY_STATE_VALIDATION_SPLIT_SCHEMA = "nimloth_sft1_query_state_connected_validation_v1"
QUERY_STATE_GENERATION_FORMAT_MANIFEST_SCHEMA = (
    "nimloth_sft1_query_state_production_generation_format_v1"
)


def _identity(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def normalize_instruction_identity(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("instruction identity requires non-empty source text")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _movement_indices() -> frozenset[int]:
    # Movement ownership comes from the canonical ordered environment table,
    # never from a historical hard-coded action-ID tuple.
    return frozenset(
        index
        for index, action in enumerate(NAVIGATION_ACTION_SPACE.actions)
        if action.key.startswith("move")
    )


def _outcome_bucket(row: SFT1V2Early4Row) -> str:
    if row.executed_action_index not in _movement_indices():
        return "non_movement"
    if row.movement_success is None:
        return "unknown"
    return "success" if row.movement_success else "failure"


def _stratum(row: SFT1V2Early4Row) -> str:
    return (
        f"step={row.step_index}/action={row.executed_action_index}/"
        f"outcome={_outcome_bucket(row)}"
    )


@dataclass(frozen=True)
class QueryStateManifestEntry:
    row_identity: str
    ordinal: int
    record_id: str
    original_image_sha256: str
    normalized_instruction: str
    step_index: int
    executed_action_index: int
    outcome_bucket: str
    stratum: str
    rendered_token_count: int
    valid_lm_token_count: int


@dataclass(frozen=True)
class QueryStateTrainingManifest:
    schema: str
    kind: str
    seed: int
    entries: tuple[QueryStateManifestEntry, ...]
    shortages: Mapping[str, int]
    identity: str

    @property
    def row_identities(self) -> tuple[str, ...]:
        return tuple(entry.row_identity for entry in self.entries)


@dataclass(frozen=True)
class QueryStateValidationSplit:
    schema: str
    seed: int
    calibration_numerator: int
    calibration_denominator: int
    target_calibration_rows: int
    calibration_row_identities: tuple[str, ...]
    holdout_row_identities: tuple[str, ...]
    calibration_component_ids: tuple[str, ...]
    holdout_component_ids: tuple[str, ...]
    row_component_ids: Mapping[str, str]
    identity: str


@dataclass(frozen=True)
class QueryStateGenerationFormatEntry:
    row_identity: str
    ordinal: int
    record_id: str
    step_index: int
    original_image_sha256: str
    prompt_identity: str


@dataclass(frozen=True)
class QueryStateGenerationFormatManifest:
    schema: str
    split: str
    entries: tuple[QueryStateGenerationFormatEntry, ...]
    max_reasoning_tokens: int
    max_output_tokens: int
    prompt_protocol_identity: str
    turn_generation_spec_identity: str
    parser_protocol_identity: str
    identity: str

    @property
    def row_identities(self) -> tuple[str, ...]:
        return tuple(entry.row_identity for entry in self.entries)


def build_generation_response_policy_prompt(row: SFT1V2Early4Row) -> Any:
    record = row.record
    required = {
        "system_prompt", "observation_texts", "image_paths", "action_indices",
        "assistant_responses",
    }
    if not isinstance(record, Mapping) or not required <= set(record):
        raise ValueError("generation-format row lacks an exact production transcript")
    step = row.step_index
    observations = record["observation_texts"]
    images = record["image_paths"]
    actions = record["action_indices"]
    responses = record["assistant_responses"]
    if (
        not isinstance(record["system_prompt"], str)
        or not isinstance(observations, list)
        or not isinstance(images, list)
        or not isinstance(actions, list)
        or not isinstance(responses, list)
        or len(observations) <= step
        or len(images) <= step
        or len(actions) <= step
        or len(responses) <= step
        or images[step] != row.original_image_path
        or actions[step] != row.executed_action_index
    ):
        raise ValueError("generation-format row production transcript is misaligned")
    transcript = AgentTranscript(
        system_prompt=record["system_prompt"],
        observation_texts=tuple(observations[: step + 1]),
        observation_images=tuple(images[: step + 1]),
        action_indices=tuple(actions[:step]),
        assistant_responses=tuple(responses[:step]),
    )
    return build_turn_response_policy_prompt(
        NimlothPromptTemplate(latent_token_count=16, action_count=8),
        transcript,
    )


def build_generation_format_manifest(
    rows: Sequence[SFT1V2Early4Row],
    *,
    validation_split: QueryStateValidationSplit,
    mode: str,
    max_reasoning_tokens: int,
    max_output_tokens: int,
    turn_generation_spec_identity: str,
) -> QueryStateGenerationFormatManifest:
    """Bind registered real rows to the exact production prompt/spec/parser."""

    values = _validate_rows(rows)
    expected_split = "calibration" if mode == "pilot" else "holdout" if mode == "formal" else None
    if expected_split is None:
        raise ValueError("generation-format mode must be pilot or formal")
    allowed = set(rows_for_validation_mode(validation_split, mode=mode))
    if any(row.identity not in allowed or not row.external_eligible for row in values):
        raise ValueError(f"{mode} generation-format rows are outside the {expected_split} split")
    if (
        isinstance(max_reasoning_tokens, bool)
        or not isinstance(max_reasoning_tokens, int)
        or max_reasoning_tokens < 1
        or isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens <= max_reasoning_tokens
    ):
        raise ValueError("generation-format reasoning/output budgets are invalid")
    if not isinstance(turn_generation_spec_identity, str) or len(turn_generation_spec_identity) != 64 or set(turn_generation_spec_identity) - set("0123456789abcdef"):
        raise ValueError("generation-format TurnGenerationSpec identity must be SHA256")
    entries = tuple(
        QueryStateGenerationFormatEntry(
            row_identity=row.identity,
            ordinal=row.ordinal,
            record_id=row.record_id,
            step_index=row.step_index,
            original_image_sha256=row.original_image_sha256,
            prompt_identity=response_policy_prompt_identity(
                build_generation_response_policy_prompt(row)
            ),
        )
        for row in values
    )
    payload = {
        "schema": QUERY_STATE_GENERATION_FORMAT_MANIFEST_SCHEMA,
        "split": expected_split,
        "entries": [asdict(entry) for entry in entries],
        "max_reasoning_tokens": max_reasoning_tokens,
        "max_output_tokens": max_output_tokens,
        "prompt_protocol_identity": TURN_RESPONSE_PROMPT_PROTOCOL_IDENTITY,
        "turn_generation_spec_identity": turn_generation_spec_identity,
        "parser_protocol_identity": TURN_RESPONSE_PARSER_PROTOCOL_IDENTITY,
    }
    return QueryStateGenerationFormatManifest(
        schema=QUERY_STATE_GENERATION_FORMAT_MANIFEST_SCHEMA,
        split=expected_split,
        entries=entries,
        max_reasoning_tokens=max_reasoning_tokens,
        max_output_tokens=max_output_tokens,
        prompt_protocol_identity=TURN_RESPONSE_PROMPT_PROTOCOL_IDENTITY,
        turn_generation_spec_identity=turn_generation_spec_identity,
        parser_protocol_identity=TURN_RESPONSE_PARSER_PROTOCOL_IDENTITY,
        identity=_identity(payload),
    )


def validate_query_state_row_audit(audit: SFT1V2RowAudit) -> None:
    if not isinstance(audit, SFT1V2RowAudit):
        raise TypeError("Query-State row audit requires SFT1V2RowAudit")
    expected = {
        "train_records": 3211,
        "validation_records": 355,
        "train_rows": 12836,
        "excluded_train_empty_cot_rows": 5,
        "raw_validation_rows": 1420,
        "excluded_validation_empty_cot_rows": 0,
        "external_validation_rows": 1413,
        "cross_split_image_hashes": 5,
        "same_image_multi_instruction_groups": 42,
        "same_instruction_multi_image_groups": 101,
    }
    for field, value in expected.items():
        if getattr(audit, field) != value:
            raise ValueError(f"Query-State row audit {field} mismatch")


def _entry(
    row: SFT1V2Early4Row,
    rendered_counts: Mapping[str, tuple[int, int]],
) -> QueryStateManifestEntry:
    counts = rendered_counts.get(row.identity)
    if (
        not isinstance(counts, tuple)
        or len(counts) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts)
    ):
        raise ValueError("manifest requires exact rendered-token and valid-LM counts")
    return QueryStateManifestEntry(
        row_identity=row.identity,
        ordinal=row.ordinal,
        record_id=row.record_id,
        original_image_sha256=row.original_image_sha256,
        normalized_instruction=normalize_instruction_identity(row.instruction),
        step_index=row.step_index,
        executed_action_index=row.executed_action_index,
        outcome_bucket=_outcome_bucket(row),
        stratum=_stratum(row),
        rendered_token_count=counts[0],
        valid_lm_token_count=counts[1],
    )


def _validate_rows(rows: Sequence[SFT1V2Early4Row]) -> tuple[SFT1V2Early4Row, ...]:
    values = tuple(rows)
    if not values or any(not isinstance(row, SFT1V2Early4Row) for row in values):
        raise ValueError("Query-State manifest requires audited early-4 rows")
    identities = [row.identity for row in values]
    ordinals = [row.ordinal for row in values]
    if len(set(identities)) != len(values) or len(set(ordinals)) != len(values):
        raise ValueError("Query-State manifest rows must have unique identity and ordinal")
    return values


def build_coverage_manifest(
    rows: Sequence[SFT1V2Early4Row],
    *,
    requested_per_stratum: Mapping[str, int],
    rendered_counts: Mapping[str, tuple[int, int]],
    seed: int,
) -> QueryStateTrainingManifest:
    """Select exact real rows with record→image→instruction uniqueness priority."""

    values = _validate_rows(rows)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("coverage seed must be a non-negative integer")
    if not requested_per_stratum or any(
        not isinstance(name, str)
        or not name
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        for name, count in requested_per_stratum.items()
    ):
        raise ValueError("coverage counts must be explicit positive per-stratum values")
    available: dict[str, list[SFT1V2Early4Row]] = {}
    for row in values:
        available.setdefault(_stratum(row), []).append(row)
    unexpected = sorted(set(available) - set(requested_per_stratum))
    if unexpected:
        raise ValueError("coverage count is unresolved for stratum: " + unexpected[0])

    used_records: set[str] = set()
    used_images: set[str] = set()
    used_instructions: set[str] = set()
    selected: list[SFT1V2Early4Row] = []
    shortages: dict[str, int] = {}
    for stratum in sorted(requested_per_stratum):
        candidates = list(available.get(stratum, ()))
        requested = requested_per_stratum[stratum]
        for _index in range(min(requested, len(candidates))):
            def priority(row: SFT1V2Early4Row) -> tuple[bool, bool, bool, bytes, str]:
                instruction = normalize_instruction_identity(row.instruction)
                digest = hashlib.sha256(f"{seed}:{row.identity}".encode()).digest()
                return (
                    row.record_id in used_records,
                    row.original_image_sha256 in used_images,
                    instruction in used_instructions,
                    digest,
                    row.identity,
                )

            chosen = min(candidates, key=priority)
            candidates.remove(chosen)
            selected.append(chosen)
            used_records.add(chosen.record_id)
            used_images.add(chosen.original_image_sha256)
            used_instructions.add(normalize_instruction_identity(chosen.instruction))
        if len(available.get(stratum, ())) < requested:
            shortages[stratum] = requested - len(available.get(stratum, ()))
    entries = tuple(_entry(row, rendered_counts) for row in selected)
    payload = {
        "schema": QUERY_STATE_TRAINING_MANIFEST_SCHEMA,
        "kind": "coverage_first_pilot",
        "seed": seed,
        "entries": [asdict(item) for item in entries],
        "requested_per_stratum": dict(sorted(requested_per_stratum.items())),
        "shortages": dict(sorted(shortages.items())),
        "selection_priority": "record_then_image_then_normalized_instruction_then_sha256",
        "movement_action_table": [action.key for action in NAVIGATION_ACTION_SPACE.actions],
    }
    return QueryStateTrainingManifest(
        schema=QUERY_STATE_TRAINING_MANIFEST_SCHEMA,
        kind="coverage_first_pilot",
        seed=seed,
        entries=entries,
        shortages=dict(sorted(shortages.items())),
        identity=_identity(payload),
    )


def build_full_training_manifest(
    rows: Sequence[SFT1V2Early4Row],
    *,
    rendered_counts: Mapping[str, tuple[int, int]],
    seed: int,
) -> QueryStateTrainingManifest:
    values = _validate_rows(rows)
    ordered = sorted(
        values,
        key=lambda row: (hashlib.sha256(f"{seed}:{row.identity}".encode()).digest(), row.identity),
    )
    entries = tuple(_entry(row, rendered_counts) for row in ordered)
    payload = {
        "schema": QUERY_STATE_TRAINING_MANIFEST_SCHEMA,
        "kind": "full_train_once_per_epoch",
        "seed": seed,
        "entries": [asdict(item) for item in entries],
    }
    return QueryStateTrainingManifest(
        schema=QUERY_STATE_TRAINING_MANIFEST_SCHEMA,
        kind="full_train_once_per_epoch",
        seed=seed,
        entries=entries,
        shortages={},
        identity=_identity(payload),
    )


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def build_connected_validation_split(
    rows: Sequence[SFT1V2Early4Row],
    *,
    calibration_numerator: int,
    calibration_denominator: int,
    seed: int,
) -> QueryStateValidationSplit:
    """Split whole exact-image/normalized-instruction connected components."""

    values = _validate_rows(rows)
    if any(not row.external_eligible for row in values):
        raise ValueError("connected validation split accepts external-eligible rows only")
    if (
        isinstance(calibration_numerator, bool)
        or isinstance(calibration_denominator, bool)
        or not isinstance(calibration_numerator, int)
        or not isinstance(calibration_denominator, int)
        or not 0 < calibration_numerator < calibration_denominator
    ):
        raise ValueError("calibration ratio must be explicit and lie strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("validation split seed must be a non-negative integer")
    identities = [row.identity for row in values]
    union = _UnionFind(identities)
    image_owner: dict[str, str] = {}
    instruction_owner: dict[str, str] = {}
    for row in sorted(values, key=lambda item: item.identity):
        instruction = normalize_instruction_identity(row.instruction)
        for key, owners in (
            (row.original_image_sha256, image_owner),
            (instruction, instruction_owner),
        ):
            owner = owners.setdefault(key, row.identity)
            union.union(owner, row.identity)
    components: dict[str, list[SFT1V2Early4Row]] = {}
    for row in values:
        components.setdefault(union.find(row.identity), []).append(row)
    if len(components) < 2:
        raise ValueError("validation graph has fewer than two connected components")

    component_rows: dict[str, tuple[SFT1V2Early4Row, ...]] = {}
    for rows_in_component in components.values():
        ordered = tuple(sorted(rows_in_component, key=lambda item: item.identity))
        component_id = _identity([row.identity for row in ordered])
        component_rows[component_id] = ordered
    target = round(len(values) * calibration_numerator / calibration_denominator)
    target = min(max(target, 1), len(values) - 1)

    calibration: set[str] = set()
    calibration_count = 0
    # Large components are placed first; the seeded digest gives a fixed tie-break.
    ordered_components = sorted(
        component_rows,
        key=lambda component: (
            -len(component_rows[component]),
            hashlib.sha256(f"{seed}:{component}".encode()).digest(),
            component,
        ),
    )
    for index, component in enumerate(ordered_components):
        size = len(component_rows[component])
        remaining_components = len(ordered_components) - index - 1
        choose = abs((calibration_count + size) - target) < abs(calibration_count - target)
        if not calibration and remaining_components == 0:
            choose = True
        if choose and calibration_count + size < len(values):
            calibration.add(component)
            calibration_count += size
    if not calibration:
        smallest = min(ordered_components, key=lambda item: (len(component_rows[item]), item))
        calibration.add(smallest)
    if calibration == set(ordered_components):
        largest_hash = max(calibration)
        calibration.remove(largest_hash)

    holdout = set(ordered_components) - calibration
    calibration_rows = tuple(sorted(
        row.identity for component in calibration for row in component_rows[component]
    ))
    holdout_rows = tuple(sorted(
        row.identity for component in holdout for row in component_rows[component]
    ))
    row_components = {
        row.identity: component
        for component, component_values in component_rows.items()
        for row in component_values
    }
    if set(calibration_rows) & set(holdout_rows) or set(calibration_rows) | set(holdout_rows) != set(identities):
        raise RuntimeError("connected validation split lost or duplicated rows")
    payload = {
        "schema": QUERY_STATE_VALIDATION_SPLIT_SCHEMA,
        "seed": seed,
        "calibration_ratio": [calibration_numerator, calibration_denominator],
        "target_calibration_rows": target,
        "calibration_components": sorted(calibration),
        "holdout_components": sorted(holdout),
        "row_components": dict(sorted(row_components.items())),
        "component_edges": "exact_image_sha256_or_normalized_instruction",
    }
    return QueryStateValidationSplit(
        schema=QUERY_STATE_VALIDATION_SPLIT_SCHEMA,
        seed=seed,
        calibration_numerator=calibration_numerator,
        calibration_denominator=calibration_denominator,
        target_calibration_rows=target,
        calibration_row_identities=calibration_rows,
        holdout_row_identities=holdout_rows,
        calibration_component_ids=tuple(sorted(calibration)),
        holdout_component_ids=tuple(sorted(holdout)),
        row_component_ids=dict(sorted(row_components.items())),
        identity=_identity(payload),
    )


def _manifest_entry_from_raw(
    raw: object,
    *,
    rows_by_identity: Mapping[str, SFT1V2Early4Row],
) -> QueryStateManifestEntry:
    fields = tuple(QueryStateManifestEntry.__dataclass_fields__)
    if not isinstance(raw, Mapping) or set(raw) != set(fields):
        raise ValueError("Query-State training manifest entry schema is invalid")
    try:
        entry = QueryStateManifestEntry(**{field: raw[field] for field in fields})
    except TypeError as error:
        raise ValueError("Query-State training manifest entry types are invalid") from error
    row = rows_by_identity.get(entry.row_identity)
    if row is None:
        raise ValueError("Query-State training manifest contains an unknown row")
    expected = _entry(
        row,
        {row.identity: (entry.rendered_token_count, entry.valid_lm_token_count)},
    )
    if entry != expected:
        raise ValueError("Query-State training manifest entry differs from its audited row")
    return entry


def deserialize_query_state_training_manifest(
    path: Path,
    *,
    rows: Sequence[SFT1V2Early4Row],
    expected_identity: str,
    expected_mode: str,
    expected_rows: int,
    expected_seed: int,
) -> QueryStateTrainingManifest:
    """Rebuild the canonical selector and identity from an immutable JSON manifest."""

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid Query-State training manifest JSON") from error
    values = _validate_rows(rows)
    rows_by_identity = {row.identity: row for row in values}
    expected_kind = (
        "coverage_first_pilot" if expected_mode == "pilot"
        else "full_train_once_per_epoch" if expected_mode == "formal"
        else None
    )
    if expected_kind is None:
        raise ValueError("Query-State training manifest mode is invalid")
    common = {"schema", "kind", "seed", "entries", "identity"}
    pilot_only = {
        "requested_per_stratum", "shortages", "selection_priority",
        "movement_action_table",
    }
    expected_fields = common | (pilot_only if expected_mode == "pilot" else set())
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise ValueError("Query-State training manifest schema is not canonical")
    if (
        raw["schema"] != QUERY_STATE_TRAINING_MANIFEST_SCHEMA
        or raw["kind"] != expected_kind
        or raw["seed"] != expected_seed
        or raw["identity"] != expected_identity
        or not isinstance(raw["entries"], list)
        or len(raw["entries"]) != expected_rows
    ):
        raise ValueError("Query-State training manifest identity/kind/count/seed mismatch")
    entries = tuple(
        _manifest_entry_from_raw(item, rows_by_identity=rows_by_identity)
        for item in raw["entries"]
    )
    if len({entry.row_identity for entry in entries}) != len(entries):
        raise ValueError("Query-State training manifest duplicates a row")
    rendered_counts = {
        entry.row_identity: (entry.rendered_token_count, entry.valid_lm_token_count)
        for entry in entries
    }
    if expected_mode == "pilot":
        if raw["selection_priority"] != "record_then_image_then_normalized_instruction_then_sha256":
            raise ValueError("Query-State pilot selection priority changed")
        if raw["movement_action_table"] != [
            action.key for action in NAVIGATION_ACTION_SPACE.actions
        ]:
            raise ValueError("Query-State pilot action table changed")
        requested = raw["requested_per_stratum"]
        shortages = raw["shortages"]
        if not isinstance(requested, Mapping) or not isinstance(shortages, Mapping):
            raise ValueError("Query-State pilot stratum/shortage fields are invalid")
        rebuilt = build_coverage_manifest(
            tuple(row for row in values if row.split == "train"),
            requested_per_stratum=requested,
            rendered_counts=rendered_counts,
            seed=expected_seed,
        )
        if dict(rebuilt.shortages) != dict(shortages):
            raise ValueError("Query-State pilot shortage report changed")
    else:
        train_rows = tuple(row for row in values if row.split == "train")
        if len(train_rows) != 12836 or expected_rows != 12836:
            raise ValueError("formal Query-State manifest does not cover all 12,836 train rows")
        rebuilt = build_full_training_manifest(
            train_rows,
            rendered_counts=rendered_counts,
            seed=expected_seed,
        )
    if rebuilt.entries != entries or rebuilt.identity != expected_identity:
        raise ValueError("Query-State training manifest canonical identity mismatch")
    return rebuilt


def deserialize_query_state_validation_split(
    path: Path,
    *,
    rows: Sequence[SFT1V2Early4Row],
    expected_identity: str,
) -> QueryStateValidationSplit:
    """Recompute all 1,413 connected components and both disjoint partitions."""

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid Query-State validation manifest JSON") from error
    fields = {
        "schema", "seed", "calibration_numerator", "calibration_denominator",
        "target_calibration_rows", "calibration_row_identities",
        "holdout_row_identities", "calibration_component_ids",
        "holdout_component_ids", "row_component_ids", "identity",
    }
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise ValueError("Query-State validation manifest schema is not canonical")
    if raw["schema"] != QUERY_STATE_VALIDATION_SPLIT_SCHEMA or raw["identity"] != expected_identity:
        raise ValueError("Query-State validation manifest identity/schema mismatch")
    external = tuple(row for row in _validate_rows(rows) if row.external_eligible)
    if len(external) != 1413:
        raise ValueError("Query-State validation audit must contain exactly 1,413 external rows")
    rebuilt = build_connected_validation_split(
        external,
        calibration_numerator=raw["calibration_numerator"],
        calibration_denominator=raw["calibration_denominator"],
        seed=raw["seed"],
    )
    expected = json.loads(json.dumps(asdict(rebuilt), sort_keys=True))
    if any(raw[field] != expected[field] for field in fields):
        raise ValueError("Query-State validation connected split or identity mismatch")
    return rebuilt


def deserialize_generation_format_manifest(
    path: Path,
    *,
    rows: Sequence[SFT1V2Early4Row],
    validation_split: QueryStateValidationSplit,
    expected_identity: str,
    expected_mode: str,
) -> QueryStateGenerationFormatManifest:
    """Reject wrong-split, duplicate, unknown, or prompt-divergent format rows."""

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid Query-State generation-format manifest JSON") from error
    fields = {
        "schema", "split", "entries", "max_reasoning_tokens",
        "max_output_tokens", "prompt_protocol_identity",
        "turn_generation_spec_identity", "parser_protocol_identity", "identity",
    }
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise ValueError("Query-State generation-format manifest schema is not canonical")
    entries = raw["entries"]
    if (
        raw["schema"] != QUERY_STATE_GENERATION_FORMAT_MANIFEST_SCHEMA
        or raw["identity"] != expected_identity
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("Query-State generation-format manifest identity/schema/rows mismatch")
    by_identity = {row.identity: row for row in _validate_rows(rows)}
    selected: list[SFT1V2Early4Row] = []
    expected_entry_fields = set(QueryStateGenerationFormatEntry.__dataclass_fields__)
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, Mapping) or set(item) != expected_entry_fields:
            raise ValueError("Query-State generation-format row schema is invalid")
        identity = item["row_identity"]
        if not isinstance(identity, str) or identity in seen:
            raise ValueError("Query-State generation-format manifest duplicates a row")
        row = by_identity.get(identity)
        if row is None:
            raise ValueError("Query-State generation-format manifest has an unregistered row")
        seen.add(identity)
        selected.append(row)
    rebuilt = build_generation_format_manifest(
        selected,
        validation_split=validation_split,
        mode=expected_mode,
        max_reasoning_tokens=raw["max_reasoning_tokens"],
        max_output_tokens=raw["max_output_tokens"],
        turn_generation_spec_identity=raw["turn_generation_spec_identity"],
    )
    expected = json.loads(json.dumps(asdict(rebuilt), sort_keys=True))
    if raw != expected:
        raise ValueError(
            "Query-State generation-format manifest row/split/protocol identity mismatch"
        )
    return rebuilt


def rows_for_training_mode(
    manifest: QueryStateTrainingManifest,
    *,
    mode: str,
) -> tuple[str, ...]:
    expected = "coverage_first_pilot" if mode == "pilot" else "full_train_once_per_epoch" if mode == "formal" else None
    if expected is None or manifest.kind != expected:
        if manifest.kind == "full_train_once_per_epoch":
            raise ValueError("full training manifest is formal-only")
        raise ValueError(f"{mode} requires {'coverage' if mode == 'pilot' else 'full'} training manifest")
    return manifest.row_identities


def rows_for_validation_mode(
    split: QueryStateValidationSplit,
    *,
    mode: str,
    requested_split: str | None = None,
) -> tuple[str, ...]:
    expected = "calibration" if mode == "pilot" else "holdout" if mode == "formal" else None
    if expected is None:
        raise ValueError("validation mode must be pilot or formal")
    requested = requested_split or expected
    if requested != expected:
        if mode == "pilot":
            raise ValueError("pilot must not open holdout validation rows")
        raise ValueError("formal primary validation must use untouched holdout rows")
    return split.calibration_row_identities if expected == "calibration" else split.holdout_row_identities


__all__ = [
    "QUERY_STATE_GENERATION_FORMAT_MANIFEST_SCHEMA",
    "QUERY_STATE_TRAINING_MANIFEST_SCHEMA",
    "QUERY_STATE_VALIDATION_SPLIT_SCHEMA",
    "QueryStateGenerationFormatEntry",
    "QueryStateGenerationFormatManifest",
    "QueryStateManifestEntry",
    "QueryStateTrainingManifest",
    "QueryStateValidationSplit",
    "build_connected_validation_split",
    "build_coverage_manifest",
    "build_generation_format_manifest",
    "build_generation_response_policy_prompt",
    "build_full_training_manifest",
    "deserialize_generation_format_manifest",
    "deserialize_query_state_training_manifest",
    "deserialize_query_state_validation_split",
    "normalize_instruction_identity",
    "rows_for_training_mode",
    "rows_for_validation_mode",
    "validate_query_state_row_audit",
]
