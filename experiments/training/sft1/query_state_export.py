#!/usr/bin/env python3
"""Thin read-only preflight for the separately gated Query-State exporter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _verify_local_environment() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("Query-State export preflight requires disabled bytecode")
    if not Path(os.environ.get("PYTHONPYCACHEPREFIX", "")).is_absolute():
        raise RuntimeError("Query-State export preflight requires external pycache")


def _load(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Query-State export contract must be a mapping")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("preflight", choices=("preflight",))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    _verify_local_environment()

    from nimloth.training.sft1.query_state_export import (
        parse_query_state_export_contract,
        verify_query_state_export_gate,
    )

    contract = parse_query_state_export_contract(_load(args.config))
    evidence = verify_query_state_export_gate(contract)
    print(json.dumps({
        "checkpoint_identity": evidence.checkpoint_identity,
        "human_decision": evidence.human_decision,
        "materialization_executed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
