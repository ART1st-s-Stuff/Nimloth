#!/usr/bin/env python3
"""Print CPU-verifiable ID176 processor/token identities for config resolution."""

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nimloth.training.sft1.identity import audit_id176_processor_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        asdict(audit_id176_processor_identity(args.actor_checkpoint)),
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
