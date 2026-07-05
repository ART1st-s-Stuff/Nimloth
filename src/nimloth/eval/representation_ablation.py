"""Compatibility entrypoint for config-driven representation ablation eval.

Use:

    python -m nimloth.eval.representation_ablation --config <yaml>
"""

from __future__ import annotations

from nimloth.representation_ablation.eval import main


if __name__ == "__main__":
    raise SystemExit(main())
