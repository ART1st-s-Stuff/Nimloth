"""Compatibility wrapper for representation ablation training."""

from nimloth.representation_ablation.train import build_arg_parser, main, train

__all__ = ["build_arg_parser", "main", "train"]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
