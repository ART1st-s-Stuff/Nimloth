"""CLI for post-hoc reconstruction decoder training."""

from __future__ import annotations

import argparse
from pathlib import Path

from nimloth.training.reconstruction.trainer import train_reconstruction_decoder


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train post-hoc WM image decoder")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--state-proj-checkpoint", type=Path, required=True)
    ap.add_argument("--wm-checkpoint", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=("l1", "mse"), default="l1")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--max-train-records", type=int, default=-1)
    ap.add_argument("--max-val-records", type=int, default=-1)
    ap.add_argument("--max-val-batches", type=int, default=-1)
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--save-samples", type=int, default=16)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--attn-implementation", default="sdpa")
    return ap


def main(argv: list[str] | None = None) -> int:
    return train_reconstruction_decoder(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
