"""Canonical JSON configuration helpers for the frozen-State WM ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nimloth.training.wm_heads.trainer import MatchedTrainerConfig
from nimloth.wm.matched_heads import MatchedHeadSpec


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def head_spec(config: dict[str, Any]) -> MatchedHeadSpec:
    vector, token = config["heads"]["vector"], config["heads"]["token"]
    shared = (int(vector["depth"]), int(vector["heads"]), int(vector["mlp_ratio"]))
    if shared != (int(token["depth"]), int(token["heads"]), int(token["mlp_ratio"])):
        raise ValueError("matched heads must share depth, heads, and mlp_ratio")
    return MatchedHeadSpec(
        state_tokens=int(token["tokens"]),
        token_dim=int(token["emb_dim"]),
        vector_hidden_dim=int(vector["hidden_dim"]),
        token_hidden_dim=int(token["hidden_dim"]),
        depth=shared[0],
        heads=shared[1],
        mlp_ratio=shared[2],
    )


def trainer_config(config: dict[str, Any]) -> MatchedTrainerConfig:
    training = config["training"]
    return MatchedTrainerConfig(
        seed=int(config["seed"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        cosine_weight=float(training["cosine_weight"]),
        grad_clip=float(training["grad_clip"]),
    )


def output_dir(config: dict[str, Any]) -> Path:
    return Path(config["output_dir"])
