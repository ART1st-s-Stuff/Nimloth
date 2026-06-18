#!/usr/bin/env python3
"""SFT2 experiment entry (thin wrapper).

Core training logic: nimloth.training.sft2.trainer
Config: configs/training/sft2/latent_wm_value.yaml
"""

from __future__ import annotations

from nimloth.training.sft2.trainer import main

if __name__ == "__main__":
    raise SystemExit(main())
