from __future__ import annotations

import csv

from nimloth.training.sft.stage3.reporting import (
    STAGE3_LOG_COLUMNS,
    SFT2Reporter,
)
from nimloth.util.csv_log import CSVRecordWriter


COMPONENT_METRICS = {
    "wm_mse": 3.0,
    "wm_spatial_mse": 1.0,
    "wm_cls_mse": 2.0,
    "dino_grid_mse": 7.0,
    "dino_spatial_mse": 3.0,
    "dino_cls_mse": 4.0,
    "predicted_dino_grid_mse": 11.0,
    "predicted_dino_spatial_mse": 5.0,
    "predicted_dino_cls_mse": 6.0,
}


def test_stage3_reporter_persists_k65_component_metrics(tmp_path, monkeypatch) -> None:
    from nimloth.training.sft.stage3 import reporting

    monkeypatch.setattr(reporting, "is_main", lambda: True)
    monkeypatch.setattr(reporting, "log_metrics", lambda *args, **kwargs: None)
    path = tmp_path / "train_step_log.csv"
    writer = CSVRecordWriter(path, STAGE3_LOG_COLUMNS)
    writer.ensure_header()
    reporter = SFT2Reporter(
        log_writer=writer,
        wandb_run=None,
        llm_tune="full",
        vision_tune="frozen",
    )

    reporter.log_train_step(
        epoch=1,
        global_step=1,
        metrics=COMPONENT_METRICS,
        wm_weight=1.0,
        sigreg_weight=0.0,
        qwen_lr=2e-7,
    )
    reporter.log_validation(
        epoch=1,
        global_step=1,
        metrics=COMPONENT_METRICS,
        best_val_wm_mse=3.0,
        checkpoint_metric="val_wm_mse",
    )

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    for row in rows:
        for name, value in COMPONENT_METRICS.items():
            assert float(row[name]) == value

