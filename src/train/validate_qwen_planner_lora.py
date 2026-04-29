"""Validate Qwen planner LoRA special-token format, action prior, and latent extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
import torch

from src.visualize.wandb_tracker import init_tracker
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.qwen_planner import (
    extract_planner_special_outputs,
    generate_planner_response,
    load_jsonl,
    validate_planner_special_output,
)

console = Console()


def _topk_contains(probs: torch.Tensor, target: int, k: int) -> bool:
    ranked = torch.argsort(probs.float(), descending=True).tolist()
    return int(target) in [int(x) for x in ranked[:k]]


def _build_metrics_table(
    *,
    total: int,
    processed: int,
    format_ok: int,
    top1: int,
    top3: int,
    latent_ok: int,
    failures: int,
    extract_latent: bool,
    current_id: str,
) -> Table:
    table = Table(title="Qwen Planner LoRA 验证指标", expand=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    denominator_total = max(1, processed)
    denominator_format = max(1, format_ok)
    table.add_row("processed", f"{processed}/{total}")
    table.add_row("current_id", current_id or "-")
    table.add_row("format_ok", str(format_ok))
    table.add_row("format_rate", f"{format_ok / denominator_total:.4f}")
    table.add_row("action_prior_top1", f"{top1 / denominator_format:.4f}")
    table.add_row("action_prior_top3", f"{top3 / denominator_format:.4f}")
    if extract_latent:
        table.add_row("latent_extract_rate", f"{latent_ok / denominator_format:.4f}")
    table.add_row("failures", str(failures))
    return table


def _build_failure_panel(failures: list[dict[str, str]]) -> Panel:
    if not failures:
        return Panel("无失败样本", title="最近失败", expand=True)
    lines = []
    for item in failures[-3:]:
        error = item.get("error", "").replace("\n", " ")[:180]
        lines.append(f"{item.get('id', '-')}: {error}")
    return Panel("\n".join(lines), title="最近失败", expand=True)


def _validate_record_schema(item: dict, idx: int) -> None:
    item_id = str(item.get("id", idx))
    if not str(item.get("prompt", "")).strip():
        raise ValueError(f"{item_id} missing required prompt")
    raw_image = str(item.get("image", ""))
    image_path = Path(raw_image)
    if not raw_image.strip() or not image_path.is_file():
        raise FileNotFoundError(f"{item_id} image file not found: {image_path}")
    action_id = int(item["action_id"])
    if action_id < 0 or action_id > 7:
        raise ValueError(f"{item_id} invalid action_id={action_id}")


def _validate_records(
    *,
    records: list[dict],
    model,
    processor,
    max_new_tokens: int,
    extract_latent: bool,
    show_progress: bool,
) -> dict:
    format_ok = 0
    top1 = 0
    top3 = 0
    latent_ok = 0
    failures: list[dict[str, str]] = []
    total = len(records)

    progress = Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    task_id = progress.add_task("Validating planner outputs", total=total)
    metrics_table = _build_metrics_table(
        total=total,
        processed=0,
        format_ok=0,
        top1=0,
        top3=0,
        latent_ok=0,
        failures=0,
        extract_latent=extract_latent,
        current_id="",
    )

    live_context = (
        Live(Group(metrics_table, _build_failure_panel(failures), progress), console=console, refresh_per_second=4)
        if show_progress
        else None
    )
    if live_context is not None:
        live_context.__enter__()
    try:
        for idx, item in enumerate(records):
            item_id = str(item.get("id", idx))
            text = ""
            try:
                _validate_record_schema(item, idx)
                text = generate_planner_response(
                    model=model,
                    processor=processor,
                    image_path=str(item["image"]),
                    prompt=str(item["prompt"]),
                    max_new_tokens=max_new_tokens,
                )
                valid, reason, _generated_action = validate_planner_special_output(text)
                if not valid:
                    raise ValueError(reason)
                format_ok += 1

                with torch.no_grad():
                    extracted = extract_planner_special_outputs(
                        model=model,
                        processor=processor,
                        image_path=str(item["image"]),
                        prompt=str(item["prompt"]),
                        response=text,
                        max_new_tokens=max_new_tokens,
                    )
                action_id = int(item["action_id"])
                top1 += int(_topk_contains(extracted.action_prior, action_id, 1))
                top3 += int(_topk_contains(extracted.action_prior, action_id, 3))
                if extract_latent:
                    latent_ok += int(torch.isfinite(extracted.latent).all().item())
            except Exception as exc:
                failures.append({"id": item_id, "error": str(exc), "text": text[:300]})

            processed = idx + 1
            progress.update(task_id, advance=1)
            if live_context is not None:
                metrics_table = _build_metrics_table(
                    total=total,
                    processed=processed,
                    format_ok=format_ok,
                    top1=top1,
                    top3=top3,
                    latent_ok=latent_ok,
                    failures=len(failures),
                    extract_latent=extract_latent,
                    current_id=item_id,
                )
                live_context.update(Group(metrics_table, _build_failure_panel(failures), progress))
    finally:
        if live_context is not None:
            live_context.__exit__(None, None, None)

    return {
        "num_records": total,
        "format_rate": format_ok / max(1, total),
        "action_prior_top1": top1 / max(1, format_ok),
        "action_prior_top3": top3 / max(1, format_ok),
        "latent_extract_rate": latent_ok / max(1, format_ok) if extract_latent else None,
        "num_failures": len(failures),
        "failures": failures[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-jsonl", default="datasets/EB-Nav/phase2_qwen_planner_sft.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--extract-latent", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable Rich UI/progress output.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B logging for this script.")
    args = parser.parse_args()

    records = load_jsonl(args.sft_jsonl, limit=int(args.limit))
    if not records:
        raise RuntimeError(f"No validation records found in {args.sft_jsonl}")
    for idx, item in enumerate(records):
        _validate_record_schema(item, idx)

    adapter = QwenVLMAdapter(
        model_name=args.model_name,
        latent_dim=4096,
        enabled=True,
        fallback_enabled=False,
    )
    adapter._ensure_model()
    if adapter._model is None or adapter._processor is None:
        raise RuntimeError(f"Failed to load Qwen model: {adapter.init_error}")
    if args.adapter_path:
        adapter.load_lora_adapter(args.adapter_path, trainable=False)
    else:
        adapter.ensure_planner_special_tokens()
    model = adapter._model
    processor = adapter._processor
    model.eval()

    tracker = None
    if not args.disable_wandb:
        tracker = init_tracker(
            task_name="validate_qwen_planner_lora",
            config={
                "sft_jsonl": args.sft_jsonl,
                "model_name": args.model_name,
                "adapter_path": args.adapter_path,
                "limit": int(args.limit),
                "max_new_tokens": int(args.max_new_tokens),
                "extract_latent": bool(args.extract_latent),
                "num_records": len(records),
            },
        )
    metrics = _validate_records(
        records=records,
        model=model,
        processor=processor,
        max_new_tokens=int(args.max_new_tokens),
        extract_latent=bool(args.extract_latent),
        show_progress=not bool(args.no_progress),
    )
    if tracker is not None:
        tracker.log_metrics(
            {f"validation/{key}": value for key, value in metrics.items() if isinstance(value, (int, float)) and value is not None},
            step=0,
        )
        failures_path = Path("outputs/qwen_planner_validation_failures.json")
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failures_path, "w") as f:
            json.dump(metrics.get("failures", []), f, ensure_ascii=False, indent=2)
        tracker.log_artifact_path("qwen-planner-validation-failures", failures_path, artifact_type="metrics")
        tracker.finish()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
