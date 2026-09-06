"""Recorded multimodal answers: teacher-forcing masks and preprocessing cache."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor

from nimloth.latent import (
    add_special_tokens,
    latent_state_tokens,
    normalize_latent_state_blocks,
)
from nimloth.util.distributed import is_main


def collect_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    imgs: list[Image.Image] = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image":
                    path = part["image"]
                    imgs.append(Image.open(path).convert("RGB"))
    return imgs


def _mask_latent_query_labels(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    processor: AutoProcessor,
    *,
    latent_token_count: int,
) -> torch.Tensor:
    if not hasattr(processor.tokenizer, "convert_tokens_to_ids"):
        return labels
    for token in latent_state_tokens(latent_token_count):
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(processor.tokenizer, "unk_token_id", None)
        if token_id is None or token_id == unk_id:
            continue
        labels = labels.masked_fill(input_ids == int(token_id), -100)
    return labels


def _encode_input_ids(
    processor: AutoProcessor,
    messages: list[dict[str, Any]],
    images: list[Image.Image],
    max_length: int,
    *,
    add_generation_prompt: bool,
    latent_token_count: int = 1,
) -> list[int]:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    text = normalize_latent_state_blocks(text, latent_token_count)
    enc = processor(
        text=[text],
        images=images or None,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return enc["input_ids"][0].tolist()


def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def assistant_token_spans(
    messages: list[dict[str, Any]],
    processor: AutoProcessor,
    max_length: int,
    *,
    latent_token_count: int = 1,
) -> list[tuple[int, int]]:
    """Return assistant token spans aligned to a single full multimodal encoding."""
    images = collect_images(messages)
    full_ids = _encode_input_ids(
        processor,
        messages,
        images,
        max_length,
        add_generation_prompt=False,
        latent_token_count=latent_token_count,
    )
    spans: list[tuple[int, int]] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prefix_images = collect_images(messages[:i])
        through_images = collect_images(messages[: i + 1])
        prefix_ids = _encode_input_ids(
            processor,
            messages[:i],
            prefix_images,
            max_length,
            add_generation_prompt=True,
            latent_token_count=latent_token_count,
        )
        through_ids = _encode_input_ids(
            processor,
            messages[: i + 1],
            through_images,
            max_length,
            add_generation_prompt=False,
            latent_token_count=latent_token_count,
        )
        start = len(prefix_ids)
        end = len(through_ids)
        if start >= end:
            continue
        if full_ids[start:end] == through_ids[start:end]:
            spans.append((start, end))
            continue
        if full_ids[:start] == prefix_ids:
            spans.append((start, end))
            continue
        anchor = prefix_ids[-min(32, len(prefix_ids)) :]
        anchor_at = _find_subsequence(full_ids, anchor)
        if anchor_at is None:
            continue
        start = anchor_at + len(anchor)
        end = start + (end - len(prefix_ids))
        if 0 <= start < end <= len(full_ids):
            spans.append((start, end))
    return spans


def collate_fn(
    batch: list[dict[str, Any]],
    processor: AutoProcessor,
    max_length: int,
    *,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> dict[str, torch.Tensor]:
    texts: list[str] = []
    spans_per_item: list[list[tuple[int, int]]] = []
    all_images: list[list[Image.Image]] = []
    for item in batch:
        messages = item["messages"]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        text = normalize_latent_state_blocks(text, latent_token_count)
        texts.append(text)
        spans_per_item.append(
            assistant_token_spans(
                messages,
                processor,
                max_length,
                latent_token_count=latent_token_count,
            )
        )
        all_images.append(collect_images(messages))

    enc = processor(
        text=texts,
        images=all_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[:] = -100
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    for row, spans in enumerate(spans_per_item):
        for start, end in spans:
            end = min(end, labels.shape[1])
            if start >= end:
                continue
            labels[row, start:end] = enc["input_ids"][row, start:end]
        if mask_latent_query_labels:
            labels[row] = _mask_latent_query_labels(
                labels[row],
                enc["input_ids"][row],
                processor,
                latent_token_count=latent_token_count,
            )
        trained = labels[row][labels[row] != -100]
        if (
            trained.numel()
            and image_pad_id is not None
            and (trained == image_pad_id).any().item()
        ):
            raise ValueError("supervised labels include <|image_pad|> tokens")
    enc["labels"] = labels
    if torch.all(labels == -100):
        raise ValueError("assistant-only SFT mask is empty for the whole batch")
    return enc


def safe_cache_name(record_id: str) -> str:
    return record_id.replace("/", "__").replace(" ", "_")


def cache_fingerprint(
    jsonl_path: Path,
    max_length: int,
    max_pixels: int,
    min_pixels: int,
    vocab_size: int,
    max_images_per_record: int,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
    cache_pixel_dtype: str = "bfloat16",
    processor_source: str = "",
    latent_query_mode: str = "inject",
) -> str:
    stat = jsonl_path.stat()
    payload = "|".join(
        [
            str(jsonl_path.resolve()),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            str(max_length),
            str(max_pixels),
            str(min_pixels),
            str(vocab_size),
            str(max_images_per_record),
            str(latent_token_count),
            str(mask_latent_query_labels),
            latent_query_mode,
            cache_pixel_dtype,
            processor_source,
            "v6-query-mode",
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def encode_sample_with_labels(
    messages: list[dict[str, Any]],
    processor: AutoProcessor,
    max_length: int,
    *,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> dict[str, torch.Tensor]:
    images = collect_images(messages)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    text = normalize_latent_state_blocks(text, latent_token_count)
    enc = processor(
        text=[text],
        images=images or None,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[:] = -100
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    for start, end in assistant_token_spans(
        messages,
        processor,
        max_length,
        latent_token_count=latent_token_count,
    ):
        end = min(end, labels.shape[1])
        if start >= end:
            continue
        labels[0, start:end] = enc["input_ids"][0, start:end]
    if mask_latent_query_labels:
        labels[0] = _mask_latent_query_labels(
            labels[0],
            enc["input_ids"][0],
            processor,
            latent_token_count=latent_token_count,
        )
    trained = labels[0][labels[0] != -100]
    if (
        trained.numel()
        and image_pad_id is not None
        and (trained == image_pad_id).any().item()
    ):
        raise ValueError("supervised labels include <|image_pad|> tokens")
    if torch.all(labels == -100):
        raise ValueError("assistant-only SFT mask is empty for sample")
    out: dict[str, torch.Tensor] = {}
    for key, value in enc.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.squeeze(0)
    out["labels"] = labels.squeeze(0)
    return out


def collate_cached_fn(
    batch: list[dict[str, torch.Tensor]], pad_token_id: int
) -> dict[str, torch.Tensor]:
    if len(batch) == 1:
        return {k: (v.unsqueeze(0) if v.ndim == 1 else v) for k, v in batch[0].items()}
    out: dict[str, torch.Tensor] = {}
    if "input_ids" in batch[0]:
        out["input_ids"] = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
    if "attention_mask" in batch[0]:
        out["attention_mask"] = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
    if "labels" in batch[0]:
        out["labels"] = pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=-100
        )
    for key in ("pixel_values", "image_grid_thw"):
        if key in batch[0]:
            out[key] = torch.cat([item[key] for item in batch], dim=0)
    return out


_CACHE_PROCESSOR: AutoProcessor | None = None
_CACHE_MAX_LENGTH = 0
_CACHE_LATENT_TOKEN_COUNT = 1
_CACHE_MASK_LATENT_QUERY_LABELS = True
_CACHE_PIXEL_DTYPE = "bfloat16"


def _init_cache_worker(
    model_path: str,
    min_pixels: int,
    max_pixels: int,
    max_length: int,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
    cache_pixel_dtype: str = "bfloat16",
) -> None:
    global \
        _CACHE_PROCESSOR, \
        _CACHE_MAX_LENGTH, \
        _CACHE_LATENT_TOKEN_COUNT, \
        _CACHE_MASK_LATENT_QUERY_LABELS, \
        _CACHE_PIXEL_DTYPE
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    add_special_tokens(processor.tokenizer, latent_token_count=latent_token_count)
    _CACHE_PROCESSOR = processor
    _CACHE_MAX_LENGTH = max_length
    _CACHE_LATENT_TOKEN_COUNT = int(latent_token_count)
    _CACHE_MASK_LATENT_QUERY_LABELS = bool(mask_latent_query_labels)
    _CACHE_PIXEL_DTYPE = cache_pixel_dtype


def _cache_one_sample(
    task: tuple[str, list[dict[str, Any]], str],
) -> tuple[str, bool, str]:
    record_id, messages, out_path = task
    try:
        assert _CACHE_PROCESSOR is not None
        encoded = encode_sample_with_labels(
            messages,
            _CACHE_PROCESSOR,
            _CACHE_MAX_LENGTH,
            latent_token_count=_CACHE_LATENT_TOKEN_COUNT,
            mask_latent_query_labels=_CACHE_MASK_LATENT_QUERY_LABELS,
        )
        if "pixel_values" in encoded:
            pixel_dtypes = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            encoded["pixel_values"] = encoded["pixel_values"].to(
                pixel_dtypes[_CACHE_PIXEL_DTYPE]
            )
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoded, path)
        return record_id, True, ""
    except Exception as exc:  # noqa: BLE001
        return record_id, False, str(exc)


def load_jsonl(path: Path, max_records: int = -1) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_records > 0 and len(records) >= max_records:
                break
    return records


class NimlothVLSFTDataset(Dataset):
    def __init__(
        self,
        path: Path,
        processor: AutoProcessor,
        max_records: int = -1,
        max_images_per_record: int = -1,
        *,
        cache_dir: Path | None = None,
    ):
        self.path = path
        self.records = load_jsonl(path, max_records=max_records)
        self.processor = processor
        self.max_images_per_record = max_images_per_record
        self.cache_dir = cache_dir
        self.use_cache = cache_dir is not None

    def __len__(self) -> int:
        return len(self.records)

    def _content_with_images(self, text: str, image_paths_iter) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        chunks = text.split("<image>")
        for i, chunk in enumerate(chunks):
            if chunk:
                parts.append({"type": "text", "text": chunk})
            if i < len(chunks) - 1:
                try:
                    path = next(image_paths_iter)
                except StopIteration as e:
                    raise ValueError(
                        "not enough image paths for <image> placeholders"
                    ) from e
                parts.append({"type": "image", "image": path})
        return parts

    def build_messages(self, rec: dict[str, Any]) -> list[dict[str, Any]]:
        image_paths = list(rec.get("image_paths", []))
        if self.max_images_per_record > 0:
            image_paths = image_paths[: self.max_images_per_record]
        image_iter = iter(image_paths)
        messages = []
        for msg in rec["messages"]:
            content = msg["content"]
            if "<image>" in content:
                content = self._content_with_images(content, image_iter)
            messages.append({"role": msg["role"], "content": content})
        return messages

    def cache_path_for_id(self, record_id: str) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir is not configured")
        return self.cache_dir / f"{safe_cache_name(record_id)}.pt"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        if self.use_cache:
            cache_path = self.cache_path_for_id(rec["id"])
            if not cache_path.is_file():
                raise FileNotFoundError(f"missing preprocess cache: {cache_path}")
            return torch.load(cache_path, map_location="cpu", weights_only=True)
        messages = self.build_messages(rec)
        image_paths = list(rec.get("image_paths", []))
        if self.max_images_per_record > 0:
            image_paths = image_paths[: self.max_images_per_record]
        return {"id": rec["id"], "messages": messages, "num_images": len(image_paths)}

    def get_messages(self, idx: int) -> list[dict[str, Any]]:
        return self.build_messages(self.records[idx])


def build_preprocess_cache(
    dataset: NimlothVLSFTDataset,
    processor: AutoProcessor,
    cache_dir: Path,
    max_length: int,
    model_path: Path,
    min_pixels: int,
    max_pixels: int,
    preprocess_workers: int,
    force: bool = False,
    *,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
    cache_pixel_dtype: str = "bfloat16",
    latent_query_mode: str = "inject",
) -> None:
    del processor
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if not force and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("count") == len(dataset)
            and manifest.get("max_length") == max_length
            and int(manifest.get("latent_token_count", 1)) == int(latent_token_count)
            and bool(manifest.get("mask_latent_query_labels", True))
            == bool(mask_latent_query_labels)
            and manifest.get(
                "latent_query_mode",
                "inject"
                if manifest.get("mask_latent_query_labels", True)
                else "generate",
            )
            == latent_query_mode
            and manifest.get("cache_pixel_dtype", "float32") == cache_pixel_dtype
        ):
            missing = 0
            for rec in dataset.records:
                if not dataset.cache_path_for_id(rec["id"]).is_file():
                    missing += 1
            if missing == 0:
                if is_main():
                    print(
                        json.dumps(
                            {
                                "preprocess_cache": "hit",
                                "dir": str(cache_dir),
                                "count": len(dataset),
                            }
                        )
                    )
                return

    tasks: list[tuple[str, list[dict[str, Any]], str]] = []
    for rec in dataset.records:
        out_path = dataset.cache_path_for_id(rec["id"])
        if not force and out_path.is_file():
            continue
        tasks.append((rec["id"], dataset.build_messages(rec), str(out_path)))

    if is_main():
        print(
            json.dumps(
                {
                    "preprocess_cache": "build",
                    "dir": str(cache_dir),
                    "total": len(dataset),
                    "to_build": len(tasks),
                    "workers": preprocess_workers,
                    "latent_token_count": latent_token_count,
                    "mask_latent_query_labels": mask_latent_query_labels,
                    "latent_query_mode": latent_query_mode,
                }
            )
        )

    manifest = {
        "count": len(dataset),
        "max_length": max_length,
        "latent_token_count": latent_token_count,
        "mask_latent_query_labels": mask_latent_query_labels,
        "latent_query_mode": latent_query_mode,
        "cache_pixel_dtype": cache_pixel_dtype,
        "dir": str(cache_dir),
    }
    if not tasks:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return

    workers = max(1, preprocess_workers)
    failures: list[str] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_cache_worker,
        initargs=(
            str(model_path),
            min_pixels,
            max_pixels,
            max_length,
            latent_token_count,
            mask_latent_query_labels,
            cache_pixel_dtype,
        ),
    ) as pool:
        futures = [pool.submit(_cache_one_sample, task) for task in tasks]
        for fut in as_completed(futures):
            record_id, ok, err = fut.result()
            if not ok:
                failures.append(f"{record_id}: {err}")

    if failures:
        raise RuntimeError(
            f"preprocess cache failed for {len(failures)} samples; first={failures[0]}"
        )

    manifest_path.write_text(json.dumps(manifest, indent=2))
    if is_main():
        print(
            json.dumps(
                {
                    "preprocess_cache": "done",
                    "dir": str(cache_dir),
                    "count": len(dataset),
                }
            )
        )
