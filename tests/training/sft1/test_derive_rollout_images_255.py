import importlib.util
import json
from pathlib import Path

from PIL import Image


SCRIPT = (
    Path(__file__).parents[3]
    / "experiments"
    / "training"
    / "sft1"
    / "derive_rollout_images_255.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("derive_rollout_images_255", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_derive_dataset_preserves_sources_and_rewrites_shared_images(tmp_path: Path):
    module = _load_module()
    source_records = tmp_path / "records"
    source_images = tmp_path / "rollout_images"
    output_records = tmp_path / "records_255"
    source_records.mkdir()
    source_images.mkdir()

    first = source_images / "first.png"
    second = source_images / "second.png"
    Image.new("RGB", (512, 512), color=(10, 20, 30)).save(first)
    Image.new("RGBA", (384, 512), color=(40, 50, 60, 70)).save(second)

    (source_records / "train_all.jsonl").write_text(
        json.dumps({"id": "train/1", "image_paths": [str(first), str(second)]}) + "\n",
        encoding="utf-8",
    )
    (source_records / "train_success.jsonl").write_text(
        json.dumps({"id": "train/1", "image_paths": [str(first)]}) + "\n",
        encoding="utf-8",
    )

    manifest = module.derive_dataset(source_records, output_records, workers=1)

    assert Image.open(first).size == (512, 512)
    assert Image.open(second).size == (384, 512)
    rewritten_all = _read_jsonl(output_records / "train_all.jsonl")[0]
    rewritten_success = _read_jsonl(output_records / "train_success.jsonl")[0]
    assert rewritten_all["id"] == "train/1"
    assert rewritten_all["image_paths"][0] == rewritten_success["image_paths"][0]
    assert len(set(rewritten_all["image_paths"])) == 2
    for path in rewritten_all["image_paths"]:
        image = Image.open(path)
        assert image.mode == "RGB"
        assert image.size == (255, 255)

    assert manifest["jsonl_files"] == 2
    assert manifest["records"] == 2
    assert manifest["image_references"] == 3
    assert manifest["unique_images"] == 2
    assert manifest["output_size"] == [255, 255]
    assert manifest["source_size_counts"] == {"384x512": 1, "512x512": 1}
    assert json.loads((output_records / "manifest.json").read_text()) == manifest

    resumed = module.derive_dataset(source_records, output_records, workers=1)
    assert resumed["unique_images"] == 2
    assert resumed["reused_images"] == 2
