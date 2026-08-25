"""Add independently generated CFM strips to a derived rollout-browser copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence


_MARKER = "nimloth-cfm-guided-successor-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        return shutil.copy2(source, destination)
    return destination


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _find_rollout(manifest: dict[str, Any], sample_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in manifest.get("rollouts", [])
        if row.get("identity", {}).get("rollout_sample_id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one rollout for {sample_id}, got {len(matches)}")
    return matches[0]


def _inject_rollout_page(page: Path, strips: Sequence[str]) -> None:
    html = page.read_text(encoding="utf-8")
    if _MARKER in html:
        raise ValueError(f"page already contains CFM reconstruction: {page}")
    css_anchor = "</style></head>"
    script_anchor = "audit.turns.forEach((t,i)=>{"
    card_anchor = "</div></div><details open><summary>All available action evidence</summary>"
    for anchor in (css_anchor, script_anchor, card_anchor):
        if anchor not in html:
            raise ValueError(f"rollout page anchor missing: {anchor}")
    css = (
        ".cfm-reconstruction{margin:16px 0;background:#0e1629;border:1px solid var(--border);"
        "border-radius:12px;padding:12px}.cfm-reconstruction img{width:100%;border-radius:8px;"
        "border:1px solid var(--border)}.cfm-reconstruction .note{margin-top:7px}"
    )
    strip_json = json.dumps(list(strips), separators=(",", ":"))
    helper = (
        f"const nimlothCfmMarker='{_MARKER}',cfmStrips={strip_json};"
        "function cfmReconstruction(i){const src=cfmStrips[i];if(!src)return '';"
        "return `<section class=\"cfm-reconstruction\"><h3>CFM reconstruction · behavior-time "
        "guided successor</h3><img src=\"${esc(src)}\" alt=\"CFM current and guided-successor "
        "comparison\"><div class=\"note\">real current · CFM current · CFM predicted successor "
        "for executed action · real next. Matched noise; frozen pre-RL ID45 CFM.</div></section>`;}"
    )
    html = html.replace(css_anchor, css + css_anchor, 1)
    html = html.replace(script_anchor, helper + script_anchor, 1)
    html = html.replace(card_anchor, "</div></div>${cfmReconstruction(i)}<details open><summary>All available action evidence</summary>", 1)
    temporary = page.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(page)


def _mark_selector(index: Path, artifact: str, seed: int) -> None:
    html = index.read_text(encoding="utf-8")
    anchor = f'data-path="{artifact}"'
    start = html.find(anchor)
    if start < 0:
        raise ValueError(f"selector artifact missing: {artifact}")
    end = html.find("</button>", start)
    if end < 0:
        raise ValueError(f"selector button is not closed: {artifact}")
    block = html[start:end]
    if "CFM reconstructed" in block:
        raise ValueError(f"selector already marked: {artifact}")
    span_end = block.rfind("</span>")
    if span_end < 0:
        raise ValueError(f"selector span missing: {artifact}")
    block = block[:span_end] + f" · seed {seed} · CFM reconstructed" + block[span_end:]
    html = html[:start] + block + html[end:]
    temporary = index.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(index)


def create_derived_view(
    *,
    source_view: Path,
    output_view: Path,
    reconstruction_browsers: Sequence[Path],
) -> dict[str, Any]:
    """Hardlink/copy a view-only Browser and inject available CFM strips."""

    source_view = source_view.resolve()
    output_view = output_view.resolve()
    if output_view.exists():
        raise FileExistsError(f"derived output already exists: {output_view}")
    source_manifest_path = source_view / "manifest.json"
    manifest = _load_json(source_manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("source Browser is not complete")
    shutil.copytree(source_view, output_view, copy_function=_hardlink_or_copy)
    records: list[dict[str, Any]] = []
    try:
        for browser in reconstruction_browsers:
            browser = browser.resolve()
            metadata_path = browser / "metadata.json"
            metadata = _load_json(metadata_path)
            if metadata.get("status") != "completed":
                raise ValueError(f"CFM Browser is not completed: {browser}")
            if metadata.get("training_uses_rl_data") is not False:
                raise ValueError(f"CFM training-data gate failed: {browser}")
            if metadata.get("state_shape") != [16, 1024]:
                raise ValueError(f"CFM state shape is not [16,1024]: {browser}")
            sample_id = str(metadata["rollout_sample_id"])
            rollout = _find_rollout(manifest, sample_id)
            if int(metadata["turn_count"]) != int(rollout["turn_count"]):
                raise ValueError(f"turn-count mismatch for {sample_id}")
            artifact = str(rollout["artifact"])
            rollout_dir = (output_view / artifact).parent
            cfm_dir = rollout_dir / "cfm"
            cfm_dir.mkdir()
            strips: list[str] = []
            for turn_index in range(int(metadata["turn_count"])):
                name = f"turn_{turn_index:02d}_comparison.png"
                source_strip = browser / name
                if not source_strip.is_file():
                    raise FileNotFoundError(source_strip)
                destination = cfm_dir / name
                shutil.copy2(source_strip, destination)
                strips.append(f"cfm/{name}")
            _inject_rollout_page(output_view / artifact, strips)
            _mark_selector(output_view / "index.html", artifact, int(rollout["seed"]))
            records.append(
                {
                    "rollout_sample_id": sample_id,
                    "data_source": rollout["data_source"],
                    "seed": rollout["seed"],
                    "artifact": artifact,
                    "turn_count": rollout["turn_count"],
                    "cfm_metadata_sha256": _sha256(metadata_path),
                }
            )
        derived = {
            "schema": "nimloth_cfm_derived_rollout_browser_v1",
            "status": "complete",
            "source_view": str(source_view),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_rollout_count": manifest["rollout_count"],
            "reconstructed_rollout_count": len(records),
            "reconstructed_turn_count": sum(row["turn_count"] for row in records),
            "reconstructions": records,
        }
        (output_view / "cfm_derived_manifest.json").write_text(
            json.dumps(derived, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return derived
    except Exception:
        shutil.rmtree(output_view)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--output-view", type=Path, required=True)
    parser.add_argument("--reconstruction-browser", action="append", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = create_derived_view(
        source_view=args.source_view,
        output_view=args.output_view,
        reconstruction_browsers=args.reconstruction_browser,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
