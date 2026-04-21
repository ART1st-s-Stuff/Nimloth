"""磁盘空间管理命令。"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.utils.run_output import refresh_run_metadata

VALID_TARGETS = ("models", "datasets", "outputs")
BASE_REF_FILE = "base_weight_ref.json"


@dataclass
class RunRecord:
    target: str
    group_dir: Path
    run_dir: Path
    run_name: str
    created_at: datetime


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _iter_group_dirs(target: str) -> list[Path]:
    root = Path(target)
    if not root.exists():
        return []
    if target == "datasets":
        return [p for p in root.iterdir() if p.is_dir()]
    if target == "models":
        groups: list[Path] = []
        for model_type in root.iterdir():
            if not model_type.is_dir():
                continue
            for cfg_name in model_type.iterdir():
                if cfg_name.is_dir():
                    groups.append(cfg_name)
        return groups
    # outputs 兼容旧目录，按两级目录扫描
    groups = []
    for level1 in root.iterdir():
        if not level1.is_dir():
            continue
        for level2 in level1.iterdir():
            if level2.is_dir():
                groups.append(level2)
    return groups


def _list_runs(group_dir: Path, target: str) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for child in group_dir.iterdir():
        if not child.is_dir():
            continue
        stat = child.stat()
        runs.append(
            RunRecord(
                target=target,
                group_dir=group_dir,
                run_dir=child,
                run_name=child.name,
                created_at=datetime.fromtimestamp(stat.st_mtime),
            )
        )
    return sorted(runs, key=lambda item: item.created_at)


def _collect_all_runs(targets: list[str]) -> dict[Path, list[RunRecord]]:
    result: dict[Path, list[RunRecord]] = {}
    for target in targets:
        for group in _iter_group_dirs(target):
            runs = _list_runs(group, target)
            if runs:
                result[group] = runs
    return result


def _build_vlm_base_refs(model_root: Path = Path("models")) -> set[Path]:
    refs: set[Path] = set()
    vlm_root = model_root / "vlm"
    if not vlm_root.exists():
        return refs
    for cfg_dir in vlm_root.iterdir():
        if not cfg_dir.is_dir():
            continue
        for run_dir in cfg_dir.iterdir():
            if not run_dir.is_dir():
                continue
            ref_path = run_dir / BASE_REF_FILE
            if not ref_path.exists():
                continue
            try:
                payload = json.loads(ref_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            base_path = payload.get("base_weight_path")
            if isinstance(base_path, str):
                refs.add(Path(base_path).resolve())
    return refs


def _is_protected(run_dir: Path, base_refs: set[Path]) -> bool:
    if not base_refs:
        return False
    for file in run_dir.iterdir():
        if file.is_file() and file.resolve() in base_refs:
            return True
    return False


def _guard_not_delete_all(runs: list[RunRecord], deletions: set[Path], group: Path) -> None:
    remain = [r for r in runs if r.run_dir not in deletions]
    if not remain:
        raise RuntimeError(f"{group} 将被全部删除，已拒绝执行。")


def _plan_trim(
    runs: list[RunRecord],
    before: datetime | None,
    keep_last: int,
    base_refs: set[Path],
) -> set[Path]:
    if keep_last < 1:
        raise ValueError("--keep-last 必须 >= 1")
    sorted_runs = sorted(runs, key=lambda r: r.created_at)
    protected_latest = set(r.run_dir for r in sorted_runs[-keep_last:])
    deletions: set[Path] = set()
    for run in sorted_runs:
        if run.run_dir in protected_latest:
            continue
        if before and run.created_at >= before:
            continue
        if _is_protected(run.run_dir, base_refs):
            continue
        deletions.add(run.run_dir)
    return deletions


def _plan_discard(
    runs: list[RunRecord],
    after: datetime | None,
    num: int,
    base_refs: set[Path],
) -> set[Path]:
    if num < 1:
        raise ValueError("--num 必须 >= 1")
    candidates = sorted(runs, key=lambda r: r.created_at, reverse=True)
    picked: list[RunRecord] = []
    for run in candidates:
        if after and run.created_at <= after:
            continue
        if _is_protected(run.run_dir, base_refs):
            continue
        picked.append(run)
        if len(picked) >= num:
            break
    return {r.run_dir for r in picked}


def _apply_deletions(deletions: set[Path], dry_run: bool = False) -> None:
    for path in sorted(deletions):
        if dry_run:
            print(f"[dry-run] remove {path}")
            continue
        shutil.rmtree(path)
        print(f"[removed] {path}")


def _update_metadata_for_groups(groups: set[Path], dry_run: bool = False) -> None:
    for group in groups:
        if dry_run:
            print(f"[dry-run] refresh metadata {group}")
            continue
        refresh_run_metadata(group)


def _resolve_targets(raw_targets: list[str]) -> list[str]:
    if not raw_targets:
        return list(VALID_TARGETS)
    for target in raw_targets:
        if target not in VALID_TARGETS:
            raise ValueError(f"非法 target: {target}")
    return raw_targets


def _confirm_reset(targets: list[str], force: bool) -> None:
    if force:
        return
    joined = ", ".join(targets)
    answer = input(f"将删除 {joined} 下所有数据。输入 YES 确认: ").strip()
    if answer != "YES":
        raise RuntimeError("reset 已取消。")


def _run_trim(args: argparse.Namespace) -> None:
    targets = _resolve_targets(args.targets)
    runs_by_group = _collect_all_runs(targets)
    base_refs = _build_vlm_base_refs()
    deletions: set[Path] = set()
    affected_groups: set[Path] = set()
    before = _parse_time(args.before)
    for group, runs in runs_by_group.items():
        plan = _plan_trim(runs, before=before, keep_last=args.keep_last, base_refs=base_refs)
        _guard_not_delete_all(runs, plan, group)
        if plan:
            deletions.update(plan)
            affected_groups.add(group)
    _apply_deletions(deletions, dry_run=args.dry_run)
    _update_metadata_for_groups(affected_groups, dry_run=args.dry_run)


def _run_discard(args: argparse.Namespace) -> None:
    targets = _resolve_targets(args.targets)
    runs_by_group = _collect_all_runs(targets)
    base_refs = _build_vlm_base_refs()
    deletions: set[Path] = set()
    affected_groups: set[Path] = set()
    after = _parse_time(args.after)
    for group, runs in runs_by_group.items():
        plan = _plan_discard(runs, after=after, num=args.num, base_refs=base_refs)
        _guard_not_delete_all(runs, plan, group)
        if plan:
            deletions.update(plan)
            affected_groups.add(group)
    _apply_deletions(deletions, dry_run=args.dry_run)
    _update_metadata_for_groups(affected_groups, dry_run=args.dry_run)


def _run_reset(args: argparse.Namespace) -> None:
    targets = _resolve_targets(args.targets)
    _confirm_reset(targets, force=args.yes_i_know_what_i_am_doing)
    for target in targets:
        root = Path(target)
        if not root.exists():
            continue
        for child in root.iterdir():
            if args.dry_run:
                print(f"[dry-run] remove {child}")
            elif child.is_dir():
                shutil.rmtree(child)
                print(f"[removed] {child}")
            else:
                child.unlink(missing_ok=True)
                print(f"[removed] {child}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="storage", description="磁盘空间管理工具")
    sub = parser.add_subparsers(dest="command", required=True)

    trim = sub.add_parser("trim", help="清理旧数据，默认只保留最后一个")
    trim.add_argument("targets", nargs="*", help="models/datasets/outputs，可省略")
    trim.add_argument("--before", type=str, default=None, help="仅删除早于该时间的运行")
    trim.add_argument("--keep-last", type=int, default=1, help="每组保留最近 N 个")
    trim.add_argument("--dry-run", action="store_true")
    trim.set_defaults(handler=_run_trim)

    discard = sub.add_parser("discard", help="删除最近若干数据")
    discard.add_argument("targets", nargs="*", help="models/datasets/outputs，可省略")
    discard.add_argument("--after", type=str, default=None, help="仅删除晚于该时间的运行")
    discard.add_argument("--num", type=int, default=1, help="每组删除最近 N 个")
    discard.add_argument("--dry-run", action="store_true")
    discard.set_defaults(handler=_run_discard)

    reset = sub.add_parser("reset", help="删除全部数据（需确认）")
    reset.add_argument("targets", nargs="*", help="models/datasets/outputs，可省略")
    reset.add_argument("--yes-i-know-what-i-am-doing", action="store_true")
    reset.add_argument("--dry-run", action="store_true")
    reset.set_defaults(handler=_run_reset)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

