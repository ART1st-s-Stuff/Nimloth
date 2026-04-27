"""参数化 DataLoader 冒烟脚本，合并旧 test_dataloader* 变体。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--with-encoder", action="store_true")
    parser.add_argument("--iterate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from dev._shared.semantic_align_common import build_dataset_from_cfg, load_cfg
    from src.application.pipelines.semantic.common import build_semantic_loader
    from src.wm.encoder import build_wm_image_encoder

    cfg = load_cfg()
    encoder = build_wm_image_encoder(wm_cfg=cfg.wm) if args.with_encoder else None
    dataset = build_dataset_from_cfg(cfg=cfg, split=args.split, use_encoder=encoder)
    loader = build_semantic_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print(f"[dataloader_smoke] dataset={len(dataset)} batch_size={args.batch_size}")
    if args.iterate_only:
        iterator = iter(loader)
        start = time.time()
        batch = next(iterator)
        print(f"[dataloader_smoke] first_batch_time={time.time() - start:.2f}s shape={tuple(batch['z_t'].shape)}")
        return
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        now = time.time()
        print(f"[dataloader_smoke] batch={i} shape={tuple(batch['z_t'].shape)} elapsed={now - start:.2f}s")
        start = now


if __name__ == "__main__":
    main()
