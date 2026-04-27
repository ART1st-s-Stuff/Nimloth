"""参数化语义评估冒烟脚本，合并旧 test_eval_* 变体。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["minimal", "small", "full"], default="minimal")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=3)
    parser.add_argument("--ckpt-path", type=str, default="")
    parser.add_argument("--enable-vlm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode not in {"minimal", "small", "full"}:
        raise ValueError(f"unsupported mode: {args.mode}")

    import torch
    from dev._shared.semantic_align_common import build_dataset_from_cfg, load_cfg
    from src.application.pipelines.semantic.common import build_qwen_vlm_adapter, build_semantic_loader
    from src.vlm.semantic_align import DeltaProjector
    from src.vlm.semantic_state import SemanticStateGenerator

    cfg = load_cfg()
    wm_cfg = cfg.wm
    train_cfg = cfg.pipeline.train.semantic_align
    vlm_cfg = cfg.vlm
    device = torch.device(str(train_cfg.device))
    dataset = build_dataset_from_cfg(cfg=cfg, split=args.split, use_encoder=None)
    if len(dataset) == 0:
        raise RuntimeError("dataset 为空，无法执行 eval_smoke")
    if args.mode == "full":
        max_batches = len(dataset) // max(1, args.batch_size)
    elif args.mode == "small":
        max_batches = min(args.max_batches, 20)
    else:
        max_batches = min(args.max_batches, 3)
    loader = build_semantic_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = DeltaProjector(latent_dim=int(wm_cfg.latent_dim), hidden_dim=int(wm_cfg.hidden_dim)).to(device)
    if args.ckpt_path:
        ckpt_path = Path(args.ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("projector", ckpt))
    model.eval()

    if args.enable_vlm:
        vlm_cfg.enabled = True
    adapter = build_qwen_vlm_adapter(vlm_cfg=vlm_cfg, train_cfg=train_cfg, latent_dim=int(wm_cfg.latent_dim))
    semantic_generator = SemanticStateGenerator(vlm_adapter=adapter)

    print(f"[eval_smoke] mode={args.mode} dataset={len(dataset)} max_batches={max_batches}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            start = time.time()
            z_t = batch["z_t"].to(device)
            z_t_pos = batch["z_t_pos"].to(device)
            z_t_neg = batch["z_t_neg"].to(device)
            pred_pos = model(z_t=z_t, z_tp=z_t_pos)
            _ = model(z_t=z_t, z_tp=z_t_neg)
            out_t = semantic_generator.infer(
                image_path=batch["image_path"][0],
                history_image_paths=[batch["image_path"][0]],
                task_text=batch["task_text"][0],
                env_context=batch["env_context"][0],
            )
            print(
                f"[eval_smoke] batch={batch_idx} z_t={tuple(z_t.shape)} pred={tuple(pred_pos.shape)} "
                f"time={time.time() - start:.2f}s s_t={tuple(out_t.s_t.shape)}"
            )


if __name__ == "__main__":
    main()
