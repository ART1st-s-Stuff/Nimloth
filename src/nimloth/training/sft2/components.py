"""Construct and place the trainable components used by SFT2."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.qwen25vl.tuning import configure_qwen_tuning, uses_lora
from nimloth.backbone.qwen25vl.vision_ema import VisionEncoderEMA
from nimloth.latent import (
    initialize_extra_latent_token_embeddings,
    install_query_embedding_adapter,
    latent_state_tokens,
)
from nimloth.model import NimlothModel
from nimloth.util.distributed import is_main
from nimloth.training.sft2.checkpoint import load_aux_checkpoint, load_lora_adapter_state
from nimloth.wm import (
    LeWMConfig,
    LatentWMPredictor,
    SIGReg,
    StateProjector,
    ValueHead,
    WorldModel,
)


SFT2_WM_HISTORY_SIZE = 1


def require_sft2_wm_history(wm_predictor: LatentWMPredictor, source: Path) -> None:
    if wm_predictor.config.history_size != SFT2_WM_HISTORY_SIZE:
        raise ValueError(
            "SFT2 one-step dynamics requires a WM checkpoint with history_size=1; "
            f"got history_size={wm_predictor.config.history_size} from {source}"
        )


@dataclass(frozen=True)
class SFT2Components:
    nimloth_model: NimlothModel
    sigreg: SIGReg
    vision_ema: VisionEncoderEMA | None
    optimizer: torch.optim.Optimizer
    base_model_path: Path
    aux_device: torch.device
    qwen_pair_parallel: bool
    ddp_static_graph: bool


def _qwen_load_kwargs(device: torch.device) -> tuple[bool, dict[str, Any]]:
    gpu_stride = int(os.environ.get("NIMLOTH_DDP_GPU_STRIDE", "1"))
    pair_parallel = gpu_stride > 1 and torch.cuda.is_available()
    if not pair_parallel:
        return False, {}
    primary_idx = int(str(device).split(":")[-1])
    pair = [primary_idx + offset for offset in range(gpu_stride)]
    if is_main():
        print(json.dumps({"qwen_pair_parallel": True, "gpu_stride": gpu_stride, "rank0_pair": pair}))
    return True, {
        "device_map": "auto",
        "max_memory": {index: "74GiB" for index in pair} | {"cpu": "64GiB"},
        "low_cpu_mem_usage": True,
    }


def _load_qwen_model(
    args,
    processor,
    token_id_map: dict[str, int],
    added_special_token_count: int,
    resume_ckpt_dir: Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, Path, Any, bool]:
    pair_parallel, load_kwargs = _qwen_load_kwargs(device)
    resume_state_path = resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir else None
    resume_adapter = resume_ckpt_dir / "adapter_config.json" if resume_ckpt_dir else None
    base_model_path = args.model

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        **load_kwargs,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.resize_token_embeddings(len(processor.tokenizer))
    if added_special_token_count > 0:
        initialize_extra_latent_token_embeddings(
            model,
            token_id_map,
            latent_token_count=args.latent_token_count,
        )
    model.config.vocab_size = len(processor.tokenizer)
    if hasattr(model, "generation_config"):
        model.generation_config.vocab_size = len(processor.tokenizer)

    if args.resume and resume_state_path is not None and resume_state_path.exists() and resume_adapter.is_file():
        saved = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        if not uses_lora(args):
            raise ValueError("--resume with LoRA adapter requires llm_tune and/or vision_tune lora")
        if saved.get("base_model_path"):
            base_model_path = Path(saved["base_model_path"])
        if is_main():
            print(
                json.dumps(
                    {
                        "resume_lora_adapter": str(resume_ckpt_dir),
                        "base_model_path": str(base_model_path),
                    }
                )
            )
        model = configure_qwen_tuning(model, args)
        load_lora_adapter_state(model, resume_ckpt_dir)
    elif (
        args.resume
        and resume_state_path is not None
        and resume_state_path.exists()
        and (resume_ckpt_dir / "config.json").is_file()
    ):
        if uses_lora(args):
            raise ValueError("cannot --resume full HF checkpoint with lora tuning")
        if is_main():
            print(json.dumps({"resume_full": str(resume_ckpt_dir)}))
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            resume_ckpt_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
            **load_kwargs,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.resize_token_embeddings(len(processor.tokenizer))
        model.config.vocab_size = len(processor.tokenizer)
        if hasattr(model, "generation_config"):
            model.generation_config.vocab_size = len(processor.tokenizer)
        model = configure_qwen_tuning(model, args)
    else:
        model = configure_qwen_tuning(model, args)
        if is_main():
            print(json.dumps({"init": "configured_tuning", "base_model_path": str(base_model_path)}))

    query_adapter = None
    if args.query_tune == "adapter":
        query_token_ids = [
            token_id_map[token] for token in latent_state_tokens(args.latent_token_count)
        ]
        query_adapter = install_query_embedding_adapter(model, query_token_ids)
        if is_main():
            print(
                json.dumps(
                    {
                        "query_tune": "adapter",
                        "query_token_ids": query_token_ids,
                        "query_lr": args.query_lr,
                    }
                )
            )

    if not pair_parallel:
        model.to(device)
    return model, base_model_path, query_adapter, pair_parallel


def build_sft2_components(
    args,
    processor,
    token_id_map: dict[str, int],
    added_special_token_count: int,
    resume_ckpt_dir: Path | None,
    *,
    device: torch.device,
    world_size: int,
    train_wm_predictor: bool,
    vision_ema_enabled: bool,
) -> SFT2Components:
    """Build Qwen, auxiliary heads, DDP wrappers, EMA, and optimizer."""

    model, base_model_path, query_adapter, pair_parallel = _load_qwen_model(
        args,
        processor,
        token_id_map,
        added_special_token_count,
        resume_ckpt_dir,
        device,
    )
    wm_config = LeWMConfig(emb_dim=args.emb_dim, history_size=SFT2_WM_HISTORY_SIZE)
    if args.wm_predictor_checkpoint is not None:
        wm_predictor = LatentWMPredictor.load_checkpoint(
            args.wm_predictor_checkpoint, map_location=device
        ).to(device)
        require_sft2_wm_history(wm_predictor, args.wm_predictor_checkpoint)
    else:
        wm_predictor = LatentWMPredictor.create(wm_config).to(device)
    if not train_wm_predictor:
        for parameter in wm_predictor.parameters():
            parameter.requires_grad = False

    aux_device = device
    if pair_parallel:
        device_map = getattr(model, "hf_device_map", {}) or {}
        mapped = device_map.get("lm_head") or device_map.get("model.language_model.norm")
        if mapped is not None:
            aux_device = torch.device(f"cuda:{mapped}")
        wm_predictor = wm_predictor.to(aux_device)

    model_dtype = next(model.parameters()).dtype
    state_proj = StateProjector(
        model.config.hidden_size,
        wm_predictor.emb_dim,
        latent_token_count=args.latent_token_count,
    ).to(device=aux_device, dtype=model_dtype)
    value_head = ValueHead(wm_predictor.emb_dim).to(device=aux_device, dtype=model_dtype)
    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_num_proj).to(device=aux_device)
    wm = WorldModel(
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
    )

    resume_state_path = resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir else None
    if args.resume and resume_state_path is not None and resume_state_path.exists():
        load_aux_checkpoint(
            resume_ckpt_dir,
            wm,
            device,
            latent_query_mode=args.latent_query_mode,
            query_tune=args.query_tune,
        )

    static_graph = world_size > 1
    if world_size > 1:
        if pair_parallel:
            model = DDP(
                model,
                device_ids=None,
                output_device=None,
                find_unused_parameters=False,
                static_graph=static_graph,
            )
        else:
            device_index = int(str(device).split(":")[-1])
            model = DDP(
                model,
                device_ids=[device_index],
                output_device=device_index,
                find_unused_parameters=False,
                static_graph=static_graph,
            )
        aux_index = int(str(aux_device).split(":")[-1])
        state_proj = DDP(
            state_proj,
            device_ids=[aux_index],
            output_device=aux_index,
            find_unused_parameters=False,
            static_graph=static_graph,
        )
        value_head = DDP(
            value_head,
            device_ids=[aux_index],
            output_device=aux_index,
            find_unused_parameters=False,
            static_graph=static_graph,
        )
        if train_wm_predictor:
            wm_predictor = DDP(
                wm_predictor,
                device_ids=[aux_index],
                output_device=aux_index,
                find_unused_parameters=False,
                static_graph=static_graph,
            )

    vision_ema: VisionEncoderEMA | None = None
    if vision_ema_enabled:
        vision_ema = VisionEncoderEMA(decay=args.vision_ema_decay)
        vision_ema.reset(model)
        ema_path = resume_ckpt_dir / "vision_ema.pt" if resume_ckpt_dir else None
        if args.resume and ema_path is not None and ema_path.is_file():
            loaded_ema = VisionEncoderEMA.load_checkpoint(ema_path, map_location=device)
            vision_ema.decay = loaded_ema.decay
            vision_ema.shadow = {key: value.to(device) for key, value in loaded_ema.shadow.items()}
        if is_main():
            print(
                json.dumps(
                    {
                        "vision_ema": True,
                        "shadow_params": len(vision_ema.shadow),
                        "decay": vision_ema.decay,
                    }
                )
            )

    query_parameter = query_adapter.delta if query_adapter is not None else None
    qwen_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter is not query_parameter
    ]
    parameter_groups: list[dict[str, Any]] = [
        {"params": qwen_parameters, "lr": args.lr_qwen_start, "name": "qwen"},
        {"params": state_proj.parameters(), "lr": args.state_proj_lr, "name": "state_proj"},
        {"params": value_head.parameters(), "lr": args.value_head_lr, "name": "value_head"},
    ]
    if query_parameter is not None:
        parameter_groups.append(
            {
                "params": [query_parameter],
                "lr": args.query_lr,
                "weight_decay": 0.0,
                "name": "query_adapter",
            }
        )
    if train_wm_predictor:
        predictor_parameters = (
            wm_predictor.module.parameters()
            if hasattr(wm_predictor, "module")
            else wm_predictor.parameters()
        )
        parameter_groups.append(
            {
                "params": list(predictor_parameters),
                "lr": args.wm_predictor_lr,
                "name": "wm_predictor",
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

    nimloth_model = NimlothModel(
        llm=model,
        wm=WorldModel(
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            value_head=value_head,
        ),
    )
    return SFT2Components(
        nimloth_model=nimloth_model,
        sigreg=sigreg,
        vision_ema=vision_ema,
        optimizer=optimizer,
        base_model_path=base_model_path,
        aux_device=aux_device,
        qwen_pair_parallel=pair_parallel,
        ddp_static_graph=static_graph,
    )
