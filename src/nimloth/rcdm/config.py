"""RCDM model/diffusion configuration for Nimloth SFT2 conditioning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from nimloth.rcdm.external import ensure_rcdm_importable


@dataclass(frozen=True)
class RCDMConfig:
    """Configuration accepted by ``guided_diffusion_rcdm.create_model_and_diffusion``.

    Defaults follow the 128x128 RCDM flags from upstream README, with
    ``feat_cond=True`` supplied by the Nimloth factory.
    """

    image_size: int = 128
    num_channels: int = 256
    num_res_blocks: int = 2
    num_heads: int = 4
    num_heads_upsample: int = -1
    num_head_channels: int = -1
    attention_resolutions: str = "32,16,8"
    channel_mult: str = ""
    dropout: float = 0.0
    class_cond: bool = False
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    use_fp16: bool = False
    use_new_attention_order: bool = False
    learn_sigma: bool = True
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    timestep_respacing: str = ""
    use_kl: bool = False
    predict_xstart: bool = False
    rescale_timesteps: bool = False
    rescale_learned_sigmas: bool = False
    g_shared: bool = False
    pretrained: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def rcdm_config_from_args(args: Any) -> RCDMConfig:
    """Build ``RCDMConfig`` from argparse args with matching attribute names."""

    fields = RCDMConfig.__dataclass_fields__
    values = {name: getattr(args, name) for name in fields if hasattr(args, name)}
    return RCDMConfig(**values)


def create_model_and_diffusion(config: RCDMConfig, *, cond_dim: int, rcdm_root: str | None = None):
    """Create an RCDM UNet and Gaussian diffusion conditioned on ``cond_dim``.

    ``cond_dim`` should match the SFT2 WM embedding dimension, typically 1024.
    """

    ensure_rcdm_importable(rcdm_root)
    from guided_diffusion_rcdm.script_util import create_model_and_diffusion as _create

    cfg = config.to_metadata()
    g_shared = bool(cfg.pop("g_shared"))
    pretrained = bool(cfg.pop("pretrained"))
    model, diffusion = _create(
        **cfg,
        feat_cond=True,
        G_shared=g_shared,
        ssl_dim=int(cond_dim),
        pretrained=pretrained,
    )
    return model, diffusion
