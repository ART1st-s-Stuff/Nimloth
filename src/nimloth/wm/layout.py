"""Explicit spatial/global state-token layout contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class GridStateLayout:
    """Describe a row-major spatial grid followed by optional global slots."""

    schema: str = "nimloth_grid_state_layout_v2"
    spatial_grid_size: int = 4
    global_tokens: int = 0
    global_role: str = "none"

    def __post_init__(self) -> None:
        if self.schema != "nimloth_grid_state_layout_v2":
            raise ValueError(f"unsupported grid state layout schema: {self.schema}")
        if self.spatial_grid_size < 1:
            raise ValueError("spatial_grid_size must be positive")
        if self.global_tokens not in (0, 1):
            raise ValueError("grid state supports zero or one global token")
        expected_role = "dino_cls" if self.global_tokens else "none"
        if self.global_role != expected_role:
            raise ValueError(
                f"global_role must be {expected_role!r} for global_tokens={self.global_tokens}"
            )

    @property
    def spatial_tokens(self) -> int:
        return self.spatial_grid_size**2

    @property
    def state_tokens(self) -> int:
        return self.spatial_tokens + self.global_tokens

    @property
    def has_global(self) -> bool:
        return self.global_tokens == 1

    def validate(self, value: torch.Tensor, *, name: str = "state") -> None:
        if value.ndim < 2 or value.shape[-2] != self.state_tokens:
            raise ValueError(
                f"{name} must have {self.state_tokens} ordered state tokens "
                f"({self.spatial_tokens} spatial + {self.global_tokens} global), "
                f"got {tuple(value.shape)}"
            )

    def spatial(self, value: torch.Tensor) -> torch.Tensor:
        self.validate(value)
        return value[..., : self.spatial_tokens, :]

    def global_state(self, value: torch.Tensor) -> torch.Tensor:
        self.validate(value)
        if not self.has_global:
            raise ValueError("state layout has no global token")
        return value[..., self.spatial_tokens, :]

    def assert_spatial_identity(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
        name: str = "observed spatial state",
    ) -> float:
        """Fail closed when a layout extension changes any existing spatial slot.

        The CLS-alignment Stage2 uses this as its epoch16 regression gate.  The
        candidate may contain the new global token, while the reference may be
        either the old spatial-only K64 tensor or a tensor with the same K65
        layout.  No implicit pooling, padding, or token reordering is allowed.
        """

        self.validate(candidate, name=f"candidate {name}")
        if reference.ndim < 2 or reference.shape[-1] != candidate.shape[-1]:
            raise ValueError(
                f"reference {name} must share the candidate feature dimension, "
                f"got {tuple(reference.shape)} versus {tuple(candidate.shape)}"
            )
        if reference.shape[-2] == self.spatial_tokens:
            reference_spatial = reference
        elif reference.shape[-2] == self.state_tokens:
            reference_spatial = self.spatial(reference)
        else:
            raise ValueError(
                f"reference {name} must contain K{self.spatial_tokens} spatial "
                f"tokens or the full K{self.state_tokens} layout, got "
                f"{tuple(reference.shape)}"
            )
        candidate_spatial = self.spatial(candidate)
        if reference_spatial.shape != candidate_spatial.shape:
            raise ValueError(
                f"{name} shapes differ: {tuple(reference_spatial.shape)} versus "
                f"{tuple(candidate_spatial.shape)}"
            )
        if not torch.isfinite(reference_spatial).all() or not torch.isfinite(
            candidate_spatial
        ).all():
            raise ValueError(f"{name} contains non-finite values")
        difference = (reference_spatial.float() - candidate_spatial.float()).abs()
        max_abs = float(difference.max().item()) if difference.numel() else 0.0
        if not torch.allclose(
            reference_spatial.float(),
            candidate_spatial.float(),
            atol=float(atol),
            rtol=float(rtol),
        ):
            raise ValueError(
                f"{name} identity gate failed: max_abs={max_abs:.9g}, "
                f"atol={atol:.9g}, rtol={rtol:.9g}"
            )
        return max_abs

    def metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            spatial_tokens=self.spatial_tokens,
            state_tokens=self.state_tokens,
            ordering="row_major_spatial_then_global",
        )
        return payload

    @classmethod
    def from_metadata(cls, payload: dict[str, object]) -> GridStateLayout:
        layout = cls(
            schema=str(payload.get("schema", "")),
            spatial_grid_size=int(payload.get("spatial_grid_size", 0)),
            global_tokens=int(payload.get("global_tokens", -1)),
            global_role=str(payload.get("global_role", "")),
        )
        claimed = {
            "spatial_tokens": layout.spatial_tokens,
            "state_tokens": layout.state_tokens,
            "ordering": "row_major_spatial_then_global",
        }
        mismatches = {
            key: (payload.get(key), expected)
            for key, expected in claimed.items()
            if payload.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"grid state layout metadata mismatch: {mismatches}")
        return layout


__all__ = ["GridStateLayout"]
