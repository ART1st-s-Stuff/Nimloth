"""Qwen + WM inference loop with slow-path / fast-path orchestration.

Slow path (every ``fast_path_interval`` steps):
    Qwen encodes the current image → StateProjector → WM latent state.
    This re-syncs the WM state with the ground-truth observation.

Fast path (steps between slow-path updates):
    WM predictor predicts next state from previous state + action.
    No Qwen forward needed — latency dominated by the lightweight
    predictor + value head.

Configurable via ``fast_path_steps``:
    - ``fast_path_steps = 0`` → slow path every step (no fast path).
    - ``fast_path_steps = N`` → Qwen every N steps, WM predictor in between.
"""

from __future__ import annotations

from typing import Any

import torch

from nimloth.wm.planning import Planner


class WMAgent:
    """Orchestrates Qwen encoding + WM planning for action selection.

    Usage::

        agent = WMAgent(
            qwen_model=qwen,
            processor=processor,
            state_proj=state_proj,
            predictor=predictor,
            value_head=value_head,
            planner_cfg={"algorithm": "greedy"},
            fast_path_steps=4,
            device=device,
        )

        for step in range(max_steps):
            action = agent.act(current_image)
            env.step(action)
    """

    def __init__(
        self,
        *,
        qwen_model: torch.nn.Module,
        processor: Any,
        token_id_map: dict[str, int],
        state_proj: torch.nn.Module,
        predictor: torch.nn.Module,
        value_head: torch.nn.Module,
        planner_cfg: dict[str, Any] | None = None,
        fast_path_steps: int = 0,
        device: torch.device | str = "cpu",
        system_prompt: str = "",
    ) -> None:
        self._qwen = qwen_model
        self._processor = processor
        self._token_id_map = token_id_map
        self._state_proj = state_proj
        self._predictor = predictor
        self._value_head = value_head
        self._fast_path_steps = fast_path_steps
        self._device = torch.device(device)
        self._system_prompt = system_prompt

        self._planner = Planner(
            planner_cfg or {"algorithm": "greedy"},
            predictor=predictor,
            value_head=value_head,
        )

        # Episode state
        self._wm_state: torch.Tensor | None = None  # (1, emb_dim)
        self._steps_since_sync: int = 0
        self._last_action: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, initial_image: Any) -> None:
        """Reset episode state and encode the initial observation via Qwen."""
        self._wm_state = self._encode_image(initial_image)  # (1, emb_dim)
        self._steps_since_sync = 0
        self._last_action = None

    def act(self, image: Any) -> int:
        """Select the next action given the current observation.

        Uses the slow path (Qwen) when ``steps_since_sync >= fast_path_steps``,
        otherwise uses the fast path (WM predictor) to advance the latent state.
        """
        if self._wm_state is None:
            raise RuntimeError("WMAgent.reset() must be called before act()")

        use_slow_path = self._fast_path_steps <= 0 or self._steps_since_sync >= self._fast_path_steps

        if use_slow_path:
            # Re-sync WM state with Qwen encoding of the real image.
            self._wm_state = self._encode_image(image)
            self._steps_since_sync = 0
        elif self._last_action is not None:
            # Fast path: predict next WM state from last (state, action).
            action_tensor = torch.tensor([self._last_action], dtype=torch.long, device=self._wm_state.device)
            self._wm_state = self._predictor.predict_next_emb(self._wm_state, action_tensor)

        # Select action from current WM state.
        action = int(self._planner.select_action(self._wm_state).item())
        self._last_action = action
        self._steps_since_sync += 1
        return action

    def act_batch(
        self,
        wm_states: torch.Tensor,
        *,
        use_planner: bool = True,
    ) -> torch.Tensor:
        """Select actions for a batch of WM states (no image encoding).

        Args:
            wm_states: (B, emb_dim) — already-projected WM latent states.
            use_planner: if True, use the planner; otherwise argmax value head.

        Returns:
            (B,) int64 — selected action indices.
        """
        if use_planner:
            return self._planner.select_action(wm_states)
        return self._value_head(wm_states.to(self._device)).argmax(dim=-1)

    @property
    def wm_state(self) -> torch.Tensor | None:
        """Current WM latent state (for external use, e.g. logging)."""
        return self._wm_state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encode_image(self, image: Any) -> torch.Tensor:
        """Run Qwen on a single image → StateProjector → WM state embedding.

        Returns:
            (1, emb_dim)
        """
        from nimloth.latent import extract_latent_state, find_last_latent_state_index, last_hidden_state
        from nimloth.latent.extraction import LatentActionTokens

        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({
            "role": "user",
            "content": [{"type": "image", "image": image}] if not isinstance(image, str) else f"<image>\n{image}",
        })

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Insert latent_state placeholder so Qwen knows where to place it.
        text = text.replace("<|im_end|>", "<|latent_state|><|im_end|>")
        inputs = self._processor(
            text=[text],
            images=[image] if not isinstance(image, str) else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._qwen(**inputs, output_hidden_states=True, return_dict=True)
        hidden = last_hidden_state(output)  # (1, seq_len, hidden_dim)
        tokens = LatentActionTokens()
        latent_idx = find_last_latent_state_index(
            inputs["input_ids"][0], self._token_id_map, tokens
        )
        latent = extract_latent_state(hidden[0:1], latent_idx)  # (1, hidden_dim)
        return self._state_proj(latent)  # (1, emb_dim)


def create_agent_from_config(
    *,
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    state_proj: torch.nn.Module,
    predictor: torch.nn.Module,
    value_head: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device | str = "cpu",
) -> WMAgent:
    """Factory: create a WMAgent from a config dict.

    Reads keys:
        ``planner``, ``fast_path_steps``, ``system_prompt``.
    """

    planner_cfg = config.get("planner", {"algorithm": "greedy"})
    fast_path_steps = config.get("fast_path_steps", 0)
    system_prompt = config.get("system_prompt", "")

    return WMAgent(
        qwen_model=qwen_model,
        processor=processor,
        token_id_map=token_id_map,
        state_proj=state_proj,
        predictor=predictor,
        value_head=value_head,
        planner_cfg=planner_cfg,
        fast_path_steps=fast_path_steps,
        device=device,
        system_prompt=system_prompt,
    )
