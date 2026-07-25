"""RL 连续窗口与表征梯度模式测试。"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import torch
import pytest

from nimloth.agent import (
    Agent,
    AgentTranscript,
    NimlothPromptTemplate,
    PolicyReplayOutput,
)
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.environment.navigation import NAVIGATION_ACTION_SPACE
from nimloth.rollout import (
    RolloutTrajectory,
    count_trajectory_windows,
    sample_trajectory_windows,
)
from nimloth.training.rl.algorithm import RLAlgorithm, RLBatch, build_rl_batch
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.wm.model import WorldModel
from nimloth.wm.grid import (
    EMATargetGridEncoder,
    GridPredictorConfig,
    GridStateProjector,
    GridWorldModel,
    LeWMGridDecoder,
    LeWMGridEncoder,
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
)
from nimloth.wm.sigreg import SequenceSIGReg
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead


class _Backbone(Backbone):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.Linear(3, 3, bias=False)

    @property
    def model(self) -> torch.nn.Module:
        return self.language_model

    def forward(self, batch: BackboneBatch, **_kwargs) -> BackboneOutput:
        return BackboneOutput(self.language_model(batch.tensors["hidden"]))

    def with_model(self, model: torch.nn.Module) -> "_Backbone":
        view = _Backbone()
        view.language_model = model
        return view

    def save_pretrained(self, output_dir: Path, **_kwargs) -> None:
        raise NotImplementedError


class _MultiTokenBackbone(_Backbone):
    def forward(self, batch: BackboneBatch, **_kwargs) -> BackboneOutput:
        hidden = self.language_model(batch.tensors["hidden"])
        return BackboneOutput(torch.stack((hidden, hidden + 0.5), dim=1))


class _InputBuilder:
    processor = None

    def __init__(self) -> None:
        self.last_hidden: torch.Tensor | None = None

    def build(self, messages, images, *, include_labels: bool) -> BackboneBatch:
        del messages, include_labels
        rows = []
        for prompt_images in images:
            step = float(str(prompt_images[-1]).rsplit("_", 1)[-1].split(".")[0])
            rows.append([step, step + 1.0, step + 2.0])
        self.last_hidden = torch.tensor(rows, requires_grad=True)
        return BackboneBatch({"hidden": self.last_hidden})

    def collate_encoded(self, rows, *, include_labels: bool) -> BackboneBatch:
        raise NotImplementedError

    def cache_key(self, messages, images) -> str:
        raise NotImplementedError


class _Predictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        _action_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(state)


class _RecordingSIGReg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        self.inputs.append(states)
        return states.pow(2).mean()


class _TokenReplay(torch.nn.Module):
    def __init__(self, token_count: int) -> None:
        super().__init__()
        self.log_probs = torch.nn.Parameter(torch.zeros(token_count))
        self.token_values = torch.nn.Parameter(torch.zeros(token_count))

    def forward(self, _samples) -> PolicyReplayOutput:
        return PolicyReplayOutput(
            selected_log_probs=self.log_probs,
            entropies=torch.ones_like(self.log_probs),
            token_values=self.token_values,
        )


class _ReferenceTokenReplay(_TokenReplay):
    def __init__(self, token_count: int, reasoning_count: int) -> None:
        super().__init__(token_count)
        self.full_log_probs = torch.nn.Parameter(
            torch.full((reasoning_count,), -0.2)
        )

    def forward(self, _samples) -> PolicyReplayOutput:
        return PolicyReplayOutput(
            selected_log_probs=self.log_probs,
            entropies=torch.ones_like(self.log_probs),
            token_values=self.token_values,
            selected_full_log_probs=self.full_log_probs,
        )


def _trajectory(record_id: str, length: int) -> RolloutTrajectory:
    system_prompt = "Follow the navigation instruction."
    observation_texts = [
        f"Observation {step}.\n<image>" for step in range(length + 1)
    ]
    image_paths = [f"{record_id}_{step}.png" for step in range(length + 1)]
    action_indices = [step % 8 for step in range(length)]
    prompt = NimlothPromptTemplate(latent_token_count=1, action_count=8)
    action_token_ids = list(range(100, 108))
    reasoning_texts = [f"Reason {step}." for step in range(length)]
    assistant_responses = [
        prompt.assistant_response(action_index, thought=reasoning_texts[step])
        for step, action_index in enumerate(action_indices)
    ]
    transcript = AgentTranscript(
        system_prompt=system_prompt,
        observation_texts=tuple(observation_texts),
        observation_images=tuple(image_paths),
        action_indices=tuple(action_indices),
        assistant_responses=tuple(assistant_responses),
    )
    return RolloutTrajectory(
        record_id=record_id,
        image_paths=image_paths,
        action_indices=action_indices,
        action_names=[
            NAVIGATION_ACTION_SPACE.key_for(index) for index in action_indices
        ],
        action_log_probs=[[-math.log(8.0)] * 8 for _ in action_indices],
        instruction="test",
        reward=1.0,
        messages=prompt.build_supervised_prompt(transcript).unbound_messages(),
        system_prompt=system_prompt,
        observation_texts=observation_texts,
        assistant_responses=assistant_responses,
        terminal_assistant_prefix=prompt.assistant_prefix(
            thought="Terminal observation."
        ),
        policy_credit_assignment="turn",
        policy_messages=[
            prompt.build_response_policy_prompt(
                transcript.policy_prefix(step)
            ).unbound_messages()
            for step in range(length)
        ],
        policy_token_ids=[
            [50 + step, action_token_ids[action_index], 200]
            for step, action_index in enumerate(action_indices)
        ],
        policy_token_log_probs=[
            [-0.2, -math.log(8.0), None] for _ in action_indices
        ],
        policy_loss_masks=[[True, True, False] for _ in action_indices],
        policy_token_roles=[
            ["reasoning", "action", "injected"] for _ in action_indices
        ],
        policy_action_token_ids=[list(action_token_ids) for _ in action_indices],
        policy_reasoning_texts=reasoning_texts,
        policy_finish_reasons=["stop" for _ in action_indices],
        policy_reasoning_truncated=[False for _ in action_indices],
        prompt_template_spec=prompt.spec,
    )


def _batch() -> RLBatch:
    windows = tuple(
        sample_trajectory_windows(
            [_trajectory(f"record_{index}", 2)],
            history_size=2,
            batch_size=1,
            seed=index,
        )[0]
        for index in range(2)
    )
    return build_rl_batch(windows, gamma=0.99, device=torch.device("cpu"))


def _algorithm(
    *,
    sigreg: SequenceSIGReg | None = None,
    representation_to_backbone: bool = True,
) -> tuple[
    RLAlgorithm,
    RLModelRuntime,
    _InputBuilder,
    _Backbone,
    torch.nn.Linear,
    _Predictor,
    ValueHead,
]:
    backbone = _Backbone()
    input_builder = _InputBuilder()
    state_proj = torch.nn.Linear(3, 2, bias=False)
    predictor = _Predictor()
    value_head = ValueHead(emb_dim=2, num_actions=8, hidden_dim=2)
    agent = Agent(
        backbone=backbone,
        wm=WorldModel(
            state_proj=state_proj,
            wm_predictor=predictor,
            value_head=value_head,
        ),
    )
    return (
        RLAlgorithm(
            history_size=2,
            sigreg=sigreg,
            sigreg_weight=0.1 if sigreg is not None else 0.0,
            value_rank_margin=0.1,
            value_rank_weight=1.0,
            ppo_clip_ratio=0.2,
            entropy_weight=0.0,
        ),
        RLModelRuntime(
            agent=agent,
            input_builder=input_builder,
            representation_to_backbone=representation_to_backbone,
            policy_replay=None,
        ),
        input_builder,
        backbone,
        state_proj,
        predictor,
        value_head,
    )


def test_sequence_batch_preserves_trajectory_boundaries_and_alignment() -> None:
    trajectories = (_trajectory("short", 1), _trajectory("long", 3))

    assert count_trajectory_windows(trajectories, history_size=2) == 2
    windows = sample_trajectory_windows(
        trajectories,
        history_size=2,
        batch_size=2,
        seed=7,
    )
    batch = build_rl_batch(windows, gamma=0.5, device=torch.device("cpu"))

    assert batch.action_indices.shape == (2, 2)
    assert all(window.record_id == "long" for window in batch.windows)
    for window, replay_inputs in zip(
        batch.windows,
        (window.policy_replay_inputs() for window in batch.windows),
        strict=True,
    ):
        assert len(window.state_prompts()) == 3
        assert len(replay_inputs) == 2
        assert replay_inputs[0].action_index == window.trajectory.action_indices[
            window.start_step
        ]


def test_rl_wm_stops_gradient_on_final_target_but_trains_backbone() -> None:
    algorithm, runtime, builder, backbone, state_proj, predictor, _ = _algorithm()
    output = algorithm.training_step(runtime, _batch())

    output.losses["wm"].backward()

    assert builder.last_hidden is not None
    assert builder.last_hidden.grad is not None
    gradient = builder.last_hidden.grad.reshape(2, 3, 3)
    assert bool(gradient[:, :-1].abs().sum() > 0)
    assert torch.count_nonzero(gradient[:, -1]) == 0
    assert backbone.model.weight.grad is not None
    assert state_proj.weight.grad is not None
    assert predictor.linear.weight.grad is not None


def test_frozen_representation_mode_blocks_only_backbone_gradient() -> None:
    algorithm, runtime, _, backbone, state_proj, _, value_head = _algorithm(
        representation_to_backbone=False
    )
    output = algorithm.training_step(runtime, _batch())

    output.losses["value"].backward()

    assert backbone.model.weight.grad is None
    assert state_proj.weight.grad is not None
    assert value_head.net[0].weight.grad is not None


def test_joint_value_loss_updates_backbone_and_state_projector() -> None:
    algorithm, runtime, _, backbone, state_proj, _, value_head = _algorithm()
    output = algorithm.training_step(runtime, _batch())

    output.losses["value"].backward()

    assert backbone.model.weight.grad is not None
    assert state_proj.weight.grad is not None
    assert value_head.net[0].weight.grad is not None


def test_rl_sigreg_receives_full_history_plus_target_sequence() -> None:
    recording = _RecordingSIGReg()
    algorithm, runtime, *_ = _algorithm(
        sigreg=SequenceSIGReg(regularizer=recording),
    )

    output = algorithm.training_step(runtime, _batch())

    assert len(recording.inputs) == 1
    assert recording.inputs[0].shape == (3, 2, 2)
    assert output.losses["sigreg"] is not None


def test_rl_preserves_multiple_latent_tokens_until_state_projection() -> None:
    backbone = _MultiTokenBackbone()
    input_builder = _InputBuilder()
    agent = Agent(
        backbone=backbone,
        wm=WorldModel(
            state_proj=StateProjector(
                qwen_hidden_dim=3,
                lewm_emb_dim=2,
                projector_hidden_dim=4,
                latent_token_count=2,
            ),
            wm_predictor=_Predictor(),
            value_head=ValueHead(emb_dim=2, num_actions=8, hidden_dim=2),
        ),
    )
    algorithm = RLAlgorithm(
        history_size=2,
        sigreg=None,
        sigreg_weight=0.0,
        value_rank_margin=0.1,
        value_rank_weight=1.0,
        ppo_clip_ratio=0.2,
        entropy_weight=0.0,
    )
    runtime = RLModelRuntime(
        agent=agent,
        input_builder=input_builder,
        representation_to_backbone=True,
        policy_replay=None,
    )

    output = algorithm.training_step(runtime, _batch())

    assert output.losses["wm"] is not None
    assert output.losses["value"] is not None


def test_grid_rl_uses_ema_targets_and_mean_pooled_sigreg_without_dino_loss() -> None:
    backbone = _MultiTokenBackbone()
    input_builder = _InputBuilder()
    online_encoder = LeWMGridEncoder(emb_dim=2, hidden_dim=4)
    state_proj = GridStateProjector(
        SharedSlotProjector(
            input_dim=3,
            output_dim=2,
            hidden_dim=4,
            grid_tokens=2,
        ),
        online_encoder,
    ).requires_grad_(False).eval()
    target_encoder = EMATargetGridEncoder(online_encoder, decay=0.99)
    dino_decoder = LeWMGridDecoder(emb_dim=2, hidden_dim=4)
    world_model = GridWorldModel(
        state_proj=state_proj,
        target_encoder=target_encoder,
        wm_predictor=TemporalSpatialGridPredictor(
            GridPredictorConfig(
                grid_tokens=2,
                emb_dim=2,
                history_size=2,
                depth=1,
                heads=1,
                dim_head=2,
                mlp_dim=4,
                dropout=0.0,
            )
        ),
        dino_decoder=dino_decoder,
        value_head=ValueHead(emb_dim=2, num_actions=8, hidden_dim=2),
        train_dino_decoder=False,
        update_target_encoder=False,
    )
    agent = Agent(backbone=backbone, wm=world_model)
    recording = _RecordingSIGReg()
    algorithm = RLAlgorithm(
        history_size=2,
        sigreg=SequenceSIGReg(regularizer=recording),
        sigreg_weight=0.1,
        value_rank_margin=0.1,
        value_rank_weight=1.0,
        ppo_clip_ratio=0.2,
        entropy_weight=0.0,
    )
    runtime = RLModelRuntime(
        agent=agent,
        input_builder=input_builder,
        representation_to_backbone=True,
        policy_replay=None,
    )

    output = algorithm.training_step(runtime, _batch())
    output.loss.backward()

    assert recording.inputs[0].shape == (3, 2, 2)
    assert set(output.losses) == {
        "wm",
        "sigreg",
        "value",
        "policy",
        "token_value",
        "action_distillation",
        "reference_kl",
    }
    assert output.losses["wm"] is not None
    assert output.losses["value"] is not None
    assert output.losses["action_distillation"] is None
    assert backbone.model.weight.grad is not None
    assert any(
        parameter.grad is not None
        for parameter in world_model.wm_predictor.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in world_model.value_head.parameters()
    )
    assert all(parameter.grad is None for parameter in state_proj.parameters())
    assert all(
        parameter.grad is None for parameter in target_encoder.parameters()
    )
    assert all(
        parameter.grad is None for parameter in dino_decoder.parameters()
    )


def test_rl_algorithm_is_pure_compute_configuration() -> None:
    algorithm, *_ = _algorithm()

    assert not isinstance(algorithm, torch.nn.Module)
    assert not hasattr(algorithm, "agent")
    assert not hasattr(algorithm, "optimizer")


def test_token_credit_trains_policy_and_token_critic_separately() -> None:
    trajectory = _trajectory("token", 2)
    trajectory.policy_credit_assignment = "token"
    trajectory.rewards = [0.0, 1.0]
    trajectory.reward = 1.0
    trajectory.terminated = True
    windows = sample_trajectory_windows(
        [trajectory],
        history_size=2,
        batch_size=1,
        seed=1,
    )
    batch = build_rl_batch(
        windows,
        gamma=0.99,
        device=torch.device("cpu"),
    )
    token_replay = _TokenReplay(token_count=4)
    _, base_runtime, *_ = _algorithm()
    runtime = replace(base_runtime, policy_replay=token_replay)
    algorithm = RLAlgorithm(
        history_size=2,
        sigreg=None,
        sigreg_weight=0.0,
        value_rank_margin=0.1,
        value_rank_weight=0.0,
        ppo_clip_ratio=0.2,
        entropy_weight=0.0,
        credit_assignment="token",
        token_gamma=1.0,
        token_gae_lambda=1.0,
        token_value_loss_weight=1.0,
    )

    output = algorithm.training_step(runtime, batch)
    output.loss.backward()

    assert output.losses["token_value"] is not None
    assert token_replay.log_probs.grad is not None
    assert token_replay.token_values.grad is not None


def test_reference_kl_is_actor_loss_only_and_uses_reasoning_tokens() -> None:
    trajectory = _trajectory("token-reference", 2)
    trajectory.policy_credit_assignment = "token"
    trajectory.rewards = [0.0, 1.0]
    trajectory.reward = 1.0
    trajectory.terminated = True
    trajectory.policy_reference_token_log_probs = [
        [-0.4, None, None],
        [-0.4, None, None],
    ]
    windows = sample_trajectory_windows(
        [trajectory],
        history_size=2,
        batch_size=1,
        seed=1,
    )
    batch = build_rl_batch(windows, gamma=1.0, device=torch.device("cpu"))
    token_replay = _ReferenceTokenReplay(token_count=4, reasoning_count=2)
    _, base_runtime, *_ = _algorithm()
    runtime = replace(base_runtime, policy_replay=token_replay)
    algorithm = RLAlgorithm(
        history_size=2,
        sigreg=None,
        sigreg_weight=0.0,
        value_rank_margin=0.1,
        value_rank_weight=0.0,
        ppo_clip_ratio=0.2,
        entropy_weight=0.0,
        credit_assignment="token",
        token_gamma=1.0,
        token_gae_lambda=1.0,
        token_value_loss_weight=1.0,
        reference_kl_loss_weight=0.001,
        reference_kl_loss_type="low_var_kl",
    )

    output = algorithm.training_step(runtime, batch)
    output.loss.backward()

    expected = math.exp(-0.2) + 0.2 - 1.0
    assert output.metrics["reference_kl_loss"] == pytest.approx(expected)
    assert output.losses["reference_kl"] is not None
    assert token_replay.full_log_probs.grad is not None
