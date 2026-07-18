"""Correctness tests for synchronized online RL rollout."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image

from nimloth.training.rl.rollout import (
    ACTION_NAMES,
    EnvRolloutCollector,
    RolloutTrajectory,
    build_nimloth_policy_messages,
    materialize_policy_images,
    multimodal_policy_messages,
    sample_action_from_logits,
    validate_rl_policy_protocol,
    validate_rollout_trajectory,
)
from nimloth.training.rl.vagen_protocol import nimloth_assistant_response


def test_trainer_replay_imports_trajectory_validator() -> None:
    from nimloth.training.rl import trainer as trainer_module

    assert trainer_module.validate_rollout_trajectory is validate_rollout_trajectory


def test_k8_inject_policy_protocol_is_supported() -> None:
    assert validate_rl_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=8,
        nimloth_latent_query_mode="inject",
    )) == 8
    with pytest.raises(ValueError, match="inject"):
        validate_rl_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=8,
            nimloth_latent_query_mode="generate",
        ))


def test_k8_pilot_uses_disjoint_fixed_heldout_protocol() -> None:
    import yaml

    config = yaml.safe_load(Path(
        "configs/training/rl/dynamic_fsdp_k8_pilot.yaml"
    ).read_text(encoding="utf-8"))
    assert config["rollout"]["eval_sets"] == ["base_train"]
    assert config["rl"]["iterations"] == 20
    assert config["rl"]["envs_per_iteration"] == 8
    assert config["rl"]["max_steps_per_episode"] == 20
    assert config["rl"]["gamma"] == 1.0
    assert config["rl"]["batch_size"] == 1
    assert config["actor"]["use_kl_loss"] is True
    assert config["actor"]["kl_loss_coef"] == 0.001
    assert config["algorithm"] == {
        "adv_estimator": "masked_gae",
        "gamma": 1.0,
        "lam": 1.0,
        "reward_placement": "final",
    }
    assert config["critic"]["backend"] == "independent_qwen"
    assert config["critic"]["micro_batch_size"] == 1
    assert config["actor"]["kl_loss_type"] == "low_var_kl"
    assert config["actor"]["use_kl_in_reward"] is False
    assert config["validation"] == {
        "enabled": True,
        "baseline": True,
        "interval": 5,
        "envs": 20,
        "max_steps_per_episode": 20,
        "eval_sets": ["base"],
        "seed_offset": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "history_window": 112,
        "max_think_tokens": 2048,
        "env_timeout": 600,
    }


def test_corrected_baseline_is_fixed_heldout_and_evaluation_only() -> None:
    import yaml

    config = yaml.safe_load(Path(
        "configs/training/rl/dynamic_fsdp_k8_baseline20.yaml"
    ).read_text(encoding="utf-8"))
    assert config["training"]["evaluation_only"] is True
    assert config["rl"]["iterations"] == 0
    assert config["validation"]["envs"] == 20
    assert config["validation"]["eval_sets"] == ["base"]
    assert config["validation"]["seed_offset"] == 1
    assert config["validation"]["temperature"] == 0.0
    assert config["validation"]["top_p"] == 1.0
    assert config["rollout"]["max_think_tokens"] == 2048
    assert config["validation"]["history_window"] == 112
    assert config["validation"]["max_think_tokens"] == 2048
    assert config["value_head"]["lambda_rank"] == 0.0


def test_k8_smoke_uses_vagen_ppo_and_flash_attention_protocol() -> None:
    import yaml

    config = yaml.safe_load(Path(
        "configs/training/rl/dynamic_fsdp_k8_smoke.yaml"
    ).read_text(encoding="utf-8"))
    assert config["rollout"]["max_think_tokens"] == 2048
    assert config["rl"]["gamma"] == 1.0
    assert config["actor"]["use_kl_loss"] is True
    assert config["actor"]["kl_loss_coef"] == 0.001
    launcher = Path(
        "experiments/training/rl/run_dynamic_fsdp_rank.sh"
    ).read_text(encoding="utf-8")
    assert "--attn-implementation flash_attention_2" in launcher
    assert "--attn-implementation sdpa" not in launcher


def test_current_fragment_launcher_has_world8_and_dedicated_preflight_env() -> None:
    launcher = Path(
        "experiments/training/rl/"
        "dynamic_fsdp_k8_fragmented_3plus2plus2plus1_env1.slurm"
    ).read_text(encoding="utf-8")
    assert launcher.count("#SBATCH --partition=normal") == 5
    assert launcher.count("#SBATCH --exclude=dgx-[18,32,52,54]") == 5
    assert "#SBATCH --nodelist=" not in launcher
    assert launcher.count("#SBATCH hetjob") == 4
    assert launcher.count("#SBATCH --ntasks=3") == 1
    assert launcher.count("#SBATCH --ntasks=2") == 2
    assert launcher.count("#SBATCH --ntasks=1") == 2
    for offset in (0, 3, 5, 7):
        assert f"RANK_OFFSET={offset}" in launcher
    assert '"${SRUN}" --het-group=4' in launcher
    assert "preflight_dynamic_env.py" in launcher
    assert "WORLD_SIZE=8" in Path(
        "experiments/training/rl/run_dynamic_fsdp_rank.sh"
    ).read_text(encoding="utf-8")


def test_fixed_6plus2_launcher_uses_current_normal_resources() -> None:
    launcher = Path(
        "experiments/training/rl/dynamic_fsdp_k8_fragmented_6plus2_env1.slurm"
    ).read_text(encoding="utf-8")
    assert launcher.count("#SBATCH hetjob") == 2
    for node in ("dgx-09", "dgx-37", "dgx-13"):
        assert f"#SBATCH --nodelist={node}" in launcher
    assert "#SBATCH --ntasks=6" in launcher
    assert "#SBATCH --ntasks=2" in launcher
    assert "RANK_OFFSET=0" in launcher
    assert "RANK_OFFSET=6" in launcher
    assert '"${SRUN}" --het-group=2' in launcher
    assert "preflight_dynamic_env.py" in launcher


def test_fixed_5plus3_launcher_uses_preflight_proven_env48() -> None:
    launcher = Path(
        "experiments/training/rl/dynamic_fsdp_k8_fragmented_5plus3_env48.slurm"
    ).read_text(encoding="utf-8")
    assert launcher.count("#SBATCH hetjob") == 2
    assert "#SBATCH --nodelist=dgx-09" not in launcher
    assert "#SBATCH --nodelist=dgx-27" not in launcher
    assert "#SBATCH --nodelist=dgx-48" in launcher
    assert launcher.count("#SBATCH --exclude=dgx-[18,32,52,54]") == 2
    assert launcher.count("#SBATCH --partition=normal") == 2
    assert launcher.count("#SBATCH --partition=preempt") == 1
    assert "#SBATCH --ntasks=5" in launcher
    assert "#SBATCH --ntasks=3" in launcher
    assert "RANK_OFFSET=0" in launcher
    assert "RANK_OFFSET=5" in launcher
    assert '"${SRUN}" --het-group=2' in launcher
    assert "preflight_dynamic_env.py" in launcher


def test_dynamic_4211_launcher_has_no_stale_trainer_node_rules() -> None:
    launcher = Path(
        "experiments/training/rl/"
        "dynamic_fsdp_k8_fragmented_4plus2plus1plus1_env48.slurm"
    ).read_text(encoding="utf-8")
    assert launcher.count("#SBATCH hetjob") == 4
    assert launcher.count("#SBATCH --partition=normal") == 4
    assert launcher.count("#SBATCH --partition=preempt") == 1
    assert launcher.count("#SBATCH --nodelist=") == 3
    assert "#SBATCH --nodelist=dgx-09" in launcher
    assert "#SBATCH --nodelist=dgx-27" in launcher
    assert "#SBATCH --nodelist=dgx-48" in launcher
    assert "#SBATCH --exclude=" not in launcher
    for tasks, count in ((4, 1), (2, 1), (1, 3)):
        assert launcher.count(f"#SBATCH --ntasks={tasks}") == count
    for offset in (0, 1, 2, 4):
        assert f"RANK_OFFSET={offset}" in launcher
    assert '"${SRUN}" --het-group=4' in launcher
    assert "preflight_dynamic_env.py" in launcher
    assert '"${RUN_MODE}" == baseline' in launcher
    assert '"${RUN_MODE}" == pilot' in launcher
    assert 'if mode == "pilot":' in launcher
    assert "deterministic-train-gradient-checkpointing-selected-logits-v1" in launcher
    assert 'expected_microbatch = 1 if mode == "pilot" else 2' in launcher


def test_actor_memory_probe_requires_real_20turn_backward_and_headroom() -> None:
    probe = Path(
        "experiments/training/rl/probe_actor_recompute_memory.py"
    ).read_text(encoding="utf-8")
    launcher = Path(
        "experiments/training/rl/actor_memory_probe_1plus1plus2plus4.slurm"
    ).read_text(encoding="utf-8")
    assert "trajectory.num_steps < 20" in probe
    assert "_temporary_deterministic_train" in probe
    assert "loss.backward()" in probe
    assert "torch.cuda.max_memory_reserved" in probe
    assert launcher.count("#SBATCH hetjob") == 3
    assert 'assert peak < 70.0' in launcher
    assert 'assert all(row["history_images"]==20' in launcher
    assert 'assert all(row["policy_tokens"]==9' in launcher


def test_placeholder_hold_runs_atomically_published_stages() -> None:
    hold = Path(
        "experiments/training/rl/hold_dynamic_fsdp_k8_1124_env48.slurm"
    ).read_text(encoding="utf-8")
    assert hold.count("#SBATCH hetjob") == 4
    assert hold.count("#SBATCH --partition=normal") == 4
    assert hold.count("#SBATCH --partition=preempt") == 1
    assert "#SBATCH --nodelist=dgx-09" in hold
    assert "#SBATCH --nodelist=dgx-27" in hold
    assert "#SBATCH --nodelist=dgx-48" in hold
    assert "#SBATCH --exclude=" not in hold
    assert 'touch "${HOLD_ROOT}/READY"' in hold
    assert '"${HOLD_ROOT}/next_stage.sh"' in hold
    assert 'touch "${stage_dir}/PASSED"' in hold
    assert 'touch "${stage_dir}/FAILED"' in hold


def test_k8_snapshot_is_immutable_and_omits_sft_optimizer(tmp_path: Path) -> None:
    from experiments.training.rl.prepare_k8_sft2_init import (
        ROOT_FILES,
        TREE_FILES,
        snapshot_checkpoint,
    )

    source = tmp_path / "latest"
    source.mkdir()
    for relative_name in (*ROOT_FILES, *TREE_FILES):
        path = source / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"content:{relative_name}".encode())
    torch.save({
        "step": 123,
        "epoch": 2,
        "latent_token_count": 8,
        "latent_query_mode": "inject",
        "base_model_path": "/base",
        "epoch_complete": True,
        "optimizer": {"huge": torch.ones(3)},
    }, source / "training_state.pt")

    output = tmp_path / "snapshot"
    manifest = snapshot_checkpoint(
        source, output, require_epoch_complete=True
    )

    assert manifest["source_step"] == 123
    assert manifest["source_epoch_complete"] is True
    assert manifest["required_epoch_complete"] is True
    assert manifest["stable_during_copy"] is True
    assert (output / "SNAPSHOT_READY").is_file()
    state = torch.load(output / "training_state.pt", weights_only=False)
    assert state["latent_token_count"] == 8
    assert "optimizer" not in state
    with pytest.raises(FileExistsError):
        snapshot_checkpoint(source, output)


def test_k8_snapshot_rejects_partial_epoch_when_complete_is_required(
    tmp_path: Path,
) -> None:
    from experiments.training.rl.prepare_k8_sft2_init import snapshot_checkpoint

    source = tmp_path / "partial"
    source.mkdir()
    torch.save({
        "step": 4799,
        "epoch": 3,
        "epoch_complete": False,
        "latent_token_count": 8,
        "latent_query_mode": "inject",
    }, source / "training_state.pt")
    with pytest.raises(ValueError, match="not a complete epoch"):
        snapshot_checkpoint(
            source,
            tmp_path / "output",
            require_epoch_complete=True,
        )


def test_llm_lora_does_not_train_suffix_matched_visual_adapters() -> None:
    from nimloth.backbone.qwen_tuning import _set_lora_trainability

    language = torch.nn.Parameter(torch.ones(1))
    visual = torch.nn.Parameter(torch.ones(1))

    class FakePeftModel:
        def named_parameters(self):
            return iter((
                ("base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight", language),
                ("base_model.model.model.visual.blocks.0.mlp.gate_proj.lora_A.default.weight", visual),
            ))

    _set_lora_trainability(
        FakePeftModel(), llm_tune="lora", vision_tune="freeze"
    )
    assert language.requires_grad is True
    assert visual.requires_grad is False

    _set_lora_trainability(
        FakePeftModel(), llm_tune="lora", vision_tune="lora"
    )
    assert language.requires_grad is True
    assert visual.requires_grad is True


def test_mixed_full_and_lora_tuning_is_rejected_before_training() -> None:
    from nimloth.training.rl.trainer import validate_policy_tune_combination

    validate_policy_tune_combination(llm_tune="lora", vision_tune="freeze")
    validate_policy_tune_combination(llm_tune="lora", vision_tune="lora")
    validate_policy_tune_combination(llm_tune="full", vision_tune="freeze")
    with pytest.raises(ValueError, match="mixed full/LoRA"):
        validate_policy_tune_combination(llm_tune="lora", vision_tune="full")
    with pytest.raises(ValueError, match="mixed full/LoRA"):
        validate_policy_tune_combination(llm_tune="full", vision_tune="lora")


def test_policy_dtype_normalization_makes_fsdp_parameters_uniform() -> None:
    from nimloth.training.rl.trainer import normalize_policy_parameter_dtype

    module = torch.nn.Module()
    module.base = torch.nn.Linear(3, 3).to(dtype=torch.bfloat16)
    module.register_parameter(
        "adapter", torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
    )
    normalize_policy_parameter_dtype(module, dtype=torch.bfloat16)
    assert {parameter.dtype for parameter in module.parameters()} == {torch.bfloat16}


def test_zero_update_run_refuses_final_checkpoint() -> None:
    from nimloth.training.rl.trainer import _require_optimizer_progress

    with pytest.raises(RuntimeError, match="zero optimizer steps"):
        _require_optimizer_progress(0)
    _require_optimizer_progress(1)


def test_policy_prompt_uses_real_windowed_transcript_and_images() -> None:
    responses = [
        nimloth_assistant_response(
            "<think>first</think>", 0, latent_token_count=8
        ),
        nimloth_assistant_response(
            "<think>second</think>", 5, latent_token_count=8
        ),
    ]
    messages, images = build_nimloth_policy_messages(
        ["obs0.png", "obs1.png", "obs2.png"],
        "system",
        ["<image>\ntask", "<image>\nfeedback1", "<image>\nfeedback2"],
        responses,
        history_window=1,
    )
    assert images == ["obs1.png", "obs2.png"]
    assert messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "<image>\nfeedback1"},
        {"role": "assistant", "content": responses[1]},
        {"role": "user", "content": "<image>\nfeedback2"},
    ]


def test_policy_image_paths_are_materialized_as_rgb(tmp_path: Path) -> None:
    path = tmp_path / "observation.png"
    Image.new("RGBA", (3, 2), (10, 20, 30, 40)).save(path)

    result = materialize_policy_images([str(path)])

    assert len(result) == 1
    assert isinstance(result[0], Image.Image)
    assert result[0].mode == "RGB"
    assert result[0].size == (3, 2)
    assert result[0].getpixel((0, 0)) == (10, 20, 30)
    # The returned copy must remain readable after the source file is closed.
    result[0].load()


def test_policy_prompt_replaces_image_placeholders_with_sft_multimodal_parts() -> None:
    image = Image.new("RGB", (2, 2), "black")
    messages, images = multimodal_policy_messages(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "before<image>after"},
        ],
        [image],
    )
    assert messages[0] == {"role": "system", "content": "system"}
    content = messages[1]["content"]
    assert content[0] == {"type": "text", "text": "before"}
    assert content[1]["type"] == "image"
    assert isinstance(content[1]["image"], Image.Image)
    assert content[1]["image"] is images[0]
    assert content[2] == {"type": "text", "text": "after"}
    assert len(images) == 1

    with pytest.raises(ValueError, match="image placeholder count"):
        multimodal_policy_messages(
            [{"role": "user", "content": "<image>"}], []
        )


def test_policy_prompt_rejects_misaligned_history() -> None:
    with pytest.raises(ValueError, match="one image per observation"):
        build_nimloth_policy_messages(
            ["obs0.png"],
            "system",
            ["<image>\ntask", "<image>\nnext"],
            [nimloth_assistant_response(
                "<think>move</think>", 0, latent_token_count=8
            )],
            history_window=112,
        )


def test_inject_runtime_generates_thought_before_inserting_query_block() -> None:
    from nimloth.latent.extraction import all_special_tokens_for_latent_count
    from nimloth.training.rl.rollout import (
        generate_nimloth_thought_and_action_logits_from_messages,
    )

    class FakeTokenizer:
        unk_token_id = -1

        def __init__(self):
            special = all_special_tokens_for_latent_count(latent_token_count=8)
            self.ids = {token: 20 + index for index, token in enumerate(special)}

        def convert_tokens_to_ids(self, token):
            return self.ids.get(token, -1)

        def encode(self, text, add_special_tokens=False):
            if text in self.ids:
                return [self.ids[text]]
            if text == "</think>":
                # Context-independent encoding intentionally differs from the
                # sampled IDs: real BPE can merge preceding punctuation with
                # the start of the closing tag.
                return [99]
            raise AssertionError(text)

        def decode(self, ids, skip_special_tokens=False):
            return {
                (10,): "<think>generated",
                (10, 11): "<think>generated from policy.</",
                (10, 11, 12): "<think>generated from policy.</think>",
            }[tuple(ids)]

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, **kwargs):
            content = messages[-1]["content"]
            assert content[0]["type"] == "image"
            assert isinstance(content[0]["image"], Image.Image)
            assert content[1] == {"type": "text", "text": "\nreal task"}
            return "rendered-training-prefix"

        def __call__(self, **kwargs):
            assert kwargs["text"] == ["rendered-training-prefix"]
            assert len(kwargs["images"]) == 1
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
            }

    class FakeModel(torch.nn.Module):
        def __init__(self, tokenizer):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.tokenizer = tokenizer

        def forward(self, input_ids, **kwargs):
            logits = torch.full((1, input_ids.shape[1], 128), -100.0)
            last = int(input_ids[0, -1])
            next_id = {1: 10, 10: 11, 11: 12}.get(last)
            if next_id is not None:
                logits[0, -1, next_id] = 0.0
            else:
                action_start = self.tokenizer.ids["<|action_start|>"]
                assert last == action_start
                for index in range(8):
                    token = self.tokenizer.ids[f"<|action_({index})|>"]
                    logits[0, -1, token] = float(index)
            return SimpleNamespace(logits=logits)

    processor = FakeProcessor()
    model = FakeModel(processor.tokenizer)
    thought, action_logits = generate_nimloth_thought_and_action_logits_from_messages(
        model,
        processor,
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "<image>\nreal task"},
        ],
        [Image.new("RGB", (2, 2), "black")],
        latent_token_count=8,
        max_think_tokens=8,
        token_selector=lambda logits, _: int(logits.argmax().item()),
    )
    assert thought == "<think>generated from policy.</think>"
    assert action_logits.tolist() == list(map(float, range(8)))

    from nimloth.training.rl.rollout import _generate_nimloth_thought_from_inputs

    traced, traced_action_logits = _generate_nimloth_thought_from_inputs(
        model,
        processor,
        {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])},
        latent_token_count=8,
        max_think_tokens=8,
        token_selector=lambda logits, _: int(logits.argmax().item()),
        log_prob_temperature=1.0,
    )
    assert traced.text == thought
    assert traced.token_ids == [10, 11, 12]
    assert len(traced.token_log_probs) == 3
    assert all(torch.isfinite(torch.tensor(traced.token_log_probs)))
    assert torch.equal(traced_action_logits, action_logits)


def test_unterminated_thought_error_keeps_bounded_generated_evidence() -> None:
    from nimloth.training.rl.rollout import _generate_nimloth_thought_from_inputs

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert text == "</think>"
            return [99]

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(f"tok{token}" for token in ids)

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    class FakeModel(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], 8)))

    with pytest.raises(RuntimeError) as exc_info:
        _generate_nimloth_thought_from_inputs(
            FakeModel(),
            FakeProcessor(),
            {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])},
            latent_token_count=8,
            max_think_tokens=4,
            token_selector=lambda _logits, token_index: token_index + 2,
        )
    message = str(exc_info.value)
    assert "generated_token_count=4" in message
    assert "generated_token_prefix=[2, 3, 4, 5]" in message
    assert "generated_text_prefix='tok2 tok3 tok4 tok5'" in message


def test_ppo_replays_every_stochastic_response_token_verbatim() -> None:
    from nimloth.latent.extraction import LatentActionTokens, latent_state_tokens
    from nimloth.training.rl.trainer import compute_policy_token_stats_for_batch

    tokens = LatentActionTokens()
    token_id_map = {
        tokens.action_start: 30,
        **{
            token: 20 + index
            for index, token in enumerate(latent_state_tokens(8, tokens))
        },
        **{token: 40 + index for index, token in enumerate(tokens.action_tokens)},
    }

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            assert token_ids == [10, 11, 12]
            return "<think>actual generated thought</think>"

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0] == {"role": "system", "content": "real system"}
            content = messages[1]["content"]
            assert content[0]["type"] == "image"
            assert isinstance(content[0]["image"], Image.Image)
            assert content[1] == {"type": "text", "text": "\nreal task"}
            assert kwargs["add_generation_prompt"] is True
            return "TRAINING_TEMPLATE_PREFIX"

        def __call__(self, **kwargs):
            assert kwargs["text"] == ["TRAINING_TEMPLATE_PREFIX"]
            assert len(kwargs["images"]) == 1
            return {
                "input_ids": torch.tensor([[1]]),
                "attention_mask": torch.tensor([[1]]),
            }

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, **kwargs):
            expected = [1, 10, 11, 12, *range(20, 28), 30]
            assert input_ids[0].tolist() == expected
            assert kwargs["logits_to_keep"].tolist() == [0, 1, 2, 12]
            logits = torch.zeros((1, 4, 64)) + self.anchor
            logits[0, 0, 10] += 3.0
            logits[0, 1, 11] += 4.0
            logits[0, 2, 12] += 5.0
            for index in range(8):
                logits[0, -1, 40 + index] += float(index)
            return SimpleNamespace(logits=logits)

    model = FakeModel()
    log_probs, entropies, token_counts = compute_policy_token_stats_for_batch(
        [{
            "image_history_paths": [Image.new("RGB", (2, 2), "black")],
            "system_prompt": "real system",
            "observation_texts": ["<image>\nreal task"],
            "assistant_responses": [],
            "current_thought": "<think>actual generated thought</think>",
            "thought_token_ids": [10, 11, 12],
            "taken_action_idx": 7,
        }],
        model,
        FakeProcessor(),
        token_id_map,
        torch.device("cpu"),
        history_window=112,
        temperature=1.0,
        latent_token_count=8,
    )
    assert token_counts == [4]
    assert log_probs.shape == (4,)
    assert entropies.shape == (4,)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropies).all()
    (-log_probs.mean()).backward()
    assert model.anchor.grad is not None


def test_deterministic_train_enables_checkpointing_without_dropout() -> None:
    from nimloth.training.rl.trainer import _temporary_deterministic_train

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.25)

        def forward(self, value):
            assert self.training is True
            assert self.dropout.training is True
            assert self.dropout.p == 0.0
            return self.dropout(value)

    model = FakeModel().eval()
    with _temporary_deterministic_train(model):
        output = model(torch.ones(4))
        assert torch.equal(output, torch.ones(4))
    assert model.training is False
    assert model.dropout.training is False
    assert model.dropout.p == 0.25


def test_reference_context_disables_and_restores_lora_without_grad_flags() -> None:
    from nimloth.training.rl.trainer import disable_lora_adapters_for_reference

    class FakeLoraLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(1, 1)
            self._disable_adapters = False

    model = torch.nn.Module()
    model.lora = FakeLoraLayer()
    parameter_flags = [parameter.requires_grad for parameter in model.parameters()]
    with disable_lora_adapters_for_reference(model):
        assert model.lora._disable_adapters is True
        assert [parameter.requires_grad for parameter in model.parameters()] == parameter_flags
    assert model.lora._disable_adapters is False


def test_sampling_is_deterministic_and_temperature_scaled() -> None:
    logits = torch.arange(8, dtype=torch.float32)
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)
    action_a, log_probs_a = sample_action_from_logits(
        logits, temperature=0.5, top_p=0.9, generator=generator_a
    )
    action_b, log_probs_b = sample_action_from_logits(
        logits, temperature=0.5, top_p=0.9, generator=generator_b
    )
    assert action_a == action_b
    assert log_probs_a == log_probs_b
    assert torch.allclose(
        torch.tensor(log_probs_a), torch.log_softmax(logits / 0.5, dim=-1)
    )


def test_trajectory_validation_rejects_nonfinite_log_probs() -> None:
    trajectory = RolloutTrajectory(
        record_id="bad",
        image_paths=["before.png", "after.png"],
        observation_texts=[
            "<image>\nHuman Instruction: Move.\nDecide your next action(s).",
            "<image>\nfeedback",
        ],
        task_instruction="Move.",
        system_prompt="system",
        assistant_responses=[nimloth_assistant_response(
            "<think>move</think>", 0, latent_token_count=8
        )],
        action_indices=[0],
        action_names=["move_forward"],
        action_log_probs=[[float("nan")] * 8],
        thought_token_ids=[[10, 11, 12]],
        thought_token_log_probs=[[-0.2, -0.3, -0.1]],
        step_rewards=[0.01],
        reward=0.01,
        latent_token_count=8,
        split="train",
    )
    with pytest.raises(ValueError, match="non-finite"):
        validate_rollout_trajectory(trajectory)


def test_distributed_wrapper_preserves_env_timeout() -> None:
    from nimloth.training.rl.distributed_rollout import DistributedEnvRolloutCollector

    collector = EnvRolloutCollector(
        None,
        None,
        "http://env",
        None,
        eval_sets=("base_train",),
        split="train",
        env_timeout=37,
    )
    wrapped = DistributedEnvRolloutCollector.from_collector(collector)
    assert wrapped._env_timeout == 37


def test_collector_can_reset_fixed_heldout_seed_cursor() -> None:
    collector = EnvRolloutCollector(
        None,
        None,
        "http://env",
        None,
        seed_offset=1,
        eval_sets=("base",),
        split="validation",
    )
    collector._ep_counter = 99
    collector.reset_seed_cursor()
    assert collector._ep_counter == 1


def test_resume_seed_cursor_accounts_for_completed_rollouts() -> None:
    collector = EnvRolloutCollector(
        None,
        None,
        "http://env",
        None,
        seed_offset=100,
        eval_sets=("base_train",),
        split="train",
    )
    collector.set_resume_iteration(
        start_iteration=6,
        envs_per_iteration=8,
        validation_enabled=True,
        validation_interval=2,
        validation_envs=3,
    )
    assert collector._ep_counter == 100 + 5 * 8 + 2 * 3


def test_k8_hidden_encoding_extracts_full_query_block(tmp_path: Path) -> None:
    from nimloth.latent.extraction import latent_state_tokens
    from nimloth.training.rl.trainer import encode_trajectory_hiddens

    latent_names = latent_state_tokens(8)
    token_id_map = {name: 101 + index for index, name in enumerate(latent_names)}
    input_ids = torch.tensor([[0, *[token_id_map[name] for name in latent_names], 9]])
    hidden = torch.arange(input_ids.numel() * 2, dtype=torch.float32).reshape(1, -1, 2)

    class FakeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            assert all(isinstance(image, Image.Image) for image in kwargs["images"])
            return {"input_ids": input_ids}

    class FakeModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(hidden_states=(hidden,))

    image_paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in image_paths:
        Image.new("RGB", (2, 2), "black").save(path)
    trajectory = RolloutTrajectory(
        record_id="k8",
        image_paths=[str(path) for path in image_paths],
        observation_texts=[
            "<image>\nHuman Instruction: Move.\nDecide your next action(s).",
            "<image>\nfeedback",
        ],
        task_instruction="Move.",
        system_prompt="Navigate.",
        assistant_responses=[nimloth_assistant_response(
            "<think>move</think>", 0, latent_token_count=8
        )],
        action_indices=[0],
        action_names=["move_forward"],
        action_log_probs=[[float(-torch.log(torch.tensor(8.0)))] * 8],
        thought_token_ids=[[10, 11, 12]],
        thought_token_log_probs=[[-0.2, -0.3, -0.1]],
        step_rewards=[0.01],
        reward=0.01,
        latent_token_count=8,
        split="train",
    )
    states = encode_trajectory_hiddens(
        trajectory,
        FakeModel(),
        FakeProcessor(),
        token_id_map,
        torch.device("cpu"),
        latent_token_count=8,
    )
    assert len(states) == 1
    assert states[0].shape == (8, 2)
    assert torch.equal(states[0], hidden[0, 1:9])


def test_state_projector_loader_infers_k8_checkpoint_width(tmp_path: Path) -> None:
    from nimloth.training.rl.cli import load_state_projector_for_rl
    from nimloth.wm.state_proj import StateProjector

    expected = StateProjector(
        qwen_hidden_dim=3,
        lewm_emb_dim=4,
        projector_hidden_dim=5,
        latent_token_count=8,
    ).to(dtype=torch.bfloat16)
    checkpoint = tmp_path / "state_proj.pt"
    torch.save(expected.state_dict(), checkpoint)

    loaded = load_state_projector_for_rl(
        checkpoint,
        qwen_hidden_dim=3,
        lewm_emb_dim=4,
        latent_token_count=8,
    )
    assert loaded.input_dim == 24
    assert loaded.latent_token_count == 8
    assert {parameter.dtype for parameter in loaded.parameters()} == {torch.bfloat16}
    assert all(
        torch.equal(expected.state_dict()[key], loaded.state_dict()[key])
        for key in expected.state_dict()
    )
    with pytest.raises(ValueError, match="input dim"):
        load_state_projector_for_rl(
            checkpoint,
            qwen_hidden_dim=3,
            lewm_emb_dim=4,
            latent_token_count=1,
        )


def test_value_head_loader_honors_checkpoint_hidden_width(tmp_path: Path) -> None:
    from nimloth.training.rl.cli import load_value_head_for_rl
    from nimloth.wm.value_head import ValueHead

    checkpoint = tmp_path / "value"
    expected = ValueHead(emb_dim=12, hidden_dim=5)
    expected.save_checkpoint(checkpoint)
    loaded = load_value_head_for_rl(checkpoint, emb_dim=12)
    assert loaded.net[0].weight.shape == (5, 12)
    assert all(
        torch.equal(expected.state_dict()[key], loaded.state_dict()[key])
        for key in expected.state_dict()
    )


def test_masked_gae_assigns_token_specific_vagen_advantages() -> None:
    from nimloth.training.rl.loss import compute_masked_gae_advantage_return

    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    values = torch.tensor([[0.2, 0.4, 0.1, 99.0]])
    loss_mask = torch.tensor([[1, 1, 1, 0]])
    advantages, returns = compute_masked_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        loss_mask=loss_mask,
        gamma=1.0,
        lam=1.0,
    )
    expected_raw = torch.tensor([0.8, 0.6, 0.9])
    expected = (expected_raw - expected_raw.mean()) / expected_raw.std(unbiased=False)
    assert torch.allclose(advantages[0, :3], expected)
    assert advantages[0, 3].item() == 0.0
    assert torch.allclose(returns[0, :3], torch.ones(3))
    assert returns[0, 3].item() == 0.0


def test_clipped_token_value_loss_matches_vagen_maximum() -> None:
    from nimloth.training.rl.loss import compute_clipped_token_value_loss

    loss, metrics = compute_clipped_token_value_loss(
        predicted_values=torch.tensor([2.0, 0.5]),
        old_values=torch.tensor([0.0, 0.0]),
        returns=torch.tensor([1.0, 1.0]),
        cliprange_value=0.5,
    )
    # First token uses the larger clipped error (0.5-1)^2 vs (2-1)^2 =>1.
    # Second token has equal unclipped/clipped error0.25.
    assert torch.isclose(loss, torch.tensor(0.3125))
    assert metrics["value_clip_fraction"] == 0.0


def test_turn_advantages_are_whitened_with_response_token_weighting() -> None:
    from nimloth.training.rl.trainer import (
        normalize_turn_advantages_for_policy_tokens,
    )

    normalized = normalize_turn_advantages_for_policy_tokens(
        torch.tensor([1.0, -1.0]), [3, 1]
    )
    flattened = torch.cat((normalized[0].expand(3), normalized[1].expand(1)))
    assert torch.isclose(flattened.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(
        flattened.std(unbiased=False), torch.tensor(1.0), atol=1e-6
    )


def test_transition_microbatches_consume_all_collected_data_once() -> None:
    from nimloth.training.rl.trainer import deterministic_transition_microbatches

    batches = deterministic_transition_microbatches(160, 2, seed=47)
    flattened = [index for batch in batches for index in batch]
    assert len(batches) == 80
    assert len(flattened) == 160
    assert sorted(flattened) == list(range(160))
    assert len(set(flattened)) == 160


def test_validation_summary_requires_all_fixed_episodes() -> None:
    from nimloth.training.rl.trainer import summarize_validation_trajectories

    trajectories = [
        RolloutTrajectory(record_id="v1", success=True, reward=10.0, action_indices=[0]),
        RolloutTrajectory(record_id="v2", success=False, reward=0.0, action_indices=[0, 1]),
    ]
    metrics = summarize_validation_trajectories(trajectories, expected_episodes=2)
    assert metrics == {
        "val_success_rate": 0.5,
        "val_avg_reward": 5.0,
        "val_avg_steps": 1.5,
        "val_num_episodes": 2.0,
    }
    with pytest.raises(RuntimeError, match="expected 3"):
        summarize_validation_trajectories(trajectories, expected_episodes=3)


class _FakeEnvClient:
    def __init__(self, call_log: Path) -> None:
        self.call_log = call_log

    def _record(self, operation: str) -> None:
        with self.call_log.open("a", encoding="utf-8") as stream:
            stream.write(operation + "\n")

    def create_environments_batch(self, configs) -> None:
        self._record("create")

    def get_system_prompts_batch(self, env_ids):
        self._record("prompt")
        return {
            env_ids[0]: (
                "You can optionally think first, then give your action. Respond in this format:\n"
                "<think>...</think><action>some_action</action>"
            )
        }

    @staticmethod
    def _observation(image, text):
        return {
            "obs_str": text,
            "multi_modal_data": {"<image>": [image]},
        }

    def reset_batch(self, seeds):
        self._record("reset")
        env_id = next(iter(seeds))
        return {env_id: (self._observation(
            Image.new("RGB", (4, 4), "black"),
            "<image>\nHuman Instruction: Move near the couch.\n"
            "Decide your next action(s).",
        ), {})}

    def step_batch(self, actions):
        self._record("step")
        env_id = next(iter(actions))
        assert "<think>generated</think>" in actions[env_id]
        assert "<action>look_down</action>" in actions[env_id]
        return {
            env_id: (
                self._observation(
                    Image.new("RGB", (4, 4), "white"),
                    "After your action, the extracted valid action is look_down.\n"
                    "The environment feedback is: Last action is executed successfully.\n"
                    "After that, the observation is:\n<image>\n"
                    "Decide your next action(s).",
                ),
                1.01,
                True,
                {
                    "last_action_success": True,
                    "task_success": True,
                    "instruction": "Move near the couch.",
                },
            )
        }

    def compute_reward_batch(self, env_ids):
        self._record("reward")
        return {env_ids[0]: 0.0}

    def close_batch(self, env_ids) -> None:
        self._record("close")


class _ForbiddenEnvClient:
    def __getattr__(self, name):
        raise AssertionError(f"nonzero rank accessed env client method {name}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_collect_worker(rank: int, world: int, port: int, root: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    try:
        import nimloth.training.rl.distributed_rollout as distributed_module
        from nimloth.training.rl.distributed_rollout import DistributedEnvRolloutCollector
        from nimloth.training.rl.rollout import GeneratedThought

        def fake_policy_turn(*args, **kwargs):
            return GeneratedThought(
                text="<think>generated</think>",
                token_ids=[10, 11, 12],
                token_log_probs=[-0.2, -0.3, -0.1],
            ), torch.tensor([-100.0] * 7 + [0.0], dtype=torch.float32)

        distributed_module.generate_nimloth_thought_and_action_logits_with_trace = (
            fake_policy_turn
        )
        collector = DistributedEnvRolloutCollector(
            object(),
            object(),
            "http://env",
            torch.device("cpu"),
            seed_offset=7,
            temperature=0.7,
            top_p=0.95,
            eval_sets=("base_train",),
            split="train",
            history_window=2,
        )
        call_log = Path(root) / "env_calls.txt"
        collector._client = _FakeEnvClient(call_log) if rank == 0 else _ForbiddenEnvClient()
        trajectories = collector.collect(
            num_episodes=1,
            max_steps_per_episode=1,
            output_dir=Path(root) / "rollout",
        )
        result_path = Path(root) / f"rank_{rank}.json"
        result_path.write_text(
            json.dumps([trajectory.to_record() for trajectory in trajectories], sort_keys=True),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_distributed_collector_rank0_env_and_identical_trajectories(tmp_path: Path) -> None:
    port = _free_port()
    mp.spawn(
        _distributed_collect_worker,
        args=(2, port, str(tmp_path)),
        nprocs=2,
        join=True,
    )
    rank0 = (tmp_path / "rank_0.json").read_text(encoding="utf-8")
    rank1 = (tmp_path / "rank_1.json").read_text(encoding="utf-8")
    assert rank0 == rank1
    records = json.loads(rank0)
    assert len(records) == 1
    assert len(records[0]["image_paths"]) == 2
    assert len(records[0]["action_log_probs"][0]) == len(ACTION_NAMES)
    assert records[0]["thought_token_ids"] == [[10, 11, 12]]
    assert records[0]["thought_token_log_probs"] == [[-0.2, -0.3, -0.1]]
    assert "Human Instruction:" in records[0]["observation_texts"][0]
    assert records[0]["step_rewards"] == [1.01]
    assert records[0]["success"] is True
    calls = (tmp_path / "env_calls.txt").read_text(encoding="utf-8").splitlines()
    assert calls == ["create", "prompt", "reset", "step", "reward", "close"]
    assert (tmp_path / "rollout" / "trajectories.jsonl").is_file()
