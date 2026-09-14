import hashlib
import json

import pytest
from PIL import Image

from experiments.training.sft.stage3.prepare_eval_records import (
    prepare_manifest, raw_record_to_stage3,
)
from experiments.training.sft.stage3.prompt_conversion import NAMES, TOKENS, rewrite_text
from experiments.training.sft1.vagen_step60_data import convert_source_prompt
from nimloth.rollout.transcript import parse_im_messages


def _write_raw(tmp_path):
    names = ["moveahead", "moveback", "moveright"]
    observations = ["<image> initial"] + [
        f"<image> state {i}\nreward: {4 if i == 3 else 0}\ndone: {i == 3}\nLast action is executed successfully."
        for i in range(1, 4)
    ]
    responses = [f"<think><observation>view{i}</observation><reasoning>reason{i}</reasoning><prediction>next{i}</prediction></think><answer>{name}</answer>" for i, name in enumerate(names)]
    prompt = "Respond <think>...</think><answer>moveahead</answer>. The action must be exactly one of these lowercase action names: " + ", ".join(NAMES) + "."
    messages = [{"role": "system", "content": prompt}]
    history = []
    for i in range(4):
        image_path = tmp_path / f"{i}.png"
        Image.new("RGB", (4, 4), color=(i, 0, 0)).save(image_path)
        image = {"path": image_path.name, "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(), "mode": "RGB", "size": [4, 4]}
        info = {} if i == 0 else {"actions": [names[i-1]], "llm_raw_response": responses[i-1], "llm_response": responses[i-1], "task_success": i == 3, "last_action_success": True}
        history.append({"obs_str": observations[i], "info": info, "reward": 4 if i == 3 else 0, "done": i == 3, "image_data": [{"image_file": image}]})
        messages.append({"role": "user", "content": observations[i]})
        if i < 3:
            messages.append({"role": "assistant", "content": responses[i]})
    output = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"
    raw = {"format": "raw_test", "source_key": "base/test", "source_index": 1, "seed": 2, "eval_set": "base", "env_config": {}, "recording": {"history": history, "output_str": output}}
    path = tmp_path / "record.json"
    path.write_text(json.dumps(raw))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "records_manifest.json"
    manifest.write_text(json.dumps({"count": 1, "records": [{"path": path.name, "sha256": digest}]}))
    return raw, path, digest, manifest


def test_eval_manifest_adapter_uses_real_images_rewards_and_b_prompt(tmp_path):
    _, _, _, source = _write_raw(tmp_path)
    output = tmp_path / "converted"
    manifest = prepare_manifest(source, output, latent_token_count=64, max_action_horizon=20)
    row = json.loads((output / "data.jsonl").read_text())
    assert row["action_value_targets"] == [4.0, 4.0]
    assert row["action_successes"] == [True, True]
    assert row["conversion_provenance"]["historical_raw_source_hash_verified"]
    assert "lowercase action names" not in row["system_prompt"]
    assert "Nimloth action tokens:" in row["system_prompt"]
    assert "<|latent_state_63|>" in row["terminal_assistant_prefix"]
    assert manifest["record_count"] == 1
    assert len(manifest["images"]) == 4
    assert (output / "COMMITTED").exists()
    with pytest.raises(FileExistsError):
        prepare_manifest(source, output, latent_token_count=64, max_action_horizon=20)


@pytest.mark.parametrize("mutation", ["reward", "action", "outcome", "response", "image"])
def test_eval_adapter_rejects_history_transcript_or_image_drift(tmp_path, mutation):
    raw, path, digest, source = _write_raw(tmp_path)
    if mutation == "image":
        (tmp_path / "1.png").write_bytes(b"changed")
    else:
        step = raw["recording"]["history"][1]
        if mutation == "reward":
            step["reward"] = 99
        elif mutation == "action":
            step["info"]["actions"] = ["moveleft"]
        elif mutation == "outcome":
            step["info"]["last_action_success"] = False
        elif mutation == "response":
            step["info"]["llm_response"] = "different"
        path.write_text(json.dumps(raw))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source.write_text(json.dumps({"count": 1, "records": [{"path": path.name, "sha256": digest}]}))
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        prepare_manifest(source, output, latent_token_count=64, max_action_horizon=20)
    assert not (output / "COMMITTED").exists()


def test_raw_helper_cannot_claim_unverified_hash(tmp_path):
    raw, path, _, _ = _write_raw(tmp_path)
    with pytest.raises(ValueError, match="manifest-pinned"):
        raw_record_to_stage3(raw, observation_image_paths=[str(tmp_path/f"{i}.png") for i in range(4)],
                            raw_path=path, raw_sha256="a"*64, latent_token_count=64,
                            max_action_horizon=20, gamma=1)


def test_b_prompt_rule_and_parser_preserve_text():
    prompt = "The answer must be exactly one of these lowercase action names: " + ", ".join(NAMES) + "."
    assert rewrite_text(prompt) == "The answer must be exactly one of these action tokens: " + ", ".join(TOKENS) + "."
    assert rewrite_text(convert_source_prompt("<answer>moveahead</answer>", latent_token_count=64)).startswith("<|latent_state|>")
    messages = parse_im_messages("<|im_start|>system\nKeep exactly.\n<|im_end|>\n<|im_start|>assistant\n")
    assert messages == [{"role": "system", "content": "Keep exactly.\n"}, {"role": "assistant", "content": ""}]


def test_check_only_produces_no_outputs(tmp_path):
    _, _, _, source = _write_raw(tmp_path)
    output = tmp_path / "never_created"
    report = prepare_manifest(source, output, latent_token_count=64,
                              max_action_horizon=20, check_only=True)
    assert report["record_count"] == 1
    assert report["retained_transitions"] == 2
    assert not output.exists()
