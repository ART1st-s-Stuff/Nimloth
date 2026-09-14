import copy
import json

import pytest

from nimloth.latent import latent_state_block
from nimloth.rollout.tail_drop import canonical_sha256, convert_jsonl, convert_sft_view, feedback
from nimloth.rollout.transitions import expand_record_transitions


def _source(n=3):
    source_messages = [{"role": "system", "content": "old prompt"}]
    messages = [{"role": "system", "content": "token prompt"}]
    names = ["moveahead", "moveback", "moveright"]
    observation = "<image> initial"
    for i in range(n):
        source_messages.extend([
            {"role": "user", "content": observation},
            {"role": "assistant", "content": f"<think>real thought {i}</think><answer>{names[i]}</answer>"},
        ])
        messages.extend([
            {"role": "user", "content": observation},
            {"role": "assistant", "content": f"<think>real thought {i}</think>{latent_state_block(16)}<|action_start|><|action_({i})|><|action_end|>"},
        ])
        observation = f"<image> step {i+1}\nreward: {4 if i == n-1 else 0}\ndone: {1 if i == n-1 else 0}\nLast action is executed successfully."
    source_messages.extend([{"role": "user", "content": observation}, {"role": "assistant", "content": ""}])
    return {
        "id": "test", "split": "train", "success": True, "reward": 4,
        "messages": messages, "image_paths": [f"{i}.png" for i in range(n)],
        "action_indices": list(range(n)),
        "source_audit": {"source_messages": source_messages, "source_record_sha256": "a" * 64},
    }


def test_convert_preserves_final_reward_thought_and_query_grid():
    row = convert_sft_view(_source(), latent_token_count=64, max_action_horizon=20, gamma=0.5)
    assert row["action_value_targets"] == [1.0, 2.0]
    assert row["rewards"] == [0.0, 0.0]
    assert row["success"] is True
    assert row["terminal_assistant_prefix"] == f"<think>real thought 2</think>{latent_state_block(64)}<|action_start|>"
    assert row["assistant_responses"][0].count("<|latent_state_63|>") == 1
    assert len(expand_record_transitions(row, value_gamma=0.5)) == 2
    assert not row["conversion_provenance"]["historical_raw_source_hash_verified"]


def test_one_action_has_no_retained_transition():
    assert convert_sft_view(_source(1), latent_token_count=64, max_action_horizon=20) is None


@pytest.mark.parametrize("mutation", ["cot", "action", "feedback", "reward", "unfinished", "query"])
def test_reject_inconsistent_source(mutation):
    row = _source()
    if mutation == "cot":
        row["messages"][2]["content"] = row["messages"][2]["content"].replace("real thought", "changed")
    elif mutation == "action":
        row["source_audit"]["source_messages"][2]["content"] = "<think>real thought 0</think><answer>moveback</answer>"
    elif mutation == "feedback":
        row["messages"][3]["content"] = row["messages"][3]["content"].replace("reward: 0", "reward: 1")
    elif mutation == "reward":
        row["reward"] = 0
    elif mutation == "unfinished":
        row["source_audit"]["source_messages"][-2]["content"] = row["source_audit"]["source_messages"][-2]["content"].replace("done: 1", "done: 0")
    elif mutation == "query":
        row["messages"][2]["content"] = row["messages"][2]["content"].replace("<|latent_state_1|>", "")
    with pytest.raises(ValueError):
        convert_sft_view(row, latent_token_count=64, max_action_horizon=20)


def test_immutable_cli_output_and_rejected_evidence(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(_source()) + "\n")
    output = tmp_path / "converted"
    manifest = convert_jsonl(source, output, latent_token_count=64, max_action_horizon=20)
    assert manifest["retained_transitions"] == 2
    assert manifest["shorter_than_t4"] == 1
    assert (output / "COMMITTED").is_file()
    with pytest.raises(FileExistsError):
        convert_jsonl(source, output, latent_token_count=64, max_action_horizon=20)
    bad = _source()
    bad["reward"] = 999
    source.write_text(json.dumps(bad) + "\n")
    failed = tmp_path / "failed"
    with pytest.raises(ValueError):
        convert_jsonl(source, failed, latent_token_count=64, max_action_horizon=20)
    assert not (failed / "COMMITTED").exists()
    assert json.loads((failed / "rejected.json").read_text())


@pytest.mark.parametrize("flag,expected", [("True", True), ("False", False), ("1.0", True), ("0.0", False)])
def test_exact_vagen_boolean_or_numeric_done_feedback(flag, expected):
    result = feedback(f"reward: 0.02\ndone: {flag}\nLast action is not executed successfully.")
    assert result == (0.02, expected, False)


@pytest.mark.parametrize("flag", ["true", "false", "2", "nan", "yes"])
def test_reject_noncanonical_done_feedback(flag):
    with pytest.raises(ValueError):
        feedback(f"reward: 0\ndone: {flag}\nLast action is executed successfully.")


def test_single_action_omission_still_validates_observation_image_binding():
    row = _source(1)
    row["messages"][1]["content"] = "initial without image"
    with pytest.raises(ValueError, match="current image"):
        convert_sft_view(row, latent_token_count=64, max_action_horizon=20)



def test_action_outcome_is_next_observation_and_source_hash_scope_is_explicit():
    source = _source()
    for messages in (source["messages"], source["source_audit"]["source_messages"]):
        messages[5]["content"] = messages[5]["content"].replace(
            "Last action is executed successfully.",
            "Last action is not executed successfully.",
        )
    before = copy.deepcopy(source)
    result = convert_sft_view(source, latent_token_count=64, max_action_horizon=20)
    assert source == before
    assert result["action_successes"] == [True, False]
    samples = expand_record_transitions(result)
    assert [(sample.action_index, sample.action_success) for sample in samples] == [(0, True), (1, False)]
    provenance = result["conversion_provenance"]
    assert provenance["source_record_sha256"] == canonical_sha256(source)
    assert provenance["source_record_sha256"] != "a" * 64
    assert provenance["historical_raw_source_hash_claim"] == "a" * 64
    assert provenance["historical_raw_source_hash_verified"] is False


@pytest.mark.parametrize("messages", ["serialized transcript", [None], [{"role": "system", "content": [None]}]])
def test_malformed_messages_produce_rejected_evidence(tmp_path, messages):
    row = _source()
    row["messages"] = messages
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps(row) + "\n")
    output = tmp_path / "result"
    with pytest.raises(ValueError):
        convert_jsonl(source, output, latent_token_count=64, max_action_horizon=20)
    assert not (output / "COMMITTED").exists()
    assert json.loads((output / "rejected.json").read_text())
