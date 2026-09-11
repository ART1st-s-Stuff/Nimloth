import json
from dataclasses import replace

import pytest
from PIL import Image

from nimloth.agent.evaluation_protocol import EarlyProtocol
from nimloth.backbone.qwen25vl.early_generation import RawGeneration
from nimloth.training.sft.evaluation.config import EvaluationConfig
from nimloth.training.sft.evaluation.format_gate import run_stage1_format_gate

ANSWER = (
    '<think><observation>chair</observation><reasoning>approach</reasoning>'
    '<prediction>nearer</prediction></think>'
    '<|action_start|><|action_(0)|><|action_end|>'
)


class Tokenizer:
    eos_token_id = 0
    pad_token_id = 1

    def decode(self, ids, **kwargs):
        assert kwargs == {
            'skip_special_tokens': False,
            'clean_up_tokenization_spaces': False,
        }
        return ''.join({2: ANSWER, 0: '<|im_end|>', 1: '<|pad|>', 9: 'bad'}[i] for i in ids)


class Generator:
    def __init__(self, correct):
        self.correct = correct
        self.calls = 0
        self.tokenizer = Tokenizer()

    def generate(self, messages, images):
        index = self.calls
        self.calls += 1
        assert '<|action_start|>' in messages[0]['content']
        assert len(images) == 1
        ids = (2, 0) if index < self.correct else (2,)
        return RawGeneration(ANSWER, ids, (), 'stop', None)


def make_config(tmp_path, *, output='output'):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / 'config.json').write_text('{}')
    image = tmp_path / 'image.png'
    Image.new('RGB', (2, 2), color='red').save(image)
    heldout = tmp_path / 'heldout.jsonl'
    rows = []
    for index in range(32):
        rows.append({
            'id': f'heldout-{index}',
            'source_identity': {
                'source_index': index,
                'source_key': f'base:{index}',
                'eval_set': 'base',
                'seed': index,
                'batch': 1,
                'split': 'val',
            },
            'messages': [
                {'role': 'system', 'content': '<answer>moveahead</answer>'},
                {'role': 'user', 'content': 'observation <image>'},
                {'role': 'assistant', 'content': ANSWER},
            ],
            'image_paths': [str(image)],
        })
    heldout.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return EvaluationConfig(
        mode='direct',
        stage='stage1',
        checkpoint=checkpoint,
        format_gate_jsonl=heldout,
        env_url='http://env',
        output_dir=tmp_path / output,
        eval_sets=('base',),
        split='test',
        episodes_per_eval_set=1,
        seed_offset=1,
        max_steps=1,
        temperature=0,
        top_p=1,
        max_response_tokens=128,
        tensor_parallel_size=1,
    )


@pytest.mark.parametrize(('correct', 'passed'), [(31, True), (30, False)])
def test_fixed_32_gate_threshold_and_evidence(tmp_path, correct, passed):
    config = make_config(tmp_path, output=f'output-{correct}')
    generator = Generator(correct)
    actual, returned = run_stage1_format_gate(
        config, EarlyProtocol('stage1'), generator=generator
    )
    assert actual is passed
    assert returned is generator
    assert generator.calls == 32
    summary = json.loads(
        (config.output_dir / 'format_gate/summary.json').read_text()
    )
    assert summary['correct'] == correct
    assert summary['denominator'] == 32
    record = json.loads(
        (config.output_dir / 'format_gate/records/000.json').read_text()
    )
    assert record['selection']['image_lineage'][0]['sha256']
    assert record['termination_validation']['raw_response'].endswith('<|im_end|>')
    assert record['termination_validation']['parsed_body'] == ANSWER
    assert record['generation']['finish_reason'] == 'stop'
    assert record['generation']['stop_reason'] is None


def test_gate_resume_reuses_generator_and_rejects_corrupt_record(tmp_path):
    config = make_config(tmp_path)
    generator = Generator(32)
    assert run_stage1_format_gate(
        config, EarlyProtocol('stage1'), generator=generator
    )[0]
    assert generator.calls == 32
    resumed = replace(config, resume=True)
    assert run_stage1_format_gate(
        resumed, EarlyProtocol('stage1'), generator=generator
    )[0]
    assert generator.calls == 32
    record_path = config.output_dir / 'format_gate/records/000.json'
    record = json.loads(record_path.read_text())
    record['termination_validation']['reason'] = 'tampered'
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='cannot be reproduced'):
        run_stage1_format_gate(
            resumed, EarlyProtocol('stage1'), generator=generator
        )
