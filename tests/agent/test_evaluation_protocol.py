import pytest

from nimloth.agent.evaluation_protocol import EarlyProtocol

COT = '<think><observation>chair</observation><reasoning>approach</reasoning><prediction>nearer</prediction></think>'
ACTION = '<|action_start|><|action_(0)|><|action_end|>'


def test_stage_protocols_and_service_text():
    for stage in ('stage1', 'stage2'):
        protocol = EarlyProtocol(stage, 2 if stage == 'stage2' else None, 'generate' if stage == 'stage2' else None)
        parsed = protocol.parse(COT + ''.join(protocol.query_tokens) + ACTION)
        assert parsed['format_correct']
        assert parsed['service_response'] == COT + '<answer>moveahead</answer>'
    assert EarlyProtocol('vagen').parse(COT + '<answer>moveahead</answer>')['format_correct']


@pytest.mark.parametrize('response', [ACTION, COT + ACTION + 'junk', COT + ACTION * 2,
    (COT + ACTION) * 2,
    COT + '<|latent_state|>' + ACTION, COT + '<answer>moveahead</answer>'])
def test_invalid_never_becomes_action(response):
    parsed = EarlyProtocol('stage1').parse(response)
    assert not parsed['format_correct']
    assert parsed['service_response'] == ''


@pytest.mark.parametrize('field', ['observation', 'reasoning', 'prediction'])
@pytest.mark.parametrize('nested', ['think', 'observation', 'reasoning', 'prediction'])
def test_cot_fields_cannot_contain_protocol_xml_boundaries(field, nested):
    crossed = COT.replace(
        f'</{field}>',
        f'<{nested}>nested</{nested}></{field}>',
        1,
    )
    assert not EarlyProtocol('stage1').parse(crossed + ACTION)['format_correct']


def test_each_cot_field_must_be_nonempty():
    contents = {
        'observation': 'chair',
        'reasoning': 'approach',
        'prediction': 'nearer',
    }
    for field, content in contents.items():
        empty = COT.replace(
            f'<{field}>{content}</{field}>',
            f'<{field}> </{field}>',
        )
        assert not EarlyProtocol('stage1').parse(empty + ACTION)['format_correct']


def test_cot_fields_cannot_hide_an_action_or_query_block():
    for marker in (
        ACTION,
        '<|action_(8)|>',
        '<|latent_state|>',
        '<answer>moveahead</answer>',
    ):
        hidden = COT.replace('chair', 'chair' + marker)
        assert not EarlyProtocol('stage1').parse(hidden + ACTION)['format_correct']


def test_query_order_and_prompt():
    protocol = EarlyProtocol('stage2', 2, 'inject')
    assert not protocol.parse(COT + ''.join(reversed(protocol.query_tokens)) + ACTION)['format_correct']
    prompt = protocol.prompt('<think>...</think><answer>moveahead</answer>')
    assert '</think><|latent_state|><|latent_state_1|><|action_start|>' in prompt
    assert '<answer>' not in prompt


def test_vagen_raw_preserved_even_invalid():
    protocol = EarlyProtocol('vagen')
    text = COT + '<answer> MoveAhead </answer>'
    assert protocol.parse(text)['service_response'] == text
    assert protocol.parse(text)['format_correct']
    assert protocol.parse('invalid')['service_response'] == 'invalid'


def test_query_prose_does_not_receive_slots():
    text = '<think>...</think><answer>moveahead</answer> inside <answer>'
    converted = EarlyProtocol('stage2', 2, 'generate').prompt(text)
    assert converted.count('<|latent_state|>') == 1
    assert 'between <|action_start|> and <|action_end|>' in converted
