"""Synthetic CPU validation mechanics; no model-quality evidence."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from experiments.training.sft1 import original_validation200 as audit
from experiments.training.sft1 import prepare_source200 as pilot
from experiments.training.sft1 import vagen_step60_data as data
from tests.training.sft1.test_vagen_step60_data import _source_rows


@pytest.fixture
def dump():
    path = Path(os.environ.get('VAGEN_ORIGINAL_DUMP_PATH', str(
        Path(__file__).resolve().parents[3] /
        'external/VAGEN/.worktree-step60-original-validation/vagen/utils/original_validation_dump.py')))
    spec = importlib.util.spec_from_file_location('original_validation_dump', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dump_validation_batch


def fixtures():
    identities, rows = [], []
    for category in ('base', 'common_sense'):
        for ordinal in range(100):
            identity = {'source_index': len(rows) * 3, 'seed': ordinal * 7 + 42,
                        'eval_set': category, 'source_key': f'{category}:{ordinal * 7 + 42}'}
            identities.append(identity)
            info = {key: value for key, value in identity.items() if key != 'eval_set'}
            info.update({'dataset_split': 'train', 'env_name': 'navigation',
                         'env_config': {**audit.ENV_CONTRACT, 'eval_set': category,
                                        'prompt_format': 'step60_source_reconstruction', 'success_reward': 10.0}})
            rows.append({'prompt': [{'role': 'user', 'content': 'actual prompt'}], 'extra_info': info})
    return rows, identities


def recording(success=True):
    image = Image.new('RGB', (3, 2), (12, 34, 56))
    return {'env_id': 'val1', 'config_id': 'navigation', 'output_str': 'verbatim response',
            'metrics': {'success': success, 'score': 10.02}, 'image_data': [image],
            'history': [{'image_data': [image], 'info': {'llm_raw_response': '<answer>moveahead</answer>'}}]}


def test_conversion_preserves_metadata_order_and_input():
    rows, identities = fixtures()
    before = copy.deepcopy(rows)
    converted = audit.convert_rows(rows, identities)
    assert rows == before
    for original, result in zip(rows, converted, strict=True):
        expected = copy.deepcopy(original)
        expected['extra_info']['env_config']['prompt_format'] = 'grounding_worldmodeling'
        del expected['extra_info']['env_config']['success_reward']
        assert result == expected
    with pytest.raises(ValueError, match='order mismatch'):
        audit.convert_rows(rows[::-1], identities)
    with pytest.raises(ValueError, match='exactly 200'):
        audit.convert_rows(rows[:-1], identities)
    identities[-1] = identities[0]
    with pytest.raises(ValueError, match='duplicate'):
        audit.convert_rows(rows, identities)


def test_real_preparation_verifier_and_hashes(tmp_path, monkeypatch):
    source = tmp_path / 'source.parquet'
    pq.write_table(pa.Table.from_pylist(_source_rows()), source)
    digest = pilot.sha256(source)
    monkeypatch.setattr(data, 'SOURCE_TRAIN_SHA256', digest)
    partition = tmp_path / 'partition'
    data.partition_source_parquet(source, partition, expected_sha256=digest)
    prepared = tmp_path / 'source200'
    pilot.prepare(partition / 'partition_manifest.json', prepared,
                  prompt_format='step60_source_reconstruction')
    source_manifest = prepared / 'prepared_manifest.json'
    original = source_manifest.read_bytes()
    output = tmp_path / 'original'
    manifest = audit.prepare(source_manifest, output)
    assert source_manifest.read_bytes() == original
    assert len(pq.read_table(output / manifest['parquet'])) == 200
    assert manifest['parquet'] == manifest['train_parquet']
    assert len(list(output.glob('*.parquet'))) == 1
    with pytest.raises(FileExistsError):
        audit.prepare(source_manifest, output)
    (prepared / 'shard_00.parquet').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        audit.prepare(source_manifest, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_dump_lossless_no_mutation_no_overwrite(tmp_path, dump):
    rows, identities = fixtures()
    info = audit.convert_rows(rows, identities)[0]['extra_info']
    result = recording()
    old_info = copy.deepcopy(info)
    original_image_state = dict(result['image_data'][0].__dict__)
    path = dump(tmp_path / 'dump', [info], [result])
    saved = json.loads(path.read_text())
    assert info == old_info and isinstance(result['image_data'][0], Image.Image)
    assert result['image_data'][0].__dict__ == original_image_state
    assert saved['recording']['metrics'] == result['metrics']
    image_ref = saved['recording']['image_data'][0]['image_file']
    with Image.open(path.parent / image_ref['path']) as image:
        assert image.tobytes() == result['image_data'][0].tobytes()
    assert len(list(path.parent.glob('*.png'))) == 1
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        dump(tmp_path / 'dump', [info], [result])
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='exactly one'):
        dump(tmp_path / 'other', [info, info], [result, result])
    with pytest.raises(ValueError, match='exactly one'):
        dump(tmp_path / 'other', [info], [])


@pytest.mark.parametrize('success', [None, 1, 0, 'true'])
def test_dump_requires_boolean(tmp_path, dump, success):
    rows, identities = fixtures()
    with pytest.raises(ValueError, match='exact boolean'):
        dump(tmp_path, [audit.convert_rows(rows, identities)[0]['extra_info']], [recording(success)])


def test_summary_exact_coverage_and_corruption(tmp_path, dump):
    rows, identities = fixtures()
    converted = audit.convert_rows(rows, identities)
    parquet = tmp_path / 'validation200.parquet'
    pq.write_table(pa.Table.from_pylist(converted), parquet)
    manifest = tmp_path / 'prepared_manifest.json'
    manifest.write_text(json.dumps({'format': 'original_validation200_prepared_v1', 'count': 200,
                                    'rows': identities, 'parquet': parquet.name,
                                    'parquet_sha256': audit.sha256(parquet)}))
    output = tmp_path / 'rollouts'
    for i, row in enumerate(converted):
        dump(output, [row['extra_info']], [recording(i % 2 == 0)])
    result = audit.summarize(manifest, output)
    assert result['successes'] == 100 and result['success_rate'] == .5
    assert result['categories']['base'] == {'count': 100, 'successes': 50, 'success_rate': .5}
    assert len(result['artifact_sha256']) == 400
    path = output / 'row_000000/record.json'
    original = path.read_text()
    record = json.loads(original)
    for invalid in [None, 1, 'true']:
        record['recording']['metrics']['success'] = invalid
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match='exact boolean'):
            audit.summarize(manifest, output)
    path.write_text(original)
    (output / 'partial').mkdir()
    with pytest.raises(ValueError, match='complete row'):
        audit.summarize(manifest, output)
    (output / 'partial').rmdir()
    next(path.parent.glob('*.png')).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='image hash'):
        audit.summarize(manifest, output)


def test_duplicate_key_and_interrupted_row_rejected(tmp_path, dump):
    rows, identities = fixtures()
    info = audit.convert_rows(rows, identities)[0]['extra_info']
    dump(tmp_path / 'dump', [info], [recording()])
    changed = copy.deepcopy(info)
    changed['source_index'] = 9999
    with pytest.raises(ValueError, match='duplicate source_key'):
        dump(tmp_path / 'dump', [changed], [recording()])
    bad = recording()
    bad['history'].append({'unsupported': object()})
    with pytest.raises(ValueError, match='unsupported audit value'):
        dump(tmp_path / 'interrupted', [info], [bad])
    assert not (tmp_path / 'interrupted/row_000000/record.json').exists()
    with pytest.raises(FileExistsError):
        dump(tmp_path / 'interrupted', [info], [recording()])


def test_original_validate_hook_does_not_mutate_results(tmp_path, dump, monkeypatch):
    import ast
    import sys
    import types

    helper = Path(sys.modules[dump.__module__].__file__) if dump.__module__ in sys.modules else (
        Path(__file__).resolve().parents[3] /
        'external/VAGEN/.worktree-step60-original-validation/vagen/utils/original_validation_dump.py')
    trainer = helper.parents[1] / 'trainer/ppo/ray_trainer.py'
    tree = ast.parse(trainer.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_validate')
    function = ast.Module(body=[method], type_ignores=[])
    rows, identities = fixtures()
    info = audit.convert_rows(rows, identities)[0]['extra_info']
    result = recording()
    events = []

    class Batch:
        non_tensor_batch = {'extra_info': [info]}
        @staticmethod
        def from_single_dict(value):
            return Batch()
        def pop(self, **kwargs):
            pass
        def __len__(self):
            return 1

    class Manager:
        def reset(self, configs):
            assert configs == [info]
            events.append('reset')
        def rollout_loop(self):
            events.append('rollout')
        def recording_to_log(self):
            events.append('record')
            return [result]

    module = types.ModuleType('vagen.utils.original_validation_dump')
    module.dump_validation_batch = dump
    monkeypatch.setitem(sys.modules, 'vagen.utils.original_validation_dump', module)
    namespace = {'DataProto': Batch}
    exec(compile(function, str(trainer), 'exec'), namespace)
    def capture(records, **kwargs):
        assert records == [result]
        return {'val/success': result['metrics']['success']}
    actor = types.SimpleNamespace(
        global_steps=0, test_rollout_manager=Manager(), val_dataloader=[{}],
        config=types.SimpleNamespace(trainer={'original_validation_output_dir': str(tmp_path / 'dump')}),
        _maybe_log_val_generations_to_wandb=capture, _maybe_log_raw_samples=capture,
        log_rst_to_metrics_dict=capture)
    assert namespace['_validate'](actor) == {'val/success': True}
    assert events == ['reset', 'rollout', 'record']
    assert isinstance(result['image_data'][0], Image.Image)
