"""CPU controller contract checks; no Slurm processes or GPU execution."""
import json
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.training.sft1 import prepare_source200 as pilot
from experiments.training.sft1 import vagen_step60_data as data
from tests.training.sft1.test_vagen_step60_data import _source_rows

SCRIPT = Path(__file__).resolve().parents[3] / 'experiments/training/sft1/run_prepared_rollout_lane.slurm'


def block(name):
    return SCRIPT.read_text().split(f"<<'{name}'\n", 1)[1].split(f'\n{name}', 1)[0]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / 'source.parquet'
    pq.write_table(pa.Table.from_pylist(_source_rows()), source)
    digest = pilot.sha256(source)
    monkeypatch.setattr(data, 'SOURCE_TRAIN_SHA256', digest)
    partition = tmp_path / 'partition'
    data.partition_source_parquet(source, partition, expected_sha256=digest)
    output = tmp_path / 'prepared'
    pilot.prepare(partition / 'partition_manifest.json', output,
                  selection=pilot.BATCH1_REMAINDER_SELECTION,
                  prompt_format='step60_source_reconstruction')
    return output / 'prepared_manifest.json'


def test_real_remainder_assignments_and_ports(tmp_path, monkeypatch, prepared):
    for name in ('REPO', 'VAGEN_DIR', 'VERL_DIR', 'PYTHON_ENV', 'INIT_HF',
                 'EXPECTED_REPO_COMMIT', 'SLURM_JOB_ID'):
        monkeypatch.setenv(name, 'test')
    monkeypatch.setenv('ROLLOUT_PREPARED_MANIFEST', str(prepared))
    lanes = []
    ports = []
    for lane in (0, 1):
        root = tmp_path / f'lane{lane}'
        root.mkdir()
        monkeypatch.setenv('ROLLOUT_LANE_ID', str(lane))
        monkeypatch.setattr(sys, 'argv', ['controller', str(root)])
        exec(compile(block('PYPLAN'), str(SCRIPT), 'exec'), {})
        contract = json.loads((root / 'launch_contract.json').read_text())
        assert contract['expected_rows'] == 900
        assert contract['indices'] == list(range(lane, 90, 2))
        assert list(map(len, contract['chunks'])) == [4] * 11 + [1]
        assert [i for chunk in contract['chunks'] for i in chunk] == contract['indices']
        assert (root / 'chunks.txt').read_text().splitlines() == [
            ','.join(map(str, chunk)) for chunk in contract['chunks']]
        lanes.append(set(contract['indices']))
        port_lines = '\n'.join(line for line in SCRIPT.read_text().splitlines()
                               if line.strip().startswith(('export PORT_BASE=', 'export RAY_PORT=')))
        for chunk in range(12):
            result = subprocess.run(['bash', '-c',
                                     f'ROLLOUT_LANE_ID={lane}; chunk={chunk}\n' + port_lines +
                                     '\nprintf "%s %s %s" "$PORT_BASE" "$RAY_PORT" "$RAY_DASHBOARD_PORT"'],
                                    text=True, capture_output=True, check=True)
            ports.extend(map(int, result.stdout.split()))
    assert not lanes[0] & lanes[1]
    assert lanes[0] | lanes[1] == set(range(90))
    assert len(set(ports)) == len(ports)
    assert all(30000 <= port <= 35000 for port in ports)


def test_plan_rejects_wrong_remainder_count(tmp_path, monkeypatch, prepared):
    manifest = json.loads(prepared.read_text())
    manifest['count'] = 200
    prepared.write_text(json.dumps(manifest))
    monkeypatch.setenv('ROLLOUT_PREPARED_MANIFEST', str(prepared))
    monkeypatch.setattr(sys, 'argv', ['controller', str(tmp_path)])
    with pytest.raises(ValueError, match='count mismatch'):
        exec(compile(block('PYPLAN'), str(SCRIPT), 'exec'), {})
    assert not (tmp_path / 'launch_contract.json').exists()


def test_completion_rejects_tampered_output(tmp_path, monkeypatch):
    output = tmp_path / 'validation/train/reconstruction1800_88'
    output.mkdir(parents=True)
    raw = output / '0.jsonl'
    raw.write_text('test-record\n')
    manifest = tmp_path / 'prepared.json'
    manifest.write_text(json.dumps({'shards': [{'sha256': 'prepared-hash'}] * 90}))
    contract = tmp_path / 'contract.json'
    contract.write_text(json.dumps({'ROLLOUT_PREPARED_MANIFEST': str(manifest),
                                   'prepared_sha256': pilot.sha256(manifest)}))
    marker = output / '0.validation.json'
    marker.write_text(json.dumps({'count': 20, 'unique_identities': 20,
                                 'identity_coverage': 'exact',
                                 'jsonl_sha256': pilot.sha256(raw),
                                 'prepared_parquet_sha256': 'prepared-hash'}))
    monkeypatch.setattr(sys, 'argv', ['controller', str(tmp_path), '88', str(contract)])
    exec(compile(block('PYCHECK'), str(SCRIPT), 'exec'), {})
    raw.write_text('tampered\n')
    with pytest.raises(ValueError, match='completion evidence'):
        exec(compile(block('PYCHECK'), str(SCRIPT), 'exec'), {})
    marker.unlink()
    with pytest.raises(ValueError, match='membership mismatch'):
        exec(compile(block('PYCHECK'), str(SCRIPT), 'exec'), {})


def test_shell_contract():
    subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
    source = SCRIPT.read_text()
    assert 'job_id=$SLURM_JOB_ID' in source
    assert '< /dev/null > "$run_root/$chunk_name.controller.log"' in source
    assert 'mkdir "$run_root"' in source
    assert 'test "$rc" -eq 0' in source
    assert 'check_sources\n' in source
    assert 'sbatch ' not in source and 'squeue ' not in source
    assert '#SBATCH --time=06:00:00' in source
