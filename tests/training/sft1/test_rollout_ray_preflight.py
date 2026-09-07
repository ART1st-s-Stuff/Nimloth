"""CPU isolation checks for the launcher gate; no real Ray/GPU evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPTIONAL_ENV = ('PATH', 'HF_HOME', 'TRANSFORMERS_CACHE', 'TORCH_HOME', 'VLLM_HOST_IP', 'NCCL_SOCKET_IFNAME', 'GLOO_SOCKET_IFNAME', 'NCCL_IB_DISABLE', 'CUDA_DEVICE_ORDER', 'FLASHINFER_WORKSPACE_DIR', 'TRITON_CACHE_DIR', 'XDG_CACHE_HOME', 'TORCH_EXTENSIONS_DIR', 'VLLM_USE_FLASHINFER_SAMPLER', 'TORCHINDUCTOR_DISABLE', 'TORCH_COMPILE_DISABLE', 'TORCHDYNAMO_DISABLE', 'PYTHONDONTWRITEBYTECODE')

SCRIPT = ROOT / 'experiments/training/sft1/rollouts_greedy_parallel.slurm'


@pytest.mark.parametrize('fault', [None, 'verl', 'vagen', 'executable', 'env_vars', 'missing', 'empty_verl', 'empty_vagen'])
def test_worker_preflight_transmits_pins_and_rejects_drift(tmp_path, monkeypatch, fault):
    source = SCRIPT.read_text()
    probe = source.split("<<'PYRAY'", 1)[1].split('\n', 1)[1].split('\nPYRAY', 1)[0]
    for module in ('verl', 'vagen'):
        root = tmp_path / module
        (root / module).mkdir(parents=True)
        (root / module / '__init__.py').write_text('')
        monkeypatch.setenv(module.upper() + '_DIR', str(root))
    pins = {name: os.environ[name] for name in ('VERL_DIR', 'VAGEN_DIR')}
    pins['PYTHONPATH'] = os.pathsep.join(pins.values())
    monkeypatch.setenv('PYTHONPATH', pins['PYTHONPATH'])
    for name in OPTIONAL_ENV:
        monkeypatch.setenv(name, str(tmp_path / name))
        pins[name] = os.environ[name]
    pins.update(TOKENIZERS_PARALLELISM='true', NCCL_DEBUG='WARN')
    monkeypatch.setenv('HF_TOKEN', 'test-secret-must-not-propagate')
    state = {}

    def init(*, runtime_env):
        state['env'] = runtime_env['env_vars']
        assert state['env'] == pins

    def remote(function):
        def invoke():
            # A fresh interpreter has no driver sys.path; imports need transmitted pins.
            child = subprocess.run(
                [sys.executable, '-c',
                 'import os,sys,json,verl,vagen; print(json.dumps(dict('
                 'verl=verl.__file__,vagen=vagen.__file__,executable=sys.executable,'
                 'env_vars={k:os.environ.get(k) for k in '
                 + repr(tuple(pins)) + '})))'],
                cwd=tmp_path, env=state['env'], capture_output=True, text=True, check=True,
            )
            result = json.loads(child.stdout)
            if fault in ('verl', 'vagen'):
                result[fault] = str(tmp_path / 'wrong' / fault / '__init__.py')
            elif fault == 'executable':
                result[fault] = str(tmp_path / 'other-venv/bin/python3')
            elif fault == 'env_vars':
                result[fault]['HF_HOME'] = '/wrong'
            return result
        return types.SimpleNamespace(remote=invoke)

    ray = types.SimpleNamespace(init=init, remote=remote, get=lambda result: result,
                                available_resources=lambda: {},
                                shutdown=lambda: state.update(shutdown=True))
    monkeypatch.setitem(sys.modules, 'ray', ray)
    if fault == 'missing':
        monkeypatch.setenv('VERL_DIR', '')
    if fault in ('empty_verl', 'empty_vagen'):
        module = fault.removeprefix('empty_')
        (tmp_path / module / module / '__init__.py').unlink()
    if fault:
        with pytest.raises(RuntimeError, match='missing pinned|mismatch'):
            exec(compile(probe, str(SCRIPT), 'exec'), {})
    else:
        exec(compile(probe, str(SCRIPT), 'exec'), {})
    assert state.get('shutdown', False) == (fault not in ('missing', 'empty_verl', 'empty_vagen'))


def test_pins_are_exported_for_both_preflight_and_rollout():
    source = SCRIPT.read_text()
    common = next(line for line in source.splitlines() if line.startswith('COMMON_ENV='))
    assert "export VAGEN_DIR='${BASEDIR}'" in common
    assert "export VERL_DIR='${VERL_DIR}'" in common
    assert "export PYTHONPATH='${BASEDIR}:${VERL_DIR}:${ROOT}/src'" in common
    assert "exec '${PYTHON_ENV}/bin/python3' -" in source
    assert source.index('PYRAY\n') < source.index('local prepared_list=')
