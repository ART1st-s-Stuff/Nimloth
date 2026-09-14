"""Wait for an existing full CPU audit, then gate and launch the approved run."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_query_gate import validate_checkout
from run_segments import process_snapshot, utc_now


def audit_ready(contract):
    root = Path(contract['run']).parent
    marker = root / 'input_audit.json'
    if not marker.exists():
        return False
    try:
        audit = json.loads(marker.read_text())
    except json.JSONDecodeError:
        # The audit process may still be closing its final report.
        return False
    assert audit['status'] == 'passed'
    assert audit['grid_size'] == contract['preparation']['grid_size']
    assert audit['query_count'] == contract['preparation']['grid_tokens']
    assert Path(audit['model']).resolve() == (root / 'base').resolve()
    prep = json.loads((root / 'preparation.json').read_text())
    assert prep['successful_subsets_equal_epoch7'] is True
    for split in ('train', 'val'):
        entry = audit['splits'][split]
        expected = prep['splits'][split]
        source = root / 'data' / f'{split}.jsonl'
        assert Path(entry['jsonl']).resolve() == source.resolve()
        assert entry['records'] == entry['checked'] == expected['records']
        assert entry['successful_lm_answers'] > 0
        assert entry['max_length'] < audit['max_length']
        assert entry['sha256'] == expected['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--audit-pid', type=int, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    checkout = Path(contract['checkout'])
    root = Path(contract['run']).parent
    validate_checkout(checkout, contract['commit'])
    assert not Path(contract['run']).exists()
    assert (root / 'base' / 'model.safetensors.index.json').is_file()
    ready = audit_ready(contract)
    identity = process_snapshot().get(args.audit_pid)
    if not ready:
        assert identity is not None and identity[2] != 'Z', 'CPU audit process is absent'
        command = (Path('/proc') / str(args.audit_pid) / 'cmdline').read_bytes().split(b'\0')
        assert any(value.endswith(b'/audit_query_inputs.py') for value in command)
        assert str(root / 'input_audit.json').encode() in command
    print(json.dumps({'time': utc_now(), 'phase': 'preflight', 'audit_ready': ready,
                      'audit_pid': args.audit_pid, 'commit': contract['commit']}), flush=True)
    if args.check_only:
        return
    while not ready:
        current = process_snapshot().get(args.audit_pid)
        if current is None or current[1] != identity[1] or current[2] == 'Z':
            if not audit_ready(contract):
                raise RuntimeError('CPU audit ended without a valid complete audit; no launch')
            break
        time.sleep(10)
        ready = audit_ready(contract)
    validate_checkout(checkout, contract['commit'])
    print(json.dumps({'time': utc_now(), 'phase': 'seven_rank_gate'}), flush=True)
    child = None
    def stop(signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signum)
            child.wait()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child = subprocess.Popen([
        sys.executable, str(Path(__file__).with_name('run_query_gate.py')),
        '--root', str(root), '--commit', contract['commit'],
        '--train-jsonl', str(root / 'data/train.jsonl'),
        '--grid-size', str(contract['preparation']['grid_size']),
    ], cwd=checkout)
    code = child.wait()
    if code:
        raise RuntimeError(f'seven-rank gate failed with {code}; no training or retry')
    passed = json.loads(Path(contract['gate_pass']).read_text())
    assert passed['commit'] == contract['commit'] and passed['world_size'] == 7
    assert audit_ready(contract)
    print(json.dumps({'time': utc_now(), 'phase': 'formal_training'}), flush=True)
    os.execve(sys.executable, [sys.executable,
              str(Path(__file__).with_name('query_control.py')),
              '--contract', str(args.contract.resolve())], os.environ.copy())


if __name__ == '__main__':
    main()
