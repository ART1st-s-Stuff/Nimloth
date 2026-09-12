"""Single bounded Stage2 gate/train/evaluate run; preserves every checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OLD_RESEARCH = Path(__file__).resolve().parents[2] / '09-10-sft1-rollout2000' / 'research'
sys.path.insert(0, str(OLD_RESEARCH))
from run_segments import (descendant, process_snapshot, remember_owned,
                          same_process, select_ranks, terminate_group, utc_now, paused_exit)


def argument(argv, name):
    values = [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == name]
    values += [token.split('=', 1)[1] for token in argv if token.startswith(name + '=')]
    if len(values) != 1:
        raise ValueError(f'expected one {name} argument')
    return values[0]


def gate_training_options(argv):
    options = []
    for name in ("--lr", "--embedding-master-dtype", "--projector-lr",
                 "--query-token-lr", "--protocol-token-lr"):
        if any(token == name or token.startswith(name + "=") for token in argv):
            options.extend([name, argument(argv, name)])
    return options


def validate_contract(contract):
    root = Path(contract['root'])
    input_root = Path(contract.get('input_root', contract['root']))
    if input_root.resolve() == root.resolve():
        raise ValueError('fresh output root must differ from immutable input_root')
    argv = contract['train_argv']
    if '--epochs' in argv or '--until-converged' not in argv:
        raise ValueError('fresh training must use convergence mode without an epoch cap')
    convergence = {
        '--convergence-min-epochs': 2,
        '--convergence-patience-epochs': 2,
        '--convergence-min-relative-improvement': .01,
    }
    for flag, expected in convergence.items():
        if float(argument(argv, flag)) != expected:
            raise ValueError(f'invalid convergence setting: {flag}')
    if ('--save-initial-checkpoint' not in argv or '--resume' in argv
            or '--continue-from-epoch' in argv):
        raise ValueError('fresh test requires epoch000 and forbids resume/continuation')
    expected_training = {
        '--lr': '5e-5', '--projector-lr': '5e-5',
        '--query-token-lr': '5e-5', '--protocol-token-lr': '1e-5',
        '--embedding-master-dtype': 'float32',
    }
    for flag, expected in expected_training.items():
        if argument(argv, flag) != expected:
            raise ValueError(f'invalid selected-row training setting: {flag}')
    if int(argument(argv, '--resume-save-steps')) != 10 or '--keep-step-checkpoints' in argv:
        raise ValueError('fresh training must prune ten-step checkpoints at epoch boundaries')
    if Path(argument(argv, '--output-dir')).resolve() != (root / 'train').resolve():
        raise ValueError('training output must be root/train')
    if int(argument(argv, '--grid-size')) != 8:
        raise ValueError('approved test uses grid_size 8')
    if int(argument(argv, '--nproc-per-node')) != 8:
        raise ValueError('approved test uses eight ranks')
    if argument(argv, '--distributed-strategy') != 'fsdp':
        raise ValueError('approved test requires FSDP')
    if not any(argv[i:i + 2] == ['-m', 'nimloth.training.sft.stage2'] for i in range(len(argv))):
        raise ValueError('unexpected training entry point')
    template = contract['eval_argv_template']
    if sum(token.count('{checkpoint}') for token in template) != 1 or sum(
            token.count('{output}') for token in template) != 1:
        raise ValueError('evaluation template must identify checkpoint and output once')
    if int(argument(template, '--nproc-per-node')) != 8:
        raise ValueError('evaluation must use eight ranks')
    for command in (argv, template):
        for flag, relative in (('--model', 'base'), ('--train-jsonl', 'data/train.jsonl'),
                               ('--val-jsonl', 'data/val.jsonl'), ('--dino-cache-root', 'dino_cache')):
            if Path(argument(command, flag)).resolve() != (input_root / relative).resolve():
                raise ValueError(f'{flag} differs from audited input')
        if int(argument(command, '--grid-size')) != 8:
            raise ValueError('evaluation/training grid differs from audit')
    total = contract.get('total_seconds', 21600)
    reserve = contract.get('evaluation_reserve_seconds', 3600)
    if not 1800 < total <= 21600 or not 900 <= reserve < total - 900:
        raise ValueError('invalid bounded train/evaluation budget')
    return total, reserve


def validate_audit(contract):
    input_root = Path(contract.get('input_root', contract['root']))
    audit_path = input_root / 'input_audit.json'
    if not audit_path.is_file():
        identity = contract.get('input_identity')
        if not isinstance(identity, dict):
            raise ValueError('input_root lacks audit and contract lacks input_identity')
        for split in ('train', 'val'):
            source = input_root / 'data' / f'{split}.jsonl'
            entry = identity.get(split, {})
            if (entry.get('records') != contract['expected_records'][split]
                    or hashlib.sha256(source.read_bytes()).hexdigest() != entry.get('sha256')):
                raise ValueError(f'changed or invalid {split} identity')
        if not 0 <= int(identity.get('train_max_sample_index', -1)) < identity['train']['records']:
            raise ValueError('input identity lacks a valid gate sample index')
        base_files = identity.get('base_files')
        if not isinstance(base_files, dict) or not base_files:
            raise ValueError('input identity lacks base file SHA256 values')
        if 'config.json' not in base_files or not any(
                name.endswith('.safetensors') for name in base_files):
            raise ValueError('input identity lacks critical base model files')
        for relative, expected_sha in base_files.items():
            path = input_root / 'base' / relative
            if (not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha):
                raise ValueError(f'changed or invalid base file: {relative}')
        dino_manifest = input_root / 'dino_cache' / 'manifest.json'
        manifest_bytes = dino_manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        if (hashlib.sha256(manifest_bytes).hexdigest()
                != identity.get('dino_cache_manifest_sha256')
                or manifest.get('fingerprint') != identity.get('dino_cache_fingerprint')):
            raise ValueError('changed or invalid DINO cache manifest identity')
        return identity
    audit = json.loads(audit_path.read_text())
    if audit['status'] != 'passed' or audit['grid_size'] != 8 or audit['query_count'] != 64:
        raise ValueError('full K64 input audit has not passed')
    if Path(audit['model']).resolve() != (input_root / 'base').resolve():
        raise ValueError('audited model differs from test base')
    for split in ('train', 'val'):
        entry = audit['splits'][split]
        source = input_root / 'data' / f'{split}.jsonl'
        if not (entry['checked'] == entry['records'] == contract['expected_records'][split] > 0):
            raise ValueError(f'incomplete {split} audit')
        if (Path(entry['jsonl']).resolve() != source.resolve()
                or hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']
                or entry['max_length'] >= audit['max_length']
                or entry['successful_lm_answers'] <= 0):
            raise ValueError(f'changed or invalid {split} audit')
    return audit


def training_ranks(launcher, run):
    snapshot = process_snapshot()
    parents = {pid: row[0] for pid, row in snapshot.items()}
    candidates = []
    for pid in snapshot:
        if not descendant(pid, launcher, parents):
            continue
        proc = Path('/proc') / str(pid)
        try:
            argv = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
            if ('torch.distributed.run' in argv
                    or not any(argv[i:i + 2] == ['-m', 'nimloth.training.sft.stage2'] for i in range(len(argv)))
                    or argument(argv, '--output-dir') != str(run)):
                continue
            env = dict(x.split(b'=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if b'=' in x)
            candidates.append((pid, int(env[b'LOCAL_RANK'])))
        except (OSError, ValueError, KeyError):
            continue
    ranks = select_ranks(launcher, candidates, parents)
    return [(pid, snapshot[pid][1]) for pid, _rank in ranks]


def checkpoints_to_evaluate(run):
    initial = run / 'epoch_000'
    if not (initial / 'COMMITTED').is_file():
        raise ValueError('exact pre-update checkpoint missing')
    marker = json.loads((initial / 'COMMITTED').read_text())
    if marker != {'epoch': 0, 'step': 0}:
        raise ValueError('invalid epoch000 marker')
    converged = run / 'CONVERGED.json'
    final = run / 'final'
    if not converged.is_file() or not (final / 'training_state.pt').is_file():
        raise ValueError('final checkpoint is not proven converged')
    evidence = json.loads(converged.read_text())
    if not evidence.get('state', {}).get('converged'):
        raise ValueError('final checkpoint is not proven converged')
    return [initial, final]


def run_phase(argv, *, phase, checkout, env, logs, deadline, event,
              training_run=None, soft_deadline=None):
    """Signal only identified ranks; reap every owned child even on failure."""
    owned = {}
    planned = capped = False
    log_path = logs / (phase + '.log')
    with log_path.open('x') as log:
        child = subprocess.Popen(argv, cwd=checkout, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        event('phase_started', phase=phase, pid=child.pid, argv=argv)
        try:
            while child.poll() is None:
                remember_owned(child.pid, process_snapshot(), owned)
                now = time.monotonic()
                # Reserve the helper's TERM/KILL/reap allowance before deadline.
                if now >= deadline - 40:
                    event('phase_deadline', phase=phase)
                    terminate_group(child, owned=owned)
                    capped = True
                    break
                if soft_deadline is not None and now >= soft_deadline and not planned:
                    ranks = training_ranks(child.pid, training_run)
                    for pid, starttime in ranks:
                        if not same_process(pid, starttime, process_snapshot()):
                            raise RuntimeError('training rank changed before pause')
                        os.kill(pid, signal.SIGUSR1)
                    planned = True
                    event('checkpoint_pause_requested', ranks=ranks)
                time.sleep(1)
            # A launcher may exit while descendants survive; never let a later
            # phase overlap those processes or rely on process-group equality.
            terminate_group(child, owned=owned)
        except BaseException:
            terminate_group(child, owned=owned)
            raise
    code = child.returncode
    event('phase_finished', phase=phase, returncode=code, capped=capped, planned=planned)
    if capped:
        if training_run is None:
            raise TimeoutError(f'{phase} exhausted allotted budget')
        return 'capped'
    if code == 0:
        return 'complete'
    if training_run is not None and paused_exit(code, planned, log_path.read_text()):
        return 'paused'
    raise RuntimeError(f'{phase} failed with code {code}; inspect {log_path}')


def main():
    from run_query_gate import validate_checkout, wait_selected_gpus_idle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    total, reserve = validate_contract(contract)
    checkout, root = Path(contract['checkout']), Path(contract['root'])
    input_root = Path(contract.get('input_root', contract['root']))
    validate_checkout(checkout, contract['commit'])
    audit = validate_audit(contract)
    if (root / 'train').exists() or (root / 'test_controller').exists():
        raise FileExistsError('test output already used; preserve it and choose a new run')
    if args.check_only:
        print(json.dumps({'preflight': 'passed'}))
        return
    logs = root / 'test_controller'
    logs.mkdir(exist_ok=False)
    def event(kind, **fields):
        with (logs / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': utc_now(), 'event': kind, **fields}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    env = os.environ.copy()
    env.update(contract.get('env', {}))
    env.update(PYTHONPATH=str(checkout / 'src'), CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    deadline = started + total
    event('test_started', total_seconds=total, evaluation_reserve_seconds=reserve,
          commit=contract['commit'], pid=os.getpid())
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        wait_selected_gpus_idle(0, 8)
        gate_base = [contract.get('python', sys.executable), '-m', 'torch.distributed.run',
                     '--standalone', '--nnodes=1', '--nproc-per-node=8',
                     str(Path(__file__).with_name('selected_rows_capacity_probe.py')),
                     '--model', str(input_root / 'base'),
                     '--train-jsonl', str(input_root / 'data/train.jsonl'),
                     '--dino-cache-root', str(input_root / 'dino_cache'),
                     '--output-dir', str(root / 'selected_rows_gate'), '--grid-size', '8',
                     '--sample-index', str(audit['train_max_sample_index'])]
        gate_base += gate_training_options(contract['train_argv'])
        gate_deadline = min(deadline, started + 900)
        run_phase(gate_base, phase='gate_initial', checkout=checkout, env=env, logs=logs,
                  deadline=gate_deadline, event=event)
        run_phase(gate_base + ['--resume'], phase='gate_resume', checkout=checkout,
                  env=env, logs=logs, deadline=gate_deadline, event=event)
        passed = json.loads((root / 'selected_rows_gate/PASSED.json').read_text())
        if passed['world_size'] != 8 or passed['schema'] != 'selected_rows_v1':
            raise ValueError('gate provenance mismatch')
        expected_cache = audit.get('dino_cache_fingerprint')
        if expected_cache is not None and passed['cache_fingerprint'] != expected_cache:
            raise ValueError('gate DINO cache fingerprint differs from input identity')
        validate_checkout(checkout, contract['commit'])
        refreshed_audit = validate_audit(contract)
        if refreshed_audit['train_max_sample_index'] != audit['train_max_sample_index']:
            raise ValueError('gate sample identity changed after gate')
        wait_selected_gpus_idle(30, 8)
        train_end = deadline - reserve
        status = run_phase(contract['train_argv'], phase='train', checkout=checkout,
            env=env, logs=logs, deadline=train_end, event=event,
            training_run=root / 'train', soft_deadline=train_end - 300)
        if status != 'complete':
            result = {'status': 'paused', 'training_status': status,
                      'elapsed_seconds': time.monotonic() - started, 'evaluations': []}
            (logs / 'PAUSED.json').write_text(json.dumps(result, indent=2) + '\n')
            event('test_paused', **result)
            return 75
        checkpoints = checkpoints_to_evaluate(root / 'train')
        if len(checkpoints) < 2:
            raise RuntimeError('test has no committed post-update checkpoint to evaluate')
        completed = []
        for checkpoint in checkpoints:
            if time.monotonic() >= deadline - 60:
                raise TimeoutError('global budget exhausted before all checkpoint evaluations')
            wait_selected_gpus_idle(min(30, deadline - time.monotonic() - 40), 8)
            output = root / 'evaluation' / checkpoint.name
            argv = [value.replace('{checkpoint}', str(checkpoint)).replace('{output}', str(output))
                    for value in contract['eval_argv_template']]
            run_phase(argv, phase='eval_' + checkpoint.name, checkout=checkout, env=env,
                      logs=logs, deadline=deadline, event=event)
            if not (output / 'COMPLETED').is_file():
                raise RuntimeError('evaluator returned without complete quality evidence')
            completed.append(str(output))
        result = {'status': 'completed', 'training_status': status,
                  'elapsed_seconds': time.monotonic() - started, 'evaluations': completed}
        (logs / 'COMPLETED.json').write_text(json.dumps(result, indent=2) + '\n')
        event('test_completed', **result)
    except BaseException as error:
        event('test_stopped', error=repr(error), elapsed_seconds=time.monotonic() - started)
        (logs / 'STOPPED.json').write_text(json.dumps({'error': repr(error)}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
