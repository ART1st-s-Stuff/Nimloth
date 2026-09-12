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


def validate_contract(contract):
    root = Path(contract['root'])
    argv = contract['train_argv']
    if int(argument(argv, '--epochs')) != 2 or '--until-converged' in argv:
        raise ValueError('test must train at most two epochs without convergence mode')
    if '--save-initial-checkpoint' not in argv or '--resume' in argv:
        raise ValueError('fresh test requires exact initialization snapshot')
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
            if Path(argument(command, flag)).resolve() != (root / relative).resolve():
                raise ValueError(f'{flag} differs from audited input')
        if int(argument(command, '--grid-size')) != 8:
            raise ValueError('evaluation/training grid differs from audit')
    total = contract.get('total_seconds', 21600)
    reserve = contract.get('evaluation_reserve_seconds', 3600)
    if not 1800 < total <= 21600 or not 900 <= reserve < total - 900:
        raise ValueError('invalid bounded train/evaluation budget')
    return total, reserve


def validate_audit(contract):
    root = Path(contract['root'])
    audit = json.loads((root / 'input_audit.json').read_text())
    if audit['status'] != 'passed' or audit['grid_size'] != 8 or audit['query_count'] != 64:
        raise ValueError('full K64 input audit has not passed')
    if Path(audit['model']).resolve() != (root / 'base').resolve():
        raise ValueError('audited model differs from test base')
    for split in ('train', 'val'):
        entry = audit['splits'][split]
        source = root / 'data' / f'{split}.jsonl'
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
    entries = []
    for path in sorted(run.glob('epoch_*')):
        if (path / 'COMMITTED').is_file():
            marker = json.loads((path / 'COMMITTED').read_text())
            if marker['epoch'] != int(path.name.split('_')[1]) or marker['epoch'] > 2:
                raise ValueError('invalid epoch checkpoint marker')
            entries.append((path, marker['step']))
    if not entries or entries[0][0].name != 'epoch_000' or entries[0][1] != 0:
        raise ValueError('exact pre-update checkpoint missing')
    steps = []
    for path in run.glob('resume_step_*'):
        if (path / 'COMMITTED').is_file():
            marker = json.loads((path / 'COMMITTED').read_text())
            if marker['step'] != int(path.name.split('_')[-1]):
                raise ValueError('invalid step checkpoint marker')
            steps.append((path, marker['step']))
    if steps:
        latest = max(steps, key=lambda entry: entry[1])
        if latest[1] > max(entry[1] for entry in entries):
            entries.append(latest)
    return [entry[0] for entry in entries]


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
    validate_checkout(checkout, contract['commit'])
    validate_audit(contract)
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
        gate = [contract.get('python', sys.executable), str(OLD_RESEARCH / 'run_query_gate.py'),
                '--root', str(root), '--commit', contract['commit'], '--world-size', '8',
                '--grid-size', '8', '--train-jsonl', str(root / 'data/train.jsonl')]
        run_phase(gate, phase='gate', checkout=checkout, env=env, logs=logs,
                  deadline=min(deadline, started + 900), event=event)
        passed = json.loads((root / 'gate_control/PASSED.json').read_text())
        if passed['commit'] != contract['commit'] or passed['world_size'] != 8:
            raise ValueError('gate provenance mismatch')
        validate_checkout(checkout, contract['commit'])
        validate_audit(contract)
        wait_selected_gpus_idle(30, 8)
        train_end = deadline - reserve
        status = run_phase(contract['train_argv'], phase='train', checkout=checkout,
            env=env, logs=logs, deadline=train_end, event=event,
            training_run=root / 'train', soft_deadline=train_end - 300)
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
    main()
