"""Own one preprocessing phase and bounded, verified pause/resume segments.

Run using the experiment Python with repository src on PYTHONPATH. This controller
must remain alive; each launcher gets its own process group. No unexplained failure
is retried. Runtime limits are not convergence evidence.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

MODULE = 'nimloth.training.sft.stage1.trainer'
SEGMENT_SECONDS = 6 * 60 * 60
PAUSE_SECONDS = SEGMENT_SECONDS - 10 * 60


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def rank_command(argv, run):
    return (any(argv[i:i+2] == ['-m', MODULE] for i in range(len(argv)-1))
            and 'torch.distributed.run' not in argv
            and any(argv[i:i+2] == ['--output-dir', str(run)] for i in range(len(argv)-1)))


def descendant(pid, ancestor, parents):
    seen = set()
    while pid in parents and pid not in seen:
        seen.add(pid)
        pid = parents[pid]
        if pid == ancestor:
            return True
    return False


def process_snapshot(proc_root=Path('/proc')):
    """Snapshot PID identity and ancestry; starttime distinguishes reused PIDs."""
    result = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            result[int(entry.name)] = (int(stat[1]), int(stat[19]), stat[0])
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return result


def select_ranks(launcher_pid, candidates, parents):
    # DataLoader children inherit the trainer argv and LOCAL_RANK. Only the
    # outermost matching trainer under this launch is a distributed rank.
    owned = [(pid, rank) for pid, rank in candidates
             if descendant(pid, launcher_pid, parents)]
    ranks = [(pid, rank) for pid, rank in owned
             if not any(other != pid and descendant(pid, other, parents)
                        for other, _ in owned)]
    if len(ranks) != 8 or {rank for _, rank in ranks} != set(range(8)):
        raise RuntimeError(f'expected exactly 8 owned training ranks, found {ranks}')
    return sorted(ranks, key=lambda item: item[1])


def find_ranks(launcher_pid, run):
    snapshot = process_snapshot()
    parents = {pid: info[0] for pid, info in snapshot.items()}
    candidates = []
    for pid in snapshot:
        entry = Path('/proc') / str(pid)
        try:
            argv = [x.decode() for x in (entry/'cmdline').read_bytes().split(b'\0') if x]
            if rank_command(argv, run):
                env = dict(x.split(b'=', 1) for x in (entry/'environ').read_bytes().split(b'\0') if b'=' in x)
                candidates.append((pid, int(env[b'LOCAL_RANK'])))
        except (FileNotFoundError, ProcessLookupError, PermissionError, KeyError, ValueError, UnicodeDecodeError):
            continue
    # torchrun workers may each lead their own group. Ownership is ancestry,
    # not equality to the launcher's PGID.
    return select_ranks(launcher_pid, candidates, parents)


def boundaries(run):
    result = {}
    for marker in run.glob('resume_step_*/COMMITTED'):
        if marker.is_symlink() or marker.parent.is_symlink():
            raise ValueError('symlink resume boundary')
        result[str(marker.parent)] = hashlib.sha256(marker.read_bytes()).hexdigest()
    return result


def new_boundary(run, before):
    after = boundaries(run)
    new = [Path(path) for path, digest in after.items() if before.get(path) != digest]
    if not new:
        raise RuntimeError('planned pause produced no new committed resume boundary')
    path = max(new, key=lambda p: int(p.name.rsplit('_', 1)[1]))
    marker = json.loads((path/'COMMITTED').read_text())
    import torch

    from nimloth.training.sft.stage1.checkpoint import validate_resume_state
    state = torch.load(path/'training_state.pt', map_location='cpu', weights_only=False)
    if (marker != {'schema': 'nimloth_early_stage_resume_v1', 'step': state['step']}
            or path.name != f"resume_step_{state['step']:08d}"
            or not isinstance(state.get('identity'), dict) or not state['identity']
            or state.get('format_objective') != 'format_answer_ce_v2'
            or state.get('training_stage') != 'format'
            or any(state.get(key) is not None for key in ('latent_token_count', 'latent_query_mode'))):
        raise RuntimeError('invalid format-only pause checkpoint')
    for key in ('optimizer', 'scheduler'):
        if not isinstance(state.get(key), dict) or not state[key]:
            raise RuntimeError(f'missing {key}')
    from nimloth.training.sft.stage1.convergence import ConvergenceState
    convergence = ConvergenceState.from_state_dict(state.get('convergence_state') or {})
    if convergence.last_epoch != state['epoch'] - 1 or convergence.converged:
        raise RuntimeError('pause convergence cursor mismatch')
    for rank in range(8):
        validate_resume_state(state, expected_identity=state['identity'], rank=rank, world=8)
        if not {'python', 'numpy', 'torch_cpu', 'torch_cuda'}.issubset(state['rank_rng_states'][rank]):
            raise RuntimeError('pause checkpoint lacks GPU RNG state')
    return path


def paused_exit(returncode, planned, log_text):
    codes = [int(code) for code in re.findall(r'exitcode\s*:\s*(-?\d+)\b', log_text)]
    return (returncode != 0 and planned and bool(codes) and set(codes) == {75}
            and not re.search(r'CUDA out of memory|OutOfMemoryError', log_text, re.IGNORECASE))


def owned_processes(launcher_pid, snapshot):
    parents = {pid: info[0] for pid, info in snapshot.items()}
    return {pid: info[1] for pid, info in snapshot.items()
            if pid == launcher_pid or descendant(pid, launcher_pid, parents)}


def same_process(pid, starttime, snapshot):
    info = snapshot.get(pid)
    return info is not None and info[1] == starttime and info[2] != 'Z'


def remember_owned(launcher_pid, snapshot, owned):
    # Once the leader's PID has been reused, its new descendants are unrelated.
    if launcher_pid in owned and not same_process(launcher_pid, owned[launcher_pid], snapshot):
        return
    for pid, starttime in owned_processes(launcher_pid, snapshot).items():
        owned.setdefault(pid, starttime)


def terminate_group(process, *, owned=None, snapshotter=process_snapshot, send=os.kill,
                    clock=time.monotonic, sleep=time.sleep):
    # Capture ownership BEFORE signaling the leader: torchrun workers have
    # separate groups and can be reparented when their launcher exits.
    owned = dict(owned or {})
    remember_owned(process.pid, snapshotter(), owned)

    def signal_owned(sig):
        for pid, starttime in owned.items():
            if same_process(pid, starttime, snapshotter()):
                try:
                    send(pid, sig)
                except ProcessLookupError:
                    pass

    signal_owned(signal.SIGTERM)
    deadline = clock() + 30
    def survivors():
        current = snapshotter()
        return [pid for pid, starttime in owned.items()
                if same_process(pid, starttime, current)]

    while survivors() and clock() < deadline:
        sleep(min(0.1, max(0, deadline - clock())))
    signal_owned(signal.SIGKILL)
    kill_deadline = clock() + 5
    while survivors() and clock() < kill_deadline:
        sleep(min(0.1, max(0, kill_deadline - clock())))
    remaining = survivors()
    if remaining:
        raise RuntimeError(f'owned processes survived SIGKILL: {remaining}')
    process.wait(timeout=5)


def wait_segment(process, run, event, *, clock=time.monotonic, sleep=time.sleep,
                 rank_finder=find_ranks, send=os.kill, snapshotter=process_snapshot,
                 terminator=None):
    if terminator is None:
        terminator = terminate_group
    start, planned = clock(), False
    cleanup_started = False
    owned = {}
    try:
        while True:
            remember_owned(process.pid, snapshotter(), owned)
            if process.poll() is not None:
                break
            elapsed = clock() - start
            if elapsed >= SEGMENT_SECONDS:
                event('deadline', pid=process.pid, elapsed=elapsed)
                raise RuntimeError('six-hour segment deadline exceeded; stopped owned group')
            if elapsed >= PAUSE_SECONDS and not planned:
                ranks = rank_finder(process.pid, run)
                event('pause_requested', ranks=ranks, elapsed=elapsed)
                for pid, _ in ranks:
                    send(pid, signal.SIGUSR1)
                planned = True
            sleep(min(5, max(0.01, SEGMENT_SECONDS - elapsed)))
        if process.returncode != 0:
            cleanup_started = True
            terminator(process, owned=owned)
        return process.returncode, planned
    except BaseException:
        if not cleanup_started:
            terminator(process, owned=owned)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('commit')
    parser.add_argument('run_id')
    parser.add_argument('policy', type=Path)
    parser.add_argument('--preprocessed', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.commit) or not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z_[A-Za-z0-9_]+', args.run_id):
        parser.error('invalid commit or run ID')
    run = Path('/mnt/nimloth/outputs/experiments/sft1-rollout2000')/args.run_id
    controller = run.parent/(args.run_id+'_controller')
    controller.mkdir(parents=True, exist_ok=False)
    policy = args.policy.resolve(strict=True)
    launch = Path(__file__).with_name('launch.sh').resolve()
    def event(kind, **fields):
        with (controller/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=utc_now(), event=kind, **fields))+'\n')
            stream.flush()
            os.fsync(stream.fileno())
    event('controller_started', pid=os.getpid(), commit=args.commit, run=str(run),
          policy=str(policy), segment_seconds=SEGMENT_SECONDS, pause_seconds=PAUSE_SECONDS)
    phase, segment = ('train', 1) if args.preprocessed else ('preprocess', 0)
    if args.preprocessed and (run/'PREPROCESS_SUCCEEDED').read_text().strip() != '0':
        raise RuntimeError('preprocessing is not complete')
    while True:
        log = controller/f'{segment:04d}_{phase}.log'
        before = boundaries(run) if phase != 'preprocess' else {}
        argv = ['bash', str(launch), args.commit, args.run_id, phase]
        if phase != 'preprocess':
            argv.append(str(policy))
        with log.open('x') as output:
            process = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            event('segment_started', phase=phase, segment=segment, pid=process.pid,
                  pgid=process.pid, argv=argv, log=str(log))
            if phase == 'preprocess':
                try:
                    returncode, planned = process.wait(), False
                except BaseException:
                    terminate_group(process)
                    raise
            else:
                returncode, planned = wait_segment(process, run, event)
        event('segment_exited', phase=phase, segment=segment, returncode=returncode, planned_pause=planned)
        if phase == 'preprocess':
            if returncode != 0 or (run/'PREPROCESS_SUCCEEDED').read_text().strip() != '0':
                raise RuntimeError('preprocessing failed; training not started')
            phase, segment = 'train', 1
            continue
        if returncode == 0:
            if ((run/'TRAIN_SUCCEEDED').read_text().strip() != '0'
                    or not (run/'CONVERGED.json').is_file()
                    or not (run/'cleanup_complete.json').is_file()):
                raise RuntimeError('launcher success lacks convergence or cleanup evidence')
            event('converged', segment=segment)
            return
        if (run/'CONVERGED.json').exists():
            raise RuntimeError('converged training launcher failed verification or cleanup; no retraining')
        if not paused_exit(returncode, planned, log.read_text()):
            raise RuntimeError('unexplained segment failure; no automatic retry')
        checkpoint = new_boundary(run, before)
        event('resume_authorized', checkpoint=str(checkpoint), prior_segment=segment)
        phase, segment = 'resume', segment+1


if __name__ == '__main__':
    # Ensure interruption executes the process-group cleanup paths above.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'controller received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
