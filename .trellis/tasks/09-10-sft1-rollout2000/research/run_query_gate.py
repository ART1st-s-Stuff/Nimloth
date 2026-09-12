"""Bound the two real seven-rank query update/save/restore phases to 15 minutes."""
import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from run_segments import process_snapshot, remember_owned, terminate_group, utc_now

from nimloth.training.sft.stage2.config import QueryAlignmentConfig


def validate_checkout(checkout: Path, expected_commit: str) -> None:
    actual_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True
    ).strip()
    assert actual_commit == expected_commit, (actual_commit, expected_commit)
    assert not subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=checkout, text=True
    ).strip()

    dependency = checkout / 'external' / 'le-wm'
    expected_dependency = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD:external/le-wm'], cwd=checkout, text=True
    ).strip()
    if not (dependency / 'module.py').is_file():
        raise RuntimeError(
            f'le-wm checkout is missing {dependency / "module.py"}; '
            'initialize the pinned submodule before launching the query gate'
        )
    actual_dependency = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=dependency, text=True
    ).strip()
    if actual_dependency != expected_dependency:
        raise RuntimeError(
            'le-wm checkout does not match the pinned gitlink: '
            f'{actual_dependency} != {expected_dependency}'
        )


def attempt_paths(root: Path, attempt: str) -> tuple[Path, Path]:
    if attempt and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', attempt) is None:
        raise ValueError(f'invalid gate attempt name: {attempt!r}')
    suffix = f'_{attempt}' if attempt else ''
    return root / f'gate_control{suffix}', root / f'gate{suffix}'


def wait_selected_gpus_idle(timeout_seconds: float, world_size: int = 7) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        rows = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            text=True,
        ).splitlines()
        busy = []
        for row in rows:
            gpu, memory, utilization = map(int, row.split(','))
            if gpu < world_size and (memory >= 100 or utilization != 0):
                busy.append(row)
        if not busy:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f'selected GPUs did not become idle: {busy}')
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--train-jsonl', type=Path, required=True)
    parser.add_argument('--attempt', default='')
    parser.add_argument('--world-size', type=int, default=7)
    parser.add_argument('--grid-size', type=int, default=4)
    args = parser.parse_args()
    if not 2 <= args.world_size <= 8:
        raise ValueError("world-size must be between 2 and 8")
    objective = QueryAlignmentConfig(grid_size=args.grid_size)
    checkout = Path(__file__).resolve().parents[4]
    validate_checkout(checkout, args.commit)
    audit = json.loads((args.root / 'input_audit.json').read_text())
    index = audit['train_max_sample_index']
    output, gate_output = attempt_paths(args.root, args.attempt)
    output.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(checkout / 'src'), CUDA_VISIBLE_DEVICES=','.join(map(str, range(args.world_size))),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               CPATH='/mnt/nimloth/dependencies/python310-dev/root/usr/include/python3.10:/mnt/nimloth/dependencies/python310-dev/root/usr/include')
    command = ['/mnt/nimloth/venv/bin/python3', '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', f'--nproc-per-node={args.world_size}', str(Path(__file__).with_name('query_capacity_probe.py')),
               '--model', str(args.root / 'base'), '--train-jsonl', str(args.train_jsonl),
               '--dino-cache-root', str(args.root / 'dino_cache'), '--output-dir', str(gate_output),
               '--sample-index', str(index), '--grid-size', str(objective.grid_size)]
    start = time.monotonic()
    with (output / 'events.jsonl').open('x') as events:
        for phase in ('initial', 'resume'):
            # CUDA utilization can remain nonzero for one sample after the
            # initial process exits even though its memory is already released.
            wait_selected_gpus_idle(0 if phase == 'initial' else 30, args.world_size)
            argv = command + (['--resume'] if phase == 'resume' else [])
            owned = {}
            with (output / f'{phase}.log').open('x') as log:
                process = subprocess.Popen(argv, cwd=checkout, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                events.write(json.dumps({'time': utc_now(), 'phase': phase, 'event': 'started',
                                         'pid': process.pid, 'argv': argv, 'commit': args.commit}) + '\n')
                events.flush()
                try:
                    while process.poll() is None:
                        remember_owned(process.pid, process_snapshot(), owned)
                        if time.monotonic() - start >= 900:
                            raise TimeoutError('query gate reached total 15 minute budget')
                        time.sleep(1)
                    if process.returncode != 0:
                        raise RuntimeError(f'{phase} failed; inspect log; no retry')
                except BaseException:
                    terminate_group(process, owned=owned)
                    raise
            events.write(json.dumps({'time': utc_now(), 'phase': phase, 'event': 'complete'}) + '\n')
            events.flush()
    result = json.loads((gate_output / 'PASSED.json').read_text())
    assert result['world_size'] == args.world_size
    (output / 'PASSED.json').write_text(json.dumps({'commit': args.commit, 'world_size': args.world_size,
                                                 'elapsed_seconds': time.monotonic() - start,
                                                 'result': result}, indent=2))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
