"""Bound the two real seven-rank query update/save/restore phases to 15 minutes."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from run_segments import process_snapshot, remember_owned, terminate_group, utc_now


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--train-jsonl', type=Path, required=True)
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[4]
    validate_checkout(checkout, args.commit)
    audit = json.loads((args.root / 'input_audit.json').read_text())
    index = audit['train_max_sample_index']
    output = args.root / 'gate_control'
    output.mkdir(exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(checkout / 'src'), CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               CPATH='/mnt/nimloth/dependencies/python310-dev/root/usr/include/python3.10:/mnt/nimloth/dependencies/python310-dev/root/usr/include')
    command = ['/mnt/nimloth/venv/bin/python3', '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', '--nproc-per-node=7', str(Path(__file__).with_name('query_capacity_probe.py')),
               '--model', str(args.root / 'base'), '--train-jsonl', str(args.train_jsonl),
               '--dino-cache-root', str(args.root / 'dino_cache'), '--output-dir', str(args.root / 'gate'),
               '--sample-index', str(index)]
    start = time.monotonic()
    with (output / 'events.jsonl').open('x') as events:
        for phase in ('initial', 'resume'):
            gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()
            for row in gpu_rows:
                gpu, memory, utilization = map(int, row.split(','))
                if gpu < 7:
                    assert memory < 100 and utilization == 0, row
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
    result = json.loads((args.root / 'gate' / 'PASSED.json').read_text())
    assert result['world_size'] == 7
    (output / 'PASSED.json').write_text(json.dumps({'commit': args.commit, 'world_size': 7,
                                                 'elapsed_seconds': time.monotonic() - start,
                                                 'result': result}, indent=2))


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
