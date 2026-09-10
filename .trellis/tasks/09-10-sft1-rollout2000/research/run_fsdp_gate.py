"""Bound the two remote eight-rank FSDP checks to 15 minutes in total."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from run_segments import terminate_group, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('commit')
    parser.add_argument('output', type=Path)
    parser.add_argument('--action-token-loss-weight', type=float, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    root = Path('/mnt/nimloth/.worktree/sft1-rollout2000')
    python = '/mnt/nimloth/venv/bin/python3'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                   text=True).strip() == args.commit
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root,
                                       text=True).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root/'src'), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
               CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               CPATH='/mnt/nimloth/dependencies/python310-dev/root/usr/include/python3.10:'
                     '/mnt/nimloth/dependencies/python310-dev/root/usr/include')
    tasks = [('roundtrip', root/'tests/integration/sft1_fsdp_roundtrip.py',
              ['--action-token-loss-weight', str(args.action_token_loss_weight)]),
             ('capacity', Path(__file__).with_name('fsdp_capacity_probe.py'),
              ['--cache-root', ('/mnt/nimloth/outputs/experiments/sft1-rollout2000/'
               '20260910T112834Z_format_only_converge/preprocess_cache'),
               '--action-token-loss-weight', str(args.action_token_loss_weight)])]
    with (args.output/'events.jsonl').open('x') as events:
        for name, script, extra in tasks:
            usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            assert len(usage) == 8 and all(int(value) < 100 for value in usage), usage
            command = [python, '-m', 'torch.distributed.run', '--standalone',
                       '--nnodes=1', '--nproc-per-node=8', str(script),
                       '--output-dir', str(args.output/name), *extra]
            with (args.output/f'{name}.log').open('x') as log:
                process = subprocess.Popen(command, cwd=root, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                events.write(json.dumps({'event': 'started', 'time': utc_now(),
                    'phase': name, 'pid': process.pid, 'commit': args.commit,
                    'command': command, 'deadline_seconds': 900})+'\n')
                events.flush()
                try:
                    code = process.wait(timeout=max(0.01, 900-(time.monotonic()-started)))
                except BaseException:
                    terminate_group(process)
                    events.write(json.dumps({'event': 'terminated', 'time': utc_now(),
                                             'phase': name})+'\n')
                    events.flush()
                    raise
            events.write(json.dumps({'event': 'exited', 'time': utc_now(),
                                     'phase': name, 'returncode': code})+'\n')
            events.flush()
            if code != 0:
                raise RuntimeError(f'{name} failed; no automatic retry')
            assert json.loads((args.output/name/'PASSED.json').read_text())['world_size'] == 8
    (args.output/'PASSED.json').write_text(json.dumps({
        'commit': args.commit, 'world_size': 8, 'phases': ['roundtrip', 'capacity'],
        'action_token_loss_weight': args.action_token_loss_weight,
        'elapsed_seconds': time.monotonic()-started})+'\n')


if __name__ == '__main__':
    main()
