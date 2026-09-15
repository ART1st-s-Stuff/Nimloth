"""Bounded no-update comparison using a completed ablation's exact argv."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != args.commit:
        raise ValueError('source commit mismatch')
    prior = json.loads((args.run_root / 'controller/progress.json').read_text())
    if prior['status'] != 'complete':
        raise ValueError('comparison requires completed source training')
    phases = {p['arm']: p for p in prior['phases'] if p['phase'] == 'formal'}
    commands = []
    for name, arm in [('stage2', 'control'), ('control', 'control'), ('treatment', 'treatment')]:
        command = phases[arm]['argv'].copy()
        for flag in ['--outcome-eval-dir', '--wandb-run-name']:
            index = command.index(flag)
            del command[index:index+2]
        command[command.index('--output-dir')+1] = str(args.output / name / 'runtime')
        port_index = next(i for i, value in enumerate(command) if value.startswith('--master_port='))
        with socket.socket() as sock:
            sock.bind(('', 0))
            command[port_index] = f'--master_port={sock.getsockname()[1]}'
        command += ['--eval-only', '--no-wandb', '--max-val-batches', '1',
                    '--feature-export-dir', str(args.output / name / 'features')]
        if name != 'stage2':
            checkpoint = Path(phases[arm]['checkpoint'])
            for relative in ['training_state.pt', 'vision_ema.pt', 'selected_token_rows.pt',
                             'state_proj.pt', 'wm_predictor/predictor.pt', 'model.safetensors.index.json']:
                if not (checkpoint / relative).is_file():
                    raise FileNotFoundError(checkpoint / relative)
            command += ['--resume', '--resume-from', str(checkpoint)]
        commands.append({'name': name, 'argv': command})
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    if len(gpu.strip().splitlines()) != 8 or any(int(line.split(',')[1]) > 10 for line in gpu.strip().splitlines()):
        raise RuntimeError(f'GPUs are occupied: {gpu}')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    contract = {'commit': args.commit, 'started': started, 'deadline': started + 900,
                'controller_pid': os.getpid(), 'commands': commands, 'gpu': gpu,
                'scope': 'first evaluation trajectory batch per rank, eight ranks; no updates',
                'status': 'running', 'phases': []}
    status_file = args.output / 'status.json'
    def save():
        temporary = status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(contract, indent=2))
        temporary.replace(status_file)
    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled',
               PYTHONPATH=str(Path.cwd() / 'src'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    child = None
    try:
        for item in commands:
            with (args.output / (item['name'] + '.log')).open('x') as log:
                child = subprocess.Popen(item['argv'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                phase = {'name': item['name'], 'pid': child.pid, 'started': time.time()}
                contract['phases'].append(phase)
                save()
                code = child.wait(timeout=max(1, contract['deadline'] - time.time()))
                phase.update(returncode=code, finished=time.time())
                save()
                if code:
                    raise RuntimeError(f'{item["name"]} exited {code}')
        render = [commands[0]['argv'][0], 'experiments/training/sft/stage3/render_dino_feature_comparison.py']
        for name in ['stage2', 'control', 'treatment']:
            render += ['--' + name, str(args.output / name / 'features')]
        source = commands[0]['argv']
        render += ['--eval-jsonl', source[source.index('--val-jsonl')+1], '--output', str(args.output / 'figures')]
        with (args.output / 'render.log').open('x') as log:
            child = subprocess.Popen(render, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait(timeout=max(1, contract['deadline'] - time.time()))
            if code:
                raise RuntimeError(f'render exited {code}')
        contract['status'] = 'complete'
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        contract.update(status='failed', error=str(error))
        raise
    finally:
        contract['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
