"""Run the approved seven-rank query stage with verified pause/resume boundaries."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import run_segments as segments

WORLD = 7


def validate_boundary(path):
    from dataclasses import asdict

    import torch
    from cleanup_checkpoints import reject_links
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY
    from nimloth.latent import latent_state_tokens
    from nimloth.training.sft.stage1.checkpoint import (
        validate_resume_stage,
        validate_resume_state,
    )
    from nimloth.training.sft.stage1.convergence import ConvergenceState
    from nimloth.wm.grid import SharedSlotProjector

    reject_links(path)
    state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=False)
    validate_resume_stage(state, path, 'query')
    assert state['world_size'] == WORLD
    assert state['identity']['stage'] == 'query'
    assert state['identity']['convergence']['monitor'] == 'validation_total_loss'
    assert state['optimizer'] and state['scheduler']
    assert len(state['rank_rng_states']) == WORLD
    assert all({'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= r.keys()
               for r in state['rank_rng_states'])
    identity = state['identity']
    grid = json.loads((path / 'grid_state_config.json').read_text())
    assert grid['training_stage'] == 'query'
    assert grid['dino_identity'] == asdict(DINOV2_LARGE_IDENTITY)
    assert grid['objective'] == {k: identity[k] for k in
                                 ('grid_size', 'projector_hidden_dim', 'weight_lm', 'weight_dino')}
    assert grid['grid_tokens'] == identity['grid_size'] ** 2 == identity['latent_token_count']
    assert grid['state_dim'] == DINOV2_LARGE_IDENTITY.hidden_size
    assert grid['projector_hidden_dim'] == identity['projector_hidden_dim']
    assert grid['ordering'] == 'row_major' and grid['shared_slot_projector'] is True
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokens = latent_state_tokens(grid['grid_tokens'])
    assert all(token in tokenizer.get_vocab() for token in tokens)
    assert grid['query_token_ids'] == [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    assert all(tokenizer.encode(t, add_special_tokens=False) == [token_id]
               for t, token_id in zip(tokens, grid['query_token_ids'], strict=True))
    for name in ('adapter_config.json', 'tokenizer_config.json', 'preprocessor_config.json'):
        assert isinstance(json.loads((path / name).read_text()), dict)
    with safe_open(path / 'adapter_model.safetensors', framework='pt', device='cpu') as weights:
        keys = list(weights.keys())
        assert keys
        for key in keys:
            assert torch.isfinite(weights.get_tensor(key)).all(), key
    projector_state = torch.load(path / 'slot_projector.pt', map_location='cpu', weights_only=True)
    assert projector_state and all(torch.isfinite(t).all() for t in projector_state.values())
    projector = SharedSlotProjector(grid['qwen_hidden_dim'], grid['state_dim'],
                                    grid['projector_hidden_dim'], grid_tokens=grid['grid_tokens'])
    projector.load_state_dict(projector_state, strict=True)
    convergence = ConvergenceState.from_state_dict(state['convergence_state'])
    marker_path = (path.parent / f"epoch_{state['epoch']:03d}" / 'COMMITTED'
                   if path.name == 'final' else path / 'COMMITTED')
    marker = json.loads(marker_path.read_text())
    if path.name.startswith('resume_step_'):
        for rank in range(WORLD):
            validate_resume_state(state, expected_identity=state['identity'], rank=rank, world=WORLD)
        assert marker == {'schema': 'nimloth_early_stage_resume_v1', 'step': state['step']}
        assert convergence.last_epoch == state['epoch'] - 1
    else:
        assert marker == {'epoch': state['epoch'], 'step': state['step']}
        assert convergence.last_epoch == state['epoch']
    if path.name == 'final':
        epoch_state = torch.load(marker_path.parent / 'training_state.pt',
                                map_location='cpu', weights_only=False)
        assert all(state[key] == epoch_state[key] for key in
                   ('identity', 'epoch', 'step', 'best_val', 'convergence_state'))
    return {k: state[k] for k in ('step', 'epoch', 'identity')}


def find_ranks(launcher, run):
    snapshot = segments.process_snapshot()
    parents = {pid: item[0] for pid, item in snapshot.items()}
    candidates = []
    for pid in snapshot:
        try:
            proc = Path('/proc') / str(pid)
            argv = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
            if not (any(argv[i:i+2] == ['-m', 'nimloth.training.sft.stage2'] for i in range(len(argv)-1))
                    and '--output-dir' in argv and argv[argv.index('--output-dir')+1] == str(run)
                    and 'torch.distributed.run' not in argv
                    and segments.descendant(pid, launcher, parents)):
                continue
            env = dict(x.split(b'=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if b'=' in x)
            candidates.append((pid, int(env[b'LOCAL_RANK'])))
        except (OSError, KeyError, ValueError):
            continue
    ranks = [(pid, rank) for pid, rank in candidates
             if not any(other != pid and segments.descendant(pid, other, parents) for other, _ in candidates)]
    assert len(ranks) == WORLD and {r for _, r in ranks} == set(range(WORLD)), ranks
    return sorted(ranks, key=lambda x: x[1])


def cleanup_epochs(run):
    from cleanup_checkpoints import cleanup

    def validate_step(path, step, identity):
        verified = validate_boundary(path)
        assert verified['step'] == step and verified['identity'] == identity

    for epoch in sorted(run.glob('epoch_*')):
        if (epoch / 'COMMITTED').is_file():
            number = int(epoch.name.split('_')[1])
            if not (run / f'cleanup_epoch_{number:03d}_complete.json').exists():
                cleanup(run, validator=validate_boundary, through_epoch=number,
                        step_validator=validate_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    run, checkout = Path(contract['run']), Path(contract['checkout'])
    controller = run.parent / 'controller'
    if args.watch:
        import time
        while not (controller / 'watch_stop').exists():
            cleanup_epochs(run)
            time.sleep(20)
        cleanup_epochs(run)
        return
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == contract['commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout, text=True).strip()
    assert Path(contract['gate_pass']).is_file()
    run.mkdir(exist_ok=False)
    controller.mkdir(exist_ok=False)
    lock = (controller / 'lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (run / 'launch.json').write_text(json.dumps({'run_output': str(run), **contract}, indent=2))
    def event(kind, **fields):
        entry = {'time': segments.utc_now(), 'event': kind, **fields}
        with (controller / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(entry), flush=True)
    env = os.environ.copy()
    env.update(contract['env'])
    selected = [int(x) for x in env['CUDA_VISIBLE_DEVICES'].split(',')]
    assert len(selected) == WORLD and len(set(selected)) == WORLD
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()
    for row in gpu_rows:
        index, memory, utilization = map(int, row.split(','))
        if index in selected:
            assert memory < 100 and utilization == 0, row
    event('started', pid=os.getpid(), contract=contract)
    # A full validation can take longer than ten minutes. Reserve one hour.
    segments.PAUSE_SECONDS = 5 * 3600
    segments.SEGMENT_SECONDS = 6 * 3600
    with (controller / 'cleanup.log').open('x') as watcher_log:
        watcher = subprocess.Popen([sys.executable, __file__, '--contract', str(args.contract), '--watch'],
                                   env=env, stdout=watcher_log, stderr=subprocess.STDOUT)
        try:
            segment = 0
            last_step = -1
            while True:
                segment += 1
                argv = contract['argv'] + (['--resume'] if segment > 1 else [])
                log = controller / f'{segment:04d}_train.log'
                with log.open('x') as output:
                    process = subprocess.Popen(argv, cwd=checkout, env=env, stdout=output,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    event('segment_started', segment=segment, pid=process.pid, log=str(log))
                    rc, planned = segments.wait_segment(process, run, event, rank_finder=find_ranks)
                event('segment_exited', segment=segment, returncode=rc, planned_pause=planned)
                if watcher.poll() is not None:
                    raise RuntimeError('checkpoint cleanup watcher exited; inspect cleanup.log')
                if rc == 0:
                    from nimloth.training.sft.stage1.convergence import ConvergenceState
                    marker = json.loads((run / 'CONVERGED.json').read_text())
                    assert marker['monitor'] == 'validation_total_loss'
                    assert ConvergenceState.from_state_dict(marker['state']).converged
                    validate_boundary(run / 'final')
                    event('converged', marker=marker)
                    break
                if not segments.paused_exit(rc, planned, log.read_text()):
                    raise RuntimeError('unexplained segment failure; no retry')
                from nimloth.training.sft.stage1.checkpoint import (
                    find_latest_resume_dir,
                )
                checkpoint = find_latest_resume_dir(run)
                verified = validate_boundary(checkpoint)
                assert verified['step'] > last_step
                last_step = verified['step']
                event('resume_authorized', checkpoint=str(checkpoint))
        finally:
            (controller / 'watch_stop').touch()
            watcher.wait(timeout=300)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
