"""Continue the validated B pilot with its exact arguments except the step cap."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from run_segments import boundaries, new_boundary, paused_exit, utc_now, wait_segment


def prepare(root, commit):
    checkout = Path(__file__).resolve().parents[4]
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip() == commit
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout, text=True).strip()
    result = json.loads((root / 'PILOT_COMPLETE.json').read_text())
    assert result['status'] == 'pilot_complete_not_converged'
    run = root / 'train'
    assert not (run / 'CONVERGED.json').exists()
    checkpoint = new_boundary(run, {})
    assert checkpoint.name == 'resume_step_00000020'
    audit = json.loads((root / 'independent_cache_validation.json').read_text())
    assert all(audit[k] is True for k in ('all_tensors_finite', 'no_latent', 'no_truncation'))
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    argv = next(e['argv'] for e in events if e.get('phase') == 'train20' and e['event'] == 'phase_started')[:]
    index = argv.index('--max-optimizer-steps')
    assert argv[index + 1] == '20'
    del argv[index:index + 2]
    argv.append('--resume')
    assert argv[argv.index('--output-dir') + 1] == str(run)
    assert '--until-converged' in argv and '--require-prebuilt-cache' in argv
    import torch
    state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)
    for flag, key in (('--model', 'model'), ('--train-jsonl', 'train_jsonl'), ('--val-jsonl', 'val_jsonl')):
        path = Path(argv[argv.index(flag) + 1])
        assert str(path.resolve()) == state['identity'][key]
        if key.endswith('jsonl'):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == state['identity'][key + '_sha256']
    assert state['identity']['action_token_loss_weight'] == 8
    assert (root / 'base' / 'INITIALIZATION_COMPLETE').exists()
    return checkout, run, checkpoint, argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    checkout, run, checkpoint, argv = prepare(args.root, args.commit)
    if args.preflight:
        print(json.dumps({'checkpoint': str(checkpoint), 'argv': argv, 'preflight': 'passed'}))
        return
    controller = args.root / 'continuation_controller'
    controller.mkdir(exist_ok=False)
    lock = (run / '.continuation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert len(rows) == 8 and all(int(r.split(',')[0]) < 100 and int(r.split(',')[1]) == 0 for r in rows)
    def event(kind, **fields):
        entry = dict(time=utc_now(), event=kind, **fields)
        with (controller / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(entry), flush=True)
    event('started', pid=os.getpid(), commit=args.commit, checkpoint=str(checkpoint), argv=argv,
          budget='6h per segment; verified pause/resume until validation LM convergence')
    # The existing cleanup validator requires explicit output ownership metadata.
    with (run / 'launch.json').open('x') as stream:
        json.dump({'run_output': str(run), 'commit': args.commit, 'args': argv, 'resumed_from': str(checkpoint)}, stream)
    stop = controller / 'watch_stop'
    scripts = Path(__file__).parent
    env = os.environ.copy()
    env.update(PYTHONPATH=str(checkout / 'src'), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', CPATH='/mnt/nimloth/dependencies/python310-dev/root/usr/include/python3.10:/mnt/nimloth/dependencies/python310-dev/root/usr/include')
    with (controller / 'cleanup.log').open('x') as cleanup_log:
        watcher = subprocess.Popen([sys.executable, str(scripts / 'watch_checkpoints.py'), str(run), str(stop)], stdout=cleanup_log, stderr=subprocess.STDOUT, env=env)
        try:
            segment = 0
            while True:
                segment += 1
                before = boundaries(run)
                log = controller / f'{segment:04d}_train.log'
                with log.open('x') as output:
                    process = subprocess.Popen(argv, cwd=checkout, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    event('segment_started', segment=segment, pid=process.pid, log=str(log))
                    rc, planned = wait_segment(process, run, event)
                event('segment_exited', segment=segment, returncode=rc, planned_pause=planned)
                if rc == 0:
                    from nimloth.training.sft.stage1.convergence import ConvergenceState
                    marker = json.loads((run / 'CONVERGED.json').read_text())
                    assert marker['monitor'] == 'validation_lm_loss'
                    assert ConvergenceState.from_state_dict(marker['state']).converged
                    event('converged', marker=marker)
                    break
                if not paused_exit(rc, planned, log.read_text()):
                    raise RuntimeError('unexpected training failure; no automatic retry')
                boundary = new_boundary(run, before)
                event('resume_authorized', checkpoint=str(boundary))
        finally:
            stop.touch()
            watcher.wait(timeout=120)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    main()
