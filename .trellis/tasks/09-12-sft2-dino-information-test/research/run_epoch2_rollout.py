"""One bounded standard Stage2 rollout evaluation; owns and reaps its server."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

from run_test import run_phase, process_snapshot, remember_owned, terminate_group, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    root = Path(contract['root'])
    checkout = Path(contract['checkout'])
    budget = contract['total_seconds']
    if not 0 < budget <= 7200:
        raise ValueError('rollout budget must be at most two hours')
    if args.check_only:
        print(json.dumps(contract, indent=2))
        return
    deadline = time.monotonic() + budget
    logs = root / 'rollout_epoch002_controller'
    logs.mkdir(exist_ok=False)
    (logs / 'contract.json').write_text(json.dumps(contract, indent=2))

    def event(kind, **details):
        with (logs / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=utc_now(), event=kind, **details)) + '\n')

    def stop(_signum, _frame):
        raise InterruptedError('rollout controller interrupted')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    env = os.environ.copy()
    env.update(contract['env'])
    server = None
    owned = {}
    done = threading.Event()
    watcher = None
    try:
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
        if actual != contract['commit'] or subprocess.check_output(
                ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=checkout, text=True).strip():
            raise ValueError('evaluation checkout differs from verified clean commit')
        event('waiting_for_dino')
        while True:
            if time.monotonic() >= deadline - 80:
                raise TimeoutError('DINO dependency did not finish within rollout budget')
            complete = all(Path(path).is_file() for path in contract['requires'])
            compute = subprocess.check_output([
                'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'
            ], text=True, timeout=15).strip()
            if complete and not compute:
                break
            time.sleep(5)
        model = root / 'rollout_epoch002_model'
        if not (model / 'config.json').is_file() or not (model / 'grid_state_config.json').is_file():
            raise ValueError('exported Stage2 model missing')
        # Claim an unused dedicated port; never reuse or stop another server.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', contract['port']))
        with (logs / 'environment.log').open('x') as stream:
            server = subprocess.Popen(contract['server_argv'], cwd=checkout, env=env,
                                      stdout=stream, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        event('server_started', pid=server.pid)

        def track_server():
            while not done.wait(0.5):
                remember_owned(server.pid, process_snapshot(), owned)

        remember_owned(server.pid, process_snapshot(), owned)
        watcher = threading.Thread(target=track_server, daemon=True)
        watcher.start()
        ready_deadline = min(deadline - 80, time.monotonic() + 180)
        while True:
            if server.poll() is not None:
                raise RuntimeError('environment server exited during startup')
            if time.monotonic() >= ready_deadline:
                raise TimeoutError('environment server startup timed out')
            try:
                with socket.create_connection(('127.0.0.1', contract['port']), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        # Keep an additional cleanup allowance for the environment after run_phase.
        run_phase(contract['rollout_argv'], phase='rollout', checkout=checkout,
                  env=env, logs=logs, deadline=deadline - 40, event=event)
    except BaseException as error:
        event('failed', error=repr(error))
        raise
    finally:
        done.set()
        if watcher is not None:
            watcher.join(timeout=2)
        if server is not None:
            terminate_group(server, owned=owned)
    event('completed')
    temporary = logs / 'COMPLETED.json.tmp'
    temporary.write_text(json.dumps({'time': utc_now()}))
    temporary.replace(logs / 'COMPLETED.json')


if __name__ == '__main__':
    main()
