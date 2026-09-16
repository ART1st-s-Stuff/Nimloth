"""Single-attempt paired decoder training followed by matched reconstruction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def run(plan_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    root, worktree = Path(plan['output']), Path(plan['worktree'])
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=worktree, text=True).strip() != plan['commit']:
        raise ValueError('source commit mismatch')
    if subprocess.check_output(['git', 'status', '--porcelain', '--ignore-submodules=untracked'],
                               cwd=worktree, text=True).strip():
        raise ValueError('worktree must be clean')
    root.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'started': time.time(), 'plan': plan, 'phases': []}
    active = []

    def publish():
        pending = root / 'progress.tmp'
        pending.write_text(json.dumps(status, indent=2))
        pending.replace(root / 'progress.json')

    def stop_children():
        for process, _entry, _log in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, entry, log in active:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if entry['status'] == 'running':
                entry.update(status='cancelled', returncode=process.returncode, finished=time.time())
            log.close()

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('controller interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for group in plan['groups']:
            active = []
            deadline = time.monotonic() + group['timeout_seconds']
            for job in group['jobs']:
                log = (root / (job['name'] + '.log')).open('xb')
                environment = dict(os.environ, **plan['environment'], **job.get('environment', {}))
                process = subprocess.Popen(job['argv'], cwd=worktree, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                entry = {'name': job['name'], 'pid': process.pid, 'status': 'running',
                         'started': time.time(), 'argv': job['argv']}
                active.append((process, entry, log))
                status['phases'].append(entry)
                publish()
            while True:
                running = False
                for process, entry, _log in active:
                    code = process.poll()
                    if code is None:
                        running = True
                    elif entry['status'] == 'running':
                        entry.update(status='complete' if code == 0 else 'failed',
                                     returncode=code, finished=time.time())
                        publish()
                    if code is not None and code != 0:
                        raise RuntimeError(f"{entry['name']} failed with exit {code}")
                if not running:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('phase runtime limit reached; no automatic restart')
                time.sleep(2)
            for _process, _entry, log in active:
                log.close()
            active = []
        status['status'] = 'complete'
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        stop_children()
        raise
    finally:
        status['finished'] = time.time()
        publish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
