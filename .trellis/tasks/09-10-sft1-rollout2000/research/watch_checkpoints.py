"""Best-effort cleanup monitor; errors are recorded without stopping training."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def watch(run, stop):
    attempted = set()
    while True:
        for path in sorted(run.glob('epoch_*')):
            if not re.fullmatch(r'epoch_[0-9]{3,}', path.name):
                continue
            if not (path/'COMMITTED').is_file() or path.name in attempted:
                continue
            attempted.add(path.name)
            epoch = int(path.name.split('_')[1])
            if (run/f'cleanup_epoch_{epoch:03d}_complete.json').is_file():
                continue
            result = subprocess.run([sys.executable, str(Path(__file__).with_name(
                'cleanup_checkpoints.py')), str(run), '--through-epoch', str(epoch)],
                check=False)
            print(json.dumps({'epoch_cleanup': path.name, 'exit_code': result.returncode}), flush=True)
        if stop.exists():
            return
        time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('stop', type=Path)
    args = parser.parse_args()
    watch(args.run, args.stop)
