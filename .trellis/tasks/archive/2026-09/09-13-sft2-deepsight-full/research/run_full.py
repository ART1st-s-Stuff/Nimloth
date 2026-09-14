"""Launch one audited run; pause on low disk and never automatically retry."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

contract = json.loads(Path(sys.argv[1]).read_text())
root = Path(contract['root'])
checkout = Path(contract['checkout'])
actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
if actual != contract['commit']:
    raise RuntimeError('checkout commit mismatch')
if root.exists():
    raise RuntimeError('new output root must not exist')
if shutil.disk_usage(root.parent).free < 150 * 1024**3:
    raise RuntimeError('insufficient free disk to launch')
root.mkdir()
(root/'contract.json').write_text(json.dumps(contract, indent=2))
env = dict(os.environ, **contract['env'])
with (root/'train.log').open('w') as log:
    child = subprocess.Popen(contract['train_argv'], cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT)
    (root/'started.json').write_text(json.dumps({'time':time.time(),'pid':child.pid,'controller':os.getpid()}))
    disk_pause = False
    while child.poll() is None:
        if not disk_pause and shutil.disk_usage(root).free < contract['disk_pause_free_gib'] * 1024**3:
            # Select only workers directly owned by this exact torchrun process.
            children = subprocess.check_output(['ps','--ppid',str(child.pid),'-o','pid='], text=True).split()
            for pid in children:
                cmd = Path('/proc',pid,'cmdline').read_bytes()
                if b'nimloth.training.sft.stage2' in cmd:
                    os.kill(int(pid), signal.SIGUSR1)
            disk_pause = True
            (root/'disk_pause_requested.json').write_text(json.dumps({'time':time.time()}))
        time.sleep(15)
    (root/'finished.json').write_text(json.dumps({'time':time.time(),'returncode':child.returncode,'disk_pause_requested':disk_pause}))
sys.exit(child.returncode)
