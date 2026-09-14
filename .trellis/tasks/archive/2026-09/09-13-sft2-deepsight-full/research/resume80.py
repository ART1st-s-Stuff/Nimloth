"""Preserve failed metrics, reconcile to committed cursor, launch once."""
import csv,json,subprocess,sys
from pathlib import Path
c=json.loads(Path('/mnt/nimloth/handoff/full-resume61.json').read_text())
root=Path('/mnt/nimloth/outputs/experiments/sft2-deepsight-full/20260913_stage1epoch2_lr2e5_r3')
train=root/'train'
assert json.loads((train/'resume_step_00000080/COMMITTED').read_text())['step']==80
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=c['checkout'],text=True).strip()==sys.argv[1]
backup=root/'train_step_log.before_resume80.csv'
assert not backup.exists()
p=train/'train_step_log.csv';backup.write_bytes(p.read_bytes())
with p.open() as f: rows=list(csv.reader(f))
retained=[rows[0]]+[r for r in rows[1:] if int(r[2])<=80]
with p.open('w',newline='') as f:csv.writer(f).writerows(retained)
c['root']=str(root/'resume_after_ram_fix_80');c['commit']=sys.argv[1]
contract=Path('/mnt/nimloth/handoff/full-resume80.json');assert not contract.exists();contract.write_text(json.dumps(c,indent=2))
with Path('/mnt/nimloth/handoff/full-resume80-controller.log').open('x') as log:
 child=subprocess.Popen(['/mnt/nimloth/venv/bin/python3','/mnt/nimloth/handoff/run_full.py',str(contract)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print('controller_pid',child.pid)
