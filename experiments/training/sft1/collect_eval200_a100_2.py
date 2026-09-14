"""Continue the migrated eval200 task using only the selected idle GPU pair."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import argparse
import collect_eval200_parallel as collector

core=collector.core
collector.ROOT=core.BASE/'outputs/experiments/vagen-eval200/20260914_a100_2_resume'
collector.GPU_PAIRS=((2,6),)
core.MODEL=core.BASE/'inputs/hf_actor'
ROOT=collector.ROOT
SNAPSHOT=core.BASE/'handoff/eval200-migration-records.json'

def prepare():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from safetensors import safe_open
    assert not ROOT.exists()
    for path,head in ((core.VAGEN,'844378ce8a5727d8274b0c7024573031f9b1296d'),
                      (core.VERL,'14b2453e2cdb859067fec4258b5657a89d7a3a9c')):
        assert subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==head
        assert not subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(core.REPO),'status','--porcelain'],text=True).strip()
    head=subprocess.check_output(['git','-C',str(core.REPO),'rev-parse','HEAD'],text=True).strip()
    index=json.loads((core.MODEL/'model.safetensors.index.json').read_text())['weight_map']
    for name in set(index.values()):
        with safe_open(core.MODEL/name,framework='pt',device='cpu') as f:
            assert all(k in f.keys() for k,v in index.items() if v==name)
    original=json.loads((core.MODEL/'nimloth_merge_manifest.json').read_text())
    assert '/global_step_60/actor' in original['source']['source_actor_dir']
    inputs=pq.read_table(core.DATA/'eval_inputs.parquet').to_pylist()
    snapshot=json.loads(SNAPSHOT.read_text());completed={}
    for name in snapshot['records']:
        p=Path(name);r=json.loads(p.read_text());idx=r['source_index']
        assert idx not in completed and r['source_key']==inputs[idx]['extra_info']['source_key']
        def check_images(x):
            if isinstance(x,dict):
                if 'image_file' in x:
                    f=x['image_file'];assert core.sha(p.parent/f['path'])==f['sha256']
                for v in x.values():check_images(v)
            elif isinstance(x,list):
                for v in x:check_images(v)
        check_images(r)
        completed[idx]={'path':name,'sha256':core.sha(p)}
    assert len(completed)==snapshot['count']==30
    remaining=[r for r in inputs if r['extra_info']['source_index'] not in completed]
    assert len(remaining)==170
    for port in list(range(24260,24265))+list(range(28000,28128)):
        with socket.socket() as s:s.bind(('0.0.0.0',port))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    assert all(int(gpu[i])<100 for i in (2,6)),gpu
    with tempfile.TemporaryDirectory(prefix='eval200-migrate-preflight-') as tmp:
        env={k:v.replace(str(ROOT),tmp) for k,v in collector.env_for(0).items()}
        subprocess.run(['cc','-x','c','-fsyntax-only','-'],input='#include <Python.h>\nint main(void){return 0;}\n',text=True,env=env,check=True)
        subprocess.run(collector.lane_command(0)+['--cfg','job','--resolve'],cwd=core.VAGEN,env=env,check=True,stdout=subprocess.DEVNULL,timeout=90)
    assert not ROOT.exists()
    lane=ROOT/'lane_0';lane.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(remaining),lane/'inputs.parquet')
    core.write(lane/'contract.json',dict(count=170,input_hash=core.sha(lane/'inputs.parquet'),
        command=collector.lane_command(0),policy_gpus=[2,6],environment_gpu=2,python=str(core.PYTHON),
        env_overrides={k:v for k,v in collector.env_for(0).items() if os.environ.get(k)!=v}))
    core.write(ROOT/'manifest.json',dict(commit=head,completed=completed,remaining=170,total=200,
        host='a100-2',checkpoint=str(core.MODEL),training_data=str(core.DATA/'train.jsonl'),
        GPU_pair=[2,6],budget='12h maximum, no auto retry',
        authorization='User requested moving collection off a100-1; preserve all other a100-2 tasks'))
    print('PREPARED migrated30 plus170remaining on GPUs2,6',flush=True)

def run():
    meta=json.loads((ROOT/'manifest.json').read_text())
    assert subprocess.check_output(['git','-C',str(core.REPO),'rev-parse','HEAD'],text=True).strip()==meta['commit']
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    assert all(int(used[i])<100 for i in (2,6)),used
    core.write(ROOT/'controller.json',dict(pid=os.getpid(),started=time.time(),host='a100-2',gpus=[2,6]))
    collector.lane_run(0)
    files=[Path(v['path']) for v in meta['completed'].values()]+list((ROOT/'lane_0/rollouts').glob('row_*/record.json'))
    ids=[json.loads(p.read_text())['source_index'] for p in files]
    assert len(ids)==200 and len(set(ids))==200
    core.write(ROOT/'records_manifest.json',dict(count=200,records=[{'path':str(p),'sha256':core.sha(p)} for p in files]))
    core.write(ROOT/'finished.json',dict(status='collection_complete',count=200,finished=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args()
    (prepare if a.mode=='prepare' else run)()
