"""Four isolated TP2 collectors resume only uncommitted eval200 rows on eight GPUs."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import collect_eval200_a100 as core

PREVIOUS=core.BASE/'outputs/experiments/vagen-eval200/20260914_parallel8'
ROOT=core.BASE/'outputs/experiments/vagen-eval200/20260914_parallel8_r2'
OLD=core.RUN

def lane_command(i):
    lane=ROOT/f'lane_{i}'
    cmd=core.command()
    replacements={'data.train_files':str(lane/'inputs.parquet'),'data.val_files':str(lane/'inputs.parquet'),
        'trainer.default_local_dir':str(lane/'unused_checkpoints'),
        '+trainer.original_validation_output_dir':str(lane/'rollouts'),
        'rollout_manager.base_url':f'http://127.0.0.1:{24260+10*i}'}
    return [s.split('=',1)[0]+'='+replacements[s.split('=',1)[0]] if '=' in s and s.split('=',1)[0] in replacements else s for s in cmd]

def env_for(i):
    lane=ROOT/f'lane_{i}'
    env={k:v.replace(str(OLD),str(lane)) for k,v in core.environment().items()}
    env['VLLM_HOST_IP']='127.0.0.1'
    return env

def prepare():
    import pyarrow as pa
    import pyarrow.parquet as pq
    assert not ROOT.exists()
    assert (OLD/'finished.json').exists(), 'old collector must be stopped'
    head=subprocess.check_output(['git','-C',str(core.REPO),'rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','-C',str(core.REPO),'status','--porcelain'],text=True).strip()
    previous=json.loads((OLD/'launch_contract.json').read_text())
    for tree,commit in previous['commits'].items():
        if tree != str(core.REPO):
            assert subprocess.check_output(['git','-C',tree,'rev-parse','HEAD'],text=True).strip()==commit
    inputs=pq.read_table(core.DATA/'eval_inputs.parquet').to_pylist()
    assert len(inputs)==200
    completed={}
    for f in sorted(list((OLD/'rollouts').glob('row_*/record.json'))+list(PREVIOUS.glob('lane_*/rollouts/row_*/record.json'))):
        r=json.loads(f.read_text());idx=r['source_index'];info=inputs[idx]['extra_info']
        assert r['source_key']==info['source_key'] and r['seed']==info['seed']
        completed[idx]={'path':str(f),'sha256':core.sha(f)}
    remaining=[r for r in inputs if r['extra_info']['source_index'] not in completed]
    shards=[remaining[i::4] for i in range(4)]
    assert len({r['extra_info']['source_index'] for rows in shards for r in rows})==len(remaining)
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert len(memory.splitlines())==8 and all(int(s)<100 for s in memory.splitlines()),memory
    mem={s.split(':')[0]:int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
    assert mem['MemAvailable']>100*1024*1024,'need at least100GiB available host RAM'
    for i in range(4):
        for port in list(range(24260+10*i,24265+10*i))+list(range(28000+128*i,28128+128*i)):
            with socket.socket() as s:s.bind(('0.0.0.0',port))
    with tempfile.TemporaryDirectory(prefix='eval200-parallel-preflight-') as temp:
        env={k:v.replace(str(ROOT),temp) for k,v in env_for(0).items()}
        subprocess.run(['cc','-x','c','-fsyntax-only','-'],input='#include <Python.h>\nint main(void){return 0;}\n',text=True,env=env,check=True)
        subprocess.run(lane_command(0)+['--cfg','job','--resolve'],cwd=core.VAGEN,env=env,check=True,stdout=subprocess.DEVNULL,timeout=90)
    assert not ROOT.exists()
    ROOT.mkdir(parents=True)
    for i,rows in enumerate(shards):
        lane=ROOT/f'lane_{i}';lane.mkdir()
        pq.write_table(pa.Table.from_pylist(rows),lane/'inputs.parquet')
        core.write(lane/'contract.json',dict(count=len(rows),input_hash=core.sha(lane/'inputs.parquet'),
            indices=[r['extra_info']['source_index'] for r in rows],command=lane_command(i),
            policy_gpus=[2*i,2*i+1],environment_gpu=2*i,ports=list(range(24260+10*i,24265+10*i)),worker_ports=[28000+128*i,28127+128*i],
            python=str(core.PYTHON),env_overrides={k:v for k,v in env_for(i).items() if os.environ.get(k)!=v}))
    core.write(ROOT/'manifest.json',dict(commit=head,completed=completed,remaining=len(remaining),
        total=200,old_run=str(OLD),training_data=str(core.DATA/'train.jsonl'),
        runtime_commits=previous['commits'],checkpoint=str(core.MODEL),
        concurrency='4 independent TP2 policies and isolated environment services, all8GPUs',
        budget='12h hard deadline; no auto retry; keep all committed records',
        authorization='User explicitly requested acceleration using8GPUs; no Trellis task'))
    print('PREPARED',len(completed),'preserved',len(remaining),'remaining',list(map(len,shards)),flush=True)

def lane_run(i):
    lane=ROOT/f'lane_{i}';cfg=json.loads((lane/'contract.json').read_text())
    assert core.sha(lane/'inputs.parquet')==cfg['input_hash']
    env=env_for(i);port=24260+10*i
    (lane/'runtime').mkdir();(lane/'env_home/.ai2thor').mkdir(parents=True)
    (lane/'env_home/.ai2thor/releases').symlink_to(core.BASE/'env_home/.ai2thor/releases')
    # Vulkan uses physical GPU ordinals; expose all devices only to the render service.
    ee=dict(env,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',HOME=str(lane/'env_home'),AI2THOR_HOME_ROOT=str(lane/'env_home'))
    pe=dict(env,CUDA_VISIBLE_DEVICES=f'{2*i},{2*i+1}',RAY_ADDRESS=f'127.0.0.1:{port+1}')
    children=[];status='failed';error=None
    def spawn(name,cmd,e):
        with (lane/f'{name}.log').open('x') as f:
            p=subprocess.Popen(cmd,cwd=core.VAGEN,env=e,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);core.write(lane/f'{name}.pid.json',dict(pid=p.pid,command=cmd));return p
    def stop(sig,frame):raise RuntimeError(f'interrupted {sig}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        p=spawn('render_probe',[str(core.PYTHON),'-m','vagen.utils.navigation_direct_render_probe','--gpu-device',str(2*i)],ee)
        assert p.wait(timeout=180)==0,'render failed'
        ep=spawn('environment',[str(core.PYTHON),'-m','vagen.server.server',f'server.port={port}',
            'use_state_reward=False','navigation.max_workers=1',f'navigation.devices=[{2*i}]'],ee)
        rp=spawn('ray',[str(core.PYTHON),'-m','ray.scripts.scripts','start','--head','--block',
            '--include-dashboard=false','--node-ip-address=127.0.0.1',f'--port={port+1}',f'--dashboard-port={port+2}',
            f'--temp-dir=/tmp/vagen-eval200-parallel8-r2-lane{i}',f'--node-manager-port={port+3}',
            f'--object-manager-port={port+4}',f'--min-worker-port={28000+128*i}',
            f'--max-worker-port={28127+128*i}','--num-gpus=2','--num-cpus=12'],pe)
        end=time.monotonic()+180
        while True:
            assert ep.poll() is None and rp.poll() is None
            try:
                for n in (port,port+1):
                    with socket.create_connection(('127.0.0.1',n),timeout=2):pass
                break
            except OSError:
                assert time.monotonic()<end,'service startup timed out'
                time.sleep(2)
        pp=spawn('policy',lane_command(i),pe);end=time.monotonic()+12*3600
        while pp.poll() is None:
            assert ep.poll() is None and rp.poll() is None,'service exited'
            assert time.monotonic()<end,'deadline exceeded'
            time.sleep(5)
        assert pp.returncode==0,f'policy exited {pp.returncode}'
        records=list((lane/'rollouts').glob('row_*/record.json'))
        assert len(records)==cfg['count'],'record count mismatch'
        status='collection_complete'
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        for p in reversed(children):
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        time.sleep(3)
        for p in reversed(children):
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            p.wait()
        core.write(lane/'finished.json',dict(status=status,error=error,finished=time.time(),records=len(list((lane/'rollouts').glob('row_*/record.json')))))

def run():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    assert subprocess.check_output(['git','-C',str(core.REPO),'rev-parse','HEAD'],text=True).strip()==manifest['commit']
    mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(v)<100 for v in mem.splitlines()),mem
    jobs=[]
    for i in range(4):
        with (ROOT/f'lane_{i}/controller.log').open('x') as f:
            jobs.append(subprocess.Popen([str(core.PYTHON),str(Path(__file__).resolve()),'lane',str(i)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
    core.write(ROOT/'controller.json',dict(pid=os.getpid(),started=time.time(),lanes=[p.pid for p in jobs]))
    def stop_all(sig,frame):
        for job in jobs:
            if job.poll() is None: job.terminate()
        for job in jobs: job.wait()
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop_all);signal.signal(signal.SIGINT,stop_all)
    for p in jobs:p.wait()
    records=[Path(v['path']) for v in manifest['completed'].values()]+list(ROOT.glob('lane_*/rollouts/row_*/record.json'))
    ids=[json.loads(p.read_text())['source_index'] for p in records]
    complete=all(p.returncode==0 for p in jobs) and len(ids)==200 and len(set(ids))==200
    core.write(ROOT/'finished.json',dict(status='collection_complete' if complete else 'partial_failure',records=len(ids),returncodes=[p.returncode for p in jobs],finished=time.time()))
    if complete:
        core.write(ROOT/'records_manifest.json',dict(records=[{'path':str(p),'sha256':core.sha(p)} for p in sorted(records)],count=200))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','lane']);p.add_argument('lane',type=int,nargs='?');a=p.parse_args()
    if a.mode=='lane':lane_run(a.lane)
    elif a.mode=='prepare':prepare()
    else:run()
