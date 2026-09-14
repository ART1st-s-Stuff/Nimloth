"""Pinned original VAGEN eval200 collection; no policy training or automatic retry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

BASE = Path('/mnt/nimloth')
REPO = Path(__file__).resolve().parents[3]
PYTHON = BASE / 'venv/bin/python3'
VAGEN = BASE / 'sources/vagen'
VERL = BASE / 'sources/verl'
MODEL = BASE / 'checkpoint/hf_actor'
RUN = BASE / 'outputs/experiments/vagen-eval200/20260914_original_step60'
DATA = BASE / 'outputs/datasets/vagen-original-train-eval/20260914_eval200'
SOURCE = BASE / 'outputs/experiments/sft2-dino-information-test/20260912_epoch2_fp32_lr5e5_r2/data'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')

def environment():
    e = dict(os.environ)
    e.pop('RAY_ADDRESS', None)
    e.pop('RAY_INIT_ADDRESS', None)
    e.update(PYTHONPATH=f'{VAGEN}:{VERL}:{REPO}', VAGEN_DIR=str(VAGEN), VERL_DIR=str(VERL),
        PATH=f'{PYTHON.parent}:'+e['PATH'], PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        HF_HOME=str(BASE/'cache/huggingface'), TORCH_HOME=str(BASE/'cache/torch'),
        VLLM_USE_V1='0', VLLM_ATTENTION_BACKEND='XFORMERS', NCCL_IB_DISABLE='1',
        TORCHINDUCTOR_DISABLE='1', TORCH_COMPILE_DISABLE='1', TORCHDYNAMO_DISABLE='1',
        CUDA_DEVICE_ORDER='PCI_BUS_ID', WANDB_MODE='disabled', OMP_NUM_THREADS='4',
        CPATH=str(BASE/'dependencies/python310-dev/root/usr/include/python3.10'),
        VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
        VK_DRIVER_FILES='/usr/share/vulkan/icd.d/nvidia_icd.json',
        ORIGINAL_VALIDATION_RUNTIME_DIR=str(RUN/'runtime'),
        TRITON_CACHE_DIR=str(RUN/'runtime/triton'),
        FLASHINFER_WORKSPACE_DIR=str(RUN/'runtime/flashinfer'),
        XDG_CACHE_HOME=str(RUN/'runtime/cache'), TORCH_EXTENSIONS_DIR=str(RUN/'runtime/extensions'))
    return e

def command():
    args = '''data.train_batch_size=32 data.val_batch_size=1 data.max_prompt_length=1024
    data.max_response_length=256 data.max_trajectory_length=16000 data.truncation=left
    algorithm.adv_estimator=reinforce_plus_plus algorithm.kl_ctrl.kl_coef=0
    actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.do_sample=True actor_rollout_ref.rollout.temperature=0.7
    actor_rollout_ref.rollout.top_p=0.95 actor_rollout_ref.rollout.top_k=-1 actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_trajectory_length=6144 actor_rollout_ref.rollout.max_model_len=6144
    actor_rollout_ref.rollout.limit_mm_per_prompt=6 actor_rollout_ref.rollout.gpu_memory_utilization=0.35
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.disable_log_stats=False +actor_rollout_ref.ref.use_ref=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 trainer.logger=[console]
    trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.val_before_train=True trainer.val_only=True
    trainer.save_freq=-1 trainer.test_freq=-1 trainer.total_epochs=1 trainer.total_training_steps=1
    trainer.resume_mode=disable trainer.val_generations_to_log_to_wandb=0
    rollout_manager.max_turns=20 rollout_manager.window_size=5 rollout_manager.max_trajectory_length=16000
    rollout_manager.n_gpus_per_node=2 rollout_manager.n_trajectory=1 rollout_manager.use_service=True
    rollout_manager.timeout=1200 rollout_manager.max_workers=2 rollout_manager.use_multi_turn_reward=False'''.split()
    return [str(PYTHON), '-m', 'vagen.trainer.main_ppo', *args,
        f'data.train_files={DATA}/eval_inputs.parquet', f'data.val_files={DATA}/eval_inputs.parquet',
        f'actor_rollout_ref.model.path={MODEL}', f'trainer.default_local_dir={RUN}/unused_checkpoints',
        f'+trainer.original_validation_output_dir={RUN}/rollouts',
        'rollout_manager.base_url=http://127.0.0.1:23860']

def prepare():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from safetensors import safe_open
    assert not RUN.exists() and not DATA.exists(), 'refuse to overwrite a previous run'
    commits = {}
    for path, expected in ((REPO, None), (VAGEN, '844378ce8a5727d8274b0c7024573031f9b1296d'),
                           (VERL, '14b2453e2cdb859067fec4258b5657a89d7a3a9c')):
        head = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        assert expected is None or head == expected
        assert not subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip()
        commits[str(path)] = head
    merge = json.loads((MODEL/'nimloth_merge_manifest.json').read_text())
    assert '/global_step_60/actor' in merge['source']['source_actor_dir']
    index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    for name in set(index.values()):
        with safe_open(MODEL/name,framework='pt',device='cpu') as f:
            assert all(k in f.keys() for k,v in index.items() if v == name)
    cfg=json.loads((MODEL/'config.json').read_text())
    assert cfg['num_attention_heads']%2 == 0 and cfg['num_key_value_heads']%2 == 0
    source_hashes = {name:sha(SOURCE/name) for name in ('train.jsonl','val.jsonl')}
    assert source_hashes == {'train.jsonl':'da102cddbb70d4094bacf6268cc1f0137d262c23eff675160a4af653ac16a06f',
                            'val.jsonl':'730361cc9825ce1f65055a1a4528243ee638de861e703b89c88a20e76047215a'}
    rows = [json.loads(line) for name in source_hashes for line in (SOURCE/name).open()]
    assert len(rows)==1902 and len({r['id'] for r in rows})==1902
    configs = dict(prompt_format='grounding_worldmodeling',render_mode='vision',use_state_reward=False,
                   max_actions_per_step=1,format_reward=0.02,invalid_action_penalty=-0.2,
                   success_threshold=1.5,step_length=0.5)
    inputs=[]; assets={}
    for cat in ('base','common_sense'):
        path=VAGEN/f'vagen/env/navigation/datasets/{cat}.json'
        tasks=json.loads(path.read_text())['tasks']; assert len(tasks)==60
        assets[cat]={'path':str(path),'sha256':sha(path),'task_count':60}
        for seed in range(100):
            task=tasks[seed%60]
            inputs.append(dict(data_source='navigation',prompt=[dict(role='user',content='')],extra_info=dict(
                split='test',env_name='navigation',env_config=dict(configs,eval_set=cat),seed=seed,
                source_index=len(inputs),source_key=f'{cat}:{seed}')))
    env=environment()
    subprocess.run(command()+['--cfg','job','--resolve'],cwd=VAGEN,env=env,check=True,
                   stdout=subprocess.DEVNULL,timeout=90)
    DATA.mkdir(parents=True); RUN.mkdir(parents=True)
    with (DATA/'train.jsonl').open('x') as f:
        for r in rows:
            r['dataset_assignment']={'split':'train','previous_split':r.get('split'),
                                      'reason':'user restores all original valid rollouts to training'}
            r['split']='train'
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    pq.write_table(pa.Table.from_pylist(inputs),DATA/'eval_inputs.parquet')
    assert pq.read_table(DATA/'eval_inputs.parquet').to_pylist()==inputs
    manifest=dict(train_count=1902,eval_requested=200,eval_status='pending',assets=assets,
        source_hashes=source_hashes,source_dir=str(SOURCE),train_sha256=sha(DATA/'train.jsonl'),
        eval_inputs_sha256=sha(DATA/'eval_inputs.parquet'),selection='base/common_sense seeds 0..99 each',
        overlap='All eval tasks also occur in the original training task pool; not held-out task generalization',
        checkpoint=str(MODEL),checkpoint_merge_manifest_sha256=sha(MODEL/'nimloth_merge_manifest.json'))
    write(DATA/'manifest.json',manifest)
    write(RUN/'launch_contract.json',dict(commits=commits,python=str(PYTHON),command=command(),
        env_overrides={k:v for k,v in env.items() if k in environment() and os.environ.get(k)!=v},
        policy_gpus=[0,1],environment_gpus=[2,3],world_size=2,tp=2,policy_training=False,
        data=str(DATA),run=str(RUN),budget='12h hard deadline, one attempt, no automatic retry',
        recovery='Completed record.json is per-row commit marker; no automatic resume or overwrite',
        authorization='User requested new eval200 with designated VAGEN ckpt and original dataset as train; no Trellis task'))
    print('PREPARED',DATA,flush=True)

def run():
    contract=json.loads((RUN/'launch_contract.json').read_text())
    for tree,head in contract['commits'].items():
        assert subprocess.check_output(['git','-C',tree,'rev-parse','HEAD'],text=True).strip()==head
    assert sha(DATA/'eval_inputs.parquet')==json.loads((DATA/'manifest.json').read_text())['eval_inputs_sha256']
    assert not (RUN/'controller.json').exists()
    for port in (23860,23861,23862):
        with socket.socket() as s: s.bind(('0.0.0.0',port))
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    assert all(int(v)<100 for v in used.splitlines()[:4]), used
    env=environment(); env['VLLM_HOST_IP']='127.0.0.1'
    (RUN/'runtime').mkdir(); (RUN/'env_home/.ai2thor').mkdir(parents=True)
    (RUN/'env_home/.ai2thor/releases').symlink_to(BASE/'env_home/.ai2thor/releases')
    ee=dict(env,CUDA_VISIBLE_DEVICES='2,3',HOME=str(RUN/'env_home'),AI2THOR_HOME_ROOT=str(RUN/'env_home'))
    pe=dict(env,CUDA_VISIBLE_DEVICES='0,1',RAY_ADDRESS='127.0.0.1:23861')
    children=[]; started=time.time(); deadline=time.monotonic()+12*3600
    write(RUN/'controller.json',dict(pid=os.getpid(),started_unix=started))
    def spawn(name,cmd,environ):
        f=(RUN/f'{name}.log').open('x')
        p=subprocess.Popen(cmd,cwd=VAGEN,env=environ,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        f.close(); children.append(p)
        write(RUN/f'{name}.pid.json',dict(pid=p.pid,command=cmd))
        return p
    def stop(sig,frame): raise RuntimeError(f'interrupted by signal {sig}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    status='failed';error=None
    try:
        p=spawn('render_probe',[str(PYTHON),'-m','vagen.utils.navigation_direct_render_probe','--gpu-device','0'],ee)
        assert p.wait(timeout=180)==0, 'render probe failed'
        envp=spawn('environment',[str(PYTHON),'-m','vagen.server.server','server.port=23860',
            'use_state_reward=False','navigation.max_workers=2','navigation.devices=[0,1]'],ee)
        rayp=spawn('ray',[str(PYTHON),'-m','ray.scripts.scripts','start','--head','--block',
            '--include-dashboard=false','--node-ip-address=127.0.0.1','--port=23861','--dashboard-port=23862',
            '--temp-dir=/tmp/vagen-eval200-20260914','--num-gpus=2','--num-cpus=32'],pe)
        ready=time.monotonic()+180
        while True:
            assert envp.poll() is None and rayp.poll() is None, 'service died'
            try:
                for port in (23860,23861):
                    with socket.create_connection(('127.0.0.1',port),timeout=2): pass
                break
            except OSError:
                assert time.monotonic()<ready, 'service startup timeout'
                time.sleep(2)
        policy=spawn('policy',command(),pe)
        while policy.poll() is None:
            assert time.monotonic()<deadline, '12h deadline exceeded'
            assert envp.poll() is None and rayp.poll() is None, 'service died'
            time.sleep(5)
        assert policy.returncode==0, f'policy exited {policy.returncode}'
        records=list((RUN/'rollouts').glob('row_*/record.json'))
        assert len(records)==200, f'expected 200 records, found {len(records)}'
        status='collection_complete'
    except BaseException as exc:
        error=repr(exc)
        raise
    finally:
        for p in reversed(children):
            try: os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError: pass
        time.sleep(3)
        for p in reversed(children):
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait()
        write(RUN/'finished.json',dict(status=status,error=error,finished_unix=time.time(),
            records=len(list((RUN/'rollouts').glob('row_*/record.json')))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run'])
    {'prepare':prepare,'run':run}[parser.parse_args().mode]()
