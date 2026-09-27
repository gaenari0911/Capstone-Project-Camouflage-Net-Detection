"""Verify heuristic mixtures and prepare, never launch, fourteen formal jobs."""
import gc
import json
from pathlib import Path
import subprocess
import sys
import torch
from .contracts import read_json, digest, code_records, contract
from .io import atomic_json, RunLock
from .epoch_preflight import verify
from .mixed import MIXTURES, AUDIT, records
from .data import ReadOnlySegDataset, effective_args, make_loader, model_batch
from .models import create_model
from .epochs import make_optimizer, epoch_update
from capstone_lab.config import sha256_file


def run(root):
    output = root/'artifacts/s8_mixed_release/run_v2'
    output.mkdir(parents=True, exist_ok=False)
    code = code_records(root)
    evidence = {}
    def record(path):
        evidence[path.relative_to(root).as_posix()] = sha256_file(path)
    with RunLock(root/'artifacts/s8_gpu.lock'):
        for exp in ('M1','L1'):
            print('VERIFY '+exp, flush=True)
            result = verify(root, output/exp, exp)
            if result['status']!='VERIFIED':
                raise ValueError('Mixture resume failed')
            record(output/exp/'summary.json')
            gc.collect(); torch.cuda.empty_cache()
        for exp in ('H0_S','H1_S'):
            print('SMOKE '+exp, flush=True)
            directory=output/exp
            directory.mkdir()
            all_rows=records(root,exp)
            real=[r for r in all_rows if r.get('kind')!='synthetic']
            synthetic=[r for r in all_rows if r.get('kind')=='synthetic']
            # Stress real instance count while guaranteeing both source types.
            real.sort(key=lambda r:len((root/r['label']).read_text().splitlines()),reverse=True)
            rows=real[:8]+synthetic[:8]
            args=effective_args(root)
            dataset=ReadOnlySegDataset(root,rows,args,augment=True)
            loader=make_loader(dataset,seed=0,workers=2,shuffle=False)
            raw=next(iter(loader))
            variant=MIXTURES[exp][3]
            model,initial=create_model(root,directory,0,variant)
            model.to('cuda:0').train()
            opt,_=make_optimizer(model)
            scaler=torch.amp.GradScaler('cuda',enabled=True)
            batch=model_batch(raw,variant,'cuda:0')
            torch.cuda.reset_peak_memory_stats()
            loss,retries=epoch_update(model,batch,opt,scaler)
            grads={name:float(p.grad.abs().sum()) for name,p in model.named_parameters() if p.grad is not None}
            if not any(v>0 for v in grads.values()): raise ValueError('No gradient')
            if variant=='H1' and (not torch.all(batch['distance_maps']==1) or not any(v>0 for k,v in grads.items() if '.cv5.' in k)):
                raise ValueError('H1 boundary gradient/target invalid')
            if variant=='H0' and hasattr(model.model[-1],'cv5'): raise ValueError('H0 still has boundary head')
            result={'status':'VERIFIED','experiment':exp,'real':8,'synthetic':8,'loss':loss,
                    'amp_retries':retries,'peak_reserved_mib':torch.cuda.max_memory_reserved()//2**20,
                    'scope':'one mixed augmented update; H0/H1 real-only epoch resume covered by previous release'}
            atomic_json(directory/'summary.json',result,immutable=True)
            record(directory/'summary.json')
            del model,opt,scaler,batch,raw,loader,dataset
            gc.collect();torch.cuda.empty_cache()
    with (output/'regression.log').open('xb') as log:
        subprocess.run([sys.executable,'-m','unittest','tests.test_s8_epochs','tests.test_s8_campaign','tests.test_s8_detached','-v'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    record(output/'regression.log')
    if code_records(root)!=code: raise ValueError('Source changed during release')
    report={'status':'VERIFIED_MIXED_EPOCH_RELEASE','code_sha256':digest(code),'evidence':evidence,
            'gpu_concurrency':1,'batch':16,'formal_jobs_started':0,
            'visual_user_review':'PENDING','scope':'M1/L1 full 2 epoch resume; H0/H1 mixed update; existing detach/checkpoint regression'}
    atomic_json(output/'summary.json',report,immutable=True)
    config=read_json(root/'configs/campaigns/s8_real_only_v1.json')
    config['inputs']={k:v for k,v in config['inputs'].items() if not k.startswith('artifacts/s8_epoch_release/')}
    for path in [output/'summary.json', root/'configs/approvals/s8_a0_repeat_exception_v1.json',
                 root/'artifacts/s8_heuristic_formal/run_v1/active_time_ledger.json',
                 *sorted((root/AUDIT).glob('*_manifest.json'))]:
        config['inputs'][path.relative_to(root).as_posix()]=sha256_file(path)
    config['output']='artifacts/s8_heuristic_training/run_v1'
    config['max_parallel']=1
    config['jobs']=[]
    previous=None
    for exp in MIXTURES:
        for seed in ((0,1,2) if exp in ('M1','L1') else (0,)):
            name=f'{exp}_seed{seed}'
            deps=[] if previous is None else [previous]
            if seed and f'{exp}_seed0' not in deps: deps.append(f'{exp}_seed0')
            config['jobs'].append({'id':name,'kind':'epoch_train','dependencies':deps,'cpu':3,
                'max_attempts':3,'duration_seconds':0,'fail_attempts':[],'partial_attempts':[],
                'training':{'experiment':exp,'seed':seed,'epochs':MIXTURES[exp][4],'stop_after_epoch':0}})
            previous=name
    config_path=output/'campaign.json'
    atomic_json(config_path,config,immutable=True)
    contract(root,config_path)
    print(json.dumps({'status':report['status'],'jobs':len(config['jobs']),'config':str(config_path)}),flush=True)


if __name__=='__main__':
    run(Path.cwd())
