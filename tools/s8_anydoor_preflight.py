"""Release checks without formal generation/training or Test. Explicit evidence."""
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import cv2

import s8_anydoor_pipeline as pipeline
from s8_anydoor_stages import candidate_plan, polygon, score, rel
from capstone_lab.campaign.io import atomic_json,RunLock
from capstone_lab.campaign.contracts import read_json
from capstone_lab.config import sha256_file


def main():
    import torch
    from capstone_lab.campaign.models import create_model
    from capstone_lab.campaign.data import ReadOnlySegDataset,effective_args,make_loader,model_batch,manifest_records
    from capstone_lab.campaign.epochs import make_optimizer,epoch_update,save_tensor
    from capstone_lab.training.dualhead import tensor_state_hash
    root=pipeline.ROOT;out=pipeline.BASE/'verification_attempts'/uuid.uuid4().hex
    out.mkdir(parents=True,exist_ok=True)
    pipeline.freeze_judge()
    starting_binding=pipeline.source_binding()
    evidence={}
    commands=[
        [str(pipeline.TRAIN_PYTHON),str(root/'tools/test_s8_anydoor_pipeline.py'),'-v'],
        [str(pipeline.TRAIN_PYTHON),str(root/'tools/test_s8_anydoor_selection.py'),'-v'],
        [str(pipeline.TRAIN_PYTHON),'-m','unittest','tests.test_s8_epochs','tests.test_s8_campaign','tests.test_s8_detached','-v'],
        [str(pipeline.GEN_PYTHON),str(root/'tools/test_anydoor_shape_control.py'),'-v'],
        [str(pipeline.GEN_PYTHON),str(root/'tools/test_anydoor_pilot.py'),'-v']]
    for index,command in enumerate(commands):
        path=out/f'regression_{index}.log'
        if path.exists(): raise ValueError('Preserve earlier verification; inspect before reusing release')
        with path.open('xb') as stream:
            subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        evidence[rel(path)]=sha256_file(path)
    print('REGRESSIONS_PASSED',flush=True)
    plans={}
    for subset in ('train748','low187'):
        rows=candidate_plan(subset)
        if rows!=candidate_plan(subset):raise ValueError('Nondeterministic candidate planner')
        if len(rows)!=6000 or len({(r['reference'],r['background']) for r in rows})!=6000:
            raise ValueError('Plan count/pair uniqueness invalid')
        plans[subset]={'count':len(rows),'plan_sha256':pipeline.digest(rows),
            'unique_pairs':6000,'foreground_count':len({r['reference'] for r in rows})}
        # Complete source-pair plan check; pixel/label geometry checked below on12 pilot cases.
        atomic_json(out/(subset+'_plan_checks.json'),plans[subset],immutable=True)
        evidence[rel(out/(subset+'_plan_checks.json'))]=sha256_file(out/(subset+'_plan_checks.json'))
    print('CANDIDATE_PLANS_PASSED',flush=True)
    source=root/'artifacts/s8_anydoor_shape_probe/run_v1'
    records=read_json(source/'manifest.json')['records']
    fixture=[]
    for row in records:
        directory=source/'samples'/row['id'];marker=read_json(directory/'COMMITTED.json')
        for file,expected in marker['files'].items():
            if sha256_file(directory/file)!=expected:raise ValueError('Reviewed pilot changed')
        mask=cv2.imdecode(np.fromfile(directory/'input_placement_mask.png',np.uint8),0)>128
        label,roundtrip=polygon(mask)
        destination=out/'labels'/(row['id']+'.txt');destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('x',encoding='utf-8') as stream:stream.write(label)
        item={'id':row['id'],'image':rel(directory/'output.png'),'image_sha256':marker['files']['output.png'],
            'mask':rel(directory/'input_placement_mask.png'),'mask_sha256':marker['files']['input_placement_mask.png'],
            'label':rel(destination),'label_sha256':sha256_file(destination),'original_split':'train','kind':'synthetic',
            'subset':row['subset'],'domain':row['domain'],'scale_bucket':row['scale_bucket'],'reference':row['reference']}
        fixture.append(item)
    atomic_json(out/'fixture_manifest.json',fixture,immutable=True)
    evidence[rel(out/'fixture_manifest.json')]=sha256_file(out/'fixture_manifest.json')
    # Real data/geometry and same independent Judge inference code; not a formal selection.
    for subset in ('train748','low187'):
        for path in score(subset,diagnostic_rows=[r for r in fixture if r['subset']==subset],diagnostic_output=out/'judge'/subset):
            evidence[rel(path)]=sha256_file(path)
        gc.collect();torch.cuda.empty_cache()
    print('JUDGE_GPU_SMOKE_PASSED',flush=True)
    with RunLock(root/'artifacts/s8_gpu.lock'):
        args=effective_args(root)
        for subset,realset in (('train748','real748'),('low187','low187')):
            real=manifest_records(root,realset)
            real.sort(key=lambda r:len((root/r['label']).read_text().splitlines()),reverse=True)
            rows=real[:10]+[r for r in fixture if r['subset']==subset]
            directory=out/'mixed_update'/subset;directory.mkdir(parents=True)
            model,initial=create_model(root,directory,0,'H2');model.cuda().train()
            optimizer,_=make_optimizer(model);scaler=torch.amp.GradScaler('cuda',enabled=True)
            dataset=ReadOnlySegDataset(root,rows,args,augment=True)
            raw=next(iter(make_loader(dataset,seed=0,workers=2,shuffle=False)))
            batch=model_batch(raw,'H2','cuda:0')
            torch.cuda.reset_peak_memory_stats()
            loss,retries=epoch_update(model,batch,optimizer,scaler)
            before=tensor_state_hash(model.state_dict())
            checkpoint=directory/'smoke.pt';checkpoint_sha=save_tensor(checkpoint,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict()})
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
            if before!=tensor_state_hash(model.state_dict()):raise ValueError('Mixed checkpoint roundtrip failed')
            path=directory/'summary.json'
            atomic_json(path,{'status':'VERIFIED','real_images':10,'pilot_images':6,'loss':loss,'amp_retries':retries,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()//2**20,'checkpoint_sha256':checkpoint_sha,
                'scope':'one augmented mixed update and state reload; exact epoch resume remains a runtime gate after actual selection'},immutable=True)
            evidence[rel(path)]=sha256_file(path)
            del model,optimizer,scaler,dataset,raw,batch,saved;gc.collect();torch.cuda.empty_cache()
    if starting_binding!=pipeline.source_binding():raise ValueError('Code/input changed during preflight; rerun new verification attempt')
    pipeline.prepare()
    path=pipeline.BASE/'implementation_release.json'
    atomic_json(path,{'status':'VERIFIED','binding':pipeline.verify_binding(),'evidence':evidence,
        'no_formal_jobs_started':True,'no_test_inference':True,'sam2':False,
        'scope':'CPU plans/contracts/regressions;12 actual pilot Judge inferences;two augmented mixed GPU updates/checkpoint reloads; previous AnyDoor shape12; runtime exact epoch preflight before formal training',
        'limitations':['Actual SSH disconnection and reboot autostart not tested/configured.',
                      'Full new23-stage campaign is not claimed executed by preflight.',
                      'Per-output semantic accuracy is not guaranteed; user accepted approximate labels.']},immutable=True)
    print(json.dumps({'status':'VERIFIED_READY_NOT_STARTED','release':str(path)}),flush=True)


if __name__=='__main__': main()
