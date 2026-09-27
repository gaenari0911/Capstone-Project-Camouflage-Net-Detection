import argparse,copy,gc,json,math,os,shutil,sys,time,traceback,types,uuid
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
BASE=ROOT/'artifacts/s9_background/run_v1';SELF=Path(__file__).resolve()
EXPS={'BG0':'M0','BG1':'M1'}
from capstone_lab.campaign.contracts import read_json,digest,code_records
from capstone_lab.campaign.io import atomic_json,RunLock
from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError

def rel(p):return Path(p).relative_to(ROOT).as_posix()
def stages():return [f'train_{e}_seed{s}' for e in EXPS for s in range(3)]+['freeze','test','report']
def guard():
    if shutil.disk_usage(ROOT).free<20*1024**3:raise ArtifactError('Free disk below20GiB')
def code_binding():
    result=code_records(ROOT)
    for p in [SELF,ROOT/'tools/s9_background_data.py',ROOT/'tools/s9_background_report.py',ROOT/'tools/s8_anydoor_pipeline.py',BASE/'plan.json',BASE/'backgrounds.json',BASE/'labels/empty.txt',ROOT/'artifacts/s8_full_backgrounds/v1/manifest.json',ROOT/'artifacts/s8_heuristic_audit/run_v1/train748_A1_manifest.json',ROOT/'artifacts/s8_anydoor_campaign/run_v1/final_freeze.json',ROOT/'artifacts/s8_anydoor_campaign/run_v1/test/summary.json',ROOT/'manifests/real_split_observed_v1.jsonl']:
        result[rel(p)]=sha256_file(p)
    return result
def verify():
    b=read_json(BASE/'binding.json')
    for p,h in b.items():
        if sha256_file(ROOT/p)!=h:raise ArtifactError('S9 input/code changed: '+p)
    return digest(b)
def prepare():
    if (BASE/'implementation_release.json').exists() or (BASE/'jobs').exists():return {'binding':verify(),'status':'ALREADY_FROZEN'}
    BASE.mkdir(parents=True,exist_ok=True);(BASE/'labels').mkdir(exist_ok=True)
    label=BASE/'labels/empty.txt'
    if not label.exists():label.write_text('',encoding='utf-8')
    source=read_json(ROOT/'artifacts/s8_full_backgrounds/v1/manifest.json')['backgrounds']
    chosen=sorted([r for r in source if Path(r['background_id']).suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}],key=lambda r:r['background_id'])
    assert len(chosen)==2129 and len({r['sha256'] for r in chosen})==2129
    real=[json.loads(x) for x in (ROOT/'manifests/real_split_observed_v1.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    assert not {r['sha256'] for r in chosen}&{r['image_sha256'] for r in real}
    from PIL import Image,ImageDraw
    records=[]
    for r in chosen:
        if sha256_file(ROOT/r['path'])!=r['sha256']:raise ArtifactError('Background integrity failure')
        with Image.open(ROOT/r['path']) as im:im.verify()
        records.append(dict(image=r['path'],image_sha256=r['sha256'],label=rel(label),label_sha256=sha256_file(label),kind='background_only',domain=r['domain'],original_split='train',background_id=r['background_id']))
    sheet=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(sheet)
    samples=[r for d in sorted({r['domain'] for r in records}) for r in [q for q in records if q['domain']==d][:6]]
    for i,r in enumerate(samples):
        with Image.open(ROOT/r['image']) as im:
            im=im.convert('RGB');im.thumbnail((195,120));x=(i%6)*200;y=(i//6)*150;sheet.paste(im,(x,y));draw.text((x,y+123),r['domain'],fill='black')
    sheet.save(BASE/'background_contact_sheet.jpg')
    atomic_json(BASE/'backgrounds.json',records,immutable=True)
    atomic_json(BASE/'plan.json',{'status':'USER_AUTHORIZED_POSTHOC_SUPPLEMENT','backgrounds':2129,'background_domains':dict(Counter(r['domain'] for r in records)),
        'historical_evidence':'S3 initial scan5811=748real+2934synth+2129background; resume epoch140 scan5667/1985background. S4 training membership unknown.',
        'historical_match':'same archive and initial extension rule/count; full historical per-image membership not proven',
        'experiments':{'BG0':{'real':748,'synthetic':0,'background':2129},'BG1':{'real':748,'synthetic':3000,'background':2129}},
        'controls':['M0','M1'],'seeds':[0,1,2],'epochs':150,'architecture':'custom H2 DualHeadSegment n','jobs':6,'gpu_concurrency':1,
        'sampling':'one visit per image per epoch; shuffle; fixed absolute background count, not fixed negative fraction',
        'evaluation':'Val100 best; reused Test200 after all6 complete and freeze; original fixed thresholds',
        'negative_holdout':'NONE; backgrounds may already occur in synthesis; no unseen-negative specificity claim',
        'annotation_limit':'background target absence assumed from archive; sample review not exhaustive annotation',
        'scope':'independent pretrained learning, current recipe; no unrelated experiments',
        'user_authorization':'User delegated background experiment execution, requesting historical background count match'},immutable=True)
    if (BASE/'binding.json').exists():
        d=BASE/'draft_bindings';d.mkdir(exist_ok=True);shutil.copy2(BASE/'binding.json',d/(uuid.uuid4().hex+'.json'))
    atomic_json(BASE/'binding.json',code_binding())
    return {'status':'PREPARED_NOT_STARTED','backgrounds':2129,'binding':verify()}
def training_rows(exp,smoke=False):
    from capstone_lab.campaign.data import manifest_records
    real=manifest_records(ROOT,'real748');bg=read_json(BASE/'backgrounds.json')
    syn=read_json(ROOT/'artifacts/s8_heuristic_audit/run_v1/train748_A1_manifest.json') if exp=='BG1' else []
    for r in bg+syn:
        for k in ['image','label']:
            if sha256_file(ROOT/r[k])!=r[k+'_sha256']:raise ArtifactError('Changed source')
    return real[:16]+bg[:16]+syn[:16] if smoke else real+bg+syn
def train(exp,seed,directory,smoke=False,stop_after=None):
    import capstone_lab.campaign.epochs as ep
    from capstone_lab.campaign.data import manifest_records
    from s9_background_data import BackgroundDataset
    rows=training_rows(exp,smoke)
    def manifest(root,subset):
        if subset=='val100':return manifest_records(root,subset)
        if subset in ['bg0','bg1','low187']:return copy.deepcopy(rows)
        raise ArtifactError('Unexpected subset')
    def binding(root,experiment,seed,epochs,smoke):
        b=ep.training_binding(root,'M0',seed,epochs,smoke)
        b.update(experiment=experiment,scope='S9_PREFLIGHT' if smoke else 'S9_FORMAL_BACKGROUND',training_subset={'experiment':exp,'backgrounds':2129,'supplement_binding':verify(),'diagnostic':smoke})
        return b
    ns=dict(ep.train_epochs.__globals__)
    ns.update(REAL_EXPERIMENTS={'BG0':('bg0','H2'),'BG1':('bg1','H2')},training_binding=binding,manifest_records=manifest,ReadOnlySegDataset=BackgroundDataset)
    f=types.FunctionType(ep.train_epochs.__code__,ns,'s9_train',ep.train_epochs.__defaults__,ep.train_epochs.__closure__);f.__kwdefaults__=ep.train_epochs.__kwdefaults__
    def pause():guard();return (BASE/'pause.request').exists()
    return f(ROOT,directory,experiment=exp,seed=seed,epochs=2 if smoke else 150,smoke=smoke,stop_after=stop_after,should_pause=pause,progress_path=directory/'progress.json')

def preflight():
    verify();guard()
    import torch
    from capstone_lab.campaign.data import effective_args,make_loader,model_batch
    from capstone_lab.campaign.models import create_model
    from capstone_lab.campaign.epochs import make_optimizer,epoch_update
    from capstone_lab.campaign.epoch_preflight import equal_state
    from s9_background_data import BackgroundDataset
    d=BASE/'preflight'/uuid.uuid4().hex;d.mkdir(parents=True);evidence={}
    with RunLock(ROOT/'artifacts/s8_gpu.lock'):
        args=effective_args(ROOT);bg=read_json(BASE/'backgrounds.json')[:16]
        ds=BackgroundDataset(ROOT,bg,args,augment=True);batch=next(iter(make_loader(ds,seed=0,workers=2)))
        assert len(batch['cls'])==0 and batch['masks'].shape[0]==0
        model,_=create_model(ROOT,d,0,'H2');model.cuda().train();model.args=args
        opt,_=make_optimizer(model);scaler=torch.amp.GradScaler('cuda',enabled=True)
        loss,retries=epoch_update(model,model_batch(batch,'H2','cuda'),opt,scaler)
        assert math.isfinite(loss)
        bad=dict(bg[0]);bad['kind']='real'
        try:BackgroundDataset(ROOT,[bad],args,augment=False)
        except ArtifactError:pass
        else:raise ArtifactError('Positive empty-label guard bypassed')
        atomic_json(d/'negative_batch.json',{'status':'VERIFIED','images':16,'instances':0,'loss':loss,'retries':retries,'positive_empty_rejected':True})
        evidence[rel(d/'negative_batch.json')]=sha256_file(d/'negative_batch.json')
        del model,opt,scaler,batch,ds;gc.collect();torch.cuda.empty_cache()
        for exp in EXPS:
            p=d/exp;a=train(exp,0,p/'continuous',True);gc.collect();torch.cuda.empty_cache()
            paused=train(exp,0,p/'resumed',True,1);assert paused['status']=='PAUSED';gc.collect();torch.cuda.empty_cache()
            b=train(exp,0,p/'resumed',True);states=[]
            for name in ['continuous','resumed']:
                ptr=read_json(p/name/'current.json');states.append(torch.load(p/name/ptr['last']['file'],map_location='cpu',weights_only=False))
            checks={k:equal_state(states[0][k],states[1][k]) for k in ['model','optimizer','scaler','scheduler','rng','initial']}
            for k in ['loss','data_stream_sha256']:checks[k]=[h[k] for h in a['history']]==[h[k] for h in b['history']]
            checks['best']=all(a['best_mask'][k]==b['best_mask'][k] for k in ['epoch','score'])
            if not all(checks.values()):raise ArtifactError('Exact background resume failed')
            atomic_json(p/'summary.json',{'status':'VERIFIED','checks':checks,'diagnostic_images':32 if exp=='BG0' else 48})
            evidence[rel(p/'summary.json')]=sha256_file(p/'summary.json');del states;gc.collect();torch.cuda.empty_cache()
    atomic_json(BASE/'implementation_release.json',{'status':'VERIFIED','binding':verify(),'evidence':evidence,'formal_jobs_started':0},immutable=True)
    return {'status':'VERIFIED','path':rel(d)}
def freeze():
    items=[]
    for exp in EXPS:
        for seed in range(3):
            job=f'{exp}_seed{seed}';d=BASE/'jobs'/job/'training_state';r=read_json(d/'result.json');best=r['best_mask'];p=d/best['file']
            if r['status']!='SUCCEEDED' or len(r['history'])!=150 or sha256_file(p)!=best['sha256'] or best['score']!=max(h['mask_map50_95'] for h in r['history']):raise ArtifactError('Invalid completed checkpoint')
            items.append(dict(job=job,experiment=exp,seed=seed,checkpoint=rel(p),sha256=best['sha256'],best_epoch=best['epoch'],validation=next(h['metrics'] for h in r['history'] if h['epoch']==best['epoch'])))
    p=BASE/'final_freeze.json';atomic_json(p,{'status':'FROZEN_BEFORE_SUPPLEMENT_TEST','post_hoc':True,'binding':verify(),'checkpoints':items,'conf':.001,'fixed_conf':.25,'mask_iou':.5,'nms_iou':.7},immutable=True);return [p]
def evaluate():
    import torch
    from capstone_lab.campaign.models import create_model,validate_masks
    from capstone_lab.campaign.data import effective_args,ReadOnlySegDataset,make_loader
    rows=[json.loads(x) for x in (ROOT/'manifests/real_split_observed_v1.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    rows=[r for r in rows if r['original_split']=='test'];assert len(rows)==200
    for r in rows:
        for k in ['image','label']:
            if sha256_file(ROOT/r[k])!=r[k+'_sha256']:raise ArtifactError('Test source changed')
    records=[];torch.set_num_threads(1)
    with RunLock(ROOT/'artifacts/s8_gpu.lock'):
        for r in read_json(BASE/'final_freeze.json')['checkpoints']:
            d=BASE/'test'/r['job'];d.mkdir(parents=True,exist_ok=True);p=d/'result.json'
            if p.exists():
                saved=read_json(p)
                if saved['freeze_sha256']!=sha256_file(BASE/'final_freeze.json'):raise ArtifactError('Test freeze changed')
            else:
                if sha256_file(ROOT/r['checkpoint'])!=r['sha256']:raise ArtifactError('Checkpoint changed')
                model,_=create_model(ROOT,d,r['seed'],'H2');state=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True);del state
                model.cuda().eval();args=effective_args(ROOT);ds=ReadOnlySegDataset(ROOT,rows,args,augment=False)
                m=validate_masks(model,make_loader(ds,seed=0,workers=2),d,args);assert m['images']==200
                saved={**r,'metrics':m,'freeze_sha256':sha256_file(BASE/'final_freeze.json')};atomic_json(p,saved,immutable=True)
                del model,ds;gc.collect();torch.cuda.empty_cache()
            records.append(saved)
    p=BASE/'test/summary.json';atomic_json(p,{'post_hoc':True,'records':records},immutable=True);return [p]+[BASE/'test'/r['job']/'result.json' for r in records]
def report():
    from s9_background_report import generate
    return generate(ROOT,BASE)
def controller():
    import s8_anydoor_pipeline as p
    p.BASE=BASE;p.__file__=str(SELF);p.stages=stages;p.verify_binding=verify;p.resource_guard=guard
    return p
def work(stage):
    p=controller()
    with RunLock(BASE/'worker.lock'):
        verify();guard();index=stages().index(stage)
        if any(not p.is_done(s) for s in stages()[:index]):raise ArtifactError('Unfinished dependency')
        if p.is_done(stage):return
        if stage.startswith('train_'):
            exp,seed=stage[6:].split('_seed');d=BASE/'jobs'/f'{exp}_seed{seed}'/'training_state';release=read_json(BASE/'implementation_release.json')
            if release['binding']!=verify() or release['status']!='VERIFIED':raise ArtifactError('Missing release')
            for f,h in release['evidence'].items():
                if sha256_file(ROOT/f)!=h:raise ArtifactError('Preflight evidence changed')
            with RunLock(ROOT/'artifacts/s8_gpu.lock'):r=train(exp,int(seed),d)
            if r['status']!='SUCCEEDED':raise ArtifactError('Paused before completion')
            files=[d/'result.json',d/'current.json',d/r['best_mask']['file']]
        else:files={'freeze':freeze,'test':evaluate,'report':report}[stage]()
        p.mark_done(stage,files)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','preflight','start','serve','watchdog','work','status']);ap.add_argument('--stage');a=ap.parse_args()
    if a.action=='prepare':r=prepare()
    elif a.action=='preflight':r=preflight()
    elif a.action=='work':
        try:r=work(a.stage)
        except Exception as exc:
            atomic_json(BASE/'errors'/f'{a.stage}.json',{'error':traceback.format_exc(),'retryable':isinstance(exc,(PermissionError,TimeoutError,ConnectionError)),'updated':time.time()});raise
    elif a.action=='status':r=read_json(BASE/'status.json') if (BASE/'status.json').exists() else {'status':'NOT_STARTED'}
    else:r=getattr(controller(),a.action)()
    if r is not None:print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
