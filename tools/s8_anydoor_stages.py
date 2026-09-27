"""Real adapters for the single-entry steps6-9 campaign; no SAM2 or cloud calls."""
from collections import Counter, defaultdict
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import uuid

from s8_anydoor_pipeline import ROOT, BASE, APPROVAL, verify_binding, resource_guard
from capstone_lab.campaign.contracts import read_json, inside, digest, code_records
from capstone_lab.campaign.io import RunLock, atomic_json
from capstone_lab.config import sha256_file


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()


def verify_files(files):
    for file, expected in files.items():
        if sha256_file(inside(ROOT, file)) != expected:
            raise ValueError('File hash mismatch: ' + file)


def pilot():
    import anydoor_pilot as api
    run = BASE / 'pilot'
    manifest = deepcopy(read_json(ROOT / 'artifacts/s8_anydoor_pilot/run_v2/manifest.json'))
    manifest['settings']['shape_control'] = True
    manifest['pilot_user_approval'] = 'ACCEPTED_METHOD_RESIDUAL_ERROR_DISCLOSED'
    manifest['annotation_status'] = 'USER_ACCEPTED_APPROXIMATE_PLACEMENT_LABEL_NOT_INDEPENDENT_GT'
    atomic_json(run / 'manifest.json', manifest, immutable=True)
    api.execute(run, 100)
    if read_json(run / 'status.json')['completed'] != 100:
        raise ValueError('Pilot count failure')
    for row in manifest['records']:
        directory = run / 'samples' / row['id']
        marker = read_json(directory / 'COMMITTED.json')
        for file, expected in marker['files'].items():
            if sha256_file(directory / file) != expected: raise ValueError('Pilot corruption')
    atomic_json(run / 'technical_review.json', {'count': 100, 'status': 'TECHNICAL_PASS',
        'quality_authority': rel(APPROVAL), 'new100_semantically_reviewed': False,
        'annotation': 'user-approved approximate labels; no automated semantic accuracy claim'}, immutable=True)
    return [run / 'technical_review.json', run / 'manifest.json', run / 'binding.json']


def candidate_plan(subset):
    """Two deterministic3000 schedules, balanced allowed objects, no repeated pair."""
    provenance_path = ROOT / f'artifacts/s8_source_audit/split_pools_v1/{subset}_object_provenance.json'
    proof = read_json(provenance_path)
    objects = proof['records']
    if proof['status'] != 'VERIFIED' or len(objects) != (187 if subset == 'low187' else 748):
        raise ValueError('Source pool contract mismatch')
    backgrounds = read_json(ROOT / 'artifacts/s8_full_backgrounds/v1/manifest.json')['backgrounds']
    bg = {r['background_id']: r for r in backgrounds}
    plans = read_json(ROOT / f'artifacts/s8_heuristic_formal/run_v1/datasets/{subset}/plan.json')['plans']
    if len(plans) != 3000: raise ValueError('Wrong original schedule size')
    uses = Counter(); pairs = set(); result = []
    for i in range(6000):
        plan = plans[i % 3000]
        background = bg[plan['background_id']]
        identifier = f'{subset}_{i+1:06d}'
        def order(obj):
            key = obj['object_image']
            return uses[key], hashlib.sha256((identifier + ':' + key).encode()).hexdigest()
        eligible = [r for r in objects if r['environment'] == background['environment']
                    and (r['object_image'], background['path']) not in pairs]
        if not eligible: raise ValueError('Unique object/background capacity exhausted')
        fg = min(eligible, key=order)
        uses[fg['object_image']] += 1
        pairs.add((fg['object_image'], background['path']))
        files = {fg['object_image']: fg['object_image_sha256'], fg['object_mask']: fg['object_mask_sha256'],
                 fg['source_image']: fg['source_sha256'], fg['label']: fg['label_sha256'],
                 background['path']: background['sha256']}
        result.append({'id': identifier, 'subset': subset, 'domain': background['domain'],
            'environment': fg['environment'], 'scale_bucket': plan['scale_bucket'], 'scale_ratio': plan['scale_ratio'],
            'seed': int(hashlib.sha256(('anydoor-formal-v1:' + identifier).encode()).hexdigest()[:8],16),
            'reference': fg['object_image'], 'reference_mask': fg['object_mask'], 'source_image': fg['source_image'],
            'background': background['path'], 'source_hashes': files,
            'provenance_path': rel(provenance_path), 'provenance_sha256': sha256_file(provenance_path)})
    return result


def polygon(mask):
    import cv2
    import numpy as np
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if len(c) >= 3 and cv2.contourArea(c) > 0]
    reconstructed = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(reconstructed, contours, -1, 1, cv2.FILLED)
    union = np.count_nonzero((mask > 0) | (reconstructed > 0))
    score = np.count_nonzero((mask > 0) & (reconstructed > 0)) / union if union else 0
    if score < .90: raise ValueError('Polygon/mask roundtrip below0.90; do not generate invalid labels')
    h,w=mask.shape
    return '\n'.join('0 ' + ' '.join(f'{v:.9f}' for v in (c[:,0,:]/np.array([w,h])).reshape(-1)) for c in contours)+'\n', score


def generate(subset):
    import cv2
    import numpy as np
    import torch
    import anydoor_pilot as api
    from anydoor_runtime import load_model, generate as synthesize
    torch.set_num_threads(4); cv2.setNumThreads(1)
    run = BASE / 'pools' / subset
    rows = candidate_plan(subset)
    manifest_path = run / 'plan.json'
    atomic_json(manifest_path, {'records': rows, 'shape_control': True, 'sam2': False,
        'sampling': 'repeated frozen domain/scale schedule; minimum-use allowed source, unique source/background pair',
        'placement': 'same centered aspect-preserving input mask policy as user-reviewed pilot'}, immutable=True)
    binding = verify_binding()
    completed = []
    with RunLock(ROOT / 'artifacts/s8_gpu.lock'):
        if api.budget()['free_bytes'] < 20*2**30: raise ValueError('Disk guard')
        from capstone_lab.campaign.resources import gpu_memory
        if gpu_memory()[1] > 2500: raise ValueError('GPU occupied; do not overlap training/generation')
        runtime = load_model(api.SOURCE, api.WEIGHTS)
        for index,row in enumerate(rows):
            directory=run/'samples'/row['id']
            marker=directory/'COMMITTED.json'
            if marker.exists():
                proof=read_json(marker)
                if proof['binding']!=binding: raise ValueError('Changed generation contract')
                for file,expected in proof['files'].items():
                    if sha256_file(directory/file)!=expected: raise ValueError('Completed image changed')
                completed.append(proof['row']); continue
            if index%20==0: resource_guard()
            verify_files(row['source_hashes'])
            reference,mask,background,target=api.inputs(row)
            label,roundtrip=polygon(target)
            image,info=synthesize(runtime,reference,mask,background,target,row['seed'],shape_control=True)
            if image.shape!=background.shape or np.array_equal(image,background): raise ValueError('Invalid generation')
            attempt=run/'attempts'/(row['id']+'_'+uuid.uuid4().hex)
            attempt.mkdir(parents=True)
            api.save_png(attempt/'image.png',image)
            api.save_png(attempt/'mask.png',target*255,rgb=False)
            with (attempt/'polygon.txt').open('x',encoding='utf-8') as stream: stream.write(label)
            atomic_json(attempt/'metadata.json',{**row,**info,'polygon_roundtrip_iou':roundtrip,
                'annotation_status':'USER_ACCEPTED_APPROXIMATE_INPUT_MASK_NOT_INDEPENDENT_GT'},immutable=True)
            training_row={'id':row['id'],'kind':'synthetic','original_split':'train','subset':subset,
                'domain':row['domain'],'scale_bucket':row['scale_bucket'],'reference':row['reference']}
            for key,name in (('image','image.png'),('mask','mask.png'),('label','polygon.txt')):
                training_row[key]=rel(directory/name); training_row[key+'_sha256']=sha256_file(attempt/name)
            proof={'binding':binding,'row':training_row,'files':{p.name:sha256_file(p) for p in attempt.iterdir()}}
            atomic_json(attempt/'COMMITTED.json',proof,immutable=True)
            directory.parent.mkdir(parents=True,exist_ok=True)
            if directory.exists(): raise ValueError('Preserve incomplete destination; inspection required')
            attempt.rename(directory)
            completed.append(training_row)
            atomic_json(run/'progress.json',{'completed':index+1,'total':6000,'updated':time.time()})
        del runtime; gc.collect(); torch.cuda.empty_cache()
    atomic_json(run/'manifest.json',completed,immutable=True)
    return [manifest_path,run/'manifest.json']


def audit():
    import cv2
    import numpy as np
    reports={}
    for subset in ('train748','low187'):
        run=BASE/'pools'/subset
        rows=read_json(run/'manifest.json'); plan=read_json(run/'plan.json')['records']
        if len(rows)!=6000 or len({r['id'] for r in rows})!=6000: raise ValueError('Pool count mismatch')
        allowed={r['object_image'] for r in read_json(ROOT/f'artifacts/s8_source_audit/split_pools_v1/{subset}_object_provenance.json')['records']}
        hashes=set(); pairs=set()
        for row,source in zip(rows,plan):
            if row['id']!=source['id'] or row['reference'] not in allowed: raise ValueError('Source identity mismatch')
            verify_files(source['source_hashes'])
            for key in ('image','mask','label'):
                if sha256_file(ROOT/row[key])!=row[key+'_sha256']: raise ValueError('Output integrity failure')
            if row['image_sha256'] in hashes: raise ValueError('Exact duplicate generation; selection must not proceed')
            hashes.add(row['image_sha256'])
            pair=(source['reference'],source['background'])
            if pair in pairs: raise ValueError('Repeated source/background')
            pairs.add(pair)
            mask=cv2.imdecode(np.fromfile(ROOT/row['mask'],np.uint8),0)>128
            label,_=polygon(mask)
            if (ROOT/row['label']).read_text(encoding='utf-8')!=label: raise ValueError('Label/mask disagreement')
        reports[subset]={'count':len(rows),'exact_duplicates':0,'repeated_pairs':0,
            'domains':dict(Counter(r['domain'] for r in rows)), 'objects':dict(Counter(r['reference'] for r in rows)),
            'annotation':'approximate user-accepted labels; no per-image semantic verification claim'}
    path=BASE/'pool_audit.json'
    atomic_json(path,{'status':'TECHNICAL_PASS','pools':reports,'sam2':False},immutable=True)
    return [path]


def score(subset, diagnostic_rows=None, diagnostic_output=None):
    import torch
    import numpy as np
    from ultralytics.models.yolo.segment.val import SegmentationValidator
    from capstone_lab.campaign.models import create_model
    from capstone_lab.campaign.data import ReadOnlySegDataset,effective_args,make_loader
    from s8_anydoor_selection import task_score
    judge=read_json(BASE/'judge_manifest.json')
    checkpoint=ROOT/judge['checkpoint']
    if sha256_file(checkpoint)!=judge['checkpoint_sha256']: raise ValueError('Frozen Judge changed')
    directory=BASE/('scores' if diagnostic_rows is None else 'implementation_preflight/judge')/subset
    if diagnostic_rows is not None and diagnostic_output is not None:
        directory=diagnostic_output
    directory.mkdir(parents=True,exist_ok=True)
    rows=read_json(BASE/'pools'/subset/'manifest.json') if diagnostic_rows is None else diagnostic_rows
    if diagnostic_rows is None and len(rows)!=6000: raise ValueError('Expected6000 candidates')
    if diagnostic_rows is not None and not 1<=len(rows)<=16: raise ValueError('Bounded diagnostic only')
    with RunLock(ROOT/'artifacts/s8_gpu.lock'):
        torch.set_num_threads(1)
        model,_=create_model(ROOT,directory,0,'H2')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True); del saved
        model.cuda().eval(); args=effective_args(ROOT)
        dataset=ReadOnlySegDataset(ROOT,rows,args,augment=False)
        validator=SegmentationValidator(save_dir=directory/'validation',args=vars(args))
        validator.device=torch.device('cuda:0');validator.training=True
        validator.data={'val':'candidate-pool-NOT-Test','names':{0:'camouflage'}}
        validator.init_metrics(model)
        results=[]; position=0
        with torch.inference_mode():
            for raw in make_loader(dataset,seed=0,workers=2):
                batch=validator.preprocess(raw)
                predictions=validator.postprocess(model(batch['img']))
                for index,pred in enumerate(predictions):
                    target=validator._prepare_batch(index,batch)
                    # Fork's matching mask geometry is shared with Validation.
                    gt=target['masks'].bool()
                    if gt.ndim==2: gt=gt[None]
                    target_union=gt.any(dim=0)
                    predicted=pred['masks'].bool()
                    if predicted.ndim==2: predicted=predicted[None]
                    if len(predicted):
                        if predicted.shape[-2:]!=target_union.shape[-2:]:
                            predicted=torch.nn.functional.interpolate(predicted[:,None].float(),size=target_union.shape[-2:],mode='nearest')[:,0].bool()
                        intersections=(predicted & target_union).flatten(1).sum(1)
                        unions=(predicted | target_union).flatten(1).sum(1).clamp_min(1)
                        ious=(intersections/unions).cpu().tolist();conf=pred['conf'].cpu().tolist()
                        best=min(range(len(ious)),key=lambda i:(-ious[i],-conf[i],i))
                        iou,confidence=ious[best],conf[best]
                    else: iou,confidence=0.,0.
                    row=rows[position];position+=1
                    results.append({**row,'valid':True,'mask_iou':float(iou),'confidence':float(confidence),
                        'score':task_score(iou,confidence),'prediction_count':len(predicted)})
                atomic_json(directory/'progress.json',{'completed':position,'total':len(rows),'updated':time.time()})
        if position!=len(rows): raise ValueError('Incomplete candidate inference')
    path=directory/'scores.json'
    atomic_json(path,{'judge_sha256':judge['checkpoint_sha256'],'records':results,
        'label_caveat':'IoU to approximate synthetic input mask, not independently annotated output GT'},immutable=True)
    return [path]


def selection():
    from s8_anydoor_selection import select
    files=[]
    for subset,prefix in (('train748','M'),('low187','L')):
        scored=read_json(BASE/'scores'/subset/'scores.json')['records']
        result=select(scored)
        byid={r['id']:r for r in read_json(BASE/'pools'/subset/'manifest.json')}
        for suffix,method in (('2','random'),('3','task')):
            path=BASE/'selection'/(prefix+suffix+'.json')
            atomic_json(path,[byid[id] for id in result[method]],immutable=True);files.append(path)
        path=BASE/'selection'/(subset+'_review.json')
        atomic_json(path,result,immutable=True);files.append(path)
    return files


def train_preflight():
    import torch
    from capstone_lab.campaign.epochs import train_epochs
    evidence={}
    with RunLock(ROOT/'artifacts/s8_gpu.lock'):
        for exp in ('M2','M3','L2','L3'):
            directory=BASE/'training_preflight'/exp
            continuous=train_epochs(ROOT,directory/'continuous',experiment=exp,seed=0,epochs=2,smoke=True)
            gc.collect();torch.cuda.empty_cache()
            paused=train_epochs(ROOT,directory/'resumed',experiment=exp,seed=0,epochs=2,smoke=True,stop_after=1)
            gc.collect();torch.cuda.empty_cache()
            resumed=train_epochs(ROOT,directory/'resumed',experiment=exp,seed=0,epochs=2,smoke=True)
            from capstone_lab.campaign.epoch_preflight import equal_state
            states=[]
            for name in ('continuous','resumed'):
                pointer=read_json(directory/name/'current.json')
                states.append(torch.load(directory/name/pointer['last']['file'],map_location='cpu',weights_only=False))
            checks={key:equal_state(states[0][key],states[1][key]) for key in ('model','optimizer','scaler','scheduler','rng','initial')}
            checks['best_epoch_score']=all(continuous['best_mask'][k]==resumed['best_mask'][k] for k in ('epoch','score'))
            checks['losses']=[h['loss'] for h in continuous['history']]==[h['loss'] for h in resumed['history']]
            checks['streams']=[h['data_stream_sha256'] for h in continuous['history']]==[h['data_stream_sha256'] for h in resumed['history']]
            if not all(checks.values()): raise ValueError('Generative mixed resume regression')
            path=directory/'summary.json';atomic_json(path,{'status':'VERIFIED','checks':checks,'images_per_diagnostic_epoch':32},immutable=True)
            evidence[rel(path)]=sha256_file(path)
            del states
            gc.collect();torch.cuda.empty_cache()
    for path in (BASE/'selection').glob('*.json'): evidence[rel(path)]=sha256_file(path)
    evidence[rel(APPROVAL)]=sha256_file(APPROVAL)
    path=BASE/'training_release.json'
    atomic_json(path,{'status':'VERIFIED_GENERATIVE_RELEASE','code_sha256':digest(code_records(ROOT)),
        'evidence':evidence,'formal_jobs_started':0,'realization':'32-image mixture diagnostic; complete6000 source/output audit precedes this gate'},immutable=True)
    return [path]


def train(stage):
    from capstone_lab.campaign.epochs import train_epochs
    exp,seed=stage.removeprefix('train_').split('_seed')
    directory=BASE/'jobs'/f'{exp}_seed{seed}'/'training_state'
    def pause():
        if (BASE/'pause.request').exists(): return True
        resource_guard(); return False
    with RunLock(ROOT/'artifacts/s8_gpu.lock'):
        result=train_epochs(ROOT,directory,experiment=exp,seed=int(seed),epochs=150,
            should_pause=pause,progress_path=directory/'progress.json')
    if result['status']!='SUCCEEDED': raise ValueError('Training safely paused; no dependent dispatch')
    return [directory/'result.json',directory/'current.json',directory/result['best_mask']['file'],directory/result['last']['file']]


def freeze():
    rows=[]
    locations=[ROOT/'artifacts/s8_real_campaign/run_v1', ROOT/'artifacts/s8_heuristic_training/run_v1', BASE]
    for location in locations:
        for path in sorted((location/'jobs').glob('*/training_state/result.json')):
            result=read_json(path)
            if result.get('status')!='SUCCEEDED' or not result.get('formal_training_completed'): continue
            best=result['best_mask']; checkpoint=path.parent/best['file']
            if sha256_file(checkpoint)!=best['sha256']: raise ValueError('Best checkpoint changed')
            history=[h for h in result['history'] if h['epoch']==best['epoch']]
            if len(history)!=1 or history[0]['mask_map50_95']!=best['score']: raise ValueError('Val selection mismatch')
            rows.append({'job':path.parent.parent.name,'experiment':result['experiment'],'seed':result['seed'],
                'checkpoint':rel(checkpoint),'sha256':best['sha256'],'validation':history[0]['metrics'],
                'variant':'H0' if result['experiment'].startswith('H0') else 'H1' if result['experiment'].startswith('H1') else 'H2'})
    if len(rows)!=34 or len({r['job'] for r in rows})!=34: raise ValueError('Expected all34 completed training jobs')
    selected=[r for r in rows if not r['experiment'].startswith('A')]
    if len(selected)!=28 or len({r['sha256'] for r in selected})!=28: raise ValueError('Expected28 unique checkpoints')
    path=BASE/'final_freeze.json'
    atomic_json(path,{'status':'FROZEN_BEFORE_TEST','all_validation':rows,'test_checkpoints':selected,
        'logical_cells':30,'H2_R_reference':'M0_seed0','H2_S_reference':'M1_seed0',
        'test_manifest_sha256':sha256_file(ROOT/'manifests/real_split_observed_v1.jsonl'),
        'evaluation':{'conf':.001,'nms_iou':.7,'max_det':300,'mask_PRF_conf':.25,'mask_PRF_iou':.5},
        'pipeline_binding':verify_binding()},immutable=True)
    return [path]


def test():
    import torch
    from capstone_lab.campaign.models import create_model,validate_masks
    from capstone_lab.campaign.data import ReadOnlySegDataset,effective_args,make_loader
    freeze=read_json(BASE/'final_freeze.json')
    manifest=ROOT/'manifests/real_split_observed_v1.jsonl'
    if sha256_file(manifest)!=freeze['test_manifest_sha256']: raise ValueError('Test manifest changed')
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows=[r for r in rows if r['original_split']=='test']
    if len(rows)!=200: raise ValueError('Expected Test200')
    for row in rows:
        for key in ('image','label'):
            if not row[key].startswith(f'Dataset/{"images" if key=="image" else "labels"}/test/'):
                raise ValueError('Test split path mismatch')
            if sha256_file(ROOT/row[key])!=row[key+'_sha256']: raise ValueError('Test integrity failure')
    results=[]
    with RunLock(ROOT/'artifacts/s8_gpu.lock'):
        torch.set_num_threads(1)
        for row in freeze['test_checkpoints']:
            directory=BASE/'test'/row['job'];path=directory/'result.json'
            if path.exists():
                saved=read_json(path)
                if saved['freeze_sha256']!=sha256_file(BASE/'final_freeze.json'): raise ValueError('Changed Test freeze')
                results.append(saved);continue
            directory.mkdir(parents=True,exist_ok=True)
            checkpoint=ROOT/row['checkpoint']
            if sha256_file(checkpoint)!=row['sha256']: raise ValueError('Test checkpoint changed')
            model,_=create_model(ROOT,directory,row['seed'],row['variant'])
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(payload['model'],strict=True);del payload
            model.cuda().eval();args=effective_args(ROOT)
            dataset=ReadOnlySegDataset(ROOT,rows,args,augment=False)
            metrics=validate_masks(model,make_loader(dataset,seed=0,workers=2),directory,args)
            if metrics['images']!=200: raise ValueError('Incomplete final Test')
            saved={'job':row['job'],'experiment':row['experiment'],'seed':row['seed'],'metrics':metrics,
                'checkpoint_sha256':row['sha256'],'freeze_sha256':sha256_file(BASE/'final_freeze.json')}
            atomic_json(path,saved,immutable=True);results.append(saved)
            del model,dataset;gc.collect();torch.cuda.empty_cache()
    path=BASE/'test'/'summary.json'
    atomic_json(path,{'unique_count':28,'logical_count':30,'records':results,'H2_references':['M0_seed0','M1_seed0']},immutable=True)
    return [path]


def report():
    tested=read_json(BASE/'test'/'summary.json')
    groups=defaultdict(list)
    for row in tested['records']: groups[row['experiment']].append(row['metrics']['library_metrics']['metrics/mAP50-95(M)'])
    lines=['# Final campaign report','','Mask mAP50-95; fixed Test200; A screening is Validation-only.','',
        '| Experiment | Seeds | Mean | Sample SD |','| --- | ---: | ---: | ---: |']
    for exp,values in sorted(groups.items()):
        sd=f'{statistics.stdev(values):.6f}' if len(values)>1 else 'N/A (single seed)'
        lines.append(f'| {exp} | {len(values)} | {statistics.mean(values):.6f} | {sd} |')
    lines += ['', 'H2_R/H2_S reference M0_seed0/M1_seed0;28 unique checkpoints,30 logical cells.',
        '', 'Caveats: AnyDoor shape-controlled input masks are user-accepted approximate labels, not independently annotated GT. SAM2 not used. '
        'Task-aware scoring may over-rank label disagreement/no-prediction cases. See selection distributions. '
        'Historical semester1 background-only/recipe differences prohibit a controlled direct comparison. '
        'Existing34-job Validation records and full per-checkpoint Test metrics are preserved in final_freeze.json and test/summary.json.']
    path=BASE/'FINAL_REPORT.md';content='\n'.join(lines)+'\n'
    if path.exists() and path.read_text(encoding='utf-8')!=content: raise ValueError('Preserve changed report')
    if not path.exists():
        with path.open('x',encoding='utf-8') as stream: stream.write(content)
    return [path]


def dispatch(stage):
    if stage=='pilot': return pilot()
    if stage.startswith('generate_'): return generate(stage.removeprefix('generate_'))
    if stage.startswith('score_'): return score(stage.removeprefix('score_'))
    if stage=='audit': return audit()
    if stage=='selection': return selection()
    if stage=='train_preflight': return train_preflight()
    if stage.startswith('train_'): return train(stage)
    if stage=='freeze': return freeze()
    if stage=='test': return test()
    if stage=='report': return report()
    raise ValueError('Unknown stage')
