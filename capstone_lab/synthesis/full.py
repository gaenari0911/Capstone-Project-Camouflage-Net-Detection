"""FULL-policy paired Heuristic preflight. Formal execution is deliberately gated."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import time
import uuid
import ctypes

import psutil
from capstone_lab.config import sha256_file
from capstone_lab.campaign.contracts import digest, read_json, inside
from capstone_lab.campaign.io import RunLock, atomic_json, atomic_replace
from capstone_lab.campaign.resources import budget_roots
from capstone_lab.campaign.supervisor import directory_bytes
from .contract import CandidatePlan, derive_seed, CONDITIONS
from .scoring import UsageState, select_candidate
from .smoke import (ActualFeatureTask, ActualRenderTask, _legacy_module, compute_actual_features,
                    render_actual_bundle, _commit_rendered_bundle, _validate_committed_bundle)

POLICY = 'configs/campaigns/s8_heuristic_policy_v1.json'
SUMMARY = 'artifacts/s8_source_audit/split_pools_v1/split_source_summary.json'
BACKGROUND = 'artifacts/s8_full_backgrounds/v1/manifest.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def memory():
    available = psutil.virtual_memory().available
    if os.name == 'nt':
        from ctypes import wintypes as w
        class Perf(ctypes.Structure):
            _fields_ = [('cb', w.DWORD)] + [(n, ctypes.c_size_t) for n in
                ('commit_total','commit_limit','commit_peak','physical_total','physical_available',
                 'system_cache','kernel_total','kernel_paged','kernel_nonpaged','page_size')] + [
                 ('handles', w.DWORD), ('processes', w.DWORD), ('threads', w.DWORD)]
        p = Perf()
        p.cb = ctypes.sizeof(p)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb):
            raise OSError('Windows commit telemetry unavailable')
        used, limit = p.commit_total*p.page_size, p.commit_limit*p.page_size
    else:
        vm, swap = psutil.virtual_memory(), psutil.swap_memory()
        used, limit = vm.total-vm.available+swap.used, vm.total+swap.total
    return {'available_bytes': available, 'commit_used': used, 'commit_limit': limit,
            'commit_fraction': used/limit, 'cpu_percent': psutil.cpu_percent(),
            'process_rss': psutil.Process().memory_info().rss}


def resource_guard(root, policy, workers, *, disk=True):
    m = memory()
    require(m['available_bytes'] >= (policy['ram_headroom_gib']+workers)*2**30, 'RAM_SAFE_WAIT')
    require(m['commit_limit']-m['commit_used'] >= (policy['commit_headroom_gib']+workers)*2**30
            and m['commit_fraction'] < policy['commit_fraction_max'], 'COMMIT_SAFE_WAIT')
    if disk:
        used = sum(directory_bytes(p) for p in budget_roots(root))
        require(used + 2**30 < policy['new_disk_gib']*2**30, 'DISK_BUDGET_STOP')
        require(shutil.disk_usage(root).free >= (policy['min_free_gib']+1)*2**30, 'DISK_FREE_STOP')
        m['new_disk_bytes'] = used
    return m


def load_sources(root, subset):
    require(subset in ('train748','low187'), 'unsupported subset')
    approval = read_json(root/'configs/approvals/s8_foreground_amendment_v1.json')
    require(approval['status']=='APPROVED_BY_USER' and approval['foreground_subsets']==
            {'A':'train748','M':'train748','H':'train748','L':'low187'}, 'source approval mismatch')
    s = read_json(root/SUMMARY)
    require(s['status']=='VERIFIED' and s['approval_sha256']==sha256_file(root/'configs/approvals/s8_foreground_amendment_v1.json'), 'source summary binding mismatch')
    ref = s['pools'][subset]
    proof_path = inside(root, ref['path'])
    require(sha256_file(proof_path)==ref['sha256'], 'source proof changed')
    proof = read_json(proof_path)
    require(proof['status']=='VERIFIED' and not proof['rejected'], 'source proof rejected')
    allowed = [json.loads(l) for l in (root/('manifests/low187_v1.jsonl' if subset=='low187' else 'manifests/real_split_observed_v1.jsonl')).read_text(encoding='utf-8').splitlines() if l.strip()]
    ids = {r['image'] for r in allowed if r['original_split']=='train'}
    require(len(ids)==(187 if subset=='low187' else 748), 'source count mismatch')
    require({r['source_image'] for r in proof['records']}==ids, 'wrong source budget')
    fg = {}
    for r in proof['records']:
        for key, h in [('object_image','object_image_sha256'),('object_mask','object_mask_sha256'),('source_image','source_sha256'),('label','label_sha256')]:
            require(sha256_file(inside(root,r[key]))==r[h], 'source bytes changed')
        fid = Path(r['object_image']).stem
        require(fid not in fg, 'duplicate foreground ID')
        image_stat=inside(root,r['object_image']).stat()
        mask_stat=inside(root,r['object_mask']).stat()
        fg[fid] = {'foreground_id':fid,'environment':r['environment'],
                   'image_path':r['object_image'],'image_sha256':r['object_image_sha256'],
                   '_image_size':image_stat.st_size,'_image_mtime_ns':image_stat.st_mtime_ns,
                   'mask_path':r['object_mask'],'mask_sha256':r['object_mask_sha256'],
                   '_mask_size':mask_stat.st_size,'_mask_mtime_ns':mask_stat.st_mtime_ns,
                   'source_image':r['source_image'],'source_sha256':r['source_sha256']}
    b = read_json(root/BACKGROUND)
    require(b['status']=='VERIFIED', 'background decode gate failed')
    bg = {}
    for r in b['backgrounds']:
        require(sha256_file(inside(root,r['path']))==r['sha256'], 'background changed')
        require(r['background_id'] not in bg, 'duplicate background ID')
        row=dict(r);stat=inside(root,r['path']).stat()
        row.update(_path_size=stat.st_size,_path_mtime_ns=stat.st_mtime_ns)
        bg[r['background_id']] = row
    require({r['domain'] for r in bg.values()}=={'snow','forest_dense','grass_field','leaf_ground','rocky','mixed'}, 'FULL domains missing')
    return fg, bg


def balanced_assignment(schedule, fg):
    """One-use/object/round feasibility certificate; NOT a fixed render choice."""
    assigned={}
    seen=set()
    for environment in ('snow','non_snow'):
        objects=sorted(k for k,r in fg.items() if r['environment']==environment)
        slots=[r for r in schedule if (r['domain']=='snow')==(environment=='snow')]
        for offset in range(0,len(slots),len(objects)):
            chunk=slots[offset:offset+len(objects)]
            edges={i:sorted([k for k in objects if (k,r['background_id']) not in seen],
                     key=lambda k:hashlib.sha256((r['sample_id']+'|balanced|'+k).encode()).hexdigest())
                   for i,r in enumerate(chunk)}
            matched={}
            def augment(i,visited):
                for k in edges[i]:
                    if k in visited: continue
                    visited.add(k)
                    if k not in matched or augment(matched[k],visited):
                        matched[k]=i
                        return True
                return False
            for i in range(len(chunk)):
                require(augment(i,set()),'balanced matching has no feasible assignment')
            for k,i in matched.items():
                assigned[chunk[i]['sample_id']]=k
                seen.add((k,chunk[i]['background_id']))
    require(len(assigned)==len(schedule),'incomplete balanced allocation')
    return assigned


def balanced_options(schedule, fg, state, current, seed, limit=8):
    env='snow' if current['domain']=='snow' else 'non_snow'
    objects=sorted(k for k,r in fg.items() if r['environment']==env)
    slots=[r for r in schedule if (r['domain']=='snow')==(env=='snow')]
    index=next(i for i,r in enumerate(slots) if r['sample_id']==current['sample_id'])
    tail=slots[index+1:min(len(slots),(index//len(objects)+1)*len(objects))]
    minimum=min(state.used_total[k] for k in objects)
    available=[k for k in objects if state.used_total[k]==minimum]
    candidates=sorted([k for k in available if not state.used_by_background[k].get(current['background_id'],0)],
                      key=lambda k:hashlib.sha256((seed+k).encode()).hexdigest())
    choices=[]
    for candidate in candidates:
        edges={i:[k for k in available if k!=candidate and not state.used_by_background[k].get(r['background_id'],0)]
               for i,r in enumerate(tail)}
        matched={}
        def augment(i,seen):
            for k in edges[i]:
                if k in seen: continue
                seen.add(k)
                if k not in matched or augment(matched[k],seen):
                    matched[k]=i
                    return True
            return False
        if all(augment(i,set()) for i in range(len(tail))):
            choices.append(candidate)
            if len(choices)>=limit: break
    return choices


def plans(root, subset, fg, bg, policy, *, legacy_root=None):
    legacy = _legacy_module(str((legacy_root or root)/'build_demo_synthetic_segmentation.py'))
    allocate = legacy._allocate_counts_by_weights
    domains = Counter(r['domain'] for r in bg.values())
    schedule = []
    for mode,n in policy['mode_counts'].items():
        for domain,count in sorted(allocate(n,dict(sorted(domains.items()))).items()):
            schedule.extend([(mode,domain)]*count)
    require(len(schedule)==3000, 'FULL target mismatch')
    rng = random.Random(derive_seed(7,subset,'schedule',0)[1])
    rng.shuffle(schedule)
    difficulties = [k for k,n in policy['difficulty_counts'].items() for _ in range(n)]
    rng.shuffle(difficulties)
    scales = [b for b in legacy.SCALE_DISTRIBUTION for _ in range(b['weight']*30)]
    rng.shuffle(scales)
    bydomain = {d:sorted(k for k,r in bg.items() if r['domain']==d) for d in domains}
    result = []
    for i,((mode,domain),difficulty,scale) in enumerate(zip(schedule,difficulties,scales)):
        sid = subset+'_'+str(i+1).zfill(6)
        d, seed = derive_seed(7,sid,'candidate_plan',0)
        srng = random.Random(seed)
        result.append({'sample_id':sid,'mode':mode,'domain':domain,'difficulty':difficulty,
                       'scale_bucket':scale['name'],
                       'scale_ratio':legacy.adjust_target_area_ratio_for_difficulty(srng.uniform(scale['min_ratio'],scale['max_ratio']),difficulty),
                       'background_id':srng.choice(bydomain[domain])})
    quotas = {}
    for environment in ('snow','non_snow'):
        n = sum((r['domain']=='snow')==(environment=='snow') for r in result)
        objects = sorted(k for k,r in fg.items() if r['environment']==environment)
        cap = math.ceil(n/len(objects))
        require(subset=='low187' or cap<=7, 'full quota exceeds legacy ceiling')
        quotas[environment]={'target':n,'objects':len(objects),'ceiling':cap,'capacity':cap*len(objects)}
    if subset=='low187':
        assignment=balanced_assignment(result,fg)
        for row in result:
            row['assigned_foreground']=assignment[row['sample_id']]
    return result, quotas


def binding(root, subset, policy, plan, *, code_root=None):
    """Bind mutable large inputs to ``root`` and executable code to its snapshot."""
    code_root = (code_root or root).resolve()
    inputs = [POLICY,SUMMARY,BACKGROUND,
              'configs/approvals/s8_foreground_amendment_v1.json',
              'configs/approvals/s8_low_diversity_v1.json']
    code = ['build_demo_synthetic_segmentation.py']
    code += [p.relative_to(code_root).as_posix()
             for p in (code_root/'capstone_lab/synthesis').glob('*.py')]
    code += ['capstone_lab/campaign/io.py','capstone_lab/campaign/detached.py',
             'capstone_lab/campaign/contracts.py','capstone_lab/campaign/resources.py',
             'capstone_lab/config.py','capstone_lab/errors.py']
    records = {p:sha256_file(root/p) for p in sorted(inputs)}
    records.update({p:sha256_file(code_root/p) for p in sorted(set(code))})
    return {'subset':subset,'policy':policy,'plan_sha256':digest(plan),
            'input_root':str(root.resolve()),'code_root':str(code_root),
            'inputs_and_code':records}


def worker_init(registry=None):
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key]='1'
    import cv2
    cv2.setNumThreads(1)
    if registry:
        process=psutil.Process()
        atomic_json(Path(registry)/(str(process.pid)+'_'+str(time.time_ns())+'.json'),
                    {'pid':process.pid,'created':process.create_time(),'parent':os.getppid(),
                     'command':process.cmdline()},immutable=True)


def feature(task):
    try:
        return {'feature':asdict(compute_actual_features(task))}
    except Exception as e:
        return {'rejected':type(e).__name__+': '+str(e)}


def roundtrip(rendered):
    import cv2
    import numpy as np
    mask = cv2.imdecode(np.frombuffer(rendered['mask'],np.uint8),0)
    raster = np.zeros_like(mask)
    h,w = mask.shape
    for line in rendered['polygon'].decode().splitlines():
        v = np.asarray([float(x) for x in line.split()[1:]]).reshape(-1,2)
        require(np.isfinite(v).all(), 'nonfinite polygon')
        points = np.rint(v*np.array([w,h])).astype(np.int32)
        cv2.fillPoly(raster,[points],255)
    union = np.count_nonzero((mask>0)|(raster>0))
    return np.count_nonzero((mask>0)&(raster>0))/max(1,union)


def execute(root, output, subset, workers, count, *, interrupt_after=None,
            interrupt_stage=None, formal_release=None, code_root=None,
            pause_path=None, checkpoint=None):
    formal = formal_release is not None
    if formal:
        require(count == 3000, 'formal dataset execution requires exactly3000 plans')
        require(formal_release.get('status') == 'VERIFIED_FORMAL_GENERATOR',
                'formal release gate missing')
        require(code_root is not None, 'formal execution requires immutable code root')
        from .formal import verify_release
        verified, runtime = verify_release(root)
        require(verified == formal_release and code_root.resolve() == runtime.resolve(),
                'unverified formal release/runtime')
        require(output.resolve().is_relative_to(
            (root/'artifacts/s8_heuristic_formal').resolve()), 'unsafe formal output')
    else:
        require(1<=count<=12, 'preflight only: formal generation not released')
    policy = read_json(root/POLICY)
    if checkpoint: checkpoint()
    worker_init()
    require(1<=workers<=policy['max_workers'], 'worker cap')
    if not formal:
        require(output.resolve().is_relative_to((root/'artifacts/s8_heuristic_preflight').resolve()), 'unsafe output')
    fg,bg = load_sources(root,subset)
    plan,quotas = plans(root,subset,fg,bg,policy,legacy_root=code_root if formal else None)
    bound = binding(root,subset,policy,plan,code_root=code_root)
    contract = {'config_sha256':digest(bound),'source_manifest_sha256':sha256_file(root/SUMMARY),
                'resolution_manifest_sha256':sha256_file(root/'manifests/object_environment_resolution_v1.jsonl'),
                'legacy_source_sha256':sha256_file((code_root or root)/'build_demo_synthetic_segmentation.py'),
                'code_sha256':digest(bound['inputs_and_code'])}
    if formal:
        contract['formal_release_sha256'] = formal_release['release_sha256']
    conditions = policy['datasets'][subset]
    output.mkdir(parents=True,exist_ok=True)
    with RunLock(output/'run.lock'):
        atomic_json(output/'contract.json',bound,immutable=True)
        atomic_json(output/'plan.json',{'plans':plan,'quotas':quotas},immutable=True)
        states = {c:UsageState(tuple(sorted(fg)),max_usage=max(q['ceiling'] for q in quotas.values())) for c in conditions}
        counts = Counter()
        started = time.monotonic()
        prior_seconds=0
        if not formal:
            for path in (root/'artifacts').glob('s8_*/*/status.json'):
                status=read_json(path)
                prior_seconds+=status.get('active_seconds',0)
        telemetry, failures, fingerprints = [], (read_json(output/'rejections.json')
            if (output/'rejections.json').exists() else []), {}
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers,mp_context=ctx,initializer=worker_init,
                                 initargs=(str(output/'.pool'),)) as pool:
            for planned in plan[:count]:
                if checkpoint: checkpoint()
                base=dict(planned)
                base.pop('assigned_foreground',None)  # static feasibility certificate only
                sid=base['sample_id']
                final=output/'samples'/sid
                if (final/'PAIR.json').exists():
                    pair=read_json(final/'PAIR.json')
                    require(pair['contract']==contract and set(pair['metadata'])==set(conditions), 'pair binding changed')
                    for c in conditions:
                        m,h=_validate_committed_bundle(final/c,contract)
                        require(m==pair['metadata'][c], 'pair metadata changed')
                        states[c].commit(m['foreground_id'],base['domain'],m['background_id'])
                        counts[c]+=1
                        fingerprints[sid+'/'+c]=h
                    continue
                requested_pause = (output/'pause.request').exists()
                if pause_path and pause_path.exists():
                    control = read_json(pause_path)
                    requested_pause = control.get('action') == 'pause'
                if requested_pause:
                    atomic_json(output/'status.json',{'status':'PAUSED','counts':dict(counts)})
                    return {'status':'PAUSED','counts':dict(counts)}
                telemetry.append(resource_guard(root,policy,workers))
                require(prior_seconds+time.monotonic()-started<policy['active_seconds_max'], 'TIME_BUDGET_STOP')
                env='snow' if base['domain']=='snow' else 'non_snow'
                env_ids=sorted(k for k,r in fg.items() if r['environment']==env)
                for c in conditions:
                    states[c].max_usage=quotas[env]['ceiling']
                committed=False
                for attempt in range(policy['sample_attempts']):
                    d,seed=derive_seed(7,sid,'candidate_plan',attempt)
                    rng=random.Random(seed)
                    background=bg[base['background_id']]
                    # Shared expansion includes available candidates for every ranked condition.
                    shared=set(rng.sample(env_ids,min(8,len(env_ids)))) if subset=='train748' else set()
                    for c in conditions:
                        if c=='A0': continue
                        eligible=[k for k in env_ids if states[c].used_total[k]<quotas[env]['ceiling']
                                  and not states[c].used_by_background[k].get(base['background_id'],0)]
                        if subset=='low187' and eligible:
                            eligible=balanced_options(plan,fg,states[c],base,d)
                        eligible.sort(key=lambda k:hashlib.sha256((d+k).encode()).hexdigest())
                        shared.update(eligible[:8])
                    p=CandidatePlan(**base,foreground_candidate_ids=tuple(sorted(shared)),
                        position_candidate_ids=tuple('p'+str(j) for j in range(16)),candidate_seed=d,
                        rendering_seed=derive_seed(7,sid,'rendering',attempt)[0],attempt=attempt)
                    tasks=[ActualFeatureTask(str((code_root or root)/'build_demo_synthetic_segmentation.py'),str(root),p.to_dict(),
                           background,fg[k],position,1280,full_policy=True)
                           for k in p.foreground_candidate_ids for position in p.position_candidate_ids]
                    features=[]
                    from .contract import CandidateFeature
                    # At most workers tasks in flight; large image arrays never travel for feature extraction.
                    for start in range(0,len(tasks),workers):
                        if checkpoint: checkpoint()
                        atomic_json(output/'progress.json',{'sample_id':sid,'attempt':attempt,
                                    'feature_tasks_completed':start,'feature_tasks_total':len(tasks),
                                    'updated':time.time(),'counts':dict(counts)})
                        futures=[pool.submit(feature,t) for t in tasks[start:start+workers]]
                        for f in futures:
                            r=f.result(timeout=policy['task_timeout_seconds'])
                            if 'feature' in r:
                                v=r['feature'];v['eligibility_reasons']=tuple(v['eligibility_reasons'])
                                features.append(CandidateFeature(**v))
                            else:
                                require('source stat changed' not in r['rejected'], 'source stat changed during execution')
                                failures.append({'sample':sid,'attempt':attempt,**r})
                    atomic_json(output/'rejections.json',failures)
                    lookup={t.foreground['foreground_id']+'@'+t.position_id:t for t in tasks}
                    selections={}
                    for c in conditions:
                        eligible=features
                        if c!='A0':
                            eligible=[f for f in features if states[c].used_total[f.foreground_id]<quotas[env]['ceiling']
                                      and not states[c].used_by_background[f.foreground_id].get(base['background_id'],0)]
                            if subset=='low187':
                                feasible=set(p.foreground_candidate_ids)
                                eligible=[f for f in eligible if f.foreground_id in feasible]
                        choice=select_candidate(c,p,eligible,states[c],7)
                        if choice is None: break
                        selections[c]=choice
                    if len(selections)!=len(conditions):
                        failures.append({'sample':sid,'attempt':attempt,'rejected':'no_common_feasible_selection'})
                        continue
                    stage=output/'.pending'/uuid.uuid4().hex
                    metadata={}
                    try:
                        for c,(f,selection) in selections.items():
                            if checkpoint: checkpoint()
                            rendered=pool.submit(render_actual_bundle,ActualRenderTask(lookup[f.candidate_id],c,selection)).result(timeout=policy['task_timeout_seconds'])
                            iou=roundtrip(rendered)
                            require(iou>=policy['polygon_roundtrip_iou_min'], 'polygon_roundtrip_mismatch:'+str(iou))
                            m={'sample_id':sid,'condition_id':c,**contract,'background_id':base['background_id'],
                               'foreground_id':f.foreground_id,'foreground':fg[f.foreground_id],
                               'background':background,'subset':subset,'plan':p.to_dict(),
                               'candidate_seed':p.candidate_seed,'rendering_seed':p.rendering_seed,
                               'selection':selection,'roundtrip_iou':iou,
                               'render':{k:v for k,v in rendered.items() if k not in ('image','mask','polygon')}}
                            _commit_rendered_bundle(output_dir=stage,plan=p,condition=c,rendered=rendered,metadata=m,contract=contract)
                            metadata[c]=m
                            if interrupt_stage == 'after_first_condition' and len(metadata) == 1:
                                raise InterruptedError('injected interruption inside paired staging')
                        bundle=stage/'samples'/sid
                        atomic_json(bundle/'PAIR.json',{'contract':contract,'metadata':metadata},immutable=True)
                        if final.exists(): raise ValueError('uncommitted final collision')
                        final.parent.mkdir(parents=True,exist_ok=True)
                        atomic_replace(bundle,final)
                        for c,m in metadata.items():
                            states[c].commit(m['foreground_id'],base['domain'],base['background_id'])
                            counts[c]+=1
                            fingerprints[sid+'/'+c]=_validate_committed_bundle(final/c,contract)[1]
                        committed=True
                        atomic_json(output/'status.json',{'status':'RUNNING','counts':dict(counts),'updated':time.time()})
                        if interrupt_after and sum(counts.values())>=interrupt_after:
                            raise InterruptedError('preflight injected interruption after atomic pair')
                        break
                    except (InterruptedError, PermissionError): raise
                    except Exception as e:
                        if committed or any(word in str(e) for word in ('STOP','SAFE_WAIT','changed','collision')):
                            raise
                        failures.append({'sample':sid,'attempt':attempt,'rejected':repr(e)})
                        atomic_json(output/'rejections.json',failures)
                require(committed,'sample attempts exhausted: '+sid)
        if formal:
            # A final complete hash pass prevents a mid-run source mutation from
            # being silently accepted as a successful formal dataset.
            load_sources(root,subset)
        report={'status':'FORMAL_DATASET_COMPLETE' if formal else 'VERIFIED_SMALL_RENDER',
                'formal_release':formal,'workers':workers,'counts':dict(counts),
                'seconds':time.monotonic()-started,'telemetry':telemetry,'failures':failures,
                'output_hashes':fingerprints,'usage':{c:s.snapshot() for c,s in states.items()},
                'contract_sha256':digest(bound),'plan_quotas':quotas}
        atomic_json(output/'report.json',report)
        atomic_json(output/'status.json',{'status':report['status'],'counts':dict(counts)})
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--subset',choices=['train748','low187'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--count',type=int,default=1)
    p.add_argument('--interrupt-after',type=int)
    a=p.parse_args()
    result=execute(Path.cwd().resolve(),a.output.resolve(),a.subset,a.workers,a.count,interrupt_after=a.interrupt_after)
    print(json.dumps({k:v for k,v in result.items() if k not in ('output_hashes','usage','failures')},indent=2))
