"""Read-only analysis of frozen runs; write only a new presentation directory."""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import math
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/s8_anydoor_campaign/run_v1'
COLORS = ['#526779', '#D98B28', '#218B82', '#7956A5']
NAMES = ['Real only', '+ Heuristic', '+ AnyDoor random', '+ AnyDoor task-aware']
KEYS = ['map', 'map50', 'precision', 'recall', 'f1', 'box_map']
LABELS = ['Mask mAP50-95', 'Mask mAP50', 'Mask precision', 'Mask recall', 'Mask F1', 'Box mAP50-95']

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def metrics(m):
    l = m['library_metrics']
    return dict(map=l['metrics/mAP50-95(M)'], map50=l['metrics/mAP50(M)'],
                precision=m['mask_precision'], recall=m['mask_recall'], f1=m['mask_f1'],
                box_map=l['metrics/mAP50-95(B)'])

def table(rows, fields):
    return '| ' + ' | '.join(fields) + ' |\n| ' + ' | '.join(['---']*len(fields)) + ' |\n' + '\n'.join(
        '| ' + ' | '.join(str(r[f]) for f in fields) + ' |' for r in rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    out = (ROOT / ap.parse_args().output).resolve()
    if not out.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Output must be a new directory under artifacts')
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    (out/'tables').mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#263746',
        'axes.titlepad':13,'savefig.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42})
    sources = {}
    def load(p):
        sources[str(Path(p).relative_to(ROOT)).replace('\\','/')] = sha(p)
        return read(p)
    def write_csv(name, rows):
        with (out/'tables'/name).open('w', newline='', encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    figures=[]
    def save(fig, name, caption):
        fig.tight_layout(rect=(0,.035,1,1))
        fig.text(.01,.008,caption,fontsize=8,color='#53606A')
        for ext in ['png','svg','pdf']:
            fig.savefig(out/'figures'/f'{name}.{ext}',dpi=300)
        figures.append((name,caption));plt.close(fig)
    freeze=load(BASE/'final_freeze.json'); tested=load(BASE/'test/summary.json')
    records={r['job']:r for r in tested['records']}
    histories={};results={};contracts={};audit=[];flat=[];epochs=[]
    for v in freeze['all_validation']:
        job=v['job'];d=(ROOT/v['checkpoint']).parent
        r=load(d/'result.json');h=load(d/'history.json');c=load(d/'contract.json')
        histories[job]=h;results[job]=r;contracts[job]=c
        expected=40 if job.startswith('A') else 150
        selected=[e for e in h if e['epoch']==r['best_mask']['epoch']]
        a={'job':job,'succeeded':r['status']=='SUCCEEDED' and r['formal_training_completed'],
           'epochs_complete':[e['epoch'] for e in h]==list(range(1,expected+1)),
           'finite_history':all(math.isfinite(e['loss']) and math.isfinite(e['mask_map50_95']) for e in h),
           'checkpoint_hash_ok':sha(ROOT/v['checkpoint'])==v['sha256'],
           'best_is_history_max':len(selected)==1 and abs(max(e['mask_map50_95'] for e in h)-r['best_mask']['score'])<1e-12,
           'validation_matches':len(selected)==1 and metrics(selected[0]['metrics'])==metrics(v['validation']),
           'test_link_ok':job not in records or records[job]['checkpoint_sha256']==v['sha256'],
           'test_freeze_ok':job not in records or records[job]['freeze_sha256']==sha(BASE/'final_freeze.json')}
        audit.append(a)
        for split,m in [('Validation',v['validation'])]+([('Test',records[job]['metrics'])] if job in records else []):
            flat.append(dict(job=job,experiment=v['experiment'],seed=v['seed'],split=split,
                epochs=len(h),best_epoch=r['best_mask']['epoch'],**metrics(m),
                tp=m['mask_tp'],fp=m['mask_fp'],fn=m['mask_fn'],images=m['images'],
                checkpoint=v['checkpoint']))
        for e in h:
            epochs.append(dict(job=job,epoch=e['epoch'],global_step=e['global_step'],loss=e['loss'],
                **metrics(e['metrics']),seconds=e['wall_seconds'],amp_retries=e.get('amp_retries',0)))
    assert len(audit)==34 and len(records)==28
    assert all(all(v for k,v in a.items() if k!='job') for a in audit), audit
    for r in tested['records']:
        m=r['metrics'];assert m['images']==200 and m['inference_boundary_calls']==0
        assert abs(m['mask_precision']-m['mask_tp']/(m['mask_tp']+m['mask_fp']))<1e-12
        assert abs(m['mask_recall']-m['mask_tp']/(m['mask_tp']+m['mask_fn']))<1e-12
        assert abs(m['mask_f1']-2*m['mask_tp']/(2*m['mask_tp']+m['mask_fp']+m['mask_fn']))<1e-12
    # Matched-seed initialization and recipe equality across every formal job.
    init=[]
    for seed in range(3):
        rr=[r for r in results.values() if r['seed']==seed]
        init.append({'seed':seed,'common_initial_equal':len({r['initial']['common'] for r in rr})==1,
            'H1_H2_boundary_equal':len({r['initial']['boundary'] for r in rr if not r['experiment'].startswith('H0')})==1})
    recipe_equal=len({json.dumps(c['recipe'],sort_keys=True) for c in contracts.values()})==1
    argkeys=['imgsz','batch','optimizer','lr0','lrf','weight_decay','mask_ratio','overlap_mask','mosaic','mixup','hsv_h','hsv_s','hsv_v','scale','translate','fliplr','flipud']
    args_equal={k:len({json.dumps(c['effective_args'].get(k),sort_keys=True) for c in contracts.values()})==1 for k in argkeys}
    # Original split membership and exact-image leakage, plus current image/label hashes.
    manifest=ROOT/'manifests/real_split_observed_v1.jsonl';sources[str(manifest.relative_to(ROOT))]=sha(manifest)
    splitrows=[json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    splitsets={s:{r['image_sha256'] for r in splitrows if r['original_split']==s} for s in ['train','val','test']}
    exact_leaks={a+'_'+b:len(splitsets[a]&splitsets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    real_hash_ok=all(sha(ROOT/r[k])==r[k+'_sha256'] for r in splitrows for k in ['image','label'])
    lowrows=[json.loads(x) for x in (ROOT/'manifests/low187_v1.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    lowstems={Path(r['image']).stem for r in lowrows};selection_audit=[]
    for exp in ['M2','M3','L2','L3']:
        s=load(BASE/'selection'/f'{exp}.json')
        selection_audit.append({'experiment':exp,'count':len(s),'unique_ids':len({r['id'] for r in s}),
            'low_source_ok':all(Path(r['reference']).stem.split('_obj')[0] in lowstems for r in s) if exp.startswith('L') else None})
    judge=load(BASE/'judge_manifest.json')
    audit_summary={'job_checks':audit,'initialization':init,'recipe_equal':recipe_equal,'effective_args_equal':args_equal,
        'split_counts':{s:sum(r['original_split']==s for r in splitrows) for s in splitsets},
        'cross_split_exact_duplicates':exact_leaks,'current_real_image_label_hashes_ok':real_hash_ok,
        'selection_checks':selection_audit,'judge_is_low187':judge['training_subset']=='low187',
        'not_verified':['scene/near-duplicate independence','independent synthetic mask annotation accuracy',
                        'equal-update compute control','new full-dataset inference replay']}
    (out/'audit.json').write_text(json.dumps(audit_summary,indent=2),encoding='utf-8')
    write_csv('per_checkpoint_metrics.csv',flat);write_csv('epoch_metrics.csv',epochs);write_csv('job_integrity.csv',audit)
    groups=defaultdict(list)
    for r in flat:groups[(r['split'],r['experiment'])].append(r)
    summary=[]
    for (split,exp),rows in sorted(groups.items()):
        for k in KEYS:
            vals=[r[k] for r in rows]
            summary.append(dict(split=split,experiment=exp,metric=k,n=len(vals),mean=st.mean(vals),
                                sample_sd=st.stdev(vals) if len(vals)>1 else '',minimum=min(vals),maximum=max(vals)))
    write_csv('summary_mean_sd.csv',summary)
    def vals(exp,key='map',split='Test'):
        return [r[key]*100 for r in sorted(groups[(split,exp)],key=lambda r:r['seed'])]
    def bars(ax,exps,key='map',split='Test'):
        means=[st.mean(vals(e,key,split)) for e in exps];sds=[st.stdev(vals(e,key,split)) if len(vals(e,key,split))>1 else 0 for e in exps]
        ax.bar(range(len(exps)),means,color=COLORS[:len(exps)],alpha=.85,yerr=sds,capsize=4)
        for i,e in enumerate(exps):
            y=vals(e,key,split);ax.scatter(np.linspace(i-.1,i+.1,len(y)),y,color='#263746',s=19,zorder=3)
            ax.text(i,means[i]+sds[i]+1,f'{means[i]:.2f}',ha='center',fontsize=10)
        ax.set_xticks(range(len(exps)),exps);ax.set_ylim(0,max(means)+max(sds)+8);ax.grid(axis='y',alpha=.18)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,p,title in zip(axs,['M','L'],['Full real data (748 images)','Low real data (187 images)']):
        bars(ax,[p+str(i) for i in range(4)]);ax.set(title=title,ylabel='Test mask mAP50-95 (%)')
    save(fig,'01_main_test','M/L: 0 Real only; 1 Heuristic; 2 AnyDoor random; 3 AnyDoor task-aware. Dots: seeds; error bars: sample SD (n=3).')
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for ax,k,label in zip(axs.flat,KEYS,LABELS):
        x=np.arange(4)
        for p,shift,color in [('M',-.18,COLORS[2]),('L',.18,COLORS[3])]:
            y=[vals(p+str(i),k) for i in range(4)]
            ax.bar(x+shift,[st.mean(v) for v in y],.34,yerr=[st.stdev(v) for v in y],capsize=3,color=color,label=p)
        ax.set(xticks=x,xticklabels=['Real','Heuristic','Random','Task-aware'],title=label,ylim=(0,100));ax.legend();ax.grid(axis='y',alpha=.15)
    save(fig,'02_test_metrics','Test200; three-seed mean +/- sample SD. Mask P/R/F1: confidence=0.25, mask IoU=0.50; AP uses confidence floor=0.001.')
    for p in ['M','L']:
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        for i in range(4):
            hs=[histories[f'{p}{i}_seed{s}'] for s in range(3)]
            for ax,key,factor in [(axs[0],'mask_map50_95',100),(axs[1],'loss',1)]:
                arr=np.array([[e[key]*factor for e in h] for h in hs]);x=np.arange(1,151)
                ax.plot(x,arr.mean(0),color=COLORS[i],label=f'{p}{i} {NAMES[i]}')
                ax.fill_between(x,arr.mean(0)-arr.std(0,ddof=1),arr.mean(0)+arr.std(0,ddof=1),color=COLORS[i],alpha=.12)
        axs[0].set(ylabel='Validation mask mAP50-95 (%)',xlabel='Epoch',title=f'{p}: Validation learning curve',ylim=(0,50))
        axs[1].set(ylabel='Recorded mean training loss',xlabel='Epoch',title=f'{p}: Training loss',ylim=(0,None))
        for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.15)
        save(fig,f'03_learning_{p}','All epochs shown without smoothing. Line: seed mean; band: +/- sample SD. Loss is aggregate, not mask-only or Validation loss.')
    fig,axs=plt.subplots(2,4,figsize=(15,7))
    for ax,exp in zip(axs.flat,[p+str(i) for p in ['M','L'] for i in range(4)]):
        for s in range(3):
            h=histories[f'{exp}_seed{s}'];best=results[f'{exp}_seed{s}']['best_mask']
            ax.plot([e['epoch'] for e in h],[e['mask_map50_95']*100 for e in h],label=f'seed{s}')
            ax.scatter(best['epoch'],best['score']*100,s=22)
        ax.set(title=exp,xlabel='Epoch',ylim=(0,50));ax.grid(alpha=.15);ax.legend(fontsize=7)
    save(fig,'04_seed_curves','Validation mask mAP50-95 (%); dots mark Validation-selected best checkpoints, not Test-selected epochs.')
    fig,axs=plt.subplots(1,2,figsize=(13,5));exps=[f'A{i}' for i in range(6)]
    yy=[vals(e,split='Validation')[0] for e in exps]
    axs[0].bar(exps,yy,color=['#708090','#218B82','#D98B28','#708090','#708090','#708090'])
    for i,y in enumerate(yy):axs[0].text(i,y+.3,f'{y:.2f}',ha='center')
    axs[0].set(ylim=(0,30),ylabel='Best Validation mask mAP50-95 (%)',title='Heuristic score ablation')
    for e in exps:
        h=histories[e+'_seed0'];axs[1].plot([r['epoch'] for r in h],[r['mask_map50_95']*100 for r in h],label=e)
    axs[1].set(xlabel='Epoch',ylabel='Validation mask mAP50-95 (%)',title='40-epoch screening');axs[1].legend(ncol=2)
    save(fig,'05_heuristic_ablation','Single seed0; Validation only. A0 random; A1 full; A2 -Tone score; A3 -Texture score; A4 -Edge score; A5 -Placement score.')
    fig,ax=plt.subplots(figsize=(10,5))
    for offset,suffix,ref,color,label in [(-.18,'R','M0',COLORS[0],'Real only'),(.18,'S','M1',COLORS[1],'Real + Heuristic')]:
        y=[vals('H0_'+suffix)[0],vals('H1_'+suffix)[0],vals(ref)[0]]
        ax.bar(np.arange(3)+offset,y,.34,color=color,label=label)
        for i,v in enumerate(y):ax.text(i+offset,v+.5,f'{v:.2f}',ha='center',fontsize=10)
    ax.set(xticks=range(3),xticklabels=['H0: no boundary','H1: ordinary BCE','H2: distance-weighted BCE'],ylim=(0,40),ylabel='Test mask mAP50-95 (%)',title='Boundary-head ablation (matched seed0)');ax.legend()
    save(fig,'06_head_ablation','Single seed, no error bars. H2_R=M0_seed0; H2_S=M1_seed0. H2 does not use three-seed means in this comparison.')
    deltas=[]
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,p in zip(axs,['M','L']):
        for i,(a,b) in enumerate([(1,0),(2,0),(3,0),(3,2)]):
            ds=np.array(vals(p+str(a)))-np.array(vals(p+str(b)))
            for s,d in enumerate(ds):deltas.append(dict(comparison=f'{p}{a}-{p}{b}',seed=s,delta_percentage_points=float(d)))
            ax.scatter(ds,[i]*3,color=COLORS[i],s=45);ax.plot([min(ds),max(ds)],[i,i],color=COLORS[i],alpha=.5)
        ax.axvline(0,color='black',lw=.8);ax.set(yticks=range(4),yticklabels=[f'{p}1 - {p}0',f'{p}2 - {p}0',f'{p}3 - {p}0',f'{p}3 - {p}2'],xlabel='Test mAP difference (percentage points)',title=f'{p}: paired-seed differences');ax.grid(axis='x',alpha=.2)
    save(fig,'07_paired_differences','Each point is a matched training-seed difference on the same Test200. These are descriptive differences, not confidence intervals or significance tests.')
    write_csv('paired_seed_differences.csv',deltas)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,p in zip(axs,['M','L']):
        for i in range(4):
            v=st.mean(vals(p+str(i),split='Validation'));t=st.mean(vals(p+str(i)))
            ax.plot([0,1],[v,t],marker='o',color=COLORS[i],label=p+str(i))
        ax.set(xticks=[0,1],xticklabels=['Selected Validation','Held-out Test'],ylim=(0,50),ylabel='Mask mAP50-95 (%)',title=f'{p}: Validation / Test');ax.legend()
    save(fig,'08_validation_test','Different fixed splits (Val100/Test200); Validation selects best epochs. A gap is not by itself proof of overfitting.')
    fig,axs=plt.subplots(1,2,figsize=(13,5));distribution=[]
    for ax,pool in zip(axs,['train748','low187']):
        r=load(BASE/'selection'/f'{pool}_review.json');domains=list(r['random_statistics']['domains']);x=np.arange(len(domains))
        for kind,offset,color in [('random',-.18,COLORS[2]),('task',.18,COLORS[3])]:
            stats=r[kind+'_statistics'];y=[stats['domains'][d]/30 for d in domains]
            ax.bar(x+offset,y,.34,color=color,label=kind)
            for d,n in stats['domains'].items():distribution.append(dict(pool=pool,selection=kind,domain=d,count=n,percent=n/30))
        ax.set(xticks=x,xticklabels=[d.replace('_','\n') for d in domains],ylabel='Selected images (%)',title=f'{pool}: selection distribution',ylim=(0,35));ax.legend()
    save(fig,'09_selection_distribution','Each selection has 3,000 images from the same pool of 6,000. Distribution shifts are observed associations, not a causal explanation of accuracy.')
    write_csv('selection_domains.csv',distribution)
    resources=[]
    for job,h in histories.items():
        resources.append(dict(job=job,epochs=len(h),optimizer_steps=h[-1]['global_step'],
            recorded_epoch_hours=sum(e['wall_seconds'] for e in h)/3600,
            peak_reserved_mib=max(e.get('train_peak_reserved_mib',0) for e in h),best_epoch=results[job]['best_mask']['epoch']))
    write_csv('training_resources.csv',resources)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,p in zip(axs,['M','L']):
        jobs=[f'{p}{i}_seed0' for i in range(4)];y=[histories[j][-1]['global_step'] for j in jobs]
        ax.bar([p+str(i) for i in range(4)],y,color=COLORS)
        for i,v in enumerate(y):ax.text(i,v+500,f'{v:,}',ha='center')
        ax.set(ylabel='Recorded optimizer steps',title=f'{p}: equal epochs, unequal updates',ylim=(0,max(y)*1.16))
    save(fig,'10_training_budget','150 epochs for all M/L jobs. Synthetic augmentation enlarges each epoch; this design does not isolate data quality at fixed compute.')
    # Per-job curves preserve all ablations and permit close inspection.
    (out/'figures'/'per_job').mkdir()
    for job,h in histories.items():
        fig,axs=plt.subplots(1,2,figsize=(10,3.5));x=[e['epoch'] for e in h]
        axs[0].plot(x,[e['loss'] for e in h],color=COLORS[0]);axs[0].set(title=job+' | train loss',xlabel='Epoch')
        axs[1].plot(x,[e['mask_map50_95']*100 for e in h],color=COLORS[2]);axs[1].set(title='Validation mask mAP50-95 (%)',xlabel='Epoch')
        fig.tight_layout();fig.savefig(out/'figures'/'per_job'/f'{job}.png',dpi=200);plt.close(fig)
    testtable=[]
    for exp in ['M0','M1','M2','M3','L0','L1','L2','L3','H0_R','H1_R','H0_S','H1_S']:
        row={'실험':exp,'seed 수':len(vals(exp))}
        for k,label in zip(KEYS,LABELS):
            v=vals(exp,k);row[label]=f'{st.mean(v):.2f} ± {st.stdev(v):.2f}' if len(v)>1 else f'{v[0]:.2f} (seed0)'
        testtable.append(row)
    report = '''# 졸업작품 최종 실험 분석 및 발표 가이드

## 종합 평가

이 프로젝트의 주된 연구 질문인 **실제 데이터 부족을 합성으로 보완할 수 있는가**에는 긍정적인 결과가 나왔다. 현재 설정에서 AnyDoor 무작위 선택이 M/L 모두 가장 높은 Test 평균을 보였다. 반면 task-aware 우위, 모든 heuristic score의 필요성, distance weighting의 일관된 이점은 입증되지 않았다. 긍정·부정 결과를 함께 제시하면 졸업작품으로 설명 가능한 실험이다. 새로운 방법이 항상 우수하다는 논문 주장까지 뒷받침하는 결과는 아니다.

이 보고서는 동결된 원본 결과를 읽어 생성했다. 재학습·threshold 변경·Test를 이용한 방법 선택은 수행하지 않았다. 오류막대는 training seed 3개의 **표본 표준편차**이며 신뢰구간이나 Test 표본 불확실성이 아니다. 3 seeds는 독립 데이터셋 3개가 아니며 유의성 검정을 주장하지 않는다.

## 실험 구성과 재현 조건

Real train/Validation/Test = 748/100/200장. Low-data는 고정된 187장 subset 하나를 사용한다. M/L은 각 3 seeds, A/H는 seed0. A는 40 epochs, M/L/H는 150 epochs. A는 Validation screening 전용이다. M0/L0 실제만; M1/L1 실제+Heuristic3000; M2/L2 실제+AnyDoor random3000; M3/L3 실제+AnyDoor task-aware3000. AnyDoor는 M/L 별도 6000 candidate에서 각각 두 selection을 구성했다.

기준 모델은 custom DualHeadSegment **n scale**이다. AdamW, batch16, 640px, cosine schedule, lr0=.001, 최종비율 .01, warmup3epochs, weight_decay=.0005, EMA/early stopping 없음. 이전 학기와 구조 계열은 같지만 scale·recipe·background-only 구성이 동일한 재현 실험이라고 주장하지 않는다. H0는 boundary 없음, H1은 일반 BCE, H2는 distance weighting. H2_R/H2_S는 M0_seed0/M1_seed0 참조이다.

## 지표 정의

- Mask mAP50–95: mask IoU .50~.95에서 AP 평균. 주 지표이며 픽셀 accuracy나 평균 IoU 자체가 아니다.
- Mask mAP50: mask IoU .50의 AP. AP 계산 confidence floor=.001, NMS IoU=.7, max_det=300.
- Mask P/R/F1: confidence=.25, mask IoU=.50의 instance matching TP/FP/FN으로 계산한다. 픽셀 Dice/F1이 아니다. 라이브러리의 자체 최적 지점 P/R과 혼합하지 않는다.
- Box mAP는 별도의 보조 지표다. 모든 수치는 표에서 100점 척도이며 증감은 percentage points이다.
- Training loss는 기록된 aggregate loss다. mask/box/boundary별 loss나 Validation loss가 저장된 것으로 표시하지 않는다.

## 전체 Test 결과

'''+table(testtable,list(testtable[0]))+'''

## 본 실험 해석

M2-M0는 +10.96점, L2-L0는 +9.17점이다. Heuristic도 M1-M0 +8.11점, L1-L0 +5.69점으로 합성의 이점이 두 정보 예산에서 관찰된다. L2의 평균22.01은 M0 평균21.03보다 높지만, 이는 동일 비용 비교나 일반적인 데이터 대체율 증명이 아니다. L2는 합성3000장과 외부 pretrained/generator/background 자원을 사용했다. '실제 라벨 25% 조건에서 합성 추가의 보완 가능성을 관찰했다'고 표현한다.

AnyDoor random은 task-aware보다 M에서2.01점, L에서1.76점 높다. 개별 seed 차이는 paired_seed_differences.csv에 전부 보존했다. 현재 Judge의 난도 점수가 유용한 학습 정보를 완전히 나타내지는 않는다는 결과다. task-aware selection은 snow 비중을 M에서 약16.6%→26.7%, L에서16.2%→27.6%로 바꾸었다. 난도, 분포 이동, 근사 mask와의 불일치가 동시에 작용했을 가능성이 있으나 인과 원인은 추가 통제 실험 없이 확정할 수 없다.

## Heuristic Ablation 해석

A0/A1/A2/A3/A4/A5 Validation mAP는22.24/21.73/23.36/21.00/22.19/21.64이다. A2(-Tone score)가 최고이며 A1이 random A0보다 낮다. 모든 구성 요소가 필요하다는 결론은 지지되지 않는다. Texture 제외와 Placement 제외의 감소는 이 screening에서의 관찰이며 작은 차이를 일반화하지 않는다. A는 seed0/40epochs이므로 150epoch M1이나 Test 수치와 직접 서열화하면 안 된다. Tone score 제거는 tone matching 렌더링 제거가 아니며 Edge score 제거도 blending 제거가 아니다. 기존 A1을 사전 고정된 Heuristic reference로 사용한 결과로 설명하고, A1이 screening에서 선택된 최적 설정이라고 서술하지 않는다.

## Head Ablation 해석

seed0 Test에서 Real-only H0/H1/H2=21.12/21.80/21.87, 합성 추가=29.16/30.71/29.90이다. boundary 보조 학습의 관찰상 이점은 있으나 distance weighting의 추가 이점은 일관되지 않는다. H2가 Validation 합성 조건에서 높고 Test에서 H1보다 낮다는 점도 선택 split과 평가 split 차이를 보여준다. 단일 seed이므로 통계적으로 확정된 architecture 우위라고 하지 않는다. H2는 추론 시 boundary head를 실행하지 않으며, 이 실험은 추론 mask 두 개를 융합하는 방식이 아니다.

## 학습 곡선과 성능 수준 평가

34개 모두 예정 epoch를 완료했고 기록된 loss/mAP는 유한하다. 개별 curve와 best epoch를 함께 확인하도록 per_job PNG와 epoch CSV를 제공한다. 손실 감소는 학습이 작동했다는 증거지만 일반화의 충분조건은 아니다. Validation-best checkpoint를 사용했으며 마지막 epoch를 best로 혼동하지 않았다. Validation/Test gap은 서로 다른 split의 차이와 선택 효과를 포함하므로 그 자체로 과적합 원인을 확정하지 않는다.

최고 Test mask mAP50–95 약32점은 본 baseline 대비 개선폭이 크지만, 운영 환경에서 충분한 정확도라는 뜻은 아니다. 엄격한 mask localization과 놓치는 객체에 개선 여지가 남는다. 배포 가능 여부는 사용 환경별 오탐·미탐 요구조건과 별도의 외부 데이터 평가가 필요하다. 1학기 숫자보다 좋다/나쁘다는 통제 비교는 현재 자료로 성립하지 않는다.

## 올바르게 수행됐는가: 검증과 한계

자동 검증 결과는 audit.json과 job_integrity.csv에 기록했다. 34개 완료/epoch 연속성/유한값/체크포인트 hash/Validation 최고값 선택/평가 연결을 검사했다. Test28개는 같은200장 및 같은 동결 record를 사용하며 P/R/F1 산술을 재계산했다. 학습 seed별 공유 초기 tensor, recipe와 주요 augmentation 인자 동일성, real 이미지/label 현재 hash와 cross-split exact duplicate, low187 AnyDoor 선택의 객체 source 범위를 검사했다. 이 검사는 전체 코드의 형식적 증명이나 독립적인 새 추론 재현이 아니다.

주요 한계:

1. **연산량 통제:** 동일150epochs지만 M0는7050updates, M1~3는35250updates; L0는1800, L1~3는30000이다. 합성 추가와 update 증가가 함께 바뀐다. 현재는 실용적인 augmentation recipe 비교이며 동일 compute에서 순수 데이터 효과를 분리하지 못했다. M1/M2/M3끼리, L1/L2/L3끼리는 이미지 수·update 수가 같다.
2. **통계 범위:** M/L3seeds는 고정 split/subset의 optimization 변동만 측정한다. A/H1seed와 Val100/Test200의 제한으로 작은 차이에 강한 주장을 하지 않는다.
3. **독립성:** exact hash 중복 검사는 near-duplicate, 동일 장면/영상 그룹 leakage 부재를 증명하지 않는다. raw source mapping이 미해결된 자료를 완전한 scene-independent benchmark라고 하지 않는다.
4. **근사 annotation:** AnyDoor mask는 사용자 승인한 입력 배치 기반 근사 GT다. 이미지와 mask의 의미적 일치를 독립 정밀 주석으로 보증하지 않았다. SAM2 미사용은 명시하고 label noise 가능성을 남긴다.
5. **추가 자원:** low-data는 대상 실제 train 라벨187장을 제한한 것이며 외부 pretrained와 background 자원까지187장만 사용한 학습이 아니다.
6. **선택 점수:** task-aware는 난도와 label noise를 구분하는 품질 보증이 아니다. 현재 부정 결과를 감추거나 Test를 보고 규칙을 바꿔 같은 Test를 새로운 최종 평가로 포장하지 않는다.

## PPT 권장 구성 (10장)

1. 연구 질문과 fixed split/모델/seed 설정.
2. M/L/A/H 실험 설계와3000장·6000후보 관계.
3. 본 실험 Test 비교: 01_main_test.
4. 부족 데이터 보완: 01_main_test L 패널과 paired differences.
5. 학습 곡선: 03_learning_M/L, 필요시04_seed_curves.
6. 다양한 Metric: 02_test_metrics (mask/box 및 fixed threshold 구분).
7. Heuristic Ablation: 05_heuristic_ablation, Full 최고 아님 명시.
8. Head Ablation: 06_head_ablation, matched seed0 명시.
9. Task-aware 분석: 07_paired_differences +09_selection_distribution.
10. 결론과 한계: augmentation 효과 확인, selection 우위 미확인,10_training_budget.

권장 결론 문장: '고정된 위장망 분할 모델과 데이터 분할에서 합성데이터 추가는 실제데이터 전량 및25% 조건 모두의 성능을 개선했다. AnyDoor 무작위 선택이 가장 높은 평균 성능을 보였으나, 난도 기반 선택과 거리 가중 boundary loss의 추가 이점은 일관되지 않았다.'

## 산출물 사용법과 없는 지표

index.html에서 모든 핵심 그림과 설명을 확인한다. PPT에는300dpi PNG 또는 SVG, 논문에는 vector PDF/SVG를 사용한다. tables CSV는 원래0~1값을 보존한다. smoothing, 가짜 seed, 유의성 별표를 추가하지 않았다. 최종 report의 요약값으로 PR 곡선·confidence 곡선·IoU histogram·pixel confusion matrix를 복원할 수 없으므로 만들지 않았다. 이런 곡선이나 실제 예측 overlay가 필요하면 동결 모델에 대한 별도 raw prediction 수집을 하고 기존 평가와 수치 일치를 검증해야 한다. 본 패키지는 새 모델 추론 없이 만들어졌다.
'''
    # Append measured checks and budget rows rather than replacing failures with narrative.
    report+='\n## 이번 검사 실제 결과\n\n```json\n'+json.dumps({k:v for k,v in audit_summary.items() if k!='job_checks'},ensure_ascii=False,indent=2)+'\n```\n'
    (out/'DETAILED_REPORT_KO.md').write_text(report,encoding='utf-8')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>졸업작품 최종 결과</title><style>body{max-width:1200px;margin:40px auto;font-family:Arial,Malgun Gothic,sans-serif;color:#233746;background:#f6f8fa}article{background:white;padding:24px;margin:25px 0;border-radius:12px}img{width:100%}a{color:#137c78}p{line-height:1.7}</style><h1>졸업작품 최종 실험 결과</h1><p>동결된 실제 기록 기반 · PNG 300dpi / SVG / PDF · seed 오차막대는 표본 표준편차</p><p><a href="DETAILED_REPORT_KO.md">한국어 상세 해석 보고서</a> | <a href="tables/per_checkpoint_metrics.csv">전체 Metric CSV</a> | <a href="audit.json">검증 결과</a></p>'
    for name,caption in figures:
        page+=f'<article><h2>{html.escape(name)}</h2><img src="figures/{name}.png"><p>{html.escape(caption)}</p><a href="figures/{name}.png">PNG</a> | <a href="figures/{name}.svg">SVG</a> | <a href="figures/{name}.pdf">PDF</a></article>'
    page+='<article><h2>개별 학습 곡선 (34개)</h2>'+''.join(f'<p><a href="figures/per_job/{j}.png">{j}</a></p>' for j in histories)+'</article></html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    sources[str(Path(__file__).relative_to(ROOT)).replace('\\','/')]=sha(__file__)
    (out/'source_manifest.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'main_figures':len(figures),'individual_curves':len(histories),'metric_rows':len(flat),'epoch_rows':len(epochs),'audit':{k:v for k,v in audit_summary.items() if k!='job_checks'}},ensure_ascii=False))

if __name__=='__main__':main()
