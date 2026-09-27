"""Consolidate completed S8/S9 and historical evidence without changing runs."""
import csv
import html
import json
import math
import re
import shutil
import statistics as st
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capstone_lab.config import sha256_file

OLD = ROOT / 'artifacts/final_presentation/end_to_end_v3_background'
OUT = ROOT / 'artifacts/final_presentation/end_to_end_v4_complete'
S8 = ROOT / 'artifacts/s8_anydoor_campaign/run_v1'
S9 = ROOT / 'artifacts/s9_background/run_v1'
TZ = timezone(timedelta(hours=9))
ORDER = ['M0','M1','M2','M3','L0','L1','L2','L3','H0_R','H1_R','H0_S','H1_S','BG0','BG1']
KEYS = ['map','map50','precision','recall','f1','box_map','box_map50','fp_per_image']
LABELS = ['Mask mAP50-95','Mask mAP50','Mask precision','Mask recall','Mask F1','Box mAP50-95','Box mAP50','FP/image']
COLORS = {'M0':'#607489','M1':'#DB922C','M2':'#16847A','M3':'#8055A5','BG0':'#879EB0','BG1':'#BD5937'}
SOURCES = {}

def load(p):
    p = Path(p)
    SOURCES[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] = sha256_file(p)
    return json.loads(p.read_text(encoding='utf-8'))

def mdtable(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def metrics(m):
    lib = m['library_metrics']
    return dict(map=lib['metrics/mAP50-95(M)'],map50=lib['metrics/mAP50(M)'],
        precision=m['mask_precision'],recall=m['mask_recall'],f1=m['mask_f1'],
        box_map=lib['metrics/mAP50-95(B)'],box_map50=lib['metrics/mAP50(B)'],
        fp_per_image=m['mask_fp']/m['images'],tp=m['mask_tp'],fp=m['mask_fp'],fn=m['mask_fn'],images=m['images'])

def write_csv(name, rows):
    with (OUT/'tables'/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def jsonout(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def timestamp(value):
    return datetime.fromtimestamp(value,TZ).strftime('%Y-%m-%d %H:%M:%S KST')

def inline(s):
    s=html.escape(s)
    s=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',s)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
    return s

def render_md(text):
    result=[];in_table=False;head=True
    for line in text.splitlines():
        if line.startswith('|'):
            if not in_table:result.append('<div class="table-wrap"><table>');in_table=True;head=True
            if re.fullmatch(r'[| :\-]+',line):continue
            tag='th' if head else 'td';head=False
            result.append('<tr>'+''.join(f'<{tag}>{inline(c.strip())}</{tag}>' for c in line.strip('|').split('|'))+'</tr>')
            continue
        if in_table:result.append('</table></div>');in_table=False
        if not line.strip():continue
        if line.startswith('#'):
            level=min(4,len(line)-len(line.lstrip('#')))
            result.append(f'<h{level}>{inline(line.lstrip("# "))}</h{level}>')
        else:result.append('<p>'+inline(line)+'</p>')
    if in_table:result.append('</table></div>')
    return '\n'.join(result)

def main():
    if OUT.exists():raise RuntimeError('Output already exists; preserve it and choose a new version.')
    status=load(S9/'status.json')
    assert status['status']=='SUCCEEDED' and status['completed_stages']==9
    assert load(S8/'status.json')['status']=='SUCCEEDED'
    frozen_files={}
    for p in sorted((S9/'done').glob('*.json')):
        stage=load(p);assert stage['status']=='SUCCEEDED'
        for rel,h in stage['files'].items():
            assert sha256_file(ROOT/rel)==h,rel
            frozen_files[rel]=h
    assert len(list((S9/'done').glob('*.json')))==9
    legacy_counts={}
    for name in ['source_manifest.json','semester1_source_manifest.json']:
        manifest=load(OLD/name)
        for path,h in manifest.items():
            target=Path(path) if Path(path).is_absolute() else ROOT/path
            assert sha256_file(target)==h,path
        legacy_counts[name]=len(manifest)
    binding=load(S9/'binding.json')
    for path,h in binding.items():assert sha256_file(ROOT/path)==h,path
    freeze8=load(S8/'final_freeze.json');freeze9=load(S9/'final_freeze.json')
    tests8=load(S8/'test/summary.json')['records'];tests9=load(S9/'test/summary.json')['records']
    tests={r['job']:r for r in tests8+tests9}
    assert len(tests8)==28 and len(tests9)==6 and len(tests)==34
    all_frozen=freeze8['all_validation']+freeze9['checkpoints']
    assert len(all_frozen)==40 and len({r['job'] for r in all_frozen})==40
    rows=[];epoch_rows=[];audits=[];resources=[];histories={};results={};contracts={};raw=[]
    for v in all_frozen:
        job=v['job'];exp=v['experiment'];directory=(ROOT/v['checkpoint']).parent
        r=load(directory/'result.json');h=load(directory/'history.json');c=load(directory/'contract.json')
        histories[job]=h;results[job]=r;contracts[job]=c
        expected=40 if exp.startswith('A') else 150
        best=r['best_mask'];selected=next(e for e in h if e['epoch']==best['epoch'])
        test=tests.get(job);is_bg=exp.startswith('BG')
        checks=dict(job=job,completed=r['status']=='SUCCEEDED' and r['formal_training_completed'],
            epochs_complete=[e['epoch'] for e in h]==list(range(1,expected+1)),
            finite_history=all(math.isfinite(e['loss']) and math.isfinite(e['mask_map50_95']) for e in h),
            checkpoint_hash=sha256_file(ROOT/v['checkpoint'])==v['sha256']==best['sha256'],
            val_selected=best['score']==max(e['mask_map50_95'] for e in h)==selected['mask_map50_95'],
            earliest_best=best['epoch']==next(e['epoch'] for e in h if e['mask_map50_95']==best['score']),
            validation_link=metrics(selected['metrics'])==metrics(v['validation']),
            test_link=not test or test.get('checkpoint_sha256',test.get('sha256'))==v['sha256'],
            freeze_link=not test or test['freeze_sha256']==sha256_file((S9 if is_bg else S8)/'final_freeze.json'))
        assert all(value for key,value in checks.items() if key!='job'),checks
        audits.append(checks)
        for split,m in [('Validation',v['validation'])]+([('Test',test['metrics'])] if test else []):
            assert m['images']==(200 if split=='Test' else 100)
            assert m['ap_min_confidence']==.001 and m['nms_iou']==.7 and m['max_det']==300
            assert m['operating_confidence']==.25 and m['operating_mask_iou']==.5
            assert m['inference_boundary_calls']==0
            tp,fp,fn=m['mask_tp'],m['mask_fp'],m['mask_fn']
            for k,value in [('precision',tp/(tp+fp) if tp+fp else 0),('recall',tp/(tp+fn) if tp+fn else 0),('f1',2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)]:
                assert math.isclose(m['mask_'+k],value,abs_tol=1e-12)
            rows.append(dict(job=job,experiment=exp,seed=v['seed'],split=split,epochs=len(h),best_epoch=best['epoch'],**metrics(m),checkpoint=v['checkpoint']))
            raw.append(dict(job=job,split=split,metrics=m,checkpoint=v['checkpoint'],sha256=v['sha256']))
        for e in h:
            epoch_rows.append(dict(job=job,epoch=e['epoch'],global_step=e['global_step'],loss=e['loss'],
                **metrics(e['metrics']),seconds=e['wall_seconds'],amp_retries=e.get('amp_retries',0)))
        resources.append(dict(job=job,epochs=len(h),optimizer_steps=h[-1]['global_step'],
            recorded_epoch_hours=sum(e['wall_seconds'] for e in h)/3600,
            mean_epoch_seconds=st.mean(e['wall_seconds'] for e in h),
            peak_reserved_mib=max(e.get('train_peak_reserved_mib',0) for e in h),
            best_epoch=best['epoch'],amp_retries=sum(e.get('amp_retries',0) for e in h)))
    assert len(rows)==74 and len(epoch_rows)==5340
    same_initial=[]
    for seed in range(3):
        js=[f'{e}_seed{seed}' for e in ['M0','M1','BG0','BG1']]
        a=dict(seed=seed,full_initial_equal=len({results[j]['initial']['full'] for j in js})==1,
               recipe_equal=len({json.dumps(contracts[j]['recipe'],sort_keys=True) for j in js})==1,
               effective_args_equal=len({json.dumps(contracts[j]['effective_args'],sort_keys=True) for j in js})==1)
        assert all(v for k,v in a.items() if k!='seed'),a
        same_initial.append(a)
    OUT.mkdir();(OUT/'tables').mkdir()
    shutil.copytree(OLD/'figures',OUT/'figures')
    shutil.copytree(OLD/'historical_evidence',OUT/'historical_evidence')
    (OUT/'supporting_reports').mkdir()
    for n in ['END_TO_END_REPORT_KO.md','BACKGROUND_REPORT_KO.md','audit.json','semester1_source_manifest.json','source_manifest.json']:
        shutil.copy2(OLD/n,OUT/'supporting_reports'/n)
    for n in ['semester1_metrics.csv','selection_domains.csv']:
        shutil.copy2(OLD/'tables'/n,OUT/'tables'/n)
    with (OLD/'tables/semester1_metrics.csv').open(encoding='utf-8-sig') as f:historical=list(csv.DictReader(f))
    write_csv('all_checkpoint_metrics.csv',rows);write_csv('all_epoch_metrics.csv',epoch_rows)
    write_csv('all_job_integrity.csv',audits);write_csv('all_training_resources.csv',resources)
    jsonout('all_metrics_raw.json',raw)
    groups=defaultdict(list)
    for r in rows:groups[(r['split'],r['experiment'])].append(r)
    def vals(exp,key='map',split='Test'):return [r[key] for r in groups[(split,exp)]]
    def mean(exp,key='map',split='Test'):return st.mean(vals(exp,key,split))
    def fmt(exp,key='map',split='Test'):
        v=vals(exp,key,split);factor=1 if key=='fp_per_image' else 100;dec=3 if factor==1 else 2
        return f'{st.mean(v)*factor:.{dec}f}'+(f' ± {st.stdev(v)*factor:.{dec}f}' if len(v)>1 else ' (seed0)')
    summary=[]
    for (split,exp),rs in sorted(groups.items()):
        for key in KEYS:
            v=[r[key] for r in rs]
            summary.append(dict(split=split,experiment=exp,metric=key,n=len(v),mean=st.mean(v),
                sample_sd=st.stdev(v) if len(v)>1 else '',minimum=min(v),maximum=max(v)))
    write_csv('all_summary_mean_sd.csv',summary)
    paired=[]
    pairs=[('M1','M0'),('M2','M0'),('M3','M2'),('L1','L0'),('L2','L0'),('L3','L2'),('BG0','M0'),('BG1','M1'),('BG1','BG0')]
    for a,b in pairs:
        for seed in range(3):
            ra=next(r for r in groups[('Test',a)] if r['seed']==seed);rb=next(r for r in groups[('Test',b)] if r['seed']==seed)
            paired.append(dict(comparison=a+' minus '+b,seed=seed,**{k:ra[k]-rb[k] for k in KEYS}))
    write_csv('all_paired_seed_differences.csv',paired)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','axes.titlepad':12,'svg.fonttype':'none','pdf.fonttype':42})
    newfigs=[]
    def save(fig,name,caption,individual=False):
        fig.tight_layout(rect=(0,.075,1,.95));fig.text(.012,.015,caption,fontsize=8,color='#53606A')
        target=OUT/'figures'/('per_job' if individual else '')
        target.mkdir(exist_ok=True)
        for ext in (['png'] if individual else ['png','svg','pdf']):fig.savefig(target/f'{name}.{ext}',dpi=300,facecolor='white')
        plt.close(fig)
        if not individual:newfigs.append((name,caption))
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for col,exps in enumerate([['M0','BG0'],['M1','BG1']]):
        for exp in exps:
            hs=[histories[f'{exp}_seed{s}'] for s in range(3)]
            for ax,key,factor in [(axs[0,col],'loss',1),(axs[1,col],'mask_map50_95',100)]:
                y=np.array([[e[key]*factor for e in h] for h in hs]);x=np.arange(1,151)
                ax.plot(x,y.mean(0),label=exp,color=COLORS[exp],lw=1.6)
                ax.fill_between(x,y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),color=COLORS[exp],alpha=.14)
                ax.set_xlabel('Epoch');ax.grid(alpha=.16)
        axs[0,col].set_title('Real only: M0 vs BG0' if col==0 else 'Real + heuristic: M1 vs BG1')
        axs[0,col].set_ylabel('Aggregate training loss');axs[1,col].set_ylabel('Validation mask mAP50-95 (%)')
        axs[0,col].legend();axs[1,col].set_ylim(bottom=0)
    save(fig,'18_background_learning','Raw epoch curves; mean +/- sample SD over 3 seeds. Loss depends on the data mixture; no smoothing.')
    fig,axs=plt.subplots(1,3,figsize=(13,4.6))
    for ax,key,title,factor in zip(axs,['map','recall','fp_per_image'],['Change in mask mAP (pp)','Change in mask recall (pp)','Change in FP / image'],[100,100,1]):
        for i,(a,b) in enumerate([('BG0','M0'),('BG1','M1')]):
            d=[r[key]*factor for r in paired if r['comparison']==a+' minus '+b]
            ax.scatter(np.array([i-.09,i,i+.09]),d,s=45,color=COLORS[a],zorder=3)
            ax.errorbar(i,st.mean(d),yerr=st.stdev(d),fmt='D',color='black',capsize=4,ms=5)
        ax.axhline(0,color='#59636C',lw=1);ax.set_xticks([0,1],['BG0 - M0','BG1 - M1']);ax.set_title(title);ax.grid(axis='y',alpha=.15)
    save(fig,'19_background_paired_effects','Dots: matched-seed differences. Black diamonds: mean +/- sample SD; not confidence intervals or significance tests.')
    fig,axs=plt.subplots(1,3,figsize=(15,5),sharey=True)
    histtest=[r for r in historical if r['split']=='Test']
    histlabels=['S4','S3','S4 + HM','S3 + HM'];histvals=[float(r['mask_map50_95'])*100 for r in histtest]
    panels=[(histlabels,histvals,[0]*4,['#647B91','#CF923A','#A3B0BC','#DEC394'],'Semester 1 archive (one run)'),
        (['M0','M1','M2','M3'],[mean(e)*100 for e in ['M0','M1','M2','M3']],[st.stdev(vals(e))*100 for e in ['M0','M1','M2','M3']],[COLORS[e] for e in ['M0','M1','M2','M3']],'Current main study (3 seeds)'),
        (['M0','BG0','M1','BG1'],[mean(e)*100 for e in ['M0','BG0','M1','BG1']],[st.stdev(vals(e))*100 for e in ['M0','BG0','M1','BG1']],[COLORS[e] for e in ['M0','BG0','M1','BG1']],'Background follow-up (3 seeds)')]
    for ax,(names,y,err,colors,title) in zip(axs,panels):
        bars=ax.bar(names,y,yerr=err,capsize=4,color=colors,width=.65)
        for bar,v,e in zip(bars,y,err):ax.text(bar.get_x()+bar.get_width()/2,v+e+.6,f'{v:.2f}',ha='center',fontsize=9)
        ax.set_title(title,fontsize=11);ax.set_ylim(0,41);ax.grid(axis='y',alpha=.15)
    axs[0].set_ylabel('Test mask mAP50-95 (%)')
    save(fig,'20_complete_research_context','Historical context, not a controlled cross-semester comparison. HM = sequential hard mining; current bars show sample SD.')
    fig,axs=plt.subplots(1,3,figsize=(13,4.8))
    for ax,key,title in zip(axs,['map','precision','recall'],['Mask mAP50-95 (%)','Mask precision (%)','Mask recall (%)']):
        for names,color,label in [(['M0','M1'],COLORS['M1'],'No background-only'),(['BG0','BG1'],COLORS['BG1'],'+ 2,129 background-only')]:
            ax.errorbar([0,1],[mean(e,key)*100 for e in names],yerr=[st.stdev(vals(e,key))*100 for e in names],fmt='o-',capsize=4,color=color,label=label)
        ax.set_xticks([0,1],['No synthetic','+ 3,000 heuristic']);ax.set_title(title);ax.set_xlim(-.2,1.2);ax.grid(alpha=.15)
    axs[0].legend(loc='lower right',fontsize=8)
    save(fig,'21_background_factorial','2 x 2 descriptive comparison; matched seed sets. Same background count, unequal negative fractions and optimizer updates.')
    fig,axs=plt.subplots(1,3,figsize=(14,6))
    exps=['M0','M1','M2','M3','L0','L1','L2','L3','BG0','BG1'];y=np.arange(len(exps))
    for ax,key,title in zip(axs,['map','f1','fp_per_image'],['Mask mAP50-95 (%)','Mask F1 @ conf .25 (%)','False positives / image']):
        factor=1 if key=='fp_per_image' else 100
        ax.errorbar([mean(e,key)*factor for e in exps],y,xerr=[st.stdev(vals(e,key))*factor for e in exps],fmt='o',capsize=3,color='#16847A')
        ax.set_yticks(y,exps);ax.invert_yaxis();ax.set_title(title);ax.set_xlim(left=0);ax.grid(axis='x',alpha=.18)
    save(fig,'22_current_results_overview','Fixed Test200, mean +/- sample SD (3 seeds). BG conditions are post-hoc; FP/image is not background-only specificity.')
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,exps,title in zip(axs,[['M0','M1','M2','M3','BG0','BG1'],['L0','L1','L2','L3']],['Full-data and background follow-up','Low-data (187 real images)']):
        for e in exps:
            rr=[r for r in resources if r['job'].startswith(e+'_seed')];x=st.mean(r['optimizer_steps'] for r in rr)/1000;v=mean(e)*100
            ax.errorbar(x,v,yerr=st.stdev(vals(e))*100,fmt='o',capsize=4,color=COLORS.get(e,'#16847A'))
            ax.annotate(e,(x,v),xytext=(7,0),textcoords='offset points',fontsize=9)
        ax.set_title(title);ax.set_xlabel('Optimizer updates (thousands)');ax.set_ylabel('Test mask mAP50-95 (%)');ax.grid(alpha=.18);ax.margins(x=.2,y=.2)
    save(fig,'23_training_budget_context','Equal epochs do not imply equal training compute. Points show current recipe outcomes, not isolated causal effects of update count.')
    for exp in ['BG0','BG1']:
        for seed in range(3):
            job=f'{exp}_seed{seed}';h=histories[job];fig,axs=plt.subplots(1,2,figsize=(10,3.7));x=[e['epoch'] for e in h]
            axs[0].plot(x,[e['loss'] for e in h],color=COLORS[exp]);axs[0].set_ylabel('Aggregate training loss')
            axs[1].plot(x,[e['mask_map50_95']*100 for e in h],color=COLORS[exp]);axs[1].set_ylabel('Validation mask mAP50-95 (%)')
            for ax in axs:ax.set_xlabel('Epoch');ax.grid(alpha=.18)
            best=results[job]['best_mask'];axs[1].axvline(best['epoch'],ls='--',color='#A44E35');fig.suptitle(job)
            save(fig,job,f"Raw 150-epoch history. Validation-selected best epoch {best['epoch']}; no Test-based checkpoint selection.",True)
    newfig_titles={
        '18_background_learning':'Background 학습 곡선: 기존 대조군과 평균·seed 변동 비교',
        '19_background_paired_effects':'동일 seed에서 background 추가에 따른 mAP·Recall·오탐 변화',
        '20_complete_research_context':'1학기·이번 본 실험·background 후속 결과의 역사적 맥락',
        '21_background_factorial':'합성 유무 × background 유무의 2×2 비교',
        '22_current_results_overview':'M/L/BG 전체 주요 Test 지표',
        '23_training_budget_context':'성능과 총 optimizer update 수의 관계'}
    (OUT/'FIGURE_INDEX.md').write_text((OLD/'FIGURE_INDEX.md').read_text(encoding='utf-8')+
        '\n\n## 최종 통합판 추가 figure\n\n총 핵심23종(PNG 300 dpi, SVG, PDF)과 학습별40종 PNG. 기존16종/17종 표기는 이전 버전의 구성이다.\n\n'+
        mdtable(['파일','내용'],[(n,newfig_titles[n]) for n,_ in newfigs])+'\n',encoding='utf-8')
    report=['# 위장 객체 분할: 1학기부터 최종 Background 실험까지 통합 결과',
        f"작성: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M KST')}. 추가 campaign 완료: {timestamp(status['updated'])}.",
        '## 1. 완료 상태와 핵심 결론',
        '**현재 학기 학습 40개가 모두 완료됐다.** 기존 34개와 background 6개, 총 5,340 epoch 기록이다. 고유 모델 34개가 같은 Test 200장으로 평가됐으며 A0~A5 6개는 계획대로 Validation screening이다. H2_R/H2_S는 M0/M1 seed0 참조이므로 중복 모델로 세지 않는다. 1학기의 보관 실행·hard-mining 연속학습 기록은 별도로 구분한다.',
        '합성 추가의 성능 개선은 두 학기 내부 비교에서 관찰됐다. 이번 본 실험에서는 AnyDoor random이 전체·25% 실제 데이터 조건 모두에서 가장 높은 평균을 보였다. 반면 task-aware 우위, 모든 heuristic score의 필요성, distance weighting의 일관된 우위는 확인되지 않았다.',
        '**최종 background 실험은 오탐과 미탐의 절충을 보여준다.** Background 추가 후 Precision은 높아지고 FP/image는 줄었지만 Recall과 mAP는 감소했다. 따라서 background를 더 넣으면 전체 분할 성능이 좋아진다는 가설은 현재 설정에서 지지되지 않았다.',
        '이는 오류로 학습이 실패한 결과와 다르다. 모든 학습의 예정 epoch·유한값·최고 Validation checkpoint·Test 연결을 검증했다. 다만 전체 코드의 정확성이나 인과적 원인까지 증명한 것은 아니다.',
        '## 2. 실험 구성과 지표',
        '현재 Real train/Validation/Test는 748/100/200장이다. L은 train의 고정 187장과 그 출처의 foreground/Judge만 사용한다. Heuristic은 A0~A5 각 3,000장과 L용 A1 3,000장으로 21,000장, AnyDoor는 M/L 후보 6,000장씩 12,000장이다. 정식 합성 총 33,000장이고, random/task-aware는 동일 후보에서 각 3,000장을 선택하므로 선택별 신규 생성량을 중복 합산하지 않는다.',
        mdtable(['분류','조건','반복/epochs','평가'],[
            ['1학기','S4 real+BG, S3 real+heuristic+BG; 각 hard mining 후속','보관된 조건별 실행, HM은 30epoch×3 연속 loop','보관 Val/Test CSV'],
            ['M0~M3','real748 + 없음/heuristic/AnyDoor random/task-aware','3 seeds ×150','Val best → Test200'],
            ['L0~L3','real187 + 같은 4가지 전략','3 seeds ×150','Val best → Test200'],
            ['A0~A5','random/full/−Tone/−Texture/−Edge/−Placement','seed0 ×40','Validation100만'],
            ['H0/H1/H2','boundary 없음/BCE/distance-weighted BCE, Real/합성','seed0 ×150, H2는 M0/M1 참조','Val best → Test200'],
            ['BG0/BG1','real748 + BG2129 / real748 + A1 3000 + 같은 BG2129','3 seeds ×150','사후 보완; 동일 Test200 재사용']]),
        '현재 기본 모델은 custom DualHeadSegment H2 n-scale이다. batch16, 640px, AdamW, lr0=.001, lrf=.01, warmup3epochs, cosine, EMA/early stopping 없음. 같은 seed의 M0/M1/BG0/BG1 초기 전체 tensor hash·recipe·effective args가 일치함을 다시 확인했다. 데이터량과 negative 비율·총 update 수는 다르다.',
        '표에서 AP/P/R/F1은 0~1값×100, 증감은 percentage points(pp), 오탐은 FP/image이다. ±는 3 training seed의 표본 표준편차이며 Test 표본에 대한 신뢰구간이 아니다. seed0 조건에는 가짜 SD를 붙이지 않았다.',
        '현재 Mask P/R/F1은 confidence=.25와 mask IoU=.5의 instance TP/FP/FN이다. AP는 confidence floor=.001, NMS IoU=.7, max_det300을 사용한다. Pixel Dice/accuracy와 다르다. 1학기 P/R/F1은 Box 기준이므로 현재 Mask P/R/F1과 직접 비교하지 않는다.',
        '## 3. 1학기 전체 보관 결과',
        mdtable(['조건','Split','Box P','Box R','Box F1','Box AP50','Box AP50–95','Mask AP50','Mask AP50–95'],
            [[r['model'],r['split']]+[f"{float(r[k])*100:.2f}" for k in ['box_precision','box_recall','box_f1','box_map50','box_map50_95','mask_map50','mask_map50_95']] for r in historical]),
        'S3−S4의 Test mask mAP 개선은 +7.10pp(27.76→34.86)이다. Hard mining은 S4 −0.29pp, S3 −1.32pp로 엄격한 mask AP의 추가 개선을 보이지 않았다. S3의 Box F1은 69.43→70.82로 증가했으므로 지표별 반응을 구분한다. 위 표는 원본의 반올림 저장값이며 이번에 새로 추론한 결과가 아니다.',
        '노트북 초기 S3 로그는 748 real+2934 synthetic+2129 background=5811장이다. epoch140 재개 시 1985 background/5667 valid 이미지 기록이 있어 1학기 전체 고정 membership으로 단정할 수 없다. 이번 background2129는 초기 S3의 archive/확장자 규칙과 수를 맞췄다. S4 전체 원본 membership, 모든 과거 best 선택 epoch는 미확인이다.',
        '## 4. 현재 학기 전체 Test 결과',
        mdtable(['조건','n']+LABELS,[[e,len(groups[('Test',e)])]+[fmt(e,k) for k in KEYS] for e in ORDER]),
        'M0/L0 실제만, M1/L1 Heuristic 추가, M2/L2 AnyDoor random, M3/L3 AnyDoor task-aware이다. H0/H1은 구조 분해, BG0/BG1은 후속 background 조건이다.',
        '## 5. 합성 전략과 실제 데이터 부족에 대한 해석']
    for a,b in [('M1','M0'),('M2','M0'),('L1','L0'),('L2','L0')]:
        report.append(f"{a}−{b}: Test mask mAP50–95 **{(mean(a)-mean(b))*100:+.2f}pp** ({mean(b)*100:.2f}→{mean(a)*100:.2f}).")
    report += [f"L2 평균 {mean('L2')*100:.2f}는 M0 평균 {mean('M0')*100:.2f}보다 수치상 높다. 실제 라벨 25% 환경에서 합성의 보완 가능성을 보여주지만, 외부 pretrained/generator/background와 추가 update를 사용했으므로 실제 데이터 75% 대체율을 증명한 것은 아니다.",
        f"Task-aware−random은 M에서 {(mean('M3')-mean('M2'))*100:+.2f}pp, L에서 {(mean('L3')-mean('L2'))*100:+.2f}pp다. 현재 Judge 난도 기준의 우위는 관찰되지 않았다. snow 분포 이동, 어려움과 label noise의 혼합은 가능한 설명이며 인과적으로 검증된 원인이 아니다.",
        '## 6. 최종 Background 2×2 실험',
        mdtable(['조건','Real','합성','BG','BG 비율','Updates/seed'],[
            ['M0',748,0,0,'0%',7050],['BG0',748,0,2129,f'{2129/2877*100:.2f}%',27000],
            ['M1',748,3000,0,'0%',35250],['BG1',748,3000,2129,f'{2129/5877*100:.2f}%',55200]])]
    for a,b in [('BG0','M0'),('BG1','M1')]:
        red=(1-mean(a,'fp_per_image')/mean(b,'fp_per_image'))*100
        ds=[r['map']*100 for r in paired if r['comparison']==a+' minus '+b]
        report.append(f"**{a}−{b}**: mAP {(mean(a)-mean(b))*100:+.2f}pp, Precision {(mean(a,'precision')-mean(b,'precision'))*100:+.2f}pp, Recall {(mean(a,'recall')-mean(b,'recall'))*100:+.2f}pp, F1 {(mean(a,'f1')-mean(b,'f1'))*100:+.2f}pp. FP/image {mean(b,'fp_per_image'):.3f}→{mean(a,'fp_per_image'):.3f}로 약 {red:.1f}% 감소. mAP seed별 차이는 {', '.join(f'{d:+.2f}' for d in ds)}pp로 3개 seed 모두 감소했다.")
    interaction=(mean('BG1')-mean('BG0'))-(mean('M1')-mean('M0'))
    report += [f"Background 없는 합성 효과 M1−M0는 {(mean('M1')-mean('M0'))*100:+.2f}pp, 있는 합성 효과 BG1−BG0는 {(mean('BG1')-mean('BG0'))*100:+.2f}pp다. 차이의 차이는 {interaction*100:+.2f}pp이며 유의성 검정이나 인과적 interaction 증명이 아닌 기술통계다.",
        'Background 추가 후 모델은 현재 threshold에서 더 적은 양성 예측과 높은 Precision, 낮은 Recall을 보였다. BG1의 F1은 소폭 높지만 mAP가 낮으므로 전체 성능 개선으로 표현하지 않는다. Background 비중이 큰 학습이 영향을 줬을 가능성은 있지만, 비율 sweep이나 equal-update 실험 없이 최적 비율이나 원인을 단정할 수 없다.',
        '이 결과는 1학기와의 성능 차이가 단순히 background 누락 때문이라고 설명하기 어렵다는 점을 보여준다. 현재 BG1의 mAP도 과거 S3보다 낮지만 과거 recipe·합성수·평가 경로·membership 차이가 남아 직접적 우열 판정은 피한다. Background를 넣는 것이 언제나 해롭다는 일반화도 하지 않는다.',
        'FP/image는 객체가 있는 real Test200에서 측정한 instance 오탐이다. 새 background-only holdout의 특이도/FPR이 아니다. 기존 Test 결과를 확인한 후 추가했으므로 사후 보완 실험이며 새로운 독립 Test 검증이라고 서술하지 않는다.',
        '## 7. Ablation 결과',
        mdtable(['조건','정의','Validation mask mAP50–95'],[[e,name,fmt(e,'map','Validation')] for e,name in zip(['A0','A1','A2','A3','A4','A5'],['Random control','Full','minus Tone score','minus Texture score','minus Edge score','minus Placement score'])]),
        'A2가 가장 높고 Full A1은 A0보다 낮다. 따라서 모든 score가 필요하다는 증거는 부족하다. seed0/40epochs/Validation screening이므로 작은 차이를 확정적 기여도로 해석하지 않는다. Score 제거는 tone matching이나 blending 렌더링 제거와 다르다. A1은 사전에 고정한 reference이지 screening 최적값으로 선택한 조건이 아니다.',
        mdtable(['데이터','H0: boundary 없음','H1: boundary BCE','H2: distance weighting'],[
            [label]+[f"{tests[j]['metrics']['library_metrics']['metrics/mAP50-95(M)']*100:.2f}" for j in jobs]
            for label,jobs in [('Real748',['H0_R_seed0','H1_R_seed0','M0_seed0']),('Real748+A1',['H0_S_seed0','H1_S_seed0','M1_seed0'])]]),
        'H0/H1/H2는 같은 seed0 비교이다. Boundary 보조 학습은 관찰상 이점이 있지만 distance weighting의 추가 이점은 일관되지 않는다. 단일 seed로 일반적인 구조 우위를 주장하지 않는다. H2는 추론 시 boundary head를 실행하지 않는다.',
        '## 8. 전체 Validation 결과',
        mdtable(['조건','n']+LABELS,[[e,len(groups[('Validation',e)])]+[fmt(e,k,'Validation') for k in KEYS] for e in ['A0','A1','A2','A3','A4','A5']+ORDER]),
        'Validation은 best checkpoint 선택용이며 Test는 해당 checkpoint의 평가다. 서로 다른 split의 차이는 과적합 원인 자체를 입증하지 않는다.',
        '## 9. 학습·평가 무결성 및 완료 판단',
        '40개 학습에서 예정 epoch 연속성, loss/mAP 유한성, Validation 최고값 및 동점 시 최초 epoch 선택, checkpoint SHA-256, Test가 동결 checkpoint를 가리키는지 검사했다. Test34개 모두 200장과 동일 AP/운영 threshold를 사용하고 P/R/F1을 TP/FP/FN으로 재계산했다. S9의 9단계 완료 표식 및 산출물 hash, 기존 보고서 입력과 1학기 근거 hash도 재검사했다. 상세는 audit_complete.json/all_job_integrity.csv이다.',
        '이번 정리는 기록 검증과 재집계이며 새 GPU 추론을 반복한 것은 아니다. 기존 audit의 real 파일 hash·exact split 중복0·low187 출처 검사를 참조하되 새 near-duplicate/scene 누수 감사나 합성 mask 독립 주석을 완료했다고 주장하지 않는다.',
        '## 10. 전체 학습 자원·선택 epoch',
        mdtable(['Job','Epochs','Updates','기록된 epoch 합(시간)','Best Val epoch','AMP 재시도'],[[r['job'],r['epochs'],r['optimizer_steps'],f"{r['recorded_epoch_hours']:.2f}",r['best_epoch'],r['amp_retries']] for r in resources]),
        '시간은 history에 기록된 epoch wall_seconds의 합이며 대기·초기화 등을 포함한 전체 campaign 경과 시간과 다르다. AMP 재시도는 수치 overflow를 자동 처리한 횟수이며 그 자체가 중단 또는 결과 무효를 뜻하지 않는다.',
        '## 11. 졸업작품으로서의 평가와 결론',
        '구현 성과는 custom 모델·출처를 기록한 33,000장 합성·전략 선택·자동 학습/재개·동결 Test·재현 가능한 보고 체계다. 실험 성과는 합성 효능, low-data 보완, 선택/score/head의 부정 결과, background 오탐–미탐 절충을 구분해 검증했다는 점이다. 이 범위면 졸업작품의 실험은 충분히 구성됐다. 추가 학습보다 데모·정성 성공/실패 예시·발표 논리 정리가 우선이다.',
        '최종 권장 결론: **고정된 위장 객체 분할 모델과 현재 학습 프로토콜에서 합성데이터는 전량 및 25% 실제 데이터 조건의 성능을 개선했다. 그러나 난도 기반 선택과 추가 head 구성의 복잡성이 항상 성능 향상으로 이어지지는 않았다. Background-only 추가는 오탐을 줄이는 대신 Recall과 mAP를 낮추어 데이터 구성과 운영 목표의 절충을 보였다.**',
        '한계: 동일 epoch·다른 update 예산, 고정 split와 작은 Test200, 3seed는 학습 난수 변동만 측정, A/H 단일 seed, scene/near-duplicate 독립성 미확인, AnyDoor 근사 mask, 1학기 완전 재현 아님, background 사후 평가. 동일 compute 우위·범용적 75% 대체·통계적 유의성·현장 배포 수준을 주장하지 않는다.',
        '## 12. 발표 구성과 파일 안내',
        '발표 순서: 연구 문제 → 1학기 모델/합성/HardMining → 데이터·모델·자동화 → M/L 결과 → A/H ablation → task-aware 부정 결과 → background2×2 → 종합 기여와 한계. Figure20은 역사적 맥락,01/22는 성능,18은 학습,19/21은 background 효과,23은 학습 예산 한계에 사용한다.',
        'index.html은 전체 보고서와 23개 figure를 오프라인으로 연다. PNG는300dpi, PPT용 SVG와 논문용 PDF도 제공한다. 모든 40개 학습 curve는 figures/per_job. tables/all_checkpoint_metrics.csv에는74개 checkpoint/split행, tables/all_epoch_metrics.csv에는5340epoch행, all_metrics_raw.json에는 library 지표를 포함한 원본 metric 객체가 있다. 1학기 원본 CSV·curve는 historical_evidence와 semester1_metrics.csv에 있다.',
        '원시 예측이 없는 PR곡선·confidence곡선·픽셀 confusion·IoU histogram을 요약 수치로 만들지 않았다. 기존 학습 곡선은 보간이나 가짜 seed 없이 저장 기록으로 그렸다.',
        '## 부록. 현재 학기 모든 seed·checkpoint 수치',
        mdtable(['Job','Split','Best epoch']+LABELS+['TP','FP','FN'],[[r['job'],r['split'],r['best_epoch']]+[f"{r[k]*(1 if k=='fp_per_image' else 100):.3f}" for k in KEYS]+[r['tp'],r['fp'],r['fn']] for r in rows])]
    report_text='\n\n'.join(report)+'\n'
    (OUT/'FINAL_REPORT_KO.md').write_text(report_text,encoding='utf-8')
    audit=dict(status='VERIFIED',current_training_jobs=40,current_unique_test_models=34,validation_only_models=6,
        epoch_rows=5340,metric_rows=74,s9_completed_at=timestamp(status['updated']),job_checks=audits,
        matched_background_controls=same_initial,verified_prior_manifests=legacy_counts,
        s9_frozen_files=len(frozen_files),new_gpu_inference=False,
        limitations=['No new near-duplicate audit','No independent mask reannotation','No equal-update control','Reused Test for post-hoc S9'])
    jsonout('audit_complete.json',audit)
    jsonout('source_manifest_complete.json',SOURCES)
    (OUT/'CHATGPT_HANDOFF_KO.md').write_text('''# 최종 보고서·PPT 작성용 전달 안내

CHATGPT_HANDOFF_v4_complete.zip 하나를 전달한다. 우선 FINAL_REPORT_KO.md, audit_complete.json, FIGURE_INDEX.md, tables/all_summary_mean_sd.csv를 읽도록 요청한다. 상세 seed는 all_checkpoint_metrics.csv, 학습은 all_epoch_metrics.csv, 1학기는 semester1_metrics.csv와 historical_evidence를 참조한다.

권장 프롬프트:

첨부 파일의 실제 수치와 figure를 근거로 소프트웨어 4학년 졸업작품 최종 보고서와 발표 구성을 작성해줘. 1학기부터 합성 전략·low-data·A/H ablation·AnyDoor 선택·background 후속 2×2까지 하나의 연구 흐름으로 설명해줘. 모든 주장에 조건/split/seed수를 명시하고 긍정·부정 결과를 함께 해석해줘. 현재 학습40개/Test34개이며 A6개는 Val-only야. Background는 오탐이 줄고 Recall/mAP도 낮아졌다는 결과를 반영해줘. 1학기 P/R/F1은 Box이고 현재는 fixed-threshold Mask이므로 혼합하지마. 학기간 통제 비교, 동일 compute, 유의성, PR곡선 등 없는 증거는 만들지마. Figure20의 학기간 비교는 역사적 맥락으로 표시하고 ±는3seed 표본SD로 설명해줘. 각 PPT 슬라이드에 해당 figure 파일명과 핵심 해석을 붙여줘.

벡터 figure PDF/SVG와 개별40개 학습PNG는 전체 FINAL_RESULTS_v4_complete.zip에 추가 포함된다. 요약 ZIP에는23개 핵심PNG와 원시 수치/전체epoch CSV/보고서/출처 검증이 포함된다.
''',encoding='utf-8')
    style='''body{font:16px/1.75 system-ui,sans-serif;background:#eef2f5;color:#213245;margin:0}main{max-width:1250px;margin:auto;padding:32px}header,article,section{background:white;padding:28px;margin:20px 0;border-radius:12px}h1{font-size:30px}h2{border-bottom:2px solid #16847a;padding-bottom:8px;margin-top:40px}a{color:#146f7b}.table-wrap{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%;margin:18px 0}th,td{padding:8px 12px;border-bottom:1px solid #dee5eb;white-space:nowrap;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf3f2}img{max-width:100%;height:auto}code{background:#edf1f5;padding:2px 4px}nav{display:flex;gap:16px;flex-wrap:wrap}@media print{body{background:white}main{padding:0}article{break-inside:avoid}a{color:inherit}}'''
    gallery=[]
    for p in sorted((OUT/'figures').glob('*.png')):
        title=newfig_titles.get(p.stem,p.stem.replace('_',' '))
        gallery.append(f'<article><h3>{html.escape(title)}</h3><img loading="lazy" src="figures/{p.name}" alt="{html.escape(title)}"><p><a href="figures/{p.stem}.png">PNG</a> · <a href="figures/{p.stem}.svg">SVG</a> · <a href="figures/{p.stem}.pdf">PDF</a></p></article>')
    page=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>위장 객체 분할 | 최종 통합 결과</title><style>{style}</style></head><body><main><header><h1>위장 객체 분할 · 최종 실험 결과</h1><p>1학기부터 Background 보완 실험까지 · 학습40개 / Test34개 / 핵심figure23종</p><nav><a href="#report">전체 보고서</a><a href="#figures">Figure 모음</a><a href="FINAL_REPORT_KO.md">Markdown</a><a href="tables/all_checkpoint_metrics.csv">전체 수치 CSV</a><a href="CHATGPT_HANDOFF_KO.md">ChatGPT 전달 안내</a></nav></header><section id="report">{render_md(report_text)}</section><section id="figures"><h2>발표·논문용 Figure</h2>{''.join(gallery)}</section></main></body></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    for target in re.findall(r'(?:href|src)="([^"]+)"',page):
        if not target.startswith('#'):assert (OUT/target).exists(),target
    assert len(list((OUT/'figures').glob('*.png')))==23
    assert len(list((OUT/'figures/per_job').glob('*.png')))==40
    jsonout('delivery_manifest.json',{p.relative_to(OUT).as_posix():sha256_file(p) for p in OUT.rglob('*') if p.is_file()})
    for name,compact in [('CHATGPT_HANDOFF_v4_complete.zip',True),('FINAL_RESULTS_v4_complete.zip',False)]:
        target=OUT.parent/name
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob('*'):
                if not p.is_file():continue
                if compact and ('per_job' in p.relative_to(OUT).parts or p.suffix in ['.svg','.pdf']):continue
                z.write(p,p.relative_to(OUT).as_posix())
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
    print(json.dumps({'output':str(OUT),'audit':audit['status'],'jobs':40,'test_models':34,'figures':23,'per_job_figures':40,'completed':timestamp(status['updated'])},ensure_ascii=False))

if __name__=='__main__':main()
