import csv,json,shutil,statistics as st,zipfile
from pathlib import Path

def generate(root,base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    old=read(root/'artifacts/s8_anydoor_campaign/run_v1/test/summary.json')['records']
    new=read(base/'test/summary.json')['records']
    assert len(new)==6 and {r['job'] for r in new}=={f'{e}_seed{s}' for e in ['BG0','BG1'] for s in range(3)}
    rows=[]
    for r in [r for r in old if r['experiment'] in ['M0','M1']]+new:
        m=r['metrics'];l=m['library_metrics']
        rows.append(dict(job=r['job'],experiment=r['experiment'],seed=r['seed'],map=l['metrics/mAP50-95(M)'],map50=l['metrics/mAP50(M)'],precision=m['mask_precision'],recall=m['mask_recall'],f1=m['mask_f1'],fp_per_image=m['mask_fp']/m['images'],fn_rate=m['mask_fn']/(m['mask_tp']+m['mask_fn'])))
    d=base/'report';d.mkdir(exist_ok=True)
    with (d/'background_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Background-only 후속 보완 실험','','기존 Test 확인 후 추가한 사후 보완 실험이다. 신규 독립 Test 검증으로 해석하지 않는다.','',
        '| 조건 | Test mask mAP50–95 평균±SD | Mask P | Mask R | Mask F1 | FP/image |',
        '| --- | --- | --- | --- | --- | --- |']
    names=['M0','BG0','M1','BG1']
    for e in names:
        rs=[r for r in rows if r['experiment']==e];vs=[r['map']*100 for r in rs]
        lines.append('| '+e+f' | {st.mean(vs):.2f} ± {st.stdev(vs):.2f} | '+' | '.join(f"{st.mean(r[k] for r in rs)*100:.2f}" for k in ['precision','recall','f1'])+f" | {st.mean(r['fp_per_image'] for r in rs):.3f} |")
    lines+=['','BG0=real748+background2129; BG1=real748+A1synthetic3000+동일background2129. 각3seeds/150epochs/H2.',
        '1학기 초기 S3 background2129 수와 archive 확장자 규칙을 맞췄다. 1학기 재개시1985장으로 바뀐 기록이 있어 전체 학습 membership까지 동일하다고 주장하지 않는다.',
        '기존1학기는 합성2934, 현재3000이며 augmentation/EMA/scheduler 등의 차이가 남는다. 1학기 완전 재현 실험이 아니다.',
        'negative 비율은 BG0 약74.00%, BG1 약36.23%로 다르다. 배경의 절대개수만 통제했다.',
        '같은150epochs라도 update 수는 M0 7050/BG0 27000, M1 35250/BG1 55200이다. 동일 compute 효과 분리는 불가능하다.',
        'FP/image는 real Test200에서의 instance 오탐이다. 독립 background-only Test의 specificity/FPR이 아니다. 배경의 target 부재는 archive에 기반한 가정이다.','', '## Seed별 효과와 해석']
    for bg,control in [('BG0','M0'),('BG1','M1')]:
        deltas={k:[] for k in ['map','recall','fp_per_image']}
        for s in range(3):
            a=next(r for r in rows if r['experiment']==bg and r['seed']==s);b=next(r for r in rows if r['experiment']==control and r['seed']==s)
            for k in deltas:deltas[k].append(a[k]-b[k])
        lines+=['',bg+'−'+control+': mAP seed별 '+', '.join(f'{x*100:+.2f}' for x in deltas['map'])+f"점; 평균 {st.mean(deltas['map'])*100:+.2f}점.",
            f"Recall 평균 변화 {st.mean(deltas['recall'])*100:+.2f}점; FP/image 변화 {st.mean(deltas['fp_per_image']):+.3f}.",
            '평균 mAP 개선이 관찰됐다.' if st.mean(deltas['map'])>0 else '평균 mAP 개선은 관찰되지 않았다.',
            '오탐 감소만으로 성공이라 하지 않고 recall 손실과 함께 해석한다. 3seed SD는 Test 표본 불확실성의 신뢰구간이 아니다.']
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    for ax,key,title,factor in zip(axs,['map','recall','fp_per_image'],['Test mask mAP50-95 (%)','Mask recall @ conf .25 (%)','False positives / Test image'],[100,100,1]):
        values=[[r[key]*factor for r in rows if r['experiment']==e] for e in names]
        ax.bar(names,[st.mean(v) for v in values],yerr=[st.stdev(v) for v in values],capsize=4,color=['#526779','#95A4AF','#218B82','#89C1B8']);ax.set_title(title);ax.set_ylim(bottom=0)
    fig.tight_layout(rect=(0,.08,1,1));fig.text(.01,.02,'Post-hoc supplement; reused Test200; n=3, sample SD. Same2129 backgrounds; unequal update counts.',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(d/f'17_background_supplement.{ext}',dpi=300)
    plt.close(fig)
    path=d/'BACKGROUND_REPORT_KO.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    original=root/'artifacts/final_presentation/end_to_end_v2';dest=root/'artifacts/final_presentation/end_to_end_v3_background'
    if not dest.exists():shutil.copytree(original,dest)
    for p in d.iterdir():
        target=dest/('figures' if p.suffix in ['.png','.svg','.pdf'] else 'tables' if p.suffix=='.csv' else '')/p.name;shutil.copy2(p,target)
    (dest/'END_TO_END_REPORT_KO.md').write_text((original/'END_TO_END_REPORT_KO.md').read_text(encoding='utf-8')+'\n\n# 추가 보완 실험\n\n'+path.read_text(encoding='utf-8'),encoding='utf-8')
    (dest/'FIGURE_INDEX.md').write_text((original/'FIGURE_INDEX.md').read_text(encoding='utf-8')+'\n\n17_background_supplement: background2129 후속2×2 비교. 재사용Test,3seed 표본SD.\n',encoding='utf-8')
    page=(original/'index.html').read_text(encoding='utf-8').replace('</html>','<article><h2>Background-only 후속 실험</h2><img src="figures/17_background_supplement.png"><p><a href="BACKGROUND_REPORT_KO.md">상세 해석</a></p></article></html>')
    (dest/'index.html').write_text(page,encoding='utf-8')
    (dest/'CHATGPT_HANDOFF_KO.md').write_text((original/'CHATGPT_HANDOFF_KO.md').read_text(encoding='utf-8').replace('CHATGPT_HANDOFF_v2.zip','CHATGPT_HANDOFF_v3_background.zip')+'\n\n추가필수: BACKGROUND_REPORT_KO.md, tables/background_metrics.csv, figures/17_background_supplement.png. 기존34개에6개 후속학습이 추가됐으며, 주 캠페인과 사후 보완실험을 분리해 서술한다.\n',encoding='utf-8')
    bundle=dest.parent/'CHATGPT_HANDOFF_v3_background.zip';tmp=bundle.with_suffix('.building.zip')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in dest.rglob('*'):
            if p.is_file() and 'per_job' not in p.parts and p.name!='epoch_metrics.csv' and p.suffix not in ['.pdf','.svg']:z.write(p,p.relative_to(dest).as_posix())
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    from capstone_lab.campaign.io import atomic_replace
    atomic_replace(tmp,bundle)
    return [path,d/'background_metrics.csv',d/'17_background_supplement.png',dest/'END_TO_END_REPORT_KO.md',dest/'index.html',bundle]
