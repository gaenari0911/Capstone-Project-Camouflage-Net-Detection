"""Add a reader guide without modifying any v5 artifact or scientific result."""
import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import zipfile
from pathlib import Path
from readable_experiment_catalog import catalog, HEADS
from build_complete_results_report import render_md, mdtable

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/final_presentation/end_to_end_v5_demo'
OUT=ROOT/'artifacts/final_presentation/end_to_end_v6_readable'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(name,text):
    (OUT/name).write_text(text,encoding='utf-8')

def dump(name,value):write(name,json.dumps(value,ensure_ascii=False,indent=2))

def table(rows,short=False):
    headers=['ID','쉽게 읽는 이름','실제','합성','배경-only','합계']
    if not short:headers+=['모델','epochs','seed','평가']
    lines=[]
    for r in rows:
        line=[r['id'],r['label']]+['미확인' if r[k] is None else f'{r[k]:,}' for k in ['real','synthetic','background','total']]
        if not short:line += [r['head'],str(r['epochs']),r['seeds'],r['evaluation']]
        lines.append(line)
    return mdtable(headers,lines)

NAV='''<div class="lookup-nav" role="navigation" aria-label="보고서 읽기 도움말"><button type="button" data-experiment="BG0">실험 ID 검색·설정 보기</button><a href="experiment_guide.html">실험·지표 사전</a><a href="index.html">전체 결과</a><a href="demo.html">예측 데모</a></div>'''

def enhance(page):
    page=page.replace('</head>','<link rel="stylesheet" href="experiment_lookup.css"></head>',1)
    page=page.replace('<body>','<body>'+NAV,1)
    return page.replace('</body>','<script src="experiment_catalog.js"></script><script src="experiment_lookup.js"></script></body>',1)

def validate_catalog(rows):
    metrics=list(csv.DictReader((OLD/'tables/all_checkpoint_metrics.csv').open(encoding='utf-8-sig')))
    assert len(metrics)==74
    for r in rows:
        if r['group']=='1학기':continue
        ids=[r['alias']] if r['alias'] else [r['id']+'_seed'+s.strip() for s in r['seeds'].split('/')]
        for job in ids:
            records=[m for m in metrics if m['job']==job]
            assert records,(r['id'],job)
            assert all(int(m['epochs'])==r['epochs'] for m in records),job
            assert {m['split'] for m in records}==({'Validation'} if r['group']=='A' else {'Validation','Test'}),job
            assert all(int(m['images'])==(100 if m['split']=='Validation' else 200) for m in records)
    return len({r['job'] for r in metrics})

def build():
    if OUT.exists():raise RuntimeError('Preserve existing output; choose a new version instead of overwriting.')
    rows=catalog();assert len(rows)==26 and len({r['id'] for r in rows})==26
    assert validate_catalog(rows)==40
    old_manifest=json.loads((OLD/'delivery_manifest.json').read_text(encoding='utf-8'))
    for name,digest in old_manifest.items():assert sha(OLD/name)==digest,name
    shutil.copytree(OLD,OUT)
    # Retain inherited delivery manifest as historical evidence, not a stale current manifest.
    shutil.copy2(OLD/'delivery_manifest.json',OUT/'supporting_reports/v5_delivery_manifest.json')
    dump('experiment_catalog.json',rows)
    write('experiment_catalog.js','window.EXPERIMENT_CATALOG='+json.dumps(rows,ensure_ascii=False)+';\nwindow.EXPERIMENT_HEADS='+json.dumps(HEADS,ensure_ascii=False)+';\n')
    with (OUT/'tables/experiment_lookup.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name in ['experiment_lookup.css','experiment_lookup.js']:shutil.copy2(ROOT/'tools'/name,OUT/name)
    guide=(ROOT/'tools/experiment_lookup_template.md').read_text(encoding='utf-8').replace('{{CATALOG_TABLE}}',table(rows))
    write('EXPERIMENT_LOOKUP_KO.md',guide)
    content=render_md(guide)
    anchors=['names','table','models','comparisons','metrics','demo','training','limits','sources']
    for number,anchor in enumerate(anchors,1):content=content.replace(f'<h2>{number}.',f'<h2 id="guide-{anchor}">{number}.',1)
    for r in rows:content=content.replace(f'<tr><td>{html.escape(r["id"])}</td>',f'<tr id="exp-{r["id"].replace("+","-")}"><td>{html.escape(r["id"])}</td>',1)
    toc=''.join(f'<a href="#guide-{a}">{n}</a>' for a,n in zip(anchors,['이름','전체 표','모델','비교 목적','지표','데모 색상','학습·seed','해석 주의','근거']))
    guide_page=f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>실험·지표 사전 | 위장 객체 분할</title></head><body style="margin:0;background:#edf2f5"><main class="lookup-page"><header><h1>실험 ID를 쉬운 말로</h1><p>M2는 어떤 데이터? BG0는 어떤 모델? 아래 이름을 누르면 설정을 바로 볼 수 있습니다.</p><nav class="lookup-topics">{toc}</nav><p><a href="EXPERIMENT_LOOKUP_KO.md">Markdown 내려받기</a> · <a href="tables/experiment_lookup.csv">전체 설정 CSV</a></p></header><section class="lookup-content">{content}</section></main></body></html>'
    write('experiment_guide.html',enhance(guide_page))
    intro='<section class="lookup-intro"><h2>처음 읽으시나요? 실험 이름부터 확인하세요</h2><p><strong>M</strong>=실제748, <strong>L</strong>=실제187(25%), <strong>A</strong>=휴리스틱 선택 점수, <strong>H</strong>=head/loss, <strong>BG</strong>=배경-only 추가입니다. 보고서의 밑줄 친 실험명을 누르면 설정이 열립니다.</p><p class="lookup-notice"><strong>예: BG0</strong>는 H2 모델을 실제748 + 배경2129 =2877장으로 학습한 조건입니다. G0라는 정식 ID는 없습니다. <strong>A5는 AnyDoor가 아닙니다.</strong></p><p><a href="experiment_guide.html">전체 실험·지표 사전 먼저 읽기</a> · <a href="experiment_guide.html#guide-comparisons">어떤 모델끼리 비교할까?</a> · <a href="experiment_guide.html#guide-metrics">mAP / IoU / TP·FP·FN 뜻</a></p><details><summary>데모에서 사용하는 10개 실험의 데이터 장수 보기</summary>'+render_md(table(rows[:10],short=True))+'<p>모두 H2 / 150epochs / seed0·1·2. 데모는 seed0, 공식 표는3seed 평균입니다.</p></details></section>'
    for name in ['index.html','demo.html','cases.html','presentation.html']:
        page=(OLD/name).read_text(encoding='utf-8')
        if name=='index.html':page=page.replace('</header>','</header>'+intro,1)
        elif name=='demo.html':
            page=page.replace('</header>','<p><a href="experiment_guide.html#guide-demo">색상·IoU·오탐을 읽는 방법</a> · <a href="experiment_guide.html#guide-table">전체 모델 설정표</a></p></header>',1)
        else:page=page.replace('<main>','<main class="case-content">',1)
        write(name,enhance(page))
    script=(OLD/'demo.js').read_text(encoding='utf-8')
    assert script.count('async function render(){')==1
    write('demo.js',script.replace('async function render(){','async function render(){window.refreshModelDetails?.();',1))
    old_report=(OLD/'FINAL_REPORT_KO.md').read_text(encoding='utf-8')
    write('FINAL_REPORT_KO.md','# v6 읽기 도움말\n\n실험 ID·지표·데모 색상은 [실험 사전](EXPERIMENT_LOOKUP_KO.md)을 먼저 참고하세요. 수치·그림·예측은 v5와 동일하며 설명 UI만 추가했습니다.\n\n'+old_report)
    demo_guide=(OLD/'DEMO_GUIDE_KO.md').read_text(encoding='utf-8').replace('tools\\serve_prediction_demo.py','tools\\serve_readable_demo.py')
    write('DEMO_GUIDE_KO.md',demo_guide+'\n\n## v6 읽기 도움말\n\n실험명 클릭으로 설정을 찾고, 선택한 모델 아래의 실제/합성/배경 장수와 두 조건의 비교 목적을 확인합니다. experiment_guide.html 또는 EXPERIMENT_LOOKUP_KO.md에 전체 사전이 있습니다. 새 이미지 서버 명령은 위의 serve_readable_demo.py를 사용합니다. v5의 추론 엔진·checkpoint를 그대로 재사용하며 이번 UI 작업에서 새 GPU 추론은 실행하지 않았습니다.\n')
    handoff=(OLD/'CHATGPT_HANDOFF_KO.md').read_text(encoding='utf-8').replace('v5_demo.zip','v6_readable.zip').replace('23개 핵심PNG','25개 핵심PNG')
    write('CHATGPT_HANDOFF_KO.md',handoff+'\n\n## v6 읽기 사전\n\nEXPERIMENT_LOOKUP_KO.md와 tables/experiment_lookup.csv를 먼저 읽어 실험 이름·데이터 수·head/loss·seed·평가 split을 확인한다. 본 설명은 새 학습/평가가 아니며 공식 수치는 기존 결과를 유지한다. 전체 HTML은 FINAL_RESULTS_v6_readable.zip에 있고, ChatGPT용 묶음은 Markdown/CSV/PNG/JPG 중심 자료다.\n')
    dump('audit_readability.json',{'status':'PENDING_BROWSER_CHECK','source_version':OLD.name,'source_delivery_sha256':sha(OLD/'delivery_manifest.json'),
        'experiment_definitions':26,'training_jobs_crosschecked':40,'checkpoint_metric_rows':74,
        'new_training':False,'new_gpu_inference':False,'prior_version_preserved':True})
    print(json.dumps({'status':'BUILT_PENDING_BROWSER_CHECK','output':str(OUT),'definitions':26},ensure_ascii=False),flush=True)

def package():
    browser=json.loads((OUT/'lookup_verification/browser.json').read_text(encoding='utf-8'))
    assert browser['status']=='VERIFIED'
    changed={'index.html','demo.html','cases.html','presentation.html','demo.js','FINAL_REPORT_KO.md','DEMO_GUIDE_KO.md','CHATGPT_HANDOFF_KO.md','delivery_manifest.json'}
    old_manifest=json.loads((OLD/'delivery_manifest.json').read_text(encoding='utf-8'))
    unchanged=0
    for name,digest in old_manifest.items():
        assert sha(OLD/name)==digest,('original_changed',name)
        if name not in changed:assert sha(OUT/name)==digest,('scientific_artifact_changed',name);unchanged+=1
    broken=[]
    for page in OUT.glob('*.html'):
        for link in re.findall(r'(?:href|src)="([^"]+)"',page.read_text(encoding='utf-8')):
            if link.startswith(('#','http:','https:','data:')):continue
            if not (page.parent/html.unescape(link.split('#')[0])).exists():broken.append((page.name,link))
    assert not broken,broken
    audit=json.loads((OUT/'audit_readability.json').read_text(encoding='utf-8'))
    audit.update(status='VERIFIED',browser=browser,unchanged_inherited_files=unchanged,broken_local_links=broken,
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'tools'/n for n in ['readable_experiment_catalog.py','experiment_lookup_template.md','experiment_lookup.css','experiment_lookup.js','build_readable_results.py','verify_readable_results.py','serve_readable_demo.py']]})
    dump('audit_readability.json',audit)
    dump('delivery_manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'})
    targets=[OUT.parent/'FINAL_RESULTS_v6_readable.zip',OUT.parent/'CHATGPT_HANDOFF_v6_readable.zip']
    assert not any(p.exists() for p in targets),'Preserve existing bundles'
    bundles=[]
    for target,compact in zip(targets,[False,True]):
        members=[]
        for p in OUT.rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(OUT)
            if compact and ('per_job' in rel.parts or p.suffix in ['.html','.js','.css','.svg','.pdf'] or rel.parts[0] in ['images','lookup_verification','demo_verification'] or (rel.parts[0]=='cases' and p.suffix=='.png')):continue
            members.append(p)
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
            for p in members:
                if compact and p.name=='delivery_manifest.json':continue
                z.write(p,p.relative_to(OUT).as_posix())
            if compact:z.writestr('bundle_manifest.json',json.dumps({p.relative_to(OUT).as_posix():sha(p) for p in members if p.name!='delivery_manifest.json'},ensure_ascii=False,indent=2))
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
        bundles.append({'path':str(target),'bytes':target.stat().st_size})
    print(json.dumps({'status':'VERIFIED','unchanged_inherited_files':unchanged,'bundles':bundles},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',action='store_true');a=p.parse_args()
    package() if a.package else build()
