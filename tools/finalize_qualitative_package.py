"""Attach actual demo test evidence and package the completed v5 deliverables."""
import json
import re
import shutil
import zipfile
from pathlib import Path
from qualitative_core import ROOT,BASE,OUTPUT,read
from build_complete_results_report import render_md
from capstone_lab.config import sha256_file
from capstone_lab.campaign.io import atomic_json

def main():
    api=read(BASE/'demo_verification/api.json');browser=read(BASE/'demo_verification/browser.json')
    assert api['status']==browser['status']=='VERIFIED'
    previous=(OUTPUT/'PRESENTATION_SCRIPT_KO.md').read_text(encoding='utf-8')
    notes=(ROOT/'tools/presentation_script_template.md').read_text(encoding='utf-8')
    (OUTPUT/'PRESENTATION_SCRIPT_KO.md').write_text(notes,encoding='utf-8')
    presentation=(OUTPUT/'presentation.html').read_text(encoding='utf-8')
    presentation=re.sub(r'<section>.*?</section>',lambda _: '<section>'+render_md(notes)+'</section>',presentation,flags=re.S)
    (OUTPUT/'presentation.html').write_text(presentation,encoding='utf-8')
    result=f"""## 실제 데모 검증 결과

고정 checkpoint의 실제 GPU 업로드 추론과 Edge headless 브라우저 상호작용을 검증했다. 전체200장/10모델 선택, 추천 사례 전환, 정답 영역 확대, 새 이미지 업로드, file:// 오프라인 예측 재생을 통과했다. JavaScript 예외는0건이고 잘못된 모델/이미지 요청은400으로 거부했다.

M2 seed0, FP32, 입력640, batch1의 워밍업 후3장×5회(15회) 측정에서 forward+후처리 중앙값은 **{api['warm_forward_postprocess_median_ms']:.2f}ms**, P95는 **{api['warm_forward_postprocess_p95_ms']:.2f}ms**였다. 표본이작은참고측정이며 모델 로딩·전처리·네트워크·브라우저 표시를 제외한다. 이를 end-to-end FPS나 모든 입력의 latency 보장으로 표현하지 않는다. 첫 서버 요청은 로딩/워밍업을 포함해 {api['cold_server_ms']:.1f}ms였다.

비교 예시1의 batch1 API 예측과 기존 batch16 평가 예측의 mask union IoU는 {api['batch1_vs_recorded_batch16_mask_iou']:.3f}였다. 별도로10개 모델 전체200장 replay의공식지표와객체TP/FP/FN이기존기록과일치했다. 근거는 demo_verification/api.json과 browser.json이다.
"""
    (OUTPUT/'DEMO_GUIDE_KO.md').write_text((OUTPUT/'DEMO_GUIDE_KO.md').read_text(encoding='utf-8')+'\n\n'+result,encoding='utf-8')
    report=(OUTPUT/'FINAL_REPORT_KO.md').read_text(encoding='utf-8').replace(previous,notes)+'\n\n'+result
    (OUTPUT/'FINAL_REPORT_KO.md').write_text(report,encoding='utf-8')
    page=(OUTPUT/'index.html').read_text(encoding='utf-8')
    page=re.sub(r'(<section id="report">).*?(</section><section id="figures">)',lambda m:m.group(1)+render_md(report)+m.group(2),page,flags=re.S)
    (OUTPUT/'index.html').write_text(page,encoding='utf-8')
    (OUTPUT/'demo_verification').mkdir(exist_ok=True)
    for name in ['api.json','browser.json','demo_browser.png']:shutil.copy2(BASE/'demo_verification'/name,OUTPUT/'demo_verification'/name)
    audit=read(OUTPUT/'audit_qualitative.json');audit['live_api_test']='VERIFIED';audit['browser_test']='VERIFIED';audit['timing']=api
    atomic_json(OUTPUT/'audit_qualitative.json',audit)
    sources=read(OUTPUT/'qualitative_source_manifest.json')
    for p in [ROOT/'tools'/n for n in ['build_qualitative_package.py','presentation_script_template.md','verify_prediction_demo.py','finalize_qualitative_package.py']]:sources[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    atomic_json(OUTPUT/'qualitative_source_manifest.json',sources)
    for name in ['index.html','cases.html','presentation.html','demo.html']:
        for link in re.findall(r'(?:href|src)="([^"]+)"',(OUTPUT/name).read_text(encoding='utf-8')):
            if not link.startswith(('#','http:','https:')):assert (OUTPUT/link).exists(),(name,link)
    assert len(list((OUTPUT/'cases').glob('*.png')))==14
    assert len(list((OUTPUT/'figures').glob('*.png')))==25
    atomic_json(OUTPUT/'delivery_manifest.json',{p.relative_to(OUTPUT).as_posix():sha256_file(p) for p in OUTPUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'})
    bundles=[]
    for name,compact in [('FINAL_RESULTS_v5_demo.zip',False),('CHATGPT_HANDOFF_v5_demo.zip',True)]:
        target=OUTPUT.parent/name
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
            for p in OUTPUT.rglob('*'):
                if not p.is_file():continue
                rel=p.relative_to(OUTPUT)
                if compact and ('per_job' in rel.parts or p.suffix in ['.svg','.pdf'] or (rel.parts[0]=='cases' and p.suffix=='.png')):continue
                z.write(p,rel.as_posix())
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
        bundles.append({'file':target.name,'bytes':target.stat().st_size})
    print(json.dumps({'status':'VERIFIED','cases':14,'models':10,'images':200,'figures':25,'bundles':bundles},ensure_ascii=False))

if __name__=='__main__':main()
