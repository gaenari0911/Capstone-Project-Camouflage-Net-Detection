"""Build real-prediction cases, offline/live demo assets and speaker notes."""
import csv
import html
import json
import re
import shutil
import statistics as st
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from qualitative_core import ROOT,BASE,OUTPUT,JOBS,read,unmask
from capstone_lab.config import sha256_file
from capstone_lab.campaign.io import atomic_json
from build_complete_results_report import render_md,mdtable

PREVIOUS=ROOT/'artifacts/final_presentation/end_to_end_v4_complete'
NAMES={f'{e}_seed0':n for e,n in [('M0','M0 · 실제 748'),('M1','M1 · + 휴리스틱'),('M2','M2 · + AnyDoor random'),
    ('M3','M3 · + task-aware'),('BG0','BG0 · 실제 + 배경'),('BG1','BG1 · 실제 + 합성 + 배경'),
    ('L0','L0 · 실제 187'),('L1','L1 · + 휴리스틱'),('L2','L2 · + AnyDoor random'),('L3','L3 · + task-aware')]}

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def overlay(rgb,gt,pred=None):
    out=rgb.copy();g=cv2.resize(gt.astype('uint8'),(640,640),interpolation=cv2.INTER_NEAREST).astype(bool)
    if pred is None:
        contours,_=cv2.findContours(g.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        out[g]=(out[g]*.72+np.array([66,190,128])*.28).astype('uint8')
        cv2.drawContours(out,contours,-1,(66,230,128),2)
    else:
        p=cv2.resize(pred.astype('uint8'),(640,640),interpolation=cv2.INTER_NEAREST).astype(bool)
        for region,color in [(p&g,[66,190,128]),(g&~p,[255,173,66]),(p&~g,[239,102,157])]:
            out[region]=(out[region]*.5+np.array(color)*.5).astype('uint8')
    return out

def main():
    assert read(BASE/'status.json')['status']=='SUCCEEDED'
    if OUTPUT.exists():raise RuntimeError('Preserve existing output; choose a new version')
    images=read(BASE/'images.json');loaded={j:read(BASE/j/'result.json') for j in JOBS}
    assert all(r['status']=='VERIFIED' and r['counts_match'] for r in loaded.values())
    preds={j:r['records'] for j,r in loaded.items()}
    for j in JOBS:assert [r['index'] for r in preds[j]]==list(range(200))
    def row(i,e):return preds[e+'_seed0'][i]
    def delta(i,a,b,key='union_iou'):return row(i,a)[key]-row(i,b)[key]
    spread=[]
    for i in range(200):
        scores=[row(i,e)['union_iou'] for e in ['M0','M1','M2','M3','BG0','BG1']]
        spread.append(dict(index=i,source=images[i]['source'],model_union_iou_range=max(scores)-min(scores),
            model_union_iou_population_sd=st.pstdev(scores),M2_minus_M0=delta(i,'M2','M0'),
            M3_minus_M2=delta(i,'M3','M2'),BG1_minus_M1=delta(i,'BG1','M1'),L2_minus_L0=delta(i,'L2','L0')))
    used=set();cases=[]
    def add(title,kind,left,right,score,condition=lambda i:True,forced=None):
        eligible=[i for i in range(200) if i not in used and condition(i)]
        fallback=False
        if not eligible:eligible=[i for i in range(200) if i not in used];fallback=True
        i=int(forced) if forced is not None else max(eligible,key=lambda i:(score(i),-i))
        used.add(i);a=row(i,left);b=row(i,right)
        reason=f"{title}: {left}→{right}의 union IoU {a['union_iou']:.3f}→{b['union_iou']:.3f} (차이 {b['union_iou']-a['union_iou']:+.3f}), 객체 TP/FP/FN {a['tp']}/{a['fp']}/{a['fn']}→{b['tp']}/{b['fp']}/{b['fn']}. "
        reason+=('고정 난수로 뽑은 보조 예시입니다.' if kind=='fixed_random' else '설명 목적의 사후 선별 사례이며 전체 성능을 대표하지 않습니다.')
        if fallback:reason+=' 해당 조건에 맞는 미선택 이미지가 없어 점수 기준 대체 사례를 사용했습니다.'
        cases.append(dict(id=f'case_{len(cases)+1:02d}_test_{i:03d}',title=title,kind=kind,index=i,
            left=left+'_seed0',right=right+'_seed0',reason=reason,selection_fallback=fallback,
            union_iou_delta=b['union_iou']-a['union_iou'],model_range=spread[i]['model_union_iou_range']))
    add('01 합성으로 놓친 객체를 찾은 사례','synthesis_gain','M0','M2',lambda i:delta(i,'M2','M0'),lambda i:delta(i,'M2','M0','tp')>0)
    add('02 합성의 큰 개선 사례','synthesis_gain','M0','M2',lambda i:delta(i,'M2','M0'))
    add('03 합성 추가 후 악화한 사례','synthesis_regression','M0','M2',lambda i:-delta(i,'M2','M0'))
    add('04 Task-aware가 random보다 낮은 사례','selection_regression','M2','M3',lambda i:-delta(i,'M3','M2'))
    add('05 Background가 오탐을 줄인 사례','background_fp_reduction','M1','BG1',lambda i:-delta(i,'BG1','M1','fp')+delta(i,'BG1','M1'),lambda i:delta(i,'BG1','M1','fp')<0 and delta(i,'BG1','M1','tp')>=0)
    add('06 Background 추가 후 객체를 놓친 사례','background_recall_loss','M1','BG1',lambda i:-delta(i,'BG1','M1','tp')-delta(i,'BG1','M1'),lambda i:delta(i,'BG1','M1','tp')<0)
    add('07 실제187장 조건에서 합성의 개선','low_data_gain','L0','L2',lambda i:delta(i,'L2','L0'))
    add('08 실제187장 조건에서 합성의 악화','low_data_regression','L0','L2',lambda i:-delta(i,'L2','L0'))
    add('09 모델 간 차이가 가장 큰 추가 사례','high_spread','M0','M2',lambda i:spread[i]['model_union_iou_range'])
    median=st.median(s['model_union_iou_range'] for s in spread)
    add('10 모델 간 차이가 중간인 사례','median_spread','M0','M2',lambda i:-abs(spread[i]['model_union_iou_range']-median))
    random_ids=np.random.default_rng(20260922).choice(200,2,replace=False)
    for k,i in enumerate(random_ids):add(f'{11+k:02d} 고정 난수 보조 사례 {k+1}','fixed_random','M0','M2',lambda _:0,forced=i)
    add('13 여러 모델에서 잘 분할한 사례','shared_success','M0','M2',lambda i:min(row(i,e)['union_iou'] for e in ['M0','M1','M2','M3','BG0','BG1']))
    add('14 여러 모델에서 어려웠던 사례','shared_failure','M0','M2',lambda i:-st.mean(row(i,e)['union_iou'] for e in ['M0','M1','M2','M3','BG0','BG1']))
    shutil.copytree(PREVIOUS,OUTPUT)
    # The v4 hash inventory describes its original package, retained under supporting_reports.
    shutil.copy2(OUTPUT/'delivery_manifest.json',OUTPUT/'supporting_reports/v4_delivery_manifest.json')
    shutil.copytree(BASE/'images',OUTPUT/'images')
    target=OUTPUT/'cases';target.mkdir()
    flat=[]
    for j in JOBS:
        for r in preds[j]:flat.append({k:v for k,v in r.items() if k not in ['mask_bits','boxes','confidences']})
    write_csv(OUTPUT/'tables/predictions_per_image.csv',flat)
    write_csv(OUTPUT/'tables/image_model_spread.csv',spread)
    write_csv(OUTPUT/'tables/selected_cases.csv',cases)
    atomic_json(OUTPUT/'case_selection.json',{'seed_policy':'fixed seed0','random_seed':20260922,'cases':cases,
        'scope':'post-hoc explanatory selection; not population-representative or seed-variance measurement',
        'spread_models':['M0','M1','M2','M3','BG0','BG1'],'spread_metric':'population SD and range of per-image union IoU across six different models'})
    (OUTPUT/'demo_data.js').write_text('window.DEMO_DATA='+json.dumps({'jobs':JOBS,'names':NAMES,'images':images,'predictions':preds,'cases':cases},ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    shutil.copy2(ROOT/'tools/qualitative_demo.html',OUTPUT/'demo.html')
    shutil.copy2(ROOT/'tools/qualitative_demo.js',OUTPUT/'demo.js')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','pdf.fonttype':42,'svg.fonttype':'none'})
    for c in cases:
        i=c['index'];rgb=np.asarray(Image.open(BASE/images[i]['image']).convert('RGB'));gt=unmask(images[i]['gt_bits'])
        models=['L0','L1','L2','L3','M0','M2'] if c['kind'].startswith('low_data') else ['M0','M1','M2','M3','BG0','BG1']
        fig,axs=plt.subplots(2,4,figsize=(15,8.4));axs=axs.ravel()
        axs[0].imshow(rgb);axs[0].set_title('Original input');axs[1].imshow(overlay(rgb,gt));axs[1].set_title('Ground truth (evaluation grid)')
        for ax,e in zip(axs[2:],models):
            r=row(i,e);ax.imshow(overlay(rgb,gt,unmask(r['mask_bits'])))
            ax.set_title(f"{e} | union IoU {r['union_iou']:.3f}\nTP {r['tp']} / FP {r['fp']} / FN {r['fn']}",fontsize=10)
        for ax in axs:ax.axis('off')
        fig.suptitle(f"{c['id']} | {c['kind'].replace('_',' ')} | fixed seed0",fontsize=15)
        fig.tight_layout(rect=(0,.05,1,.96));fig.text(.015,.016,'Green: mask overlap. Orange: GT-only area. Pink: prediction-only area. Post-hoc example, not a representative performance estimate.',fontsize=9)
        for ext in ['png','pdf']:fig.savefig(target/f"{c['id']}.{ext}",dpi=220,facecolor='white')
        plt.close(fig)
        with Image.open(target/f"{c['id']}.png") as im:im.thumbnail((1500,900));im.convert('RGB').save(target/f"{c['id']}.jpg",quality=93)
    # A compact figure with clear gains, regressions and background effects.
    fig,axs=plt.subplots(3,4,figsize=(12,9.6))
    for axes,c in zip(axs,[cases[0],cases[2],cases[4]]):
        i=c['index'];rgb=np.asarray(Image.open(BASE/images[i]['image']).convert('RGB'));gt=unmask(images[i]['gt_bits'])
        axes[0].imshow(rgb);axes[0].set_title(f"Test {i:03d}: {c['kind'].replace('_',' ')}",fontsize=9)
        axes[1].imshow(overlay(rgb,gt));axes[1].set_title('Ground truth')
        for ax,j in zip(axes[2:],[c['left'],c['right']]):
            r=preds[j][i];ax.imshow(overlay(rgb,gt,unmask(r['mask_bits'])));ax.set_title(f"{j[:-6]} | union IoU {r['union_iou']:.3f}")
        for ax in axes:ax.axis('off')
    fig.tight_layout(rect=(0,.045,1,1));fig.text(.01,.01,'Fixed seed0; selected explanatory cases. Green overlap / orange GT only / pink prediction only. See all Test200 in demo.html.',fontsize=8)
    for ext in ['png','svg','pdf']:fig.savefig(OUTPUT/'figures'/f'24_prediction_examples.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4.4))
    for ax,a,b in zip(axs,['M2','BG1','L2'],['M0','M1','L0']):
        values=[delta(i,a,b) for i in range(200)]
        ax.hist(values,bins=np.linspace(-1,1,21),color='#277c86',alpha=.85);ax.axvline(0,color='#953936',ls='--')
        ax.set_title(f'{a} - {b}: all Test200');ax.set_xlabel('Change in per-image union mask IoU');ax.set_ylabel('Image count')
    fig.tight_layout(rect=(0,.1,1,1));fig.text(.01,.015,'Descriptive pixel-union IoU at confidence .25, fixed seed0. Not mAP, not independent seeds; all 200 images are included.',fontsize=8)
    for ext in ['png','svg','pdf']:fig.savefig(OUTPUT/'figures'/f'25_per_image_effect_distribution.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    analysis=['# 실제 예측 사례: 차이가 큰 장면과 전체 분포',
        '기존 v4 보고서에는 발표 순서만 있었고 실제 예측 overlay와 실행 데모는 없었다. 이번 v5에는 실제 고정 모델 예측14개 사례, 전체200장 비교 데모, 새 이미지 추론 서버와 발표 대본을 추가했다.',
        '## 분석 범위와 재현 확인',
        'M0/M1/M2/M3/BG0/BG1/L0/L1/L2/L3의 seed0를 결과 확인 전에 고정했다. 각 모델의 Validation-best checkpoint로 같은 Test200장을 다시 추론했다. 모든10개 모델에서 기존 Test의 전체 library 지표 차이 절댓값은1e-10 미만이고, 객체 TP/FP/FN은 정확히 일치했다. 모델별 checkpoint hash를 확인했고 boundary head의 추론 호출은0회였다. 재학습이나 threshold 조정은 없었다.',
        '이번의 재추론은 기존 결과의 재생·설명을 위한 것으로 독립 Test 평가10개가 새로 생긴 것이 아니다. 기존 학습40개/Test고유34개라는 수는 그대로다.',
        '## 차이가 큰 사례의 의미',
        '사용자가 요청한 차이가 두드러지는 예시는 모델 간 이미지별 예측 차이가 큰 경우로 구현했다. Full-data 모델6개(M0/M1/M2/M3/BG0/BG1)의 이미지별 union-mask IoU 범위와 모집단 SD를 기록했다. 서로 다른 모델의 한 이미지 내 차이이며, training seed의 분산 또는 전체 성능의 SD와는 다르다.',
        'Union-mask IoU는 conf=.25를 통과한 모든 예측 instance 마스크의 합집합과 모든 정답 마스크 합집합의 IoU다. mAP와 다르고, 중복 instance를 따로 벌점 주지 않으므로 반드시 객체 TP/FP/FN도 함께 표시한다. 화면의 초록/주황/분홍은 픽셀 겹침/정답만/예측만 영역이며 객체 단위 TP/FP/FN 색상이라고 해석하지 않는다.',
        '선정은 합성 개선2, 합성 악화1, task-aware 악화1, background 오탐 감소1, background 미탐 증가1, low-data 개선/악화 각1, 추가 큰 차이1, 중간 차이1, 고정 난수2, 여러 모델의 성공/실패 각1이다. 난수 seed20260922의2개 이미지는 전체200장에서 먼저 정해지는 난수 표본이며, 다른 선정과 겹칠 수 있다. 중복을 숨기거나 표본 수를 독립 관측 수로 과장하지 않는다.',
        '큰 차이 사례는 사후 선별이므로 전체 성능을 대표하지 않는다. 유리한 예시만 제시하지 않도록 악화·미탐·중간/난수 사례를 함께 제공하며, 모든200장의 모델별 수치를 CSV와 데모에서 공개한다. 이 사례를 보고 학습·checkpoint·confidence를 다시 선택하지 않았다.',
        mdtable(['사례','Test index','모델 비교','Union IoU 차이','6모델 IoU 범위'],[[c['title'],c['index'],c['left']+' → '+c['right'],f"{c['union_iou_delta']:+.3f}",f"{c['model_range']:.3f}"] for c in cases]),
        '## 전체200장 분포와 대표성']
    for a,b in [('M2','M0'),('BG1','M1'),('L2','L0')]:
        values=[delta(i,a,b) for i in range(200)]
        analysis.append(f"{a}−{b}: 이미지별 union IoU가 증가한 경우 {sum(v>1e-12 for v in values)}장, 감소 {sum(v< -1e-12 for v in values)}장, 동일 {sum(abs(v)<=1e-12 for v in values)}장. 평균 차이 {st.mean(values):+.4f}. 이는 고정 seed0의 이미지별 픽셀 지표 기술통계이며, 3seed 평균 mAP 결론을 대체하지 않는다.")
    analysis += ['## 사례별 설명']
    for c in cases:
        analysis += [f"### {c['title']}",c['reason'],f"원본 출처: `{images[c['index']]['source']}`. 비교 패널: [PNG](cases/{c['id']}.png), [PDF](cases/{c['id']}.pdf)."]
    analysis += ['## 시각화의 범위',
        '입력은 기존 평가와 동일한640px resize/letterbox이며 정답·예측은 기존 mask 평가에 사용한160×160 격자다. 화면 확대는 nearest-neighbor 표시로 점수에 사용한 마스크를 그대로 보여준다. 정답 영역 확대는 보기 기능이며 IoU나 객체 수는 전체 이미지 기준으로 유지된다. 확대 화면에서 배경의 오탐이 잘릴 수 있으므로 발표 때 전체 화면도 함께 제시한다.',
        '1학기 예측은 이번에 재생하지 않았다. 검증한1학기 원본 수치와 학습 곡선은 기존 보고서에서 다루며, 현재 모델 예측을1학기 결과로 표시하지 않는다.']
    analysis_text='\n\n'.join(analysis)+'\n'
    (OUTPUT/'QUALITATIVE_REPORT_KO.md').write_text(analysis_text,encoding='utf-8')
    guide='''# 데모 실행·발표 안내

## 1. 설치 없이 예측 비교

전체 ZIP을 폴더로 압축 해제한 다음 demo.html을 브라우저로 연다. 인터넷이나 GPU 없이 실제로 저장된 Test200장·모델10개 예측을 비교할 수 있다. demo.html만 이동하면 images/demo_data.js/demo.js가 없어 작동하지 않는다.

추천 사례를 선택하면 설명과 두 모델이 자동으로 바뀐다. 왼쪽/오른쪽 모델을 직접 바꾸고, 정답 영역 확대 또는 전체 화면으로 전환할 수 있다. 원본만/정답 윤곽과 예측/차이 영역 모드를 제공한다. 현재 이미지 저장 버튼은 발표용 PNG를 내려받는다.

이 부분은 저장된 예측 재생이며 브라우저 안에서 모델을 새로 돌리는 것이 아니다. 화면에 표시된 Test 이미지 IoU는 설명용 union-mask 지표이고 평균 성능은 기존3seed 보고서를 사용한다.

## 2. 새 이미지 GPU 추론

Windows 서버에서 프로젝트 루트로 이동한 뒤 실행한다.

    .venv-s5\\python.exe -X utf8 tools\\serve_prediction_demo.py --port 8765

서버 PC에서는 http://127.0.0.1:8765/demo.html 로 접속한다. SSH로 다른 PC에서 접속 중이라면 사용자 로컬 터미널에 포트 포워딩을 연다.

    ssh -N -L 8765:127.0.0.1:8765 sshuser@<Windows서버주소>

그 후 사용자 로컬 브라우저에서 http://127.0.0.1:8765/demo.html 을 연다. VS Code의 Ports 탭에서8765를 포워딩해도 된다. 서버는127.0.0.1에만 바인딩하므로 별도 공개 서버 배포가 아니다. 서버를 실행한 터미널에서 Ctrl+C로 종료한다.

아래 새 이미지 영역에서 JPG/PNG/WebP와 모델을 선택하고 추론 실행을 누른다. 새 이미지는 메모리에서 처리되고 서버에 업로드 원본을 저장하지 않는다. 정답이 없으므로 IoU/정확도는 표시하지 않는다. 입력640, conf=.25, NMS=.7을 고정한다. GPU 작업 lock으로 다른 학습과 동시에 충돌하지 않도록 한다.

모델을 바꾸면 첫 요청에 로딩/워밍업 시간이 추가된다. 화면은 forward+후처리와 서버 처리 시간을 구분한다. 네트워크/브라우저 표시까지 포함한 실시간 FPS나 배포 성능으로 환산하지 않는다. 분할 결과와 입력을 확인한 뒤 예측 이미지 저장 버튼으로 PNG를 받는다.

## 3. 권장 시연 순서 (약2분)

1. 합성 개선 사례: M0→M2에서 놓친 영역이 어떻게 바뀌는지 설명한다.
2. 합성 악화 또는 task-aware 악화 사례: 모든 장면에서 좋아지는 것은 아님을 보인다.
3. Background 오탐 감소와 미탐 증가 사례를 연속으로 보여 절충을 설명한다.
4. 전체 Test 목록 또는 고정 난수 사례로 이동해 특정 예시만 있는 화면이 아님을 보여준다.
5. 시간이 허용되면 새 이미지를 업로드해 실제 추론을 시연한다. 정답 없는 이미지의 정확도를 주장하지 않는다.

발표 현장 연결 문제에 대비해 demo.html의 오프라인 사례 재생을 준비한다. 오프라인 재생을 실시간 추론이라고 소개하지 않는다.
'''
    (OUTPUT/'DEMO_GUIDE_KO.md').write_text(guide,encoding='utf-8')
    slides=[
        ('연구 질문','데이터가 부족한 위장 객체 분할을 합성으로 보완할 수 있는가?','문제의 어려움과 실제 라벨 확보 비용에서 출발했습니다. 최종 목표는 합성 전략에 따른 성능과 한계를 검증하는 것입니다.','20_complete_research_context'),
        ('1학기의 출발점','Custom DualHeadSegment와 휴리스틱 합성','1학기에는 real+background 대비 합성 추가로 Test mask mAP가27.76에서34.86으로 올랐습니다. 이번 학기의 seed 평균과 직접적인 통제 비교는 아닙니다.','12_semester1_results'),
        ('왜 후속 연구가 필요한가','어떤 구성과 데이터 조건에서 효과가 나는가?','합성이 도움이 된다는 관찰에서 더 나아가 데이터 부족, 합성 방식, 선택 전략, 모델 구조를 분리해 비교했습니다.','16_within_study_gains'),
        ('시스템 구현','생성→선택→학습→동결→평가→보고','모델을 단순 실행한 것에 그치지 않고 출처·해시 기록과 중단 복구를 포함한 자동 실험 흐름을 구현했습니다. 초기화는 이전 학기 best 계승이 아니라 공통 pretrained 기준입니다.','10_training_budget'),
        ('실험 설계','고정 split, 모델, 조건별 seed','Real748/Val100/Test200, Low187입니다. M/L/BG는3seeds, A/H는seed0입니다. 합성33,000장, 이번 학기 학습40개이며 A6개는Val-only입니다.','01_main_test'),
        ('주요 성능','AnyDoor random이 현재 M/L에서 가장 높은 평균','M0 21.03→M2 31.99, L0 12.85→L2 22.01입니다. 모든 값은 Test mask mAP50–95이며 ±는training seed 표본SD입니다.','01_main_test'),
        ('학습이 정상적으로 됐는가','계획된 epoch, best checkpoint, 평가 연결 검증','40개 모두 예정 epoch를 마쳤습니다. loss 감소만으로 일반화를 주장하지 않고 Validation-best의 Test를 비교했습니다.','03_learning_M'),
        ('합성 효과를 눈으로 보기','같은 이미지의 M0와M2 예측','demo.html의01과03을 연달아 보여줍니다. 개선 사례 뒤 악화 사례도 제시하며, 차이가 큰 설명용 사례를 사후 선별했다고 밝힙니다.','24_prediction_examples'),
        ('Heuristic ablation','Full이 항상 최고는 아니었다','A2 −Tone score가Val23.36으로 가장 높았고 Full은21.73입니다. 렌더링 자체를 제거한 것이 아닌 선택 score ablation이며40epoch/seed0의탐색 결과입니다.','05_heuristic_ablation'),
        ('Head ablation','Boundary와거리 가중의효과는구분해야한다','합성조건 H0/H1/H2는29.16/30.71/29.90입니다. Boundary 보조학습은관찰상이점이있지만 distance weighting의추가우위는일관되지않았습니다.','06_head_ablation'),
        ('Task-aware의부정결과','어려운합성이반드시좋은학습자료는아니다','Random 대비task-aware는M−2.01pp,L−1.76pp였습니다. 분포변화나label noise는가능한설명이며원인으로입증한것은아닙니다.','09_selection_distribution'),
        ('Background2×2','합성유무×background유무','M0/M1을대조군으로같은배경2129장을추가했습니다. 오탐은61.9%/51.9%감소했지만mAP는3.31/2.30pp낮아졌습니다.','21_background_factorial'),
        ('오탐–미탐절충시연','오탐을줄이는대신객체를놓칠수있다','demo05에서오탐감소를,06에서미탐증가를보입니다. Precision상승만으로전체분할성능이개선됐다고말하지않습니다.','19_background_paired_effects'),
        ('실제소프트웨어시연','새이미지입력→마스크표시→PNG저장','서버를실행한경우새이미지를업로드합니다. 화면의실측추론시간과모델첫로딩시간을구분합니다. 새이미지에는정답이없어정확도를표시하지않습니다.','demo.html'),
        ('해석의한계','동일epoch는동일compute가아니다','학습량차이,작은고정Test,A/H단일seed,근사mask,학기간recipe차이,background사후평가를명시합니다. 동일compute통제는후속과제입니다.','23_training_budget_context'),
        ('최종기여','합성의효능과복잡한구성의한계를검증','합성은데이터부족을보완했지만더복잡한선택·구조가항상유리하지는않았습니다. Background의오탐–미탐절충까지검증한시스템과재현가능한실험기록을제시합니다.','22_current_results_overview')]
    notes=['# 발표 대본과 데모 연결',
        '약10~12분 발표에 맞춘16장 구성이다. 아래 멘트는 실제 수치와 한계에 근거한다. 필요하면 배경·구조 설명을 합쳐12장으로 줄이고, 데모는2분 내로 준비한다.',
        mdtable(['슬라이드','핵심 메시지','사용 자료'],[[f'{i+1}. {s[0]}',s[1],s[3]] for i,s in enumerate(slides)])]
    for i,s in enumerate(slides):notes += [f'## {i+1}. {s[0]}',s[2]]
    notes += ['## 예상 질문과 답변',
        '**1학기보다 수치가 낮은데 실패인가?** 역사적S3의34.86은이번M2평균31.99보다높다. 다만학습recipe,background,평가경로등이달라통제된방법비교가아니다. 이번기여는조건별검증과재현성확장이고,각학기내합성효과는확인됐다.',
        '**왜 task-aware가 더 낮았나?** 현재 결과는우위를지지하지않는다. 난도점수가유용한정보와생성오류를구분하지못하거나분포를바꿨을가능성은있지만원인을입증한추가실험은없다.',
        '**그림을 좋은 것만 골랐나?** 큰차이를보여주는설명용선정임을공개했고악화·미탐·중간·고정난수사례와전체200장예측도제공했다. 평균성능주장은선별예시가아닌전체Test와3seed표에서한다.',
        '**왜 같은 epoch인데 공정하다고 하나?** 학습설정은같지만데이터추가로업데이트수가달라완전한동일compute비교는아니다. 실용적데이터증강recipe의효과이며순수데이터효과를분리한주장은하지않는다.',
        '**화면의IoU와mAP는왜다른가?** 사례화면은한이미지의합집합mask IoU다. 공식mAP는instance별matching과confidence순위,IoU .50~.95평균을사용한다. 서로대체할수없는지표다.',
        '**내가 직접 이해하고 설명할 구현 지점은?** source제한과split,custom head의학습전용boundary,고정초기화,checkpoint복구,Validation선택과Test평가분리,동일후보random/task-aware선택,background빈라벨처리다. 실제코드를가리키며설명할수있도록준비한다.']
    notes_text=(ROOT/'tools/presentation_script_template.md').read_text(encoding='utf-8')
    (OUTPUT/'PRESENTATION_SCRIPT_KO.md').write_text(notes_text,encoding='utf-8')
    report=(PREVIOUS/'FINAL_REPORT_KO.md').read_text(encoding='utf-8')
    report=report.replace('이번 정리는 기록 검증과 재집계이며 새 GPU 추론을 반복한 것은 아니다.',
        'v4 집계 단계는 기록 검증이었다. v5에서는 별도로 고정 seed0 모델10개를 Test200장에 재추론해 기존 지표와 일치를 확인하고 실제 예측 사례를 추가했다.')
    report=report.replace('원시 예측이 없는 PR곡선·confidence곡선·픽셀 confusion·IoU histogram을 요약 수치로 만들지 않았다.',
        'PR곡선·confidence곡선·픽셀 confusion을 요약 수치로 만들지 않았다. v5의 이미지별 IoU 분포는 실제 재추론한 원시 마스크로 계산했고 공식 mAP와 구분했다.')
    report=report.replace('23개 figure','25개 핵심 figure와14개 예측 사례')
    report+='\n\n# 실제 예측·데모·발표 보완 (v5)\n\n'+analysis_text+'\n\n'+guide+'\n\n'+notes_text
    (OUTPUT/'FINAL_REPORT_KO.md').write_text(report,encoding='utf-8')
    intro='<section><h2>실제 예측·데모·발표 자료 추가</h2><p><a href="demo.html">예측 비교 데모 열기</a> · <a href="cases.html">14개 사례 그림과 설명</a> · <a href="presentation.html">발표 대본 보기</a> · <a href="DEMO_GUIDE_KO.md">새 이미지 추론 실행 안내</a></p><p>모델10개 × Test200장의 실제 예측을 재생합니다. 큰 차이·악화·중간·고정 난수 사례를 함께 공개하며, 공식 성능 수치는 기존3seed 결과를 유지합니다.</p></section>'
    page=(PREVIOUS/'index.html').read_text(encoding='utf-8')
    page=page.replace('핵심figure23종','핵심figure25종 · 예측사례14개')
    page=page.replace('<section id="report">',intro+'<section id="report">',1)
    page=re.sub(r'(<section id="report">).*?(</section><section id="figures">)',lambda m:m.group(1)+render_md(report)+m.group(2),page,flags=re.S)
    extra=''.join(f'<article><h3>{name}</h3><img src="figures/{name}.png"><p><a href="figures/{name}.pdf">PDF</a> · <a href="figures/{name}.svg">SVG</a></p></article>' for name in ['24_prediction_examples','25_per_image_effect_distribution'])
    page=page.replace('</section></main>',extra+'</section></main>')
    (OUTPUT/'index.html').write_text(page,encoding='utf-8')
    style='<style>body{font:16px/1.7 system-ui,sans-serif;background:#edf2f5;color:#203746}main{max-width:1250px;margin:auto;padding:20px}article,section{background:white;padding:24px;margin:20px 0;border-radius:12px}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}.table-wrap{overflow:auto}a{color:#09747a}</style>'
    gallery=''.join(f'<article><h2>{html.escape(c["title"])}</h2><p>{html.escape(c["reason"])}</p><img loading="lazy" src="cases/{c["id"]}.jpg"><p><a href="cases/{c["id"]}.png">PNG 원본</a> · <a href="cases/{c["id"]}.pdf">PDF</a> · <a href="demo.html">모델 직접 비교</a></p></article>' for c in cases)
    (OUTPUT/'cases.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>실제 예측 사례</title>'+style+'<main><h1>실제 예측 사례 14개</h1><p>사후 선정된 설명용 사례입니다. 평균 성능이나 seed 분산을 대표하지 않습니다. <a href="index.html">전체 보고서</a></p>'+gallery+'</main></html>',encoding='utf-8')
    (OUTPUT/'presentation.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>발표 대본</title>'+style+'<main><section>'+render_md(notes_text)+'</section></main></html>',encoding='utf-8')
    with (OUTPUT/'FIGURE_INDEX.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## v5 실제 예측 보완\n\n핵심 figure는25종으로 확장됐다. 24_prediction_examples는실제예측비교,25_per_image_effect_distribution은전체Test200의이미지별union IoU변화다. 별도 cases 폴더에14개PNG/PDF/JPG가있다. cases.html과demo.html에서선정이유및전체예측을확인한다.\n')
    with (OUTPUT/'CHATGPT_HANDOFF_KO.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## v5 추가 전달 사항\n\n최신 전달 파일은 CHATGPT_HANDOFF_v5_demo.zip이다. QUALITATIVE_REPORT_KO.md, PRESENTATION_SCRIPT_KO.md, DEMO_GUIDE_KO.md, case_selection.json과cases JPG를함께읽고발표를정리한다. 큰차이사례는사후선별임을명시하며전체200장·10모델의predictions_per_image.csv를근거로대표성을구분한다. 그림IoU는union-mask진단지표이고공식mAP/seed SD가아니다.\n')
    (OUTPUT/'CHATGPT_HANDOFF_KO.md').write_text((OUTPUT/'CHATGPT_HANDOFF_KO.md').read_text(encoding='utf-8').replace('CHATGPT_HANDOFF_v4_complete.zip','CHATGPT_HANDOFF_v5_demo.zip').replace('FINAL_RESULTS_v4_complete.zip','FINAL_RESULTS_v5_demo.zip'),encoding='utf-8')
    audit={'status':'VERIFIED_PREDICTION_REPLAY','models':10,'unique_test_images':200,'prediction_image_model_pairs':2000,
        'seed_policy':'seed0 fixed before case selection','preprocessing_live_vs_evaluation':'pixel-exact on indices0,49,99,149,199',
        'metric_checks':{j:{'checkpoint_sha256':loaded[j]['checkpoint_sha256'],'counts_match':loaded[j]['counts_match'],
            'max_library_metric_abs_difference':max(abs(v) for v in loaded[j]['library_differences'].values()),'boundary_calls':loaded[j]['boundary_calls']} for j in JOBS},
        'selected_cases':14,'unique_selected_images':len({c['index'] for c in cases}),'fixed_random_seed':20260922,
        'new_training':False,'new_independent_test':False,'live_api_test':'PENDING','selection_fallback_count':sum(c['selection_fallback'] for c in cases)}
    atomic_json(OUTPUT/'audit_qualitative.json',audit)
    atomic_json(OUTPUT/'qualitative_source_manifest.json',{p.relative_to(ROOT).as_posix():sha256_file(p) for p in [BASE/'plan.json',BASE/'images.json']+[BASE/j/'result.json' for j in JOBS]+[ROOT/'tools'/n for n in ['qualitative_core.py','collect_qualitative_predictions.py','build_qualitative_package.py','serve_prediction_demo.py','qualitative_demo.html','qualitative_demo.js']]})
    for p in [OUTPUT/'index.html',OUTPUT/'cases.html',OUTPUT/'demo.html',OUTPUT/'presentation.html']:
        for link in re.findall(r'(?:href|src)="([^"]+)"',p.read_text(encoding='utf-8')):
            if not link.startswith(('#','http:','https:')):assert (OUTPUT/link).exists(),(p.name,link)
    print(json.dumps({'output':str(OUTPUT),'cases':cases,'replay':'10/10 VERIFIED'},ensure_ascii=False))

if __name__=='__main__':main()
