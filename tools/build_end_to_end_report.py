"""Extend presentation assets with sourced historical evidence; no training changes."""
from pathlib import Path
import csv
import hashlib
import html
import json
import shutil
import statistics as st
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('D:/Capstone_Projects/Capstone(Dataset)/Heuristic_Output/result')
OUT=ROOT/'artifacts/final_presentation/end_to_end_v2'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:
        return [{k.strip():v.strip() for k,v in r.items()} for r in csv.DictReader(f)]
def mdtable(rs,keys):
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(str(r[k]) for k in keys)+' |' for r in rs)

def main():
    if OUT.exists():raise FileExistsError('Preserve existing output; use a new version')
    shutil.copytree(ROOT/'artifacts/final_presentation/run_v1',OUT)
    evidence=OUT/'historical_evidence';evidence.mkdir()
    source={};legacy=[];curves={};figures=[]
    def copy(p,d):
        d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d);source[str(p)]=sha(p)
        assert sha(p)==sha(d)
    for split,file in [('Validation','val_results.csv'),('Test','test_results.csv')]:
        copy(OLD/file,evidence/file)
        for r in rows(OLD/file):legacy.append(dict(semester='1',split=split,model=r['Model'],
            box_precision=float(r['Precision']),box_recall=float(r['Recall']),box_f1=float(r['F1']),
            box_map50=float(r['Box mAP50']),box_map50_95=float(r['Box mAP50-95']),
            mask_map50=float(r['Mask mAP50']),mask_map50_95=float(r['Mask mAP50-95']),
            provenance='archived rounded CSV; not rerun; one recorded run per condition'))
    for d in sorted(OLD.iterdir()):
        if d.is_dir() and (d/'results.csv').exists():
            copy(d/'results.csv',evidence/d.name/'results.csv')
            copy(d/'args.yaml',evidence/d.name/'args.yaml')
            curves[d.name]=rows(d/'results.csv')
    for name in ['RECOVERY_REPORT.md','S5_VERIFICATION_REPORT.md','S8_EPOCH_RECIPE.md','FINAL_RESEARCH_PLAN.md','EXPERIMENT_PROTOCOL.md','HEAD_ABLATION_PLAN.md','S8_FOREGROUND_AMENDMENT.md','S8_SEED_AMENDMENT.md']:
        p=ROOT/'docs'/name
        if p.exists():copy(p,evidence/'reference_docs'/name)
    nb=Path('D:/Capstone_Projects/Capstone(Dataset)/Capstone_Colab_Clean.ipynb')
    source[str(nb)]=sha(nb);notebook=json.loads(nb.read_text(encoding='utf-8'))
    excerpts=[]
    for i,c in enumerate(notebook['cells']):
        src=''.join(c.get('source',[]))
        if 'def eval_on_split' in src or 'src_pairs_s3 =' in src or 'src_pairs_s4 =' in src:
            excerpts.append(f'## Notebook cell {i} (zero-based)\n\n```python\n{src}\n```')
    (evidence/'notebook_source_excerpts.md').write_text('\n\n'.join(excerpts),encoding='utf-8')
    assert any('prec = m.box.mp' in s and 'm.seg.map' in s for s in excerpts)
    with (OUT/'tables/semester1_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(legacy[0]));w.writeheader();w.writerows(legacy)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42})
    def save(fig,name,caption):
        fig.tight_layout(rect=(0,.06,1,1));fig.text(.015,.018,caption,fontsize=8,color='#53606A')
        for ext in ['png','svg','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=300,facecolor='white')
        figures.append((name,caption));plt.close(fig)
    models=['S4 Baseline','S3 Synth','S4+HardMining','S3+HardMining']
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,split in zip(axs,['Validation','Test']):
        rs=[next(r for r in legacy if r['split']==split and r['model']==m) for m in models]
        y=[r['mask_map50_95']*100 for r in rs];ax.bar(range(4),y,color=['#526779','#218B82','#A8B3BC','#8BC1B9'])
        for i,v in enumerate(y):ax.text(i,v+.6,f'{v:.2f}',ha='center')
        ax.set(xticks=range(4),xticklabels=['S4','S3','S4 + HM','S3 + HM'],ylim=(0,45),ylabel='Mask mAP50-95 (%)',title=f'Semester 1: archived {split}')
    save(fig,'12_semester1_results','Historical CSV values, not new evaluation; no multi-seed variance available. HM: final recorded hard-mining model.')
    fig,axs=plt.subplots(1,3,figsize=(15,4.8))
    for name,label,color in [('scenario4_baseline','S4','#526779'),('scenario3_synth','S3','#218B82')]:
        rr=curves[name];x=[int(r['epoch']) for r in rr]
        assert x==list(range(1,151))
        for ax,key,title,factor in zip(axs,['metrics/mAP50-95(M)','train/seg_loss','val/seg_loss'],['Validation mask mAP50-95 (%)','Recorded train/seg_loss','Recorded val/seg_loss'],[100,1,1]):
            ax.plot(x,[float(r[key])*factor for r in rr],label=label,color=color);ax.set(title=title,xlabel='Epoch');ax.legend();ax.grid(alpha=.15)
    save(fig,'13_semester1_learning','Original 150-epoch CSVs; raw curves, no smoothing. Historical component losses are not directly comparable to current aggregate loss.')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ax,tag in zip(axs,['s4','s3']):
        for loop in range(1,4):
            rr=curves[f'hard_mining_{tag}_loop{loop}'];ax.plot([int(r['epoch']) for r in rr],[float(r['metrics/mAP50-95(M)'])*100 for r in rr],label=f'Loop {loop}')
        ax.set(title=f'Semester 1: {tag.upper()} hard mining',xlabel='Within-loop epoch',ylabel='Validation mask mAP50-95 (%)');ax.legend();ax.grid(alpha=.15)
    save(fig,'14_semester1_hard_mining','Three sequential 30-epoch fine-tuning loops per branch, not independent seeds. Each loop uses its own within-loop epoch axis.')
    current=rows(OUT/'tables/per_checkpoint_metrics.csv')
    fig,axs=plt.subplots(1,2,figsize=(14,5.5))
    for ax,split in zip(axs,['Validation','Test']):
        historical=[next(r['mask_map50_95']*100 for r in legacy if r['split']==split and r['model']==m) for m in models[:2]]
        exps=['M0','M1','M2','M3'];vs=[[float(r['map'])*100 for r in current if r['split']==split and r['experiment']==e] for e in exps]
        y=historical+[st.mean(v) for v in vs];sd=[0,0]+[st.stdev(v) for v in vs]
        ax.bar(range(6),y,yerr=sd,capsize=3,color=['#B3BAC1','#91BAB4','#526779','#D98B28','#218B82','#7956A5'])
        ax.axvline(1.5,color='#53606A',ls='--');ax.axvspan(-.5,1.5,color='#EEEEEE',alpha=.3)
        for i,v in enumerate(y):ax.text(i,v+sd[i]+.6,f'{v:.2f}',ha='center',fontsize=9)
        ax.set(xticks=range(6),xticklabels=['S1\nS4','S1\nS3','S2\nM0','S2\nM1','S2\nM2','S2\nM3'],ylim=(0,50),title=f'{split}: historical context only',ylabel='Mask mAP50-95 (%)')
    save(fig,'15_cross_semester_context','NOT a controlled semester comparison: different background-only data / recipes / provenance. S1: archived run; S2: mean +/- SD (3 seeds).')
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,split in zip(axs,['Validation','Test']):
        old=[r for r in legacy if r['split']==split];get=lambda m:next(r['mask_map50_95'] for r in old if r['model']==m)
        delta=[100*(get('S3 Synth')-get('S4 Baseline'))]
        for p in ['M','L']:
            val=lambda i:st.mean(float(r['map']) for r in current if r['split']==split and r['experiment']==p+str(i))
            delta.append(100*(val(1)-val(0)))
        ax.bar(['S1: S3-S4','S2: M1-M0','S2: L1-L0'],delta,color=['#91BAB4','#D98B28','#7956A5'])
        for i,v in enumerate(delta):ax.text(i,v+.15,f'+{v:.2f}',ha='center')
        ax.set(title=f'{split}: within-study heuristic gains',ylabel='Mask mAP gain (percentage points)',ylim=(0,12))
    save(fig,'16_within_study_gains','Each gain uses its own study baseline. Different training conditions prohibit causal comparison of gain magnitudes across semesters.')
    hist_table=[{'실험':r['model'],'split':r['split'],'Box P':f"{r['box_precision']*100:.2f}",'Box R':f"{r['box_recall']*100:.2f}",'Box F1':f"{r['box_f1']*100:.2f}",'Mask mAP50':f"{r['mask_map50']*100:.2f}",'Mask mAP50–95':f"{r['mask_map50_95']*100:.2f}"} for r in legacy]
    intro='''# 위장망 분할을 위한 합성데이터 연구: 1·2학기 통합 최종보고서

## 초록

본 프로젝트는 실제 위장망 데이터의 부족을 합성데이터로 보완하고, 생성·선택 방식과 boundary 보조 학습의 효과를 평가한다. 1학기에는 custom DualHeadSegment와 heuristic 합성 및 hard mining을 탐색했다. 2학기에는 재현 가능한 실행 체계를 구축하고, 생성 score Ablation, 실제데이터 전량/25% 비교, AnyDoor와 task-aware selection, head/loss Ablation을 수행했다. 이번 캠페인은34회 학습·총4440 epoch,28개 고유 최종 모델의 Test 평가를 완료했다. 합성 추가의 이점은 두 학기 기록에서 관찰되지만 task-aware와 distance weighting의 일관된 우위는 확인되지 않았다.

## 1. 문제 정의와 연구 흐름

위장망 객체는 배경과 시각적으로 유사해 segmentation 학습에 어려움이 있다. 본 연구 질문은 '고정된 분할 모델에서 합성데이터가 실제데이터 부족을 보완하는가, 그리고 어떤 생성·선택 요소가 도움이 되는가'이다. 최첨단 성능이나 모든 환경에서의 배포 가능성을 입증하는 benchmark는 아니다.

연구의 흐름은 **1학기 기반 모델·합성 가능성 탐색 → 결과·코드 복구와 구조 검증 → score별 생성기 통제 → 전체/부족 데이터 비교 → AnyDoor pilot과 근사 mask 승인 → 공통 후보 생성·선택 → 독립 학습 → Validation checkpoint 동결 → Test 평가 → 결과 해석**이다. 2학기 learner는 1학기 학습완료 checkpoint를 이어 학습한 것이 아니라 공통 pretrained에서 독립 초기화했다.

## 2. 1학기: 구현과 원본 성과

1학기 노트북 소스에서 S4는 real+background-only, S3는 real+synthetic+background-only를 합치는 경로가 확인된다. 실행 시점의 완전한 병합 manifest는 없어 최종 합성/배경 개수를 확정하지 않는다. S4/S3 각150epoch와 두 branch별 hard mining30epoch×3loop 기록이 보존돼 있다. hard mining loop는 이전 모델을 계승하는 후속 학습이며 독립 seed가 아니다.

`S5_VERIFICATION_REPORT.md`에서 **1학기 S4 checkpoint의 실제 구조는 n-scale DualHeadSegment,3,291,620 parameters**로 검증됐다. pretrained 파일명 yolov8m-seg만 보고 실제 모델을 m-scale로 쓰면 안 된다. 현재 모델은 동일 n 계열이며, 원본 S3/모든 hard-mining checkpoint 각각을 이번에 새로 구조 검사한 것은 아니다.

다음 표는 원본 val_results.csv/test_results.csv의 반올림된 저장값이다. 새로 추론한 결과가 아니며, Precision/Recall/F1은 노트북의 m.box에서 가져온 **Box 지표**다.

'''+mdtable(hist_table,list(hist_table[0]))+'''

1학기 S3−S4 mask mAP50–95는 Validation +8.64점, Test +7.10점이다. hard mining 이후 Test는 S4 27.76→27.47, S3 34.86→33.54로 감소했다. S3 hard mining에서 Box F1은69.43→70.82로 증가했으므로 detection과 엄격한 mask localization의 변화가 같지 않다는 점도 논의할 수 있다. 특정 hard mining이 모든 지표를 개선했다고 주장하지 않는다. 과거 best 정확한 선택 epoch/전체 입력 provenance에 미확인 부분이 있으므로 마지막 CSV 행을 best 재평가로 대체하지 않는다.

## 3. 학기 간 관계와 비교의 경계

| 항목 | 1학기 보관 기록 | 2학기 최종 캠페인 |
| --- | --- | --- |
| baseline 데이터 | S4: real+background-only (노트북 근거) | M0: real748 only |
| 합성 | S3 heuristic; 정확한 병합량 미확정 | 조건별3000; M/L provenance 분리 |
| head/scale | S4 checkpoint n DualHead 검증 | n DualHead 고정, H-series 분해 |
| 반복 | 각 조건 하나의 보관 실행 | M/L3seeds; A/H seed0 |
| augmentation/schedule | S4 mosaic .5, degrees10, hsv_s .2, cosine off | mosaic0, degrees0, hsv_s .7, cosine on |
| early stopping/EMA | 과거 recipe 및 framework 동작; S4 patience20/S3 100 | early stopping 없음, EMA 없음 |
| P/R/F1 | Box 기준 | fixed confidence의 Mask instance 기준 |
| 학습 후 조정 | hard mining 후속 loop | 사전 정의된 독립 실험과 최종 freeze |
| provenance | 일부 과거 입력·선택 epoch 미확정 | 학습·checkpoint·selection hash 기록 |

1학기 S3의 Test34.86은 이번 M2 평균31.99보다 **기록값으로2.87점 높다**. 이를 숨기지 않는다. 하지만 위 조건 차이 때문에 2학기 방법이 퇴보했다거나1학기 방법이 통제된 조건에서 우월하다는 인과 결론은 낼 수 없다. 학기별 절대 수치는 역사적 맥락 figure15로 보여주고, 연구 주장은 각 학기 내부 baseline 대비 변화에 둔다. 현재 M2가1학기 S3를 능가했다고 발표하면 사실과 다르다.

1학기의 의미는 합성 가능성과 시스템의 출발점,2학기의 의미는 요인별 통제·low-data 검증·generative/selection 비교·실행 재현성의 확장이다. 모든 단계에서 성능이 단조롭게 상승해야 연구가 성립하는 것은 아니다.

## 4. 2학기 상세 실험·결과·타당성 평가

아래는 현재 동결 캠페인에 대한 전체 지표와 검증이다. 1학기 비교에 관한 표현은 위 원본 근거와 함께 읽는다.

'''
    current_report=(OUT/'DETAILED_REPORT_KO.md').read_text(encoding='utf-8')
    full=intro+current_report.replace('# 졸업작품 최종 실험 분석 및 발표 가이드','### 2학기 분석 상세',1)+'''

## 5. 전체 프로젝트의 결론과 후속 과제

졸업작품의 주요 성과는 (1) custom segmentation과 합성 pipeline 구현, (2) 실제 부족 데이터에서 합성의 보완 효과, (3) 생성·선택·head 구성에 대한 긍정/부정 결과의 분리, (4) 중단 복구와 출처·checkpoint 검증을 갖춘 실험 자동화다. 성능 관점의 현재 결론은 augmentation 유용성 관찰이며, 모든 heuristic score의 필요성이나 task-aware 우위를 확증한 것은 아니다.

후속 연구 우선순위는 동일 update budget 통제, A/H 다중 seed 또는 full-budget 재확인, synthetic label 품질의 독립 평가, scene/group 기반 누수 감사, 외부 데이터 일반화 평가다. 이는 미실행 과제이며 완료 실험으로 쓰지 않는다. 이미 확인한 Test에 맞춰 수정한 방법은 같은 Test로 독립적인 최종 검증을 했다고 표현하지 않는다.

## 6. 통합 발표 권장 순서 (14장)

1. 문제 정의·연구 질문.
2.1학기 시스템과 custom DualHead 구조.
3.1학기 S4/S3 및 hard mining 결과(figure12~14).
4.1학기 한계와2학기 연구 질문.
5. 전체 pipeline과 source 정보 예산.
6. A/M/L/H 설계·고정 조건.
7. M/L 최종 Test(figure01).
8. 학습 곡선·seed 안정성(figure03~04).
9. 다양한 Metric(figure02).
10. Heuristic Ablation(figure05).
11. Head Ablation(figure06).
12. Task-aware 부정 결과·분포(figure07/09).
13. 학기 간 역사적 맥락·비교 한계(figure15/16,10).
14. 기여·한계·후속 계획.

## 7. Figure 사용 원칙

figure12~16은 이번에1학기 원본 CSV로 다시 그린 그림이다. 기존1학기 PNG의 PR곡선/정성비교는 출처와 split 검증 범위가 달라 현재2학기 그림으로 재사용하지 않는다. 현재 주요 그림은300dpi PNG와vector PDF/SVG를 함께 제공한다. 핵심 입력·역사적 증거 hash는 semester1_source_manifest.json에 보존했다. 표 값은 과거 CSV의 반올림 정밀도를 넘는 정밀도로 해석하지 않는다.
'''
    (OUT/'END_TO_END_REPORT_KO.md').write_text(full,encoding='utf-8')
    handoff='''# ChatGPT 전달 안내 및 첫 프롬프트

## 전달 파일

가장 편한 방법: 이 폴더 밖의 `CHATGPT_HANDOFF_v2.zip`을 업로드한다. ZIP을 읽지 못하는 경우 아래 파일을 개별 업로드한다. 학습 dataset/weights, 수십 GB의 artifacts, 전체 notebook의 이미지 출력은 필요 없다.

필수:
1. END_TO_END_REPORT_KO.md — 연구 전체 흐름·수치·해석·한계의 기준 문서.
2. tables/semester1_metrics.csv — 과거 지표. Box P/R/F1임을 유지.
3. tables/per_checkpoint_metrics.csv 및 summary_mean_sd.csv — 현재 seed별/평균 수치.
4. FIGURE_INDEX.md — 각 그림의 목적과 원본 파일명.
5. 사용할 figures/*.png — 발표 구성을 논의할 때 실제 그림도 첨부. 01,02,03_learning_M,03_learning_L,05,06,12,15를 우선한다.

근거 검토용: audit.json, job_integrity.csv, paired_seed_differences.csv, training_resources.csv, selection_domains.csv, historical_evidence의 val/test CSV·args.yaml·reference_docs·notebook_source_excerpts.md. epoch CSV는 곡선을 다시 그릴 때만 별도 제공한다(가벼운 ZIP에는 제외).

최종 문서 스타일을 맞추려면 사용자가 가진 기존1학기 PPT/PDF, 학과 보고서 template, 발표 시간·페이지 제한·팀/지도교수 표기도 함께 전달한다. 이 파일들은 현재 패키지에 없으며 제공됐다고 가정하면 안 된다. 논문 참고문헌은 별도 검증해서 추가하고 존재하지 않는 인용을 만들지 않는다.

## 그대로 붙여넣을 첫 프롬프트

```text
첨부한 자료를 바탕으로 위장망 segmentation 졸업작품의 1·2학기 통합 최종보고서와 PPT 구성을 작성해줘.

먼저 ZIP의 파일 목록을 확인하고 END_TO_END_REPORT_KO.md, FIGURE_INDEX.md, semester1_metrics.csv, per_checkpoint_metrics.csv, summary_mean_sd.csv, audit.json을 읽어. 읽지 못한 파일은 추정하지 말고 정확히 알려줘. 원본 CSV를 수치의 근거로 삼고 보고서 문장의 오류가 있으면 구분해서 지적해줘.

연구 서사는 문제 정의 → 1학기 custom DualHead·heuristic·hard mining → 2학기 통제 실험과 low-data·AnyDoor·task-aware → Ablation → 평가·한계·결론 순서로 구성해줘. 1학기는 과거 보관 결과,2학기는 이번 동결 캠페인이며 이어 학습한 단일 실험처럼 쓰지 마.

작성할 산출물:
1. 초록/문제 정의/방법/실험 설계/결과/논의/한계/결론을 갖춘 한국어 최종보고서 초안.
2.14장 내외 PPT의 slide별 제목,핵심 주장,3~5개 bullet,해당 figure 파일명,발표 대본.
3. 각 figure의 한국어·영어 캡션,축/단위/split/seed/오차막대 정의.
4. 심사 예상 질문10개와 근거 중심 답변.
5. 실험으로 확인된 사실,가능한 해석,미검증 가설,미실행 후속 과제를 구분한 점검표.

필수 정확성:
- 1학기 P/R/F1은Box,현재 주 P/R/F1은Mask instance 기준. pixel Dice와 혼동 금지.
- Validation/Test,마지막 epoch/best checkpoint,seed SD/신뢰구간을 구분.
- 1학기 S3 Test34.86,현재 M2 평균31.99를 그대로 보존. background-only·recipe 차이가 있어 학기 간 통제된 우열을 주장하지 않기.
- 1학기 S4 checkpoint는n-scale DualHead로 검증됨. pretrained 이름만 보고m-scale로 쓰지 않기.
- A/H는seed0; M/L은3seeds. A40epochs/Validation-only. H2 비교는M0/M1 seed0 참조.
- A1 Full 최고 아님,task-aware는random보다 낮음,distance weighting 일관된 이점 없음. 부정 결과 누락 금지.
- 동일150epochs라도 합성 추가군 update 수가 많음. 동일 compute 실험이라고 쓰지 않기.
- L은실제 train187장 정보 예산이며외부 pretrained/배경까지187장만 쓴 것은 아님.
- AnyDoor는근사 mask,near-duplicate/scene leakage와외부 일반화는미검증.
- 가짜 PR곡선/유의성 별표/숫자/문헌/실험을 추가하지 않기. 보관된 과거 PR이미지를이번 결과로 쓰지 않기.

더 필요한 자료는 기존 제공 파일을 읽은 뒤 목록으로 정리해줘. 재학습이나새 실험이 완료됐다고 쓰지 말고,현재 증거로 가장 설득력 있고 정직한 보고서와figure 구성을 완성해줘.
```
'''
    (OUT/'CHATGPT_HANDOFF_KO.md').write_text(handoff,encoding='utf-8')
    captions={
        '01_main_test':'현재 M/L 최종 Test: 합성 추가의 효능', '02_test_metrics':'현재6개 지표: Box/Mask·fixed threshold 구분',
        '03_learning_M':'현재 full-data 학습 곡선','03_learning_L':'현재 low-data 학습 곡선','04_seed_curves':'현재 seed별 변동과Validation-best epoch',
        '05_heuristic_ablation':'A0~A5: score Ablation,seed0,Val40epochs','06_head_ablation':'H0/H1/H2:seed0 Test 비교',
        '07_paired_differences':'동일seed간 Test 차이;유의성 검정 아님','08_validation_test':'현재Validation/Test 차이',
        '09_selection_distribution':'random/task-aware의배경분포 변화','10_training_budget':'동일epoch·상이한update 한계',
        '12_semester1_results':'1학기S4/S3/HardMining Validation·Test 저장값','13_semester1_learning':'1학기 S4/S3 원본학습곡선',
        '14_semester1_hard_mining':'1학기30epoch×3loop;독립seed 아님','15_cross_semester_context':'학기별 역사적 맥락;통제비교 아님',
        '16_within_study_gains':'학기 내부Heuristic 이득;학기간 gain의인과비교 금지'}
    (OUT/'FIGURE_INDEX.md').write_text('# Figure 목록\n\n핵심16종;각PNG/SVG/PDF. 모든퍼센트는0~1지표×100. 현재오차막대=seed 표본SD.\n\n'+mdtable([{'파일명':k,'용도':v} for k,v in captions.items()],['파일명','용도']),encoding='utf-8')
    page=(OUT/'index.html').read_text(encoding='utf-8')
    banner='<article><h1>1·2학기 End-to-end 최종 보고서</h1><p><a href="END_TO_END_REPORT_KO.md">통합 최종보고서</a> | <a href="CHATGPT_HANDOFF_KO.md">ChatGPT 전달 안내·프롬프트</a> | <a href="FIGURE_INDEX.md">Figure 목록</a></p><p>1학기와 현재 결과는 학습 조건이 달라 역사적 맥락으로 비교합니다.</p></article>'
    page=page.replace('<h1>졸업작품 최종 실험 결과</h1>',banner)
    extra=''
    for name,caption in figures:extra+=f'<article><h2>{html.escape(captions[name])}</h2><img src="figures/{name}.png"><p>{html.escape(caption)}</p><a href="figures/{name}.png">PNG</a> | <a href="figures/{name}.svg">SVG</a> | <a href="figures/{name}.pdf">PDF</a></article>'
    page=page.replace('</html>',extra+'</html>');(OUT/'index.html').write_text(page,encoding='utf-8')
    source[str(Path(__file__))]=sha(__file__)
    (OUT/'semester1_source_manifest.json').write_text(json.dumps(source,indent=2,ensure_ascii=False),encoding='utf-8')
    bundle=OUT.parent/'CHATGPT_HANDOFF_v2.zip'
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if not p.is_file():continue
            rel=p.relative_to(OUT)
            if 'per_job' in rel.parts or p.name=='epoch_metrics.csv' or p.suffix in ['.svg','.pdf']:continue
            z.write(p,rel.as_posix())
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
    print(json.dumps({'output':str(OUT),'historical_rows':len(legacy),'historical_training_runs':len(curves),
        'new_figures':len(figures),'total_main_png':len(list((OUT/'figures').glob('*.png'))),
        'handoff_zip':str(bundle),'zip_mib':round(bundle.stat().st_size/1024**2,2)},ensure_ascii=False))

if __name__=='__main__':main()
