"""Reader-facing names; not training configuration. Sources: completed v5 report."""
HEADS = {
    'H0': '같은 n-scale backbone/neck + Segment. 보조 boundary head/loss 없음.',
    'H1': 'custom DualHeadSegment n-scale. 일반 boundary BCE 보조 학습.',
    'H2': 'custom DualHeadSegment n-scale. 거리 가중 boundary BCE 보조 학습. 추론 때 boundary head는 사용하지 않음.',
}

def catalog():
    rows = []
    def add(id, group, label, real, synthetic, background, strategy, head='H2', epochs=150,
            seeds='0 / 1 / 2', evaluation='Validation 100장으로 best 선택 → Test 200장 평가',
            compare='', note='', alias=''):
        rows.append(dict(id=id, group=group, label=label, real=real, synthetic=synthetic,
            background=background, total=real+synthetic+background if real is not None else None,
            strategy=strategy, head=head, architecture=HEADS.get(head,head), epochs=epochs,
            seeds=seeds, evaluation=evaluation, compare=compare, note=note, alias=alias))
    strategies = ['합성 없음', '휴리스틱 A1 Full', 'AnyDoor 후보 중 무작위 3,000장 선택',
                  'AnyDoor 후보 중 task-aware 3,000장 선택']
    labels = ['실제 데이터만', '휴리스틱 합성 추가', 'AnyDoor 무작위 선택 추가', 'AnyDoor 난도 기반 선택 추가']
    for prefix,real in [('M',748),('L',187)]:
        for n in range(4):
            note = ('전체 train748 객체 출처.' if prefix=='M' else
                    '같은 고정 low187을 공유. 합성 객체와 Judge의 실제 학습 정보도 low187로 제한. 외부 배경·pretrained/generator는 사용.')
            if n>=2: note += ' 해당 M/L의 같은 6,000장 후보 pool에서 random/task-aware가 각각 3,000장 선택. 두 선택 집합은 겹칠 수 있음.'
            add(f'{prefix}{n}',prefix,('전체 실제 데이터 · ' if prefix=='M' else '실제 데이터 25% · ')+labels[n],
                real,0 if n==0 else 3000,0,strategies[n],compare=f'{prefix}0 ↔ {prefix}1/2/3: 합성 전략 효과; {prefix}2 ↔ {prefix}3: 선택 방식 효과.',note=note)
    add('BG0','BG','실제 + background-only',748,0,2129,'합성 없음',
        compare='M0 ↔ BG0: 실제-only 조건에 배경을 추가한 효과. BG0 ↔ BG1: 배경이 있는 상태에서 합성 추가 효과.',
        note='배경-only는 목표 객체 라벨이 없는 별도 학습 이미지. 전체 2,877장 중 74.00%. 기존 Test를 다시 쓴 사후 보완 실험. G0라는 정식 ID는 없음.')
    add('BG1','BG','실제 + 휴리스틱 + background-only',748,3000,2129,'휴리스틱 A1 Full',
        compare='M1 ↔ BG1: 합성 혼합 조건에 배경을 추가한 효과. BG0 ↔ BG1: 합성 추가 효과.',
        note='BG0와 같은 배경 2,129장. 전체 5,877장 중 36.23%. 동일 Test를 재사용한 사후 보완 실험.')
    anames=['무작위 control','Full 선택 점수','Tone 점수 제외','Texture 점수 제외','Edge 점수 제외','Placement 점수 제외']
    for n,label in enumerate(anames):
        add(f'A{n}','A','휴리스틱 점수 ablation · '+label,748,3000,0,'휴리스틱 '+label,
            epochs=40,seeds='0',evaluation='Validation 100장만 평가. Test 결과 없음.',
            compare='A0 ↔ A1: 무작위/Full; A1 ↔ A2~A5: 각 선택 점수의 기여.',
            note='모두 H2 모델. 점수 제거는 색상 보정·blending 렌더링 제거가 아님. A5는 AnyDoor가 아님. A1의 40-epoch 모델과 M1의 150-epoch 모델은 별도 학습.')
    for suffix,synth in [('R',0),('S',3000)]:
        for h in range(3):
            alias=('M0_seed0' if suffix=='R' else 'M1_seed0') if h==2 else ''
            add(f'H{h}_{suffix}','H',f'모델 구조 ablation · H{h} / '+('실제만' if suffix=='R' else '휴리스틱 추가'),
                748,synth,0,'합성 없음' if suffix=='R' else '휴리스틱 A1 Full',head=f'H{h}',seeds='0',
                compare=f'H0_{suffix} ↔ H1_{suffix} ↔ H2_{suffix}: 같은 데이터에서 boundary 학습 구성 비교.',
                note='R=Real-only, S=Synthetic 추가. '+(f'{alias}의 같은 checkpoint를 참조하며 추가 학습하지 않음. 3seed 평균과 비교하지 말고 seed0끼리 비교.' if alias else 'A/M/L 데이터 생성 실험과 구분. 단일 seed 결과이며 구조의 보편적 우위를 보장하지 않음.'),alias=alias)
    for id,label in [('S4','1학기 baseline: 실제 + 배경'),('S3','1학기 합성 추가: 실제 + 휴리스틱 + 배경'),
                     ('S4+HardMining','S4 이후 hard mining'),('S3+HardMining','S3 이후 hard mining')]:
        add(id,'1학기',label,None,None,None,'보관 기록 참고',head='과거 custom DualHeadSegment 계열',
            epochs='보관 로그 참고',seeds='독립 3seed 실험 아님',evaluation='보관된 Validation/Test CSV. 이번에 다시 평가한 수치가 아님.',
            compare='S4 ↔ S3: 1학기 내부 합성 효과. 현재 M0와 S4는 데이터 조건이 다름.',
            note='S3 초기 로그: 실제748 + 합성2934 + 배경2129 =5811장. 재개 로그의 배경 수는1985로 달라 전체 고정 membership은 미확인. S4 초기 전체 membership도 미확인. Hard mining은 30epoch×3 연속 loop이지 독립3seed가 아님.')
    return rows
