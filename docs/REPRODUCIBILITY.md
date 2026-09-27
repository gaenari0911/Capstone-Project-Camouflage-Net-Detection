# 재현 범위와 실행 안내

## 1. 세 가지 재현 수준

| 수준 | 공개 저장소에서 가능한 것 | 필요한 것 |
| --- | --- | --- |
| 결과 읽기 | 실험표,7개수치그래프,seed/epoch별CSV | 브라우저 또는 텍스트/CSV 도구 |
| CPU 검증 | 집계 재계산,무결성,스케줄러·선택 fixture | Python3.11 표준 라이브러리 |
| GPU 연구 전체 재현 | 원래 pipeline과 custom model 코드 제공 | 별도 데이터/가중치/생성기,새 경로 설정·freeze/preflight |

첫 두 수준과 기존 서버에서의 완료 기록을 구분한다. 세 번째 수준의 fresh-clone end-to-end 실행은 이번 공개 작업에서 검증하지 않았다. 원본 학습을 다시 시작하거나 기존 checkpoint를 바꾸지 않았다.

## 2. 데이터·GPU 없는 검증

저장소 루트에서 다음을 실행한다. 별도 pip 설치가 필요 없는 범위다.

```bash
python tools/verify_public_results.py
python -m unittest tests.test_s0_s1 tests.test_s3_synthesis -v
python tools/test_s8_anydoor_selection.py -v
```

fixture 테스트는 임시 폴더의 작은 가짜 입력만 사용하고 실제 학습/GPU를 실행하지 않는다. 이름에 gpu가 등장하는 fixture job도 실제 모델 추론이 아니다. `python -m unittest discover`로 전체 테스트를 무작정 실행하면 서버 자산과 custom GPU 환경을 요구하는 테스트가 섞이므로, 일반 사용자는 위 명시 목록을 사용한다.

`verify_public_results.py`는 다음을 검사한다.

- 공개 복사본이 source_manifest의 파일 hash와 일치하는지.
- 현재 학기40개 학습,5,340 epoch,Val40/Test34=74행의 수와 epoch 연속성.
- 각 checkpoint의 seed·epoch 및 각 조건의 mean/sample-SD/min/max를 raw CSV로 재계산.

이 검사는 checkpoint tensor나 비공개 원본 이미지의 현재 hash를 검증하거나 실제 AP 추론을 다시 수행하지 않는다. `results/evidence/`는 당시 연구 서버에서 저장한 감사 기록이지 깨끗한 컴퓨터에서 방금 실행한 결과가 아니다. `.gitattributes`는 export의 SHA-256을 유지하기 위해 줄바꿈 자동 변환을 끈다.

## 3. 현재 연구 환경과 custom 모델

실제 검증 환경은 Windows, Python3.11.9, RTX5070Ti, PyTorch2.10.0+cu128, torchvision0.25.0+cu128, custom Ultralytics8.4.31이다. 로컬 서버의 가상환경 자체는 배포하지 않는다.

`configs/environments/s5-environment-lock.txt`는 당시 환경의 감사 기록이다. `python==...`, `ultralytics-custom==...` 등이 포함돼 있으므로 일반 `pip install -r`용 파일로 취급하지 않는다. `legacy-root-requirements.txt`도 이전 환경 기록이며 실제 custom fork를 PyPI ultralytics로 덮어쓰는 설치 안내가 아니다.

GPU 환경을 별도로 구성할 때 PyTorch/torchvision CUDA 조합을 먼저 확인한 후 필요한 dependencies를 설치하고 custom 패키지를 사용한다.

```bash
# CUDA/PyTorch와 필요한 의존성을 먼저 구성한 별도 연구 환경에서만 실행
python -m pip install --no-deps -e third_party/ultralytics_custom
```

위 한 줄만으로 전체 환경 설치가 완료되지는 않는다. 특히 일반 pip ultralytics로 대체하면 DualHeadSegment, boundary target/loss와 fork-native evaluator가 달라진다. 모델 YAML은 `models/yolov8-seg-custom.yaml`에 있다. 원본 YAML의nc80과scale목록은 템플릿 값이고 실제 campaign은 **n-scale/nc1**로 명시적으로 해석한다. pretrained 파일명m으로 실제scale을 추정하지 않는다.

## 4. 전체 생성·학습의 입력 계약

현재 `configs/campaigns/`는 완료된 연구의 역사적 계약이다. 일부 경로는 원래 서버의 `artifacts/`, source ZIP, manifests, pretrained hash를 참조한다. 공개 저장소에는 그 자산을 복사하지 않았다. 빈 파일이나 성공 marker를 만들어 hash 검사를 우회해서는 안 된다.

새 환경에서 수행하려면 권한 있는 원본 데이터와 split을 확보하고, model/pretrained/AnyDoor의 정확한 source와환경을 구성한 뒤 **새 버전**의 경로 설정·manifest·preflight를 만들어야 한다. 기존 완료 campaign을 덮어쓰거나 pretrained검증을 끄지 않는다. 공개 branch의 복사 경로가 달라진 만큼 원래 freeze와 새 실행을 동일 run이라고 부르지 않는다.

M/L foreground 출처는 train748/low187로 분리한다. AnyDoor는 M/L 별도6,000후보에서 random/task-aware가 각각3,000선택한다. Judge는L0_seed0이며 최종learner의초기값으로 계승하지 않는다. A5는AnyDoor가아니다.

AnyDoor는 공식 revision `44ca2b2a70ec2cf107f3d26a5b46def6670fb0a5`를 별도로 취득한다. 그 source/weights, 외부 배경과 원본 사진의 권한은 [제3자 고지](../THIRD_PARTY_NOTICES.md)를 확인한다. 다른 AnyDoor revision/새 라이브러리 버전이 같은 결과를 보장하지 않는다.

## 5. 데모와 보고서

`tools/qualitative_demo.html`·JS와 `serve_readable_demo.py` 등 구현 코드는 포함했다. 원래 full package의 `demo_data.js`, Test200사진, 모델checkpoint는 미공개이므로 이 clone만으로 사진 데모를 바로 열거나 새 이미지 추론을 할 수는 없다.

`docs/EXPERIMENT_LOOKUP_KO.md`는 full package의 사전을 내용 변경 없이 보존한 자료다. 그 안의 index.html/demo.html 설명은 원래 비공개 full package 기준이다. 공개 수치 경로는 `results/tables/`, 정리된 결과·한계는 [RESULTS_AND_LIMITATIONS](RESULTS_AND_LIMITATIONS.md)다.

이번 GitHub 공개는 hosted service, GitHub Pages, 원본 데이터 배포, 연구 전체의 재학습을 포함하지 않는다.
