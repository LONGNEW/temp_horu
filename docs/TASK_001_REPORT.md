# TASK 001 구현 및 진단 보고서

## 1. 목적과 범위

TASK 001의 목적은 HDZoo-compatible nonlinear encoder, UCI-HAR cache, client별
prototype classifier, 그리고 로컬 smoke 실행 경로를 만드는 것이다. 이 문서는
구현된 기준 경로와 실행 중 발견된 prototype update-scale 문제, 그리고 이를
분리해 검증한 prototype-only 진단을 기록한다.

이 문서의 accuracy는 모두 subject 1, 2, 3만 사용하는 smoke 진단이다. 논문
결과, baseline 비교 또는 유효한 hyperparameter tuning 결과가 아니다.

## 2. TASK 001 요구사항과 구현

| 요구사항 | 구현 | 상태 |
|---|---|---|
| Gaussian projection | CPU `torch.Generator`로 `(D, F)` Gaussian을 만든 뒤 transpose | 구현됨 |
| nonlinear encoder | `H = cos(XE)`, sign/phase/bias 없음 | 구현됨 |
| UCI-HAR | 공식 archive download, SHA-256, 10,299/561/6/30 검증, offline cache | 구현됨 |
| client split | subject 1--3, subject 내부 class-stratified 70/30 deterministic split | 구현됨 |
| default prototype | class별 encoded train mean, nonzero row L2 normalization | 구현됨 |
| default prediction | cosine similarity; zero prototype은 `-inf` | 구현됨 |
| default update | 현재 memory에 대해 sample별 재예측 후 즉시 push--pull | 구현됨 |
| device | CPU, CUDA, auto | 구현됨 |
| artifacts | JSON, CSV, resolved config, environment, results document | 구현됨 |

기준 smoke config는 `D=256`, `η=0.035`, one local epoch, batch size 128,
cosine similarity, sample-wise update, row normalization이다.

## 3. 데이터와 재현성

공식 UCI-HAR archive SHA-256은
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`이다.
cache는 10,299 samples, 561 features, 6 classes, 30 subjects를 검증한다.

smoke에는 3개 subject만 사용한다.

| Client / subject | 전체 | train | test |
|---:|---:|---:|---:|
| 1 | 347 | 243 | 104 |
| 2 | 302 | 212 | 90 |
| 3 | 341 | 238 | 103 |
| 합계 | 990 | 693 | 297 |

같은 seed와 CPU config에서 projection hash, split index, update 수, accuracy는
재현됨을 확인했다. CPU와 CUDA는 동일 projection hash와 update 수를 사용했으며,
initial accuracy 차이는 약 `4e-8`이었다.

## 4. 구현 변경 이력과 해결

### 4.1 Stale batch prediction 수정

초기 구현은 한 batch 전체 prediction을 먼저 계산한 뒤 개별 update했다. 앞 sample의
update가 memory를 바꾼 뒤에도 뒤 sample이 변경 전 prediction을 사용하므로,
TASK의 sample-level push--pull 의미와 맞지 않았다.

현재 `PrototypeMemory.update`는 매 sample마다 현재 memory로 prediction을 다시
계산하고, 오분류일 때만 true/predicted row를 즉시 갱신한다. regression test는
첫 update가 두 번째 sample의 prediction을 바꾸는 경우를 검증한다.

### 4.2 HDZoo 공개 구현과의 구분

현재 공개 [HDZoo-official](https://github.com/CELL-POSTECH/HDZoo-official)는
nonlinear encoder에서 Gaussian base와 cosine을 사용한다. 그러나 일반 retraining은
zero/sum model, 기본 dot similarity, batch prediction 고정, class별 batch update
aggregation을 사용한다. TASK에 적힌 commit은 현재 공개 저장소에서 확인되지 않아,
그 commit과 완전한 동등성은 주장하지 않는다.

TASK의 mean-normalized prototype + cosine + sample-wise update는 HDZoo encoder를
사용하되, 학습 semantics는 TASK가 별도로 고정한 경로이다. HDZoo batch retraining은
별도 `hdzoo_batch` diagnostic mode로만 추가했다.

## 5. Update-scale 진단

### 5.1 원인

sample-wise L2-normalized input과 Gaussian-cosine encoder에서 실제 norm은 다음과
같았다.

| D | mean `||h||` | mean `||ηh||`, η=0.035 |
|---:|---:|---:|
| 256 | 12.11 | 0.424 |
| 2000 | 33.65 | 1.178 |

row-normalized prototype의 초기 norm은 1이다. 따라서 D=2000에서 한 update의
크기가 prototype norm보다 크다. D=256에서도 update는 약 42%이다. D=256
sample-wise diagnostic에서 class row는 한 epoch에 최대 24번 push/pull 되었고,
initial/final row cosine은 0.937--0.994였다.

### 5.2 진단 결과

아래의 D=2000 결과는 동일 data/split/seed/η/local epoch/batch size(32)를 사용한
prototype-only 비교다. historical baseline 파일은 stale-batch 수정 전 생성되었으므로
이 표에는 포함하지 않았다.

| Initialization / similarity / update | Initial mean acc. | Final mean acc. | Updates | 해석 |
|---|---:|---:|---:|---|
| row norm / cosine / sample-wise | 0.97274 | 0.70403 | 239 | norm-1 memory에서 update가 과대 |
| row norm / dot / sample-wise | 0.97274 | 0.67720 | 295 | dot-only 변경은 악화 |
| row norm / dot / HDZoo batch | 0.97274 | 0.31723 | 433 | aggregation만 이식하면 불안정 |
| **raw class mean / cosine / sample-wise** | **0.97274** | **0.98291** | **17** | 상대 update scale이 작아져 안정화 |
| **row norm / cosine / sample-wise / unit update target** | **0.97274** | **0.98291** | **17** | initialization을 유지하고 update scale만 제어 |

raw class mean의 초기 row norm은 32.15--33.46이었다. 따라서 `||ηh||≈1.18`은
row norm의 약 3.6%였고, cosine prediction의 초기 argmax는 row normalization 유무와
동일했다. 차이는 update가 prototype 방향을 바꾸는 상대 크기에서 발생했다.

동일한 scale 제어는 row normalization을 유지하면서도 가능하다. 새
`normalize_update_hypervectors: true` mode는 encoder `H = cos(XE)`를 바꾸지 않고,
update 때만 `h / ||h||₂`를 사용한다. 그러면 update norm은 정확히 `η=0.035`가
된다. D=2000 diagnostic은 initial 0.97274, final 0.98291, 17 updates를 기록했다.
이는 raw class mean diagnostic과 동일한 accuracy/update 수였지만, TASK의
`M ← M ± ηh` 식을 `M ← M ± η h/||h||₂`로 변경한다.

## 6. 채택 상태와 후속 결정

현재 기본 `configs/smoke_ucihar.yaml`은 TASK 001 명세대로
`normalize_prototypes: true`의 기본값을 사용한다. `normalize_prototypes: false`는
명세를 변경하는 USER_SPECIFIED prototype diagnostic이며, 기본 smoke 결과나 논문
주장으로 승격하지 않았다.

향후 선택지는 셋이다.

1. TASK의 row normalization을 유지한다면, raw `ηh` 대신 unit update target을
   명시적으로 채택할 수 있다. 이는 `η=0.035`를 유지하지만 update 식을 바꾸는
   USER_SPECIFIED prototype 방식이다.
2. raw `ηh`도 유지하려면, D=2000에서 raw-mean과 비슷한 상대 step을 맞추는 근사
   시작점은 `η≈0.0011`이다. 이 값은 검증되지 않은 prototype 가설이다.
3. raw class mean을 채택한다면, TASK 001의 row-normalization 요구사항을 명시적으로
   변경하고, 동일 조건에서 multi-seed 및 전체 subject 평가로 별도 검증해야 한다.

어느 선택도 현재 진단만으로 논문 설정 또는 최종 실험 설정이라고 주장해서는 안 된다.

## 7. 검증과 산출물

`PYTHONPATH=src python3 -m pytest -q`의 최신 결과는 `12 passed`다. CUDA가 있는
환경에서는 encoder parity test도 실행된다. 실제 실행 명령과 각 diagnostic의 상세
수치는 [RESULTS.md](../RESULTS.md)에 기록되어 있다.

주요 config와 결과:

- `configs/smoke_ucihar.yaml` / `results/smoke_ucihar/`
- `configs/prototype_ucihar_samplewise_d2000.yaml` /
  `results/prototype_ucihar_samplewise_d2000/`
- `configs/prototype_ucihar_samplewise_d2000_dot.yaml` /
  `results/prototype_ucihar_samplewise_d2000_dot/`
- `configs/prototype_ucihar_hdzoo_batch_d2000.yaml` /
  `results/prototype_ucihar_hdzoo_batch_d2000/`
- `configs/prototype_ucihar_samplewise_d2000_no_row_norm.yaml` /
  `results/prototype_ucihar_samplewise_d2000_no_row_norm/`
- `configs/prototype_ucihar_samplewise_d2000_unit_update.yaml` /
  `results/prototype_ucihar_samplewise_d2000_unit_update/`
