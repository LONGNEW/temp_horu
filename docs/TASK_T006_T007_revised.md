# TASK.md

## 작업 식별자

- **Task ID:** T006–T007
- **작업명:** 6개 dataset pipeline 구축 및 accuracy reproduction
- **기준 문서:** `RESEARCH_SPEC.md`, revised paper, T001–T005 보고서
- **상태:** Ready

## 1. 현재 수행할 단일 작업

6개 dataset의 source·client·split·전처리·cache를 고정하고 세 method를 동일 protocol로 실행해 accuracy table과 validation report를 만든다.

```bash
python -m horu_artifact prepare-data all   --config configs/datasets.yaml --data-root data

python -m horu_artifact run-suite   --config configs/accuracy_full.yaml   --data-root data --output results/accuracy_full

python -m horu_artifact validate-results   --results results/accuracy_full   --reference references/paper_accuracy.csv
```

`prepare-data`만 network를 사용할 수 있다. Raw/processed hash 또는 dataset
statistics가 맞지 않으면 실험을 시작하지 않는다.

## 2. 공통 dataset interface

Loader는 ordered client IDs, train/test tensor, sample ID와 split hash를 반환한다.

```text
download_raw
verify_raw
build_processed
load_federated
dataset_statistics
validate_cache
```

각 dataset은 다음을 생성한다.

```text
data/<dataset>/{raw,processed}/
data/<dataset>/manifest.json
data/<dataset>/statistics.json
docs/datasets/<dataset>.md
```

Manifest에는 source/license, raw hash, parser, client/label/feature 정의, split hash, normalization, cap, seed와 config/Git hash를 기록한다.

## 3. Dataset별 고정 규칙

### UCI-HAR

- 공식 UCI archive, 30 subjects = 30 clients
- 561 features, 6 classes
- 원본 split을 결합한 뒤 client 내부 stratified 70/30; singleton은 train-only
- sample-wise L2 normalization
- provenance: `USER_SPECIFIED_PERSONALIZED_SPLIT`

### ISOLET

- 공식 전체 7,797 samples, 617 features, 26 classes
- 공식 train/test 파일을 결합
- class-wise Dirichlet 8-client partition, `alpha=0.05`, seed 0
- client 최소 50 samples와 2 classes; 실패 시 재시도
- client 내부 class-stratified 70/30
- sample-wise L2 normalization
- partition attempt와 client class histogram 기록
- provenance: `USER_SPECIFIED_DIRICHLET_SPLIT`

### FEMNIST

- LEAF writer = client natural split
- LEAF commit과 생성 JSON SHA-256 고정
- `niid`, sample-level 80/20 split, sampling/split seed 0
- train/test 공통 writer ID 정렬 후 first 200 clients
- 28×28 image를 784 vector로 flatten
- pixel `/255`, 이후 sample-wise L2 normalization
- 기존 LEAF split을 다시 나누지 않음

### WISDM

- UCI WISDM official transformed examples
- phone accelerometer modality only
- subject IDs 1600–1650 = 51 clients
- 공식 10-second window row를 sample로 사용
- 43 features: 30 bins, axis mean/peak interval/absolute deviation/std 각 3개, resultant 1개
- activity order `[A..M,O..S]`를 `0..17`로 remap
- client 내부 class-stratified 70/30
- nonfinite row 제거; 1% 초과 시 실패
- pooled train-only feature standardization
- client별 class-stratified cap train 5,000/test 1,000
- provenance: `USER_SPECIFIED_WISDM_BASIC_FEATURES`

### Synthetic

- LEAF-style generator 또는 보존된 JSON
- 30 clients, 10 classes, 60 features
- `alpha=0.5`, `beta=0.5`, generator/split seed 0
- client 내부 70/30, 추가 normalization 없음
- generator/version/hash와 client statistics 기록; 생성 JSON 고정
- provenance: `USER_SPECIFIED_MODERATE_HETEROGENEITY`

### NinaPro DB1

- subjects 1–27 = 27 clients
- EMG 10 + glove 22 channels
- `restimulus`/`rerepetition`, rest 제외, 52 classes
- test repetitions `[2,5,7]`, 나머지 train
- 100 Hz, 200 ms non-overlap window = 20×32 = 640 features
- gesture/repetition 경계를 넘는 window 제거
- pooled train-only feature standardization
- client별 class-stratified cap train 5,000/test 1,000
- provenance: `USER_SPECIFIED_NINAPRO_REPETITION_SPLIT`

## 4. 공통 accuracy protocol

```yaml
rounds: 25
participation: 1.0
local_epochs: 3
batch_size: 32
seeds: [0, 1, 2]
hd_dim: 2000
hd_learning_rate: 0.035
horu:
  common_rank: 24
  global_rank: 8
  personal_rank: 64
```

Train sample이 100,000을 넘으면 client 내부 stratified 방식으로 총 50,000개까지 줄이고 ID/hash를 기록한다.

Method 정의:

- FedHDC: T002 controlled baseline
- HyperFeel: T003 retraining semantics + shared Gaussian-cosine encoder
- HoRU: T004/T005 구현

세 method는 동일 cache, client order, projection, budget과 metric을 사용하며 고유 update는 유지한다.

## 5. 실행 모드

Quick:

```yaml
datasets: [ucihar, isolet]
methods: [fedhdc, hyperfeel, horu]
seeds: [0]
rounds: 3
```

Full은 6×3×3, 25 rounds다. Run별 checkpoint를 사용하고 resume은 round/RNG/state를 복원한다.

## 6. Metric과 산출물

필수 metric:

- client별/pooled accuracy, client mean/std/P10/worst
- seed별 final/mean/std, convergence, runtime과 payload

```text
results/accuracy_full/
├── runs/<dataset>/<method>/<seed>/
├── summary/accuracy_by_seed.csv
├── summary/accuracy_table.csv
├── summary/client_tail_table.csv
├── summary/validation_report.json
└── figures/accuracy_comparison.pdf
```

`paper_accuracy.csv`에는 확인된 target만 넣고 미확보 값은 `NA`로 둔다.

## 7. 자동 검증

- raw/processed hash와 expected statistics 일치
- 세 method의 sample/split hash 동일
- 동일 seed 재실행 시 projection, split, final metric 재현
- failed/missing run을 summary에 명시
- dataset target이 있으면 절대 오차 ≤ 2.0 percentage points
- 6-dataset method mean 절대 오차 ≤ 1.0 point
- HoRU mean > HyperFeel mean > FedHDC mean
- aggregate target:
  HoRU 72.66%, HyperFeel 68.28%, FedHDC 51.38%

기준은 사후 완화하지 않으며 tuning으로 수치를 맞추지 않는다.

## 8. 변경 범위와 테스트

```text
src/horu_artifact/datasets/{isolet,femnist,wisdm,synthetic,ninapro}.py
src/horu_artifact/experiments/{suite,validation,reporting}.py
configs/{datasets,accuracy_quick,accuracy_full}.yaml
docs/datasets/*.md
references/paper_accuracy.csv
tests/test_datasets_*.py
tests/test_accuracy_suite.py
```

테스트는 offline parser, mapping, split, train-only normalization, cap, hash, resume, summary 재계산과 regression을 포함한다.

## 9. 완료 조건

1. 6개 dataset cache와 provenance 문서 완성
2. quick suite 성공
3. full 6×3×3 실행 완료 또는 실패 run 명시
4. accuracy/tail/convergence 결과 생성
5. 자동 validation report 생성
6. deterministic rerun/resume 검증
7. CPU-only `pytest -q` 전체 통과
8. 실제 결과와 차이를 `RESULTS.md`에 기록

비범위: FedAvg/DFL, rank sweep, ablation, participation/late-client 연구,
controlled systems benchmark, Docker와 DOI packaging.

다음 작업: **T008 — controlled synthetic systems benchmark**
