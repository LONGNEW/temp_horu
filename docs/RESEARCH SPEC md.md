# RESEARCH_SPEC.md

## 1. 연구 목적

본 프로젝트의 목적은 CASES 2026에 게재되는 논문
“Personalized Federated Hyperdimensional Computing via
Shared–Personal Prototype Memory Decomposition”의 핵심 방법과
실험 결과를 독립적인 환경에서 재현 가능한 Artifact로 구현하는 것이다.

Artifact는 다음 결과를 검증할 수 있어야 한다.

1. HoRU의 shared–local prototype memory decomposition 구현
2. 6개 federated benchmark에서의 정확도 평가
3. FedHDC 및 HyperFeel 대비 정확도 개선 확인
4. Controlled synthetic benchmark에서 client-round latency 측정
5. Full prototype 통신 대비 communication payload 감소 확인
6. 한 번의 명령으로 데이터 준비, 실행, 결과 생성 및 검증 가능

본 Artifact의 최소 목표 badge는 다음과 같다.

- Artifacts Available
- Artifacts Evaluated – Functional
- Results Validated – Reproduced

코드 구조와 문서화가 충분할 경우
Artifacts Evaluated – Reusable까지 신청한다.

---

## 2. 연구 가설과 수식

### 2.1 연구 가설

#### H1. 정확도 가설

클라이언트의 class prototype memory를 하나의 global memory로
통합하는 대신 shared state와 local state로 분해하면,
non-IID 환경에서 personalized client accuracy가 향상된다.

주요 비교 대상은 다음과 같다.

- FedHDC
- HyperFeel
- HoRU

논문에서 보고된 6개 데이터셋 평균 정확도는 다음과 같다.

- HoRU: 72.66%
- HyperFeel: 68.28%
- FedHDC: 51.38%

따라서 기대되는 평균 정확도 차이는 다음과 같다.

- HoRU − HyperFeel ≈ +4.38 percentage points
- HoRU − FedHDC ≈ +21.28 percentage points

:contentReference[oaicite:1]{index=1}

#### H2. 통신량 가설

Full prototype memory가 클래스 수 K와 HD dimension D에 대해
K × D 값을 전송하는 반면, HoRU는 shared coefficients만 전송한다.

HoRU의 round당 upload 크기는 다음과 같다.

payload_HoRU = K × (r_c + r_g)

Full prototype upload는 다음과 같다.

payload_full = K × D

따라서 상대 통신량은 다음과 같다.

payload_HoRU / payload_full ≈ (r_c + r_g) / D

기본 설정에서:

- D = 2000
- r_c = 24
- r_g = 8

이므로:

(24 + 8) / 2000 = 0.016

즉, HoRU의 recurring payload는 full prototype 방식보다
약 62.5배 작아야 한다.

:contentReference[oaicite:2]{index=2}

#### H3. 실행시간 가설

HoRU는 full D-dimensional prototype에서 반복 연산하는 대신
low-rank coefficient space에서 prediction과 update를 수행한다.

따라서 bootstrap 이후 client-round latency가 FedHDC와
HyperFeel보다 작아야 한다.

논문의 controlled benchmark 기준 목표값은 다음과 같다.

- HoRU client round: 약 1.63 ms
- FedHDC client round: 약 4.72 ms
- HyperFeel client round: 약 4.87 ms
- HoRU upload: 6.4 KB
- FedHDC / HyperFeel upload: 400 KB

절대 시간은 하드웨어에 따라 달라질 수 있으므로,
Artifact의 기본 검증에서는 절대 시간보다 상대 speedup을 우선한다.

목표 상대 성능:

- recurring client-round speedup: 약 3.0×
- 25-round bootstrap-inclusive speedup: 약 1.36–1.37×

:contentReference[oaicite:3]{index=3}

---

## 3. HoRU 구현 명세

Git hub의 HDZOO가 제공하는 nonlinear encoding 방식에서 시작되어야 한다. 
서버가 제공하는 projection matrix 가 이에 해당한다. 

### 3.1 Client prototype memory

클라이언트 i의 class prototype memory는 다음 행렬로 표현한다.

M_i ∈ R^(K×D)

각 행은 한 클래스의 normalized class hypervector이다.

### 3.2 Shared basis 추출

서버는 모든 클라이언트 prototype의 second-order statistics를
집계한다.

Σ = Σ_i M_i^T M_i

Σ의 eigendecomposition으로 다음 basis를 생성한다.

B = [B_c, B_g]

- B_c: common basis
- B_g: global basis

:contentReference[oaicite:4]{index=4}

### 3.3 Shared state와 local state

각 클라이언트의 prototype representation은 다음 coefficient로 구성한다.

Shared state:

- C_i: common coefficients
- G_i: global coefficients

Local state:

- Δ_i: local correction in common basis
- P_i: personal coefficients
- B_p,i: personal basis

Class k의 coefficient representation은 다음과 같다.

u_i,k = [C_i[k] + Δ_i[k], G_i[k], P_i[k]]

### 3.4 Prediction

입력 hypervector h를 각 basis에 projection한다.

z_c = h B_c  
z_g = h B_g  
z_p = h B_p,i

Prediction은 coefficient space의 cosine similarity로 수행한다.

ŷ = argmax_k cosine(
    [z_c, z_g, z_p],
    [C_i[k] + Δ_i[k], G_i[k], P_i[k]]
)

:contentReference[oaicite:5]{index=5}

### 3.5 Federated aggregation

클라이언트는 C_i와 G_i만 서버에 업로드한다.

서버는 class sample count n_i,k를 이용해 class-wise weighted
aggregation을 수행한다.

C̄[k] = Σ_i n_i,k C_i[k] / Σ_i n_i,k  
Ḡ[k] = Σ_i n_i,k G_i[k] / Σ_i n_i,k

Δ_i, P_i, B_p,i는 클라이언트에 유지한다.

클라이언트는 local class error ratio를 기반으로 aggregated shared
state를 선택적으로 반영한다.

:contentReference[oaicite:6]{index=6}

---

## 4. 실험 설계

실험은 세 단계로 구분한다.

### 4.1 Smoke test

목적:

- 설치 및 데이터 흐름 검증
- HoRU bootstrap, local update, aggregation 검증
- 5분 이내 실행

설정:

- 생성형 toy dataset
- 3 clients
- 3 classes
- 100 samples/client 이하
- D = 128 또는 256
- 2 federated rounds

검증 항목:

- 모든 tensor shape 정상
- loss 또는 misclassification count 계산 가능
- shared coefficients만 서버로 전송
- local coefficients가 서버로 전달되지 않음
- 결과 JSON 생성

### 4.2 Main accuracy experiments

대상 데이터셋:

1. UCI-HAR
2. ISOLET
3. FEMNIST
4. WISDM
5. Synthetic
6. NinaPro DB1

공통 학습 설정:

- communication rounds: 25
- client participation: 100%
- local epochs: 3
- batch size: 32
- random seeds: 최소 3개
- HD dimension: 2000
- HDC learning rate: 0.035
- common rank: 24
- global rank: 8
- personal rank: 64

이는 논문의 기본 실험 설정을 따른다.

:contentReference[oaicite:7]{index=7}

필수 비교 방법:

- HoRU
- FedHDC
- HyperFeel

확장 비교 방법:

- FedAvg
- DFL

Artifact review의 제한된 시간에 대응하기 위해 두 실행 모드를 제공한다.

#### Quick reproduction

- 대표 데이터셋 2개 사용
- 권장 데이터셋: FEMNIST, WISDM
- seed 1개
- 핵심 정확도 경향 확인
- 예상 실행시간 1–3시간 이내

#### Full reproduction

- 전체 6개 데이터셋
- 논문에 사용한 모든 seed
- 전체 accuracy table 생성
- 장시간 실행 허용

### 4.3 Controlled systems benchmark

논문의 latency 및 payload 실험을 재현하는 synthetic benchmark를
구현한다.

설정:

- clients: 50
- classes: 50
- training samples/client: 1000
- batch size: 32
- local epochs: 1
- initial misclassified samples/client: 500
- D: 2000
- common rank: 24
- global rank: 8
- personal rank: 64
- CPU execution
- GPU 사용 금지

:contentReference[oaicite:8]{index=8}

측정 구성요소:

- one-time bootstrap time
- local similarity time
- coefficient update time
- final prediction time
- error-statistics time
- server aggregation time
- total client-round time
- upload payload bytes
- download payload bytes
- bootstrap-inclusive total runtime

Timing은 다음 조건을 적용한다.

- warm-up 실행 제외
- 최소 30회 반복
- median과 interquartile range 보고
- Python wall-clock은 `time.perf_counter_ns()` 사용
- CPU thread 수 고정
- 시스템 및 CPU 정보 결과에 기록

Intel RAPL energy 측정은 지원 가능한 시스템에서만 수행하는
optional experiment로 둔다.

---

## 5. 데이터셋 명세

각 데이터셋 loader는 다음 기능을 제공해야 한다.

- 자동 다운로드 또는 명시적 수동 다운로드 안내
- source URL 및 라이선스 기록
- archive checksum 검증
- deterministic preprocessing
- deterministic client split
- train/test split 저장
- 캐시 재사용
- 데이터 통계 출력

각 데이터셋에 대해 다음 metadata를 저장한다.

- 데이터셋 버전
- 원본 파일 checksum
- client 수
- class 수
- feature dimension
- train/test sample 수
- normalization 방법
- sample cap
- random seed

논문의 데이터 처리 조건을 따른다.

- UCI-HAR: subject split, 30 clients, L2 normalization
- ISOLET: original federated split, 8 clients, L2 normalization
- FEMNIST: LEAF writer split, first 200 clients
- WISDM: user split, 51 clients, standardized features
- Synthetic: LEAF-style 30-client split
- NinaPro DB1: subject split, 27 clients, standardized features

:contentReference[oaicite:9]{index=9}

---

## 6. 평가 지표

### 6.1 Accuracy 지표

필수:

- mean personalized client accuracy
- standard deviation across seeds
- per-client accuracy
- 10th-percentile client accuracy
- worst-client accuracy

### 6.2 Efficiency 지표

- bootstrap latency
- client-round latency
- server aggregation latency
- total round latency
- bootstrap-inclusive runtime
- upload payload bytes
- download payload bytes
- peak memory usage

Optional:

- Intel RAPL package energy

### 6.3 결과 파일

모든 실험은 다음 형식으로 결과를 저장한다.

results/
├── raw/
│   ├── run_config.json
│   ├── client_metrics.csv
│   ├── round_metrics.csv
│   └── timing_samples.csv
├── summary/
│   ├── accuracy_table.csv
│   ├── efficiency_table.csv
│   └── validation_report.json
└── figures/

모든 결과 파일에는 다음 정보가 포함되어야 한다.

- Git commit hash
- 실행 명령
- seed
- hostname
- CPU/GPU 정보
- OS
- Python 버전
- dependency 버전
- 시작 및 종료 시각

---

## 7. 최종 성공 기준

### 7.1 기능 성공 기준

- Docker image가 clean machine에서 build된다.
- smoke test가 오류 없이 완료된다.
- 모든 데이터셋 loader가 동일 split을 재생성한다.
- HoRU의 bootstrap, local training, aggregation, inference가 동작한다.
- FedHDC와 HyperFeel이 동일한 실험 interface로 실행된다.
- 실험 결과가 CSV와 JSON으로 저장된다.

### 7.2 정확도 성공 기준

Random seed 및 환경 차이를 고려해 다음 조건을 사용한다.

필수 조건:

- 6개 데이터셋 평균에서 HoRU가 FedHDC보다 높다.
- 6개 데이터셋 평균에서 HoRU가 HyperFeel보다 높다.
- 논문 평균값과 Artifact 평균값의 차이가 사전에 정한 허용 오차 이내다.

초기 허용 오차:

- 개별 데이터셋 accuracy: ±2.0 percentage points
- 6개 데이터셋 평균 accuracy: ±1.0 percentage point
- 방법 간 평균 차이: 논문과 동일한 방향

최종 허용 오차는 최초 full experiment 결과 이후 확정한다.

### 7.3 시스템 성공 기준

하드웨어 차이를 고려해 절대 latency는 badge 성공 조건으로
강제하지 않는다.

필수 조건:

- HoRU payload가 수식과 정확히 일치
- HoRU payload가 FedHDC/HyperFeel의 1/62.5 수준
- 동일 머신에서 HoRU recurring client round가 두 HDC baseline보다 빠름
- bootstrap time과 recurring time이 분리되어 보고됨

목표 조건:

- recurring round speedup ≥ 2.5×
- 25-round bootstrap-inclusive runtime에서 HoRU가 baseline보다 빠름

### 7.4 재현성 성공 기준

다음 명령이 문서에 명시되고 실제로 동작해야 한다.

1. 환경 설치
2. smoke test
3. 데이터셋 다운로드
4. quick reproduction
5. full reproduction
6. systems benchmark
7. 결과 검증
8. 논문 표와 figure 생성

---

## 8. 전체 구현 순서

### Phase 1. Repository와 실행 인터페이스

1. Python package 구조 생성
2. configuration schema 생성
3. CLI 생성
4. logging 및 result schema 생성
5. Docker 환경 생성
6. smoke test 생성

### Phase 2. Core HDC

1. encoder 구현
2. class hypervector 생성
3. HDC prediction 구현
4. error-driven update 구현
5. unit test 작성

### Phase 3. Baseline

1. FedHDC 구현
2. HyperFeel 구현
3. 공통 federated runner 구현
4. baseline sanity test 작성

### Phase 4. HoRU

1. shared basis bootstrap
2. shared/local state initialization
3. coefficient-space prediction
4. coefficient error-driven update
5. class-wise error statistics
6. server aggregation
7. client shared-state absorption
8. inference
9. communication accounting
10. unit 및 integration test

### Phase 5. Dataset pipeline

1. UCI-HAR
2. ISOLET
3. Synthetic
4. WISDM
5. FEMNIST
6. NinaPro DB1

구현 난도가 낮고 다운로드 검증이 쉬운 데이터셋부터 진행한다.

### Phase 6. Accuracy reproduction

1. 각 데이터셋 단일 seed 실행
2. 설정 오류 수정
3. 전체 seed 실행
4. 논문 결과와 비교
5. 허용 오차 결정
6. 자동 validation script 작성

### Phase 7. Systems benchmark

1. controlled synthetic generator 구현
2. timing instrumentation
3. payload accounting
4. repeated benchmark
5. bootstrap-inclusive 분석
6. 시스템 결과 표 생성

### Phase 8. Artifact packaging

1. README.md
2. INSTALL.md
3. REQUIREMENTS.md
4. STATUS.md
5. LICENSE
6. Artifact Appendix
7. Docker image 검증
8. clean-machine reproduction
9. DOI archive 생성
