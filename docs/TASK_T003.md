# TASK.md

## 작업 식별자

- **Task ID:** T003
- **작업명:** HyperFeel horizontal personalized retraining 구현
- **기준 문서:** `RESEARCH_SPEC.md`, T001/T002 보고서
- **상태:** Ready

## 1. 현재 수행할 단일 작업

HyperFeel 원문을 기준으로 central AM, delta 통신, personalized update를 T002 runner에 독립 method로 구현한다.

```bash
python -m horu_artifact federated   --method hyperfeel   --config configs/hyperfeel_ucihar_smoke.yaml   --data-root data   --output results/hyperfeel_ucihar_smoke   --device cpu
```

Smoke는 UCI-HAR subject 1--3, 2 retraining rounds를 사용한다. Vertical FL과 논문 accuracy 재현은 비범위다.

## 2. 고정 참고 문헌

```text
H. Li, F. Liu, Y. Chen, and L. Jiang,
“HyperFeel: An Efficient Federated Learning Framework
Using Hyperdimensional Computing,” ASP-DAC 2024,
pp. 716–721. DOI: 10.1109/ASP-DAC58780.2024.10473907
PDF SHA-256:
c298831aec5d1c929e67e6670ebcd5823be125ad6440b805a1cc9a7c2d4aa734
```

`hyperfeel_reference.md`에 Fig. 1, Eq. (1)–(3), Algorithm 1 대응을 기록한다.

## 3. 논문 기반 알고리즘

### 3.1 공통 표현

T001의 공유 projection과 nonlinear encoder를 사용한다.

```text
Q = cos(XE), Q ∈ R^D
Client AM: C_i ∈ R^(K×D)
Client delta: δ_i ∈ R^(K×D)
Server delta: Δ ∈ R^(K×D)
```

원문은 high-precision vector에 cosine을 설명하지만 class 비교에서는 inner product로 대체하므로 dot product로 고정한다.

```text
ŷ = argmax_k C_i[k] · Q
```

원문은 update에 raw sample hypervector `Q`를 사용하며 prototype row normalization을 명시하지 않는다. Paper-faithful mode에서는 unit update와 row normalization을 적용하지 않는다.

### 3.2 One-time central-AM bootstrap

각 client의 AM을 0으로 초기화하고 local train sample을 정답 class에 bundle한다.

```text
C_i^local[y] += Q
```

Static dataset에서는 local train sample을 한 번 사용한 뒤 AM을 upload한다. Server는 client AM을 element-wise sum하여 central AM을 만든다.

```text
C_central = Σ_i C_i^local
```

Server는 `C_central`을 broadcast하고 client는 초기 `C_i`로 복사해 지속 유지한다. 가중 평균과 row normalization을 추가하지 않는다.

Bootstrap timing:

- client encoded-cache read와 class bundling
- initial AM upload copy
- server accumulation
- central AM broadcast copy
- sequential 및 client-parallel estimate

Bootstrap payload:

```text
upload = N × K × D × 4
download = N × K × D × 4
```

Data download, preprocessing과 encoding-cache 생성 시간은 제외한다.

### 3.3 Personalized retraining round

상태는 round 사이에 유지한다.

```text
C_i: personalized local AM
Δ_prev: 이전 round의 global delta
error_ratio_i[k]: 이전 round의 class error ratio
```



Round `r` 시작 시 client는 이전 global delta를 받아 class별 personalization update를 수행한다.

```text
C_i[k] += error_ratio_i[k] × lr × Δ_prev[k]
error_ratio_i[k] = error_i[k] / cnt_i[k]
```

`cnt_i[k]=0`이면 ratio는 0이다. 그 후 `δ_i`, error count를 0으로 초기화하고 local sample을 **한 개씩 순차 처리**한다.

오분류 `(Q, y, ŷ)`에 대해 Eq. (2)를 그대로 적용한다.

```text
C_i[y] += lr × Q
C_i[ŷ] -= lr × Q
δ_i[y] += lr × Q
δ_i[ŷ] -= lr × Q
```

정분류 sample은 변경하지 않는다. `batch_size=16`은 loader chunk이며 각 sample은 현재 `C_i`로 다시 prediction한다.

Local pass 종료 후 client는 `δ_i`만 upload한다. Server는 Algorithm 1처럼 평균이 아닌 합으로 global delta를 생성한다.

```text
Δ[k] = Σ_i δ_i[k]
```

Server는 `Δ`만 broadcast하며 `C_i`를 덮어쓰지 않는다.

### 3.4 통신량

실제로 통신하는 float32 tensor만 계산한다.

```text
round_upload = participants × K × D × 4
round_download = participants × K × D × 4
```

Local personalized AM, error count와 class count는 통신하지 않는다. Zero delta를 sparse compression하지 않는다.

## 4. 변경 범위

```text
src/horu_artifact/methods/hyperfeel.py
src/horu_artifact/federated/{interfaces,runner,metrics}.py
src/horu_artifact/{cli,config}.py
configs/hyperfeel_ucihar_{smoke,controlled}.yaml
docs/baselines/hyperfeel_reference.md
tests/test_{hyperfeel,hyperfeel_cli,federated_runner}.py
```

T001/T002 API와 schema를 유지하고 Runner는 method hook만 호출한다.

## 5. 기본 smoke 설정

```yaml
method: hyperfeel
implementation_mode: paper_faithful
dataset: ucihar
subject_ids: [1, 2, 3]
rounds: 2
participation: 1.0
local_epochs: 1
batch_size: 16
hd_dim: 256
learning_rate: 0.035
similarity: dot
normalize_update_hypervectors: false
normalize_prototypes: false
server_aggregation: sum
seed: 0
device: cpu
```

원 논문 실험은 `D=1000`, `N=30`을 사용한다. `controlled` config는 HoRU 비교용 `D=2000`이며 T003 결과로 주장하지 않는다.

## 6. 산출물

```text
results/hyperfeel_ucihar_smoke/
├── result.json
├── config.resolved.yaml
├── environment.json
├── bootstrap_metrics.csv
├── round_metrics.csv
├── client_metrics.csv
├── communication.csv
└── checkpoints/
```

필수 기록:

- source PDF, dataset, projection, split, config hash
- initial central AM과 client AM hash
- round별 `Δ`, personalized AM과 upload hash
- personalized pooled/mean/P10/worst accuracy
- class별 count, error와 personalization weight
- bootstrap/round timing과 payload
- Git commit, device와 dependency version

## 7. 테스트

- local class bundling과 central AM sum 수작업 일치
- bootstrap 후 client가 동일 central AM clone으로 시작
- first round의 zero-delta personalization은 무변경
- Eq. (2)의 `C_i`와 `δ_i` 두-row update 일치
- sample-wise current-model prediction 검증
- Eq. (3)의 error ratio와 personalized update 일치
- `cnt=0` class update 없음
- server delta가 client delta의 합과 일치
- 전체 local AM이 upload되지 않음
- client personalized AM이 round 사이에 유지됨
- payload가 `N×K×D×4`와 일치
- deterministic rerun과 checkpoint resume 동일
- FedHDC regression test 통과
- 3-client, 2-round offline CPU CLI 성공

## 8. 완료 조건

1. reference 문서에 Fig./Eq./Algorithm 대응 기록
2. central-AM bootstrap과 delta retraining 분리 구현
3. raw-Q sample-wise update와 server sum 검증
4. class error-ratio personalization 검증
5. personalized state persistence와 resume 검증
6. timing, payload, state hash와 personalized metric 생성
7. 3-client 2-round smoke 성공
8. CPU-only `pytest -q` 전체 통과
9. 결과와 논문 대비 차이를 `RESULTS.md`에 기록

비범위: vertical FL, FEMNIST/Synthetic 재현, unit-normalized controlled update의 성능 주장, HoRU, energy/network benchmark와 Docker packaging.
