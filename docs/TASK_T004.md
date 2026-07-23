# TASK.md

## 작업 식별자

- **Task ID:** T004
- **작업명:** HoRU one-time bootstrap 및 shared/local decomposition 구현
- **기준 문서:** `RESEARCH_SPEC.md`, HoRU revised paper, T001–T003 보고서
- **상태:** Ready

## 1. 현재 수행할 단일 작업

T001의 Gaussian-cosine encoder와 UCI-HAR cache를 사용해 HoRU의 federated round 이전 bootstrap만 구현한다. Client class prototype으로 shared bases를 추출하고 shared/local coefficient state와 query projection cache를 생성한다.

```bash
python -m horu_artifact federated   --method horu   --config configs/horu_ucihar_bootstrap_smoke.yaml   --data-root data   --output results/horu_ucihar_bootstrap_smoke   --device cpu   --bootstrap-only
```

T004에서는 coefficient retraining, shared-state round aggregation과 personalization update를 구현하지 않는다.

## 2. 논문 기준 상태

Client `i`, classes `K`, HD dimension `D`:

```text
M_i ∈ R^(K×D)
B_c ∈ R^(D×r_c)
B_g ∈ R^(D×r_g)
B_p,i ∈ R^(D×r_p)
C_i ∈ R^(K×r_c)
G_i ∈ R^(K×r_g)
Δ_i ∈ R^(K×r_c)
P_i ∈ R^(K×r_p)
```

모든 client는 동일 projection matrix와 `h=cos(xE)`를 사용한다. 각 class의 encoded train mean을 계산하고 nonempty row를 L2 normalize해 `M_i`를 만든다. 빈 class row는 0이며 class sample count `n_i,k`를 함께 저장한다.

## 3. Server shared-basis bootstrap

Server는 client class prototype 전체를 받아 pooled covariance를 계산한다.

```text
Σ = Σ_i M_i^T M_i
```

`torch.linalg.eigh` 결과를 eigenvalue 내림차순으로 정렬하여 상위 `r_c+r_g` eigenvector를 선택한다.

```text
B = [B_c, B_g]
```

- 앞 `r_c`: common basis
- 다음 `r_g`: global basis

Eigenvector sign은 각 column에서 절댓값이 가장 큰 원소가 양수가 되도록 canonicalize한다. 동률은 가장 작은 index를 사용한다. Raw basis hash와 함께 sign-invariant projector `BB^T` hash를 기록한다.

검증:

```text
B_c^T B_c ≈ I
B_g^T B_g ≈ I
B_c^T B_g ≈ 0
```

`r_c+r_g ≤ D`를 강제하며 선택 eigenvalue, explained-energy ratio와 수치 tolerance를 기록한다.

## 4. Shared-state 초기화

Client prototype을 common basis에 projection한다.

```text
C_i^tot = M_i B_c
```

Server는 class sample count로 common consensus를 계산한다.

```text
C_global[k] =
  Σ_i n_i,k C_i^tot[k] / Σ_i n_i,k
```

분모가 0이면 row는 0이다. 모든 client state를 다음처럼 초기화한다.

```text
C_i^(0) = C_global
G_i^(0) = 0
Δ_i^(0) = 0
```

Server는 `B_c`, `B_g`, `C_global`을 client에 broadcast한다. 각 client는 독립 clone을 가져야 한다.

## 5. Local personal-state 초기화

`B=[B_c,B_g]`에 대해 identity matrix를 직접 만들지 않고 다음과 같이 residual을 계산한다.

```text
R_i = M_i - (M_i B) B^T
```

Client는 `R_i`의 right singular vectors로 personal basis를 만든다.

```text
B_p,i = top-r_p right singular vectors of R_i
P_i^(0) = R_i B_p,i
```

기본 smoke는 `r_p≤min(K,D)`를 사용하고 reduced SVD로 계산한다.

논문 기본 설정은 일부 dataset에서 `r_p>K`이므로 source 식만으로 zero-singular subspace 선택이 유일하지 않다. `full_svd` mode는 full right-singular matrix의 앞 `r_p` vectors를 사용하는 literal completion으로 제공하되 `USER_SPECIFIED_NUMERICAL_COMPLETION`으로 기록한다. Reduced/full mode, effective numerical rank, zero-singular direction 수와 projector hash를 출력한다.

초기 reconstruction:

```text
M_hat_i =
 row_normalize(
   C_global B_c^T +
   P_i^(0) B_p,i^T
 )
```

`G_i`와 `Δ_i`는 0이므로 reconstruction에서 기여하지 않는다. `M_hat_i`의 finite 여부, nonempty row norm과 `M_i` 대비 reconstruction cosine/error를 기록한다.

## 6. Query coefficient cache

각 client의 encoded train/test hypervector를 한 번 projection해 저장한다.

```text
z_c = h B_c
z_g = h B_g
z_p = h B_p,i
```

Cache는 dtype, shape, source split hash, basis projector hash와 연결한다. T005는 full `D` hypervector를 반복 projection하지 않고 이 cache를 사용한다.

## 7. Timing과 payload

HoRU Table I에 대응해 round 이전 one-time cost로 분리한다.

Server:

- pooled covariance
- eigendecomposition/shared basis
- client prototype projection
- common consensus
- server bootstrap total

Client:

- residual construction
- personal basis SVD
- residual coefficient projection
- train/test query coefficient cache
- client별 total, sum과 max

단일 프로세스에서 다음을 기록한다.

```text
bootstrap_sequential_ms
bootstrap_parallel_estimate_ms =
  server_total + max(client_total) + broadcast_copy
```

실제 network latency로 간주하지 않는다.

Per-client payload:

```text
upload = K×D×4
download = [D×(r_c+r_g) + K×r_c]×4
```

전체 payload는 client 수를 곱해 기록한다. Local `B_p,i`, `P_i`, `Δ_i`와 query cache는 통신하지 않는다.

## 8. 변경 범위

```text
src/horu_artifact/methods/horu.py
src/horu_artifact/horu/{basis,state,bootstrap}.py
src/horu_artifact/federated/{interfaces,runner,metrics}.py
src/horu_artifact/{cli,config}.py
configs/horu_ucihar_bootstrap_{smoke,paper_rank}.yaml
docs/methods/horu_bootstrap_reference.md
tests/test_{horu_basis,horu_bootstrap,horu_cli}.py
```

Runner는 method hook만 호출하고 FedHDC/HyperFeel regression을 깨지 않는다.

## 9. 기본 smoke 설정

```yaml
method: horu
dataset: ucihar
subject_ids: [1, 2, 3]
hd_dim: 256
common_rank: 3
global_rank: 2
personal_rank: 4
personal_basis_policy: reduced_svd
seed: 0
device: cpu
bootstrap_only: true
```

별도 paper-rank diagnostic은 `D=2000`, `r_c=24`, `r_g=8`, `r_p=64`, `full_svd`를 사용하되 논문 재현 결과로 주장하지 않는다.

## 10. 산출물과 테스트

```text
results/horu_ucihar_bootstrap_smoke/
├── result.json
├── config.resolved.yaml
├── environment.json
├── bootstrap_metrics.csv
├── state_manifest.json
├── reconstruction_metrics.csv
└── checkpoints/bootstrap.pt
```

필수 테스트:

- class mean/row normalization과 `n_i,k`
- covariance 및 eigenpair 수작업 비교
- basis orthogonality와 sign canonicalization
- class-weighted `C_global`
- `C/G/Δ/P` shape와 zero initialization
- residual orthogonality `R_i B≈0`
- personal SVD와 projector 검증
- reconstruction과 cache shape/finite
- payload 수식
- 동일 CPU config 재실행 hash 일치
- checkpoint load 후 동일 state
- 3-client offline CLI 성공
- 기존 21개 이상 regression test 통과

## 11. 완료 조건

1. 논문 Eq. (5)–(16) 대응 문서화
2. shared basis와 shared/local state bootstrap 구현
3. query coefficient cache 생성
4. server/client timing과 bootstrap payload 기록
5. personal-rank ambiguity와 completion policy 명시
6. deterministic rerun/checkpoint 검증
7. CPU-only `pytest -q` 전체 통과
8. 실제 결과와 미확인 사항을 `RESULTS.md`에 기록

비범위: coefficient prediction/update, class-error statistics, recurring shared-state communication, 25-round accuracy, systems benchmark와 Docker packaging.

