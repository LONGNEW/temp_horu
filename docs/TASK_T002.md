TASK.md
작업 식별자

    Task ID: T002

    작업명: FedHDC global-model bootstrap 및 공통 federated runner 구현

    기준 문서: RESEARCH_SPEC.md, T001 결과 보고서

    상태: Ready

1. 현재 수행할 단일 작업

T001의 encoder, UCI-HAR cache와 prototype 연산을 재사용하여 다음 두 단계를 구현한다.

    federated round 전에 client local prototype으로 global model을 만드는 one-time bootstrap

    global broadcast, local batch training, full-model aggregation을 반복하는 FedHDC runner

python -m horu_artifact federated   --method fedhdc   --config configs/fedhdc_ucihar_smoke.yaml   --data-root data   --output results/fedhdc_ucihar_smoke   --device cpu

Smoke는 UCI-HAR subject 1--3, 2 rounds를 사용한다. 30-subject, D=2000, 5-round acceptance config도 제공하되 논문 정확도 재현은 완료 조건이 아니다.
2. 알고리즘 기준

참고 문헌:

K. Ergun, R. Chandrasekaran, and T. Rosing,
“Federated Hyperdimensional Computing,” arXiv:2312.15966, 2023.

원 논문은 class sample hypervector bundling으로 local class hypervector를 만들고, server가 local models를 aggregation해 global model을 갱신한다. global model의 timing은 명시되지 않아 아래 bootstrap을 Artifact 규칙으로 정의한다.

다음은 USER_SPECIFIED 기준이다.

    similarity: dot product

    update target: unit-normalized encoded hypervector

    local batch size: 16

    batch 시작 model로 prediction 고정

    batch delta를 종료 시 한 번 적용

    변경 prototype row와 aggregated global row L2 normalization

    server aggregation: client train-sample weighted average

차이는 docs/baselines/fedhdc_reference.md에 기록하며 원 논문의 완전한 재현이라고 주장하지 않는다.
3. One-time global bootstrap
3.1 Client 단계

client는 T001의 동일 projection matrix E와 encoder를 사용한다.

H_i = cos(X_i E)
S_i[k] = Σ_{j:y_j=k} H_i[j]
M_i^0[k] = normalize(S_i[k])

빈 class row는 0이다. S_i는 원 논문의 bundling에 대응하고 row normalization은 USER_SPECIFIED 결정이다.

각 client는 전체 M_i^0 ∈ R^(K×D)와 train sample 수 n_i를 server에 upload한다.
3.2 Server 단계

M_global^0 =
  normalize_rows(
    Σ_i n_i M_i^0 / Σ_i n_i
  )

server는 M_global^0을 모든 참여 client에 broadcast한다. 첫 federated round는 이 model의 clone에서 시작한다. Client local bootstrap model을 round 1에 그대로 이어서 사용해서는 안 된다.
3.3 Bootstrap timing scope

논문의 HoRU Table I과 같은 “round 이전 one-time cost”로 분리 보고하되, 동일한 알고리즘 단계라고 표현하지 않는다. 다음을 측정한다.

Client:

    local encoded-cache read

    class bundling

    row normalization

    initial-model serialization/copy

Server:

    initial models 수신/copy

    weighted aggregation

    global row normalization

    broadcast serialization/copy

Data download, archive extraction, preprocessing과 nonlinear encoding cache 생성은 data_prepare_ms로 별도 기록하고 bootstrap total에서 제외한다.

단일 프로세스 simulation에서는 다음을 저장한다.

client_bootstrap_sum_ms
client_bootstrap_max_ms
server_bootstrap_ms
broadcast_ms
bootstrap_sequential_ms
bootstrap_parallel_estimate_ms

parallel_estimate = client_max + server + broadcast로 정의한다. 실제 network latency로 간주하지 않는다.

초기 통신량:

bootstrap_upload = clients × K × D × 4
bootstrap_download = clients × K × D × 4

4. Federated round

Round t에서:

    M_global^t를 참여 client에 clone

    client local model을 clone으로 완전히 교체

    deterministic shuffle 후 batch size 16으로 local training

    전체 local model upload

    sample-weighted aggregation 및 row normalization

    global/client accuracy와 payload 기록

Prediction:

score[k] = M[k] · h
ŷ = argmax_k score[k]

zero row score는 -inf이다.

Batch update:

u = h / max(||h||₂, ε)
Δ[y] += ηu
Δ[ŷ] -= ηu

오분류 sample만 누적한다. Batch 종료 후 M←M+Δ를 수행하고 변경된 nonzero row만 normalize한다. Delta는 합계이며 평균으로 나누지 않는다. Incomplete batch도 동일하다.

Server aggregation:

M_global^(t+1) =
 normalize_rows(Σ_i n_i M_i^(t+1) / Σ_i n_i)

Recurring payload:

round_upload = participants × K × D × 4
round_download = participants × K × D × 4

Bootstrap과 recurring-round timing을 섞지 않는다. R rounds total은 다음 두 값을 모두 보고한다.

sequential_total = bootstrap_sequential + Σ round_sequential
parallel_estimate_total = bootstrap_parallel_estimate + Σ round_parallel_estimate

5. 변경 범위

src/horu_artifact/federated/{interfaces,runner,metrics}.py
src/horu_artifact/methods/fedhdc.py
src/horu_artifact/{cli,config}.py
configs/fedhdc_ucihar_{smoke,acceptance}.yaml
docs/baselines/fedhdc_reference.md
tests/test_{fedhdc,federated_runner,fedhdc_cli}.py

Runner는 method interface를 사용하고 T001 API 호환성을 유지한다.
6. 산출물

results/fedhdc_ucihar_smoke/
├── result.json
├── config.resolved.yaml
├── environment.json
├── bootstrap_metrics.csv
├── round_metrics.csv
├── client_metrics.csv
├── communication.csv
└── checkpoints/

필수 기록:

    dataset, projection, split, config와 client-selection hash

    initial local/global model hash

    bootstrap client/server component time

    bootstrap 및 round별 upload/download bytes

    round별 global hash, client accuracy, update 수

    mean, P10, worst accuracy

    Git commit, device와 dependency version

7. 테스트

    local class bundling과 row normalization 수작업 일치

    initial weighted aggregation과 global normalization 일치

    client가 동일 M_global^0 clone으로 round 1 시작

    bootstrap이 한 번만 실행되고 resume 시 반복되지 않음

    bootstrap timing field가 존재하고 음수가 아님

    bootstrap payload가 clients×K×D×4와 일치

    dot prediction과 unit update target 검증

    batch size 16, stale-batch semantics와 incomplete batch 검증

    changed row만 normalize

    recurring aggregation과 payload 검증

    broadcast tensor aliasing 없음

    동일 seed 재실행 및 checkpoint resume 결과 동일

    3-client, 2-round CPU integration 성공

    smoke 중 network 미사용

8. 완료 조건

    reference 문서에 논문 근거와 USER_SPECIFIED 차이 기록

    one-time global bootstrap과 recurring round를 분리 구현

    bootstrap client/server timing 및 initial payload 산출

    dot, unit update, batch 16을 config와 test로 고정

    global model 수식과 round aggregation 검증

    3-client 2-round smoke 성공

    deterministic 재실행과 resume 검증

    CPU-only pytest -q 전체 통과

    실제 결과, bootstrap breakdown, 실패와 미확인 사항을 RESULTS.md에 기록