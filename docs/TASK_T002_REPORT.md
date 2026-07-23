# TASK T002 구현 및 검증 보고서

## 1. 목적과 범위

TASK T002는 T001의 nonlinear encoder와 offline UCI-HAR cache를 재사용하여,
FedHDC의 one-time global bootstrap 및 공통 federated runner를 구현하는 작업이다.

이 보고서는 subject 1--3, 2-round CPU smoke의 기능 검증 기록이다. 정확도는
논문 재현, baseline 비교 또는 hyperparameter tuning 결과가 아니다. 공식 성능값은
**매 round 이후 global model을 모든 참여 client의 test sample에 적용한 pooled
accuracy**로 정의한다. client별 accuracy는 진단 지표다.

## 2. 구현된 알고리즘

### 2.1 One-time bootstrap

client `i`는 동일 projection `E`와 `H=cos(XE)`를 사용해 class-wise encoded
sample sum을 만들고 nonzero row를 L2 normalize한다.

`M_i^0[k] = normalize(sum(H_i[j] for y_i[j] = k))`

server는 client의 train sample 수 `n_i`로 local model 전체를 가중 평균하고
nonzero row를 L2 normalize한다.

`M_global^0 = normalize_rows(sum(n_i M_i^0) / sum(n_i))`

각 client는 round 1 시작 시 자신의 bootstrap model을 이어서 쓰지 않고,
`M_global^0`의 독립 clone에서 시작한다.

### 2.2 Recurring federated round

각 round에서 client는 global clone을 deterministic shuffle한 local train data로
학습한다. TASK T002의 USER_SPECIFIED 규칙을 다음과 같이 고정했다.

- dot-product prediction, zero row score는 `-inf`
- batch size 16
- batch 시작 model로 prediction 고정
- 오분류 sample만 `h / max(||h||₂, eps)`를 이용해 true/predicted row delta에 누적
- delta는 batch 종료 시 한 번 적용하며 평균으로 나누지 않음
- 변경된 nonzero row만 L2 normalize
- server는 전체 client model을 train-sample-weighted aggregation 후 row normalize

## 3. 산출물과 측정

`results/fedhdc_ucihar_smoke/`에는 다음을 기록한다.

- `result.json`, `config.resolved.yaml`, `environment.json`
- `bootstrap_metrics.csv`: encoded-cache copy, bundling/normalization,
  model copy/receive, server aggregation, broadcast의 분해 시간
- `round_metrics.csv`: global model hash, pooled global-test accuracy,
  client 진단 통계, round timing
- `client_metrics.csv`, `communication.csv`, `checkpoints/latest.pt`

Bootstrap과 recurring round는 별도로 측정한다. `parallel_estimate`는 client
최대 시간과 server/broadcast 시간을 더한 단일 프로세스 simulation 추정치이며,
실제 network latency가 아니다. data download, extraction, preprocessing 및
nonlinear encoding cache 생성은 bootstrap에 포함하지 않았고, smoke run에서는
이미 준비된 cache를 읽으므로 `data_prepare_ms=0.0`이다.

## 4. Smoke 설정과 결과

실행 명령은 다음과 같다.

```bash
PYTHONPATH=src python3 -m horu_artifact federated --method fedhdc \
  --config configs/fedhdc_ucihar_smoke.yaml --data-root data \
  --output results/fedhdc_ucihar_smoke --device cpu
```

설정은 UCI-HAR subject 1--3, 2 rounds, `D=256`, `η=0.035`, local epoch 1,
batch size 16, seed 0이다. `η`, seed, local epoch 및 split ratio는 T001에서
재사용한 smoke/prototype 설정이며, T002의 논문 재현 설정이라고 주장하지 않는다.

| 항목 | 결과 |
|---|---:|
| 참여 client / test samples | 3 / 297 |
| initial global model hash | `0fec4314…5009def` |
| final global model hash | `782ae482…d1d4596` |
| bootstrap upload / download | 18,432 / 18,432 bytes |
| bootstrap sequential / parallel estimate | 0.825515 / 0.598920 ms |
| round 1 pooled global-test accuracy | 0.92929292 |
| round 2 official pooled global-test accuracy | **0.94612795** |
| round 2 client diagnostic mean / P10 / worst | 0.94460487 / 0.91111112 / 0.91111112 |
| total sequential / parallel estimate | 19.618426 / 7.347425 ms |

Payload는 bootstrap 및 각 recurring round에서 동일하게
`3 × 6 × 256 × 4 = 18,432` bytes이다. 이는 full model의 float32 payload만
계산한 값이다.

## 5. 재현성과 테스트

동일 config를 별도 output directory에서 재실행했을 때 final global hash가
동일했다. `--resume`은 checkpoint의 완료 round를 읽어 bootstrap 또는 완료 round를
반복하지 않았으며 같은 hash를 유지했다.

`PYTHONPATH=src python3 -m pytest -q` 결과는 `18 passed in 4.67s`였다. 테스트는
다음을 포함한다.

- local class bundling 및 weighted global aggregation 수식
- dot prediction, unit update, stale batch, incomplete batch 및 changed-row normalization
- bootstrap payload와 nonnegative timing fields
- round-1 global clone 시작, aliasing 방지, deterministic rerun 및 resume
- global model을 모든 test sample에 적용하는 pooled accuracy 정의
- 3-client/2-round offline CPU CLI integration

## 6. 참고 기준과 차이

문헌 기준은 Ergun, Chandrasekaran, Rosing, *Federated Hyperdimensional
Computing* (arXiv:2312.15966, 2023)이다. 구현은 local class bundling과 server
aggregation 원칙을 따르되, bootstrap timing, dot similarity, unit update target,
batch semantics, train-sample weights 및 pooled evaluation은 TASK T002의
USER_SPECIFIED artifact 규칙이다. 상세 차이는
[`fedhdc_reference.md`](baselines/fedhdc_reference.md)에 기록했다.

## 7. 상태와 미확인 범위

T002의 기능 완료 조건인 bootstrap 분리, timing/payload 기록, batch-16 semantics,
aggregation 검증, deterministic/resume, CPU smoke 및 전체 pytest 통과는 충족했다.

아직 실행하거나 주장하지 않은 항목은 다음과 같다.

- 30 client, `D=2000`, 5-round acceptance-scale config 실행
- multi-seed 평가와 paper-level accuracy reproduction
- 실제 multi-process/network latency 및 network transfer measurement
- FedHDC 논문과의 완전한 동등성 또는 baseline 비교

따라서 현재 산출물의 상태는 **SMOKE_TEST_ONLY**다.
