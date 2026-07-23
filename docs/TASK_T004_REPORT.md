# TASK T004 구현 및 검증 보고서

## 목적과 범위

T004는 HoRU의 federated round **이전** one-time bootstrap만 구현하고 검증한다.
T001의 Gaussian projection 및 `h = cos(xE)` encoder와 준비된 UCI-HAR cache를
사용하여 client class prototype, server shared basis, shared/local coefficient
state, 개인 basis, query coefficient cache를 만든다.

본 결과는 **SMOKE_TEST_ONLY**다. coefficient retraining, shared-state round
aggregation, personalization update, 정확도 비교 및 논문 재현은 수행하지 않았다.
아래 수치는 논문 성능 또는 통신 지연/energy benchmark가 아니다.

## 구현 대응

논문 Eq. (5)--(16)과 T004 구현의 대응은
[HoRU bootstrap reference](methods/horu_bootstrap_reference.md)에 정리했다.

```text
M_i: class-wise normalized train prototype, shape K×D
C_i: C_global clone, shape K×r_c
G_i: zero, shape K×r_g
Δ_i: zero, shape K×r_c
B_p,i: local personal basis, shape D×r_p
P_i: residual coefficient, shape K×r_p
```

Server는 `Σ_i M_iᵀM_i`를 `eigh`하고 eigenvalue 내림차순 상위 vectors를
`[B_c, B_g]`로 선택한다. 각 column은 최대 절댓값 원소가 양수가 되도록 sign
canonicalization한다(동률은 가장 작은 index). `C_global`은 class sample count를
가중치로 하여 `M_i B_c`에서 계산한다. `B_c`, `B_g`, `C_global`은 독립 clone으로
broadcast하며, `B_p,i`, `P_i`, `Δ_i`, query cache는 local-only다.

개인 basis의 smoke policy는 `reduced_svd`이다. Paper-rank diagnostic의
`full_svd`는 `r_p > K`의 zero-singular subspace가 유일하지 않은 문제에 대한
literal numerical completion이며, `USER_SPECIFIED_NUMERICAL_COMPLETION`으로
기록한다. 이를 논문이 지정한 유일한 basis 또는 재현 설정으로 해석하지 않는다.

## Smoke 설정과 산출물

```bash
PYTHONPATH=src python3 -m horu_artifact federated --method horu \
  --config configs/horu_ucihar_bootstrap_smoke.yaml --data-root data \
  --output results/horu_ucihar_bootstrap_smoke --device cpu --bootstrap-only
```

| 항목 | 값 | Provenance |
|---|---:|---|
| dataset / clients | UCI-HAR / subjects 1, 2, 3 | USER_SPECIFIED |
| HD dimension | 256 | USER_SPECIFIED |
| `r_c`, `r_g`, `r_p` | 3, 2, 4 | USER_SPECIFIED |
| personal policy / seed / device | reduced SVD / 0 / CPU | USER_SPECIFIED |
| train/test split ratio | 0.3 | REPO_EXISTING |

생성 산출물은 [smoke result](../results/horu_ucihar_bootstrap_smoke/result.json),
`config.resolved.yaml`, `environment.json`, `bootstrap_metrics.csv`,
`state_manifest.json`, `reconstruction_metrics.csv`, `checkpoints/bootstrap.pt`다.
Manifest는 cache dtype/shape, source split hash, shared/personal basis projector
hash를 연결한다.

## Bootstrap 결과

| 항목 | 결과 |
|---|---:|
| selected eigenvalues | 16.58744049, 1.08063412, 0.09967704, 0.06686905, 0.04316397 |
| explained-energy ratio | 0.99321055 |
| `BᵀB-I` max abs | 7.15e-7 |
| `B_cᵀB_c-I` / `B_gᵀB_g-I` max abs | 4.77e-7 / 7.15e-7 |
| `B_cᵀB_g` max abs | 2.34e-7 |
| residual `R_iB` max abs (clients 1--3) | 5.27e-7 / 5.67e-7 / 5.93e-7 |
| reconstruction cosine mean (clients 1--3) | 0.99514121 / 0.99535292 / 0.99619907 |
| reconstruction error mean (clients 1--3) | 0.09404074 / 0.09291107 / 0.08491767 |
| reconstruction finite / nonempty norm minimum | all true / at least 0.99999988 |

Shared raw basis hash는
`163b0adda77954e84706c80c163a6f378418b37b031e296ef42198fcad112d6e`,
sign-invariant shared projector hash는
`6f7821189b5bad0643c2f374b2f77f8dd4a4257ac955ec220e6e6c605a0a5d67`,
`C_global` hash는
`a7aa57b41203946ab8f9892b24ce5df9e9bdf956630b6fd212025ae72e985e26`이다.

## Cache, timing, payload

Train/test cache는 client 1에서 각각 `243×{3,2,4}` / `104×{3,2,4}`, client
2에서 `212×{3,2,4}` / `90×{3,2,4}`, client 3에서 `238×{3,2,4}` /
`103×{3,2,4}` shape의 `z_c`, `z_g`, `z_p`다. 모든 cache는 `torch.float32`다.

| 항목 | 값 |
|---|---:|
| server bootstrap total | 5.774264 ms |
| client bootstrap sum / max | 1.946878 / 0.787216 ms |
| sequential / parallel estimate | 8.708285 / 6.588417 ms |
| upload per client | `6×256×4 = 6,144` bytes |
| download per client | `[256×(3+2)+6×3]×4 = 5,192` bytes |
| all-client upload / download | 18,432 / 15,576 bytes |

Timing은 단일 프로세스의 compute/copy 값이다. `parallel estimate`는
`server_total + max(client_total) + broadcast_copy`이며 실제 network latency가 아니다.

## 검증과 한계

`PYTHONPATH=src pytest -q` 결과는 **25 passed**였다. Test는 class mean/count,
covariance/eigenbasis, sign canonicalization, residual/personal projector,
state/cache shape, payload, checkpoint load, offline 3-client CLI를 다룬다.
동일 CPU config를 `/tmp/horu_t004_repeat`에 재실행하여 projection, raw basis,
projector, common-consensus hash가 모두 일치함을 확인했다.

이 smoke는 `D=256`, 3 clients, split ratio 0.3이며 paper-rank diagnostic도
실행하지 않았다. 그 어떤 accuracy, round cost, comparison, fair baseline,
paper-matched 또는 final-result 주장도 이 보고서 범위에 포함되지 않는다.
