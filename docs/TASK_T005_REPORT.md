# TASK T005 구현 및 검증 보고서

## 목적과 범위

T005는 T004 bootstrap checkpoint로부터 HoRU의 recurring federated round를
구현하고 검증한다. 각 client는 cached coefficient에서 prediction 및 local
push--pull update를 수행하고, server에는 shared coefficient `C_i`, `G_i`만
전송한다. Server aggregate는 client의 local class error ratio로만 흡수한다.

본 문서의 수치와 timing은 **SMOKE_TEST_ONLY**다. 논문 정확도 재현, FedHDC나
HyperFeel과의 비교, multi-seed 결과, 실제 네트워크 latency 또는 energy 측정은
수행하지 않았으며, 아래 accuracy를 그러한 결과로 해석해서는 안 된다.

## 구현 대응

논문 Eq. (17)--(32)와의 상세 대응은
[HoRU recurring-round reference](methods/horu_round_reference.md)에 정리했다.

| 기능 | T005 구현 |
|---|---|
| prediction | cached `z_c,z_g,z_p`만 사용하여 coefficient space에서 수행 |
| local update | deterministic epoch shuffle; loader chunk 16 내부에서도 sample-wise current-model prediction |
| error statistic | final local state로 train set 전체를 재예측하여 `w_i,k / max(t_i,k,1)` 계산 |
| upload / aggregation | `C_i,G_i`만 class sample count로 row-wise weighted average |
| absorption | `eta_global * e_i,k * (U_bar-U_i)`를 `C_i,G_i`에만 적용 |
| local persistence | `Δ_i,P_i,B_p,i`, cache, error statistic은 local-only이며 absorption 전후 hash 검사 |

### Coefficient score

시스템 비교의 hot path는 `q=[z_c,z_g,z_p]`와
`u=[C+Δ,G,P]` 사이의 direct dot product만 사용한다. Full-dimensional
prototype reconstruction, cosine normalization, induced Gram metric은
similarity timing 및 prediction에서 제외한다.

Local coefficient update는 식 (14)–(17)의 additive push–pull만 수행하며,
batch 이후 row normalization은 적용하지 않는다.

## Smoke 설정과 provenance

```bash
PYTHONPATH=src python3 -m horu_artifact federated --method horu \
  --config configs/horu_ucihar_round_smoke.yaml --data-root data \
  --bootstrap-checkpoint results/horu_ucihar_bootstrap_smoke/checkpoints/bootstrap.pt \
  --output results/horu_ucihar_round_smoke --device cpu
```

| 항목 | 값 | Provenance |
|---|---:|---|
| dataset / clients | UCI-HAR / subjects 1, 2, 3 | USER_SPECIFIED_SMOKE |
| HD dimension / ranks | `D=256`, `r_c=3`, `r_g=2`, `r_p=4` | USER_SPECIFIED_SMOKE |
| rounds / participation | 2 / full | USER_SPECIFIED_SMOKE |
| local epochs / loader chunk | 1 / 16 | USER_SPECIFIED_SMOKE |
| `eta_shared`, `eta_personal`, `eta_global` | 0.035 / 0.035 / 0.035 | USER_SPECIFIED_SMOKE |
| split ratio / seed / device | 0.3 / 0 / CPU | REPO_EXISTING / USER_SPECIFIED_SMOKE |

Bootstrap checkpoint SHA-256은
`0fdbc1e19d54d2aeaa79949c09bb02d36994dcff73d5221bba46519da9eae64c`이고,
projection SHA-256은
`d5e11b2a9bb6bca5746068175277c35d89e382518b7600549b55ab92e8d5f087`이다.
Checkpoint의 config, projection, split, basis 및 consensus hash가 현재 실행과
불일치하면 runner는 실패한다.

## 결과

| Metric | Round 1 | Round 2 |
|---|---:|---:|
| personalized pooled accuracy (297 samples) | 0.77441078 | 0.70033669 |
| client mean / P10 / worst accuracy | 0.77005090 / 0.67777777 / 0.67777777 | 0.70748486 / 0.60194176 / 0.60194176 |
| client round sum / max | 92.394698 / 33.390162 ms | 97.666519 / 34.374025 ms |
| server aggregation total | 0.166708 ms | 0.103346 ms |
| client absorption sum | 0.071798 ms | 0.062669 ms |
| synchronization total | 0.238506 ms | 0.166015 ms |

Round-2 aggregate hash는 다음과 같다.

```text
C_bar: 70fac0e195004904d9963b30df84b6c06ffb200011a2b63e546cd4c6563588e3
G_bar: fa316554009c663fb8fd085b3932599bb7726b6ecc956d600fd741c0daf73c18
```

Timing은 단일 process의 CPU compute/copy 측정값이다. `parallel estimate`는
network latency가 아니다. Accuracy의 round 간 변화도 이 single smoke의
관찰값일 뿐 scale correction의 일반적인 성능 향상·유지 주장이 아니다.

## 통신, 상태 및 scale diagnostic

Recurring transport는 client별 정확히
`K × (r_c+r_g) × 4 = 6 × (3+2) × 4 = 120` bytes이다. 세 client의 round별
upload/download는 각각 360 / 360 bytes다. Metadata 및 local state는 이
계산에서 제외했다.

`state_manifest.json`은 server state를 `C_bar,G_bar`로만 기록하고, client
local state를 `delta,personal,personal_basis,query_caches,error_statistics`로
기록한다. 각 round/client별 `C/G/Δ/P` hash와 aggregate `C_bar/G_bar` hash도
보존한다. Round-2 coefficient norm mean은 다음과 같다.

| Client | `||z_c||` | `||z_g||` | `||z_p||` | `||[z_c,z_g,z_p]||` | `||C+Δ||` | `||G||` | `||P||` |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.79462 | 0.97462 | 1.01184 | 11.89343 | 1.37636 | 0.53399 | 0.79649 |
| 2 | 11.81877 | 0.96469 | 1.05867 | 11.92587 | 1.11607 | 0.22169 | 0.43105 |
| 3 | 11.85887 | 0.89539 | 0.98262 | 11.94712 | 1.40976 | 0.49246 | 0.59925 |

이 값은 scale을 관찰하기 위한 diagnostic이며, parameter tuning 또는
cross-method comparison의 근거가 아니다.

## 검증

`PYTHONPATH=src python3 -m pytest -q`는 **31 passed**였다. 포함된 검사는 다음을
직접 다룬다.

- direct coefficient dot score의 수작업 계산 일치
- 오분류에서 네 coefficient branch의 true/predicted row만 변경
- 두 번째 sample이 stale batch score가 아닌 current model을 사용함
- final-train class error count 및 ratio의 수작업 일치
- class-wise weighted aggregation, zero denominator, error-ratio absorption
- absorption 전후 `Δ/P/B_p` 불변 및 `ω=0` shared row 불변
- payload 수식, server-local state 분리, checkpoint hash/config 거부
- 3-client 2-round offline CLI, independent deterministic rerun, `--resume`
- T001--T004 regression 포함 전체 CPU suite

독립 재실행 `/tmp/horu_t005_final_repeat`은 위 round-2 `C_bar/G_bar` hash와
일치했고, `--resume`은 완료된 two-round checkpoint를 변경하지 않고 재사용했다.
생성 산출물은 [T005 smoke result](../results/horu_ucihar_round_smoke/result.json),
`round_metrics.csv`, `client_metrics.csv`, `timing_samples.csv`,
`communication.csv`, `state_manifest.json`, `config.resolved.yaml`,
`environment.json`, `checkpoints/latest.pt`다.

## 미확인 사항과 비범위

- 논문 main setting에서의 accuracy 재현 및 multi-seed variance
- six-dataset 결과, rank sweep, ablation, participation/late-client 연구
- FedHDC/HyperFeel과의 controlled comparison
- 실제 distributed network latency, payload metadata overhead, energy
- `r_p=64`의 paper-rank diagnostic에서 full-SVD numerical completion의 영향

따라서 본 문서는 T005 functional smoke와 state/scale semantics의 검증 보고서이며,
paper-matched, final, fair comparison 또는 reproduced-result 보고서가 아니다.
