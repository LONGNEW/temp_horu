# TASK T003 구현 및 검증 보고서

## 목적과 범위

TASK T003는 T001 encoder와 T002 runner를 이용해 HyperFeel horizontal
personalized retraining을 독립 method로 구현했다. 본 보고서는 UCI-HAR subject
1--3, 2-round CPU smoke의 기능 검증 기록이다. 이는 논문 정확도 재현,
FedHDC/HoRU 비교, tuning 또는 vertical FL 검증이 아니며 상태는
**SMOKE_TEST_ONLY**다.

기준 문헌은 Li et al., *HyperFeel: An Efficient Federated Learning Framework
Using Hyperdimensional Computing*, ASP-DAC 2024, DOI
`10.1109/ASP-DAC58780.2024.10473907`이다. 제공 PDF SHA-256은
`c298831aec5d1c929e67e6670ebcd5823be125ad6440b805a1cc9a7c2d4aa734`이다.

## 구현

공유 projection `E`와 T001 nonlinear encoder로 `Q = cos(XE)`를 만들고,
dot product `argmax_k C_i[k] · Q`로 예측한다. Bootstrap에서 client는 raw
sample hypervector를 정답 class에 bundle하고, server는 local AM의 element-wise
sum을 central AM으로 만든다. Row normalization 및 sample-count 가중 평균은 없다.

`C_i^local[y] += Q`, `C_central = Σ_i C_i^local`

각 client는 central AM의 독립 clone을 persistent personalized AM으로 보관한다.
Round 시작에는 이전 delta와 이전 pass의 class error ratio를 적용한다.

`C_i[k] += error_i[k] / cnt_i[k] × lr × Δ_prev[k]`

`cnt_i[k]=0`이면 해당 weight는 0이다. Loader batch size 16은 chunk일 뿐이며,
각 sample은 현재 변경된 AM으로 순차 prediction된다. 오분류는 raw-Q Eq. (2)의
two-row update를 AM과 local delta에 똑같이 적용한다.

`C_i[y] += lr × Q; C_i[ŷ] -= lr × Q`

`δ_i[y] += lr × Q; δ_i[ŷ] -= lr × Q`

Server는 full AM 대신 `δ_i`만 받아 평균 없이 `Δ = Σ_i δ_i`를 broadcast한다.
Personalized AM은 덮어쓰지 않는다. Fig. 1, Eq. (1)--(3), Algorithm 1 대응은
[hyperfeel_reference.md](baselines/hyperfeel_reference.md)에 정리했다.

## 산출물과 측정

주요 구현은 `src/horu_artifact/methods/hyperfeel.py`와
`src/horu_artifact/federated/runner.py`에 있다. 결과는
`results/hyperfeel_ucihar_smoke/`에 result/resolved config/environment,
bootstrap/round/client/communication CSV, checkpoint로 저장한다. Source PDF,
dataset/projection/config hash, central/local AM hash, round delta,
personalized AM/upload hash, class count/error/personalization weight를 기록한다.

Timing은 단일 프로세스의 compute/copy 시간이며 `parallel_estimate`는 실제
network latency가 아니다. Data download, preprocessing, encoded-cache 생성은
제외했다.

## Smoke 설정과 결과

```bash
PYTHONPATH=src python3 -m horu_artifact federated --method hyperfeel \
  --config configs/hyperfeel_ucihar_smoke.yaml --data-root data \
  --output results/hyperfeel_ucihar_smoke --device cpu
```

설정: `paper_faithful`, subject 1--3, 2 rounds, `D=256`, `lr=0.035`, local
epoch 1, batch size 16, seed 0, CPU. `test_ratio=0.3`은 기존 artifact 설정이며,
config SHA-256은 `810a0e1b1736e4366e043d39380a5437163344491de43697d445822382b6fe2e`다.

| 항목 | 결과 |
|---|---:|
| 참여 client / pooled test sample | 3 / 297 |
| central AM hash | `d78679f6…29a9348f68` |
| bootstrap upload / download | 18,432 / 18,432 bytes |
| bootstrap sequential / parallel estimate | 0.423830 / 0.334367 ms |
| round 1 / 2 personalized pooled accuracy | 0.26599327 / **0.34343433** |
| round 2 mean / P10 / worst client accuracy | 0.34278276 / 0.28155339 / 0.28155339 |
| final global delta hash | `063b34dd…91738bedd` |
| total sequential / parallel estimate | 54.894952 / 23.837243 ms |

Bootstrap과 round payload는 각 direction별 `3 × 6 × 256 × 4 = 18,432` bytes다.
Local AM, class/error count는 통신하지 않으며 zero delta도 압축하지 않았다.

## Raw-Q scale 및 class-count scale 진단

동일한 subject/split/seed/round에서 bootstrap train partition을 읽어 확인했다.
Encoded `Q`의 client별 평균 L2 norm은 12.0925, 12.1175, 12.1113이었다. 따라서
paper-faithful `lr=0.035` raw-Q update 하나의 평균 norm은 약 `0.424`다.

Central AM의 class count는 `[149, 112, 101, 101, 118, 112]`, row norm은
`[1729.80, 1300.42, 1149.85, 1228.34, 1436.36, 1348.16]`이었다. Count와 row
norm의 Pearson correlation은 **0.9841**이다. Raw central AM을 이용한 bootstrap
train prediction의 class histogram은 `[693, 0, 0, 0, 0, 0]`이었다. 즉 이 split에서
dot product score는 class-count/AM-norm scale의 강한 영향을 받으며, scale bias가
없다고 볼 수 없다.

두 counterfactual은 논문 충실 설정이 아닌 **ASSUMED_FOR_PROTOTYPE / DEBUG_ONLY**
진단이다. 다른 설정은 모두 smoke와 동일하고 각 경우 한 가지 요소만 바꿨다.

| Run | 변경 | round-2 pooled | 평균 | P10 / worst | 기준 대비 pooled |
|---|---|---:|---:|---:|---:|
| paper-faithful | raw Q, AM 비정규화 | 0.34343433 | 0.34278276 | 0.28155339 / 0.28155339 | — |
| unit-update diagnostic | update만 `Q / ||Q||₂` | 0.21212122 | 0.21142644 | 0.16504854 / 0.16504854 | -0.13131311 |
| row-normalized diagnostic | raw Q 유지, bootstrap 및 update 뒤 AM row normalize | 0.46801347 | 0.47374285 | 0.40776700 / 0.40776700 | +0.12457913 |

따라서 이 제한된 smoke에서는 raw-Q update 크기를 unit update로 줄이면 성능이
낮아졌고, AM row normalization으로 class-scale bias를 제거하면 성능이 높아졌다.
이는 원문과의 우열이나 일반화된 tuning 결론이 아니라, 현재 UCI-HAR split에서
두 scale 효과가 결과에 실질적으로 영향을 준다는 진단이다.

## 검증과 한계

`PYTHONPATH=src python3 -m pytest -q`는 **21 passed in 6.77s**였다. Test는
bundling/central sum, clone bootstrap, zero-delta personalization, raw-Q
two-row update, current-model prediction, absent-class weight, server sum,
AM 미전송, persistent state, payload, deterministic rerun/resume, FedHDC
regression, offline CPU CLI를 검증한다.

동일 config를 별도 output에 재실행해 final delta hash 일치를 확인했고,
`--resume`도 완료된 round를 재실행하지 않고 동일 hash를 유지했다.

논문은 `D=1000`, 30 clients인 반면 이 smoke는 `D=256`, 3 clients, 2 rounds다.
split ratio, local epoch, learning rate, seed는 artifact smoke 설정으로
paper-matched 값이 아니다. Vertical FL, FEMNIST/Synthetic, controlled `D=2000`
성능 주장, HoRU 비교, 실제 energy/network benchmark는 범위 밖이다.
