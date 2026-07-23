# TASK.md

## 작업 식별자

- **Task ID:** T005
- **작업명:** HoRU coefficient-space local training 및 shared-state synchronization
- **기준 문서:** `RESEARCH_SPEC.md`, HoRU revised paper, T004 보고서
- **상태:** Ready

## 1. 현재 수행할 단일 작업

T004 checkpoint에서 HoRU recurring round를 구현한다. Client는 coefficient-space prediction/update 후 shared coefficients만 upload하고, server aggregation 결과를 local error ratio로 반영한다.

```bash
python -m horu_artifact federated   --method horu   --config configs/horu_ucihar_round_smoke.yaml   --data-root data   --bootstrap-checkpoint results/horu_ucihar_bootstrap_smoke/checkpoints/bootstrap.pt   --output results/horu_ucihar_round_smoke   --device cpu
```

Smoke는 UCI-HAR subject 1--3, 2 rounds, full participation이다. Checkpoint가 없거나 hash/config가 다르면 실패한다.

## 2. 상태와 불변 조건

Client `i`는 다음 state와 T004 query cache를 보유한다.

```text
C_i ∈ R^(K×r_c)       shared common coefficients
G_i ∈ R^(K×r_g)       shared global coefficients
Δ_i ∈ R^(K×r_c)       local common correction
P_i ∈ R^(K×r_p)       local personal coefficients
B_p,i                  local-only personal basis
z_c, z_g, z_p          train/test query coefficient cache
n_i,k                  local class sample count
```

Server는 aggregated shared coefficients만 보유한다.

```text
C_bar ∈ R^(K×r_c)
G_bar ∈ R^(K×r_g)
```

`Δ_i`, `P_i`, `B_p,i`, error statistics와 cache는 local-only다. Client state는 round 사이에 유지된다.

## 3. Coefficient-space prediction

Encoded query의 cached coefficient를 연결한다.

```text
q_i(h) = [z_c, z_g, z_p]
u_i,k  = [C_i[k] + Δ_i[k], G_i[k], P_i[k]]
ŷ = argmax_k dot(q_i(h), u_i,k)
```

Hot path에서 full prototype을 재구성하거나 Gram metric을 계산하지 않는다.
시스템 비교에서는 연결된 coefficient 사이의 direct dot product를 사용한다.

## 4. Client local training

Epoch마다 deterministic shuffle을 사용한다. `batch_size=16`은 loader chunk이며 update는 sample-wise current-model 방식이다.

오분류 `(h,y,ŷ)`에 대해 raw projected coefficients를 사용한다.

```text
C_i[y] += η_s z_c       C_i[ŷ] -= η_s z_c
G_i[y] += η_s z_g       G_i[ŷ] -= η_s z_g
Δ_i[y] += η_p z_c       Δ_i[ŷ] -= η_p z_c
P_i[y] += η_p z_p       P_i[ŷ] -= η_p z_p
```

정분류 sample은 변경하지 않는다. Unit normalization, row normalization과 clipping은 추가하지 않는다.

Scale 진단을 위해 round/client별 다음 norm 통계를 기록한다.

```text
||z_c||, ||z_g||, ||z_p||, ||[z_c,z_g,z_p]||
||C+Δ||, ||G||, ||P||
```

## 5. Class error statistics

Local training 후 최종 state로 train set 전체를 재예측한다.

```text
t_i,k = class k train sample count
w_i,k = class k misclassified sample count
e_i,k = w_i,k / max(t_i,k, 1)
```

`t_i,k=0`이면 synchronization weight는 0이다. 통계는 local-only이며 upload하지 않는다.

## 6. Shared-state synchronization

Client upload:

```text
U_i = [C_i, G_i]
```

Server는 class sample count로 row별 aggregation한다.

```text
C_bar[k] = Σ_i n_i,k C_i[k] / Σ_i n_i,k
G_bar[k] = Σ_i n_i,k G_i[k] / Σ_i n_i,k
```

분모가 0이면 row는 0이며 server는 `[C_bar,G_bar]`만 broadcast한다.

Client absorption:

```text
ω_i,k = e_i,k if t_i,k>0 else 0
U_i_new[k] =
 U_i[k] + η_g ω_i,k (U_bar[k] - U_i[k])
```

`C_i`, `G_i`만 갱신하며 `Δ_i`, `P_i`, `B_p,i` hash는 전후 동일해야 한다.

## 7. Learning-rate provenance

Smoke는 성능 재현이 아니므로 세 learning rate를 config와 provenance에 명시한다.

```yaml
eta_shared: 0.035
eta_personal: 0.035
eta_global: 0.035
provenance: USER_SPECIFIED_SMOKE
```

Paper-matched 값은 별도 config로 추가한다.

## 8. Timing과 payload

Table II에 대응해 recurring cost를 분리한다.

Client:

- coefficient similarity
- coefficient update
- final train prediction
- class error statistics
- local round total
- shared-state absorption

Server:

- common aggregation
- global aggregation
- server aggregation total

Synchronization total은 server aggregation과 client absorption이다. Sequential/parallel estimate를 기록하되 network latency로 간주하지 않는다.

Per participating client:

```text
upload_bytes   = K×(r_c+r_g)×4
download_bytes = K×(r_c+r_g)×4
```

Metadata와 local state는 제외한다.

## 9. 변경 범위

```text
src/horu_artifact/methods/horu.py
src/horu_artifact/horu/{training,synchronization,inference}.py
src/horu_artifact/federated/{runner,metrics,interfaces}.py
src/horu_artifact/{cli,config}.py
configs/horu_ucihar_round_{smoke,paper_rank}.yaml
docs/methods/horu_round_reference.md
tests/test_{horu_training,horu_sync,horu_round_cli}.py
```

T001–T004 API/checkpoint 호환성을 유지한다.

## 10. 산출물

```text
results/horu_ucihar_round_smoke/
├── result.json
├── config.resolved.yaml
├── environment.json
├── round_metrics.csv
├── client_metrics.csv
├── timing_samples.csv
├── communication.csv
├── state_manifest.json
└── checkpoints/latest.pt
```

기록:

- bootstrap/config/data/projection/split hash
- round별 `C/G/Δ/P` 및 aggregated shared-state hash
- client별 accuracy, update 수, error ratio
- mean/P10/worst/pooled personalized accuracy
- norm diagnostics
- timing과 payload
- Git commit과 환경

## 11. 테스트

- coefficient prediction과 reconstructed reference 일치
- 오분류 시 네 branch의 true/predicted row만 변경
- current-model sample-wise prediction
- final-train error statistics 수작업 일치
- `C/G`만 upload되고 local state는 server에 없음
- class-wise weighted aggregation과 zero-denominator 처리
- error-ratio absorption 식 수작업 일치
- `ω=0`인 class shared state 무변경
- synchronization 전후 `Δ/P/B_p` 동일
- payload 수식 일치
- deterministic rerun과 checkpoint resume 동일
- 3-client, 2-round offline CPU CLI 성공
- T001–T004 regression test 통과

## 12. 완료 조건

1. 논문 Eq. (17)–(32) 대응 문서화
2. coefficient-space prediction과 local update 구현
3. class error statistics와 adaptive synchronization 구현
4. shared-only communication 및 local-state persistence 검증
5. recurring timing/payload와 norm diagnostic 생성
6. 3-client 2-round smoke 성공
7. deterministic rerun/resume 검증
8. CPU-only `pytest -q` 전체 통과
9. 실제 결과와 미확인 사항을 `RESULTS.md`에 기록

비범위: 6-dataset accuracy, rank sweep, ablation, participation/late-client 연구, controlled latency/energy benchmark와 Docker packaging.
