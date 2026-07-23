# TASK 001 results

## TASK T006–T007 progress: federated caches and UCI-HAR seed-0 protocol run

Status: **INCOMPLETE — do not use as a six-dataset reproduction result.**

The T006 cache contract (`processed/federated.pt`) now retains ordered client
IDs, train/test tensors, sample IDs, and a split SHA-256.  The following
caches have been built and independently loaded back successfully:

| Dataset | Clients | Features | Classes | Train / test | Split SHA-256 |
|---|---:|---:|---:|---:|---|
| UCI-HAR | 30 | 561 | 6 | 7,207 / 3,092 | `ae07fca824daed40ee5c55faaaf080fc8ec05f10e9e8c5f4a929dce970d10fef` |
| ISOLET | 8 | 617 | 26 | 5,460 / 2,337 | `940d65fdc61dafad0b3221fddb8a8e6464039487da6c7ec0c0ccbe98dac8b228` |
| FEMNIST | 200 | 784 | 62 | 63,900 / 7,203 | `c54eeda0dfdb1642e826a92751589835f69133e6846f7535b3b51c62b8f34f75` |
| Synthetic | 30 | 60 | 10 | 12,601 / 5,399 | `8b80e14655727f89f21c310015e441b936a1182a8478ab75902ddf4105397eb7` |
| NinaPro DB1 | 27 | 640 | 52 | 135,000 / 27,000 | `47a15bf59f4de665e3ad06683d91a8b157eabbd56cae63331b272490b29aaba0` |

WISDM preparation correctly failed before any experiment because the supplied
official transformed phone-accelerometer archive contains 50 rather than the
specified 51 subjects: subject 1614 is absent.  The loader does not silently
drop that client or substitute a raw-signal preprocessing pipeline.

The subsequent USER_SPECIFIED adjustment requests subject 1599 in place of
1614.  `configs/datasets.yaml` and the loader now require the 51-client set
`[1599] + ([1600..1650] - {1614})`; the supplied archive has no transformed or
raw phone accelerometer file for 1599, so preparation now fails explicitly with
`missing requested subjects: [1599]`.  No WISDM cache or WISDM result has been
created.

Update: a fresh official UCI archive download matched the original archive
SHA-256, confirming that 1614's transformed ARFF is an upstream omission, not
a corrupted local copy. The archive does contain 1614 raw phone accelerometer
data. The bundled `arffmagic` source was used to recover only the T006-selected
43 basic features; recovery of supplied client 1600 matched the existing ARFF
in row count/labels and within `4.96e-5` maximum float difference. WISDM is
therefore now prepared as the original 1600–1650, 51-client set with explicit
`USER_SPECIFIED_RAW_RECOVERY` manifest provenance: 16,807 train and 7,085 test
rows; split SHA-256
`76ef9620a4feabbd9d392188d6ec94ef2ac711d74c44098b3e0321d90fd306ea`.

### UCI-HAR, seed 0

These runs use the T006 user-specified 30-client, 25-round, full-participation,
three-local-epoch, batch-32, D=2000 protocol.  They are one dataset and one
seed only, therefore are not aggregate/paper reproduction results.

| Method | Evaluation definition | Final pooled accuracy | Client mean / P10 / worst |
|---|---|---:|---:|
| FedHDC | global model on client tests | 96.5071% | 96.4603% / 91.2088% / 86.5979% |
| HyperFeel | personalized local AM on client tests | 69.6313% | 69.5271% / 62.5000% / 53.4091% |
| HoRU | personalized coefficient state on client tests | 95.5045% | 95.4855% / 92.0455% / 86.9565% |

Result files are under `results/accuracy_full/runs/ucihar/`.  FedHDC and
HyperFeel retain their T002/T003 method-defined metrics; the distinction is
recorded rather than silently treated as a fair common metric.

### UCI-HAR, all required seeds

The three configured seeds completed for the UCI-HAR-only protocol.  The
standard deviations below are sample standard deviations over seeds 0, 1, and
2.  These numbers remain **not comparable as a common-method accuracy table**:
FedHDC evaluates its global model, while HyperFeel and HoRU evaluate their
personalized states.  `validate-results` therefore correctly reports
`incomplete_or_failed` instead of accepting an unfair ordering or aggregate.

| Method | seed 0 | seed 1 | seed 2 | Mean ± std |
|---|---:|---:|---:|---:|
| FedHDC (global pooled) | 96.5071% | 96.2160% | 96.2160% | 96.3131% ± 0.1681 |
| HyperFeel (personalized pooled) | 69.6313% | 74.1915% | 72.1539% | 71.9922% ± 2.2844 |
| HoRU (personalized pooled) | 95.5045% | 93.9845% | 95.3752% | 94.9547% ± 0.8427 |

## Commands executed

```bash
PYTHONPATH=src python3 -m horu_artifact prepare-data ucihar --data-root data
PYTHONPATH=src python3 -m horu_artifact smoke --config configs/smoke_ucihar.yaml --data-root data --output results/smoke_ucihar --device cpu
PYTHONPATH=src python3 -m horu_artifact smoke --config configs/smoke_ucihar.yaml --data-root data --output results/smoke_ucihar_cuda --device cuda
```

The official UCI archive validated as 10,299 samples, 561 features, six classes,
and 30 subjects. Its downloaded archive SHA-256 was
`c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.

## Smoke output

The CPU run passed with a shared projection SHA-256 of
`d5e11b2a9bb6bca5746068175277c35d89e382518b7600549b55ab92e8d5f087`, 94
push-pull updates, initial mean accuracy 0.96676485, and final mean accuracy
0.71332947. These are smoke-test-only measurements, not a reported experiment.

CUDA was available. CUDA used the same projection hash, splits, and 94 updates.
Its final mean accuracy was identical to the CPU value; initial mean accuracy
differed by about `4e-8`, within the specified floating-point tolerance.

## Verification notes

`--device auto` selected CUDA and passed. A second smoke invocation to a
non-empty output directory without `--overwrite` failed as intended. The host
does not provide a `python` executable (only `python3`). Pytest 9.1.1 was
installed in the user Python site, and `PYTHONPATH=src python3 -m pytest -q`
passed: `8 passed in 3.51s`. The CUDA-specific encoder parity test is
conditionally skipped only when CUDA is unavailable; CUDA was available on
this host.

## Batch-size diagnostic (prototype only)

```bash
PYTHONPATH=src python3 -m horu_artifact smoke \
  --config configs/prototype_ucihar_batch32.yaml \
  --data-root data --output results/prototype_ucihar_batch32 --device cpu
```

This USER_SPECIFIED diagnostic changed only `batch_size` from 128 to 32. It
passed, but final mean accuracy fell to 0.49409524 and updates increased to 206
(baseline: 0.71332947 and 94). This is a prototype diagnostic, not a valid
experiment or tuning result.

## Sample-wise update and D=2000 diagnostic (prototype only)

`PrototypeMemory.update` was corrected to predict each sample with the current
prototype memory immediately before deciding whether to update. The regression
test verifies that a second sample does not use a stale prediction calculated
before the first sample's update. The complete test suite then passed:
`9 passed in 3.39s`.

```bash
PYTHONPATH=src python3 -m horu_artifact smoke \
  --config configs/prototype_ucihar_samplewise_d2000.yaml \
  --data-root data --output results/prototype_ucihar_samplewise_d2000 --device cpu
```

This USER_SPECIFIED prototype used `hd_dim: 2000` and `batch_size: 32`. It
passed with projection SHA-256
`c3e84b6a70ce32f92652f4ec2acaabdd0fef7a5b730dc7970a42c572cf940ff6`, initial
mean accuracy 0.97273878, final mean accuracy 0.70402592, and 239 updates.
The remaining accuracy decrease must not be interpreted as a valid experimental
comparison; it requires a separate, provenance-controlled update-stability
investigation.

## Dot-similarity diagnostic (prototype only)

```bash
PYTHONPATH=src python3 -m horu_artifact smoke \
  --config configs/prototype_ucihar_samplewise_d2000_dot.yaml \
  --data-root data --output results/prototype_ucihar_samplewise_d2000_dot --device cpu
```

This USER_SPECIFIED diagnostic changed only `similarity` from `cosine` to `dot`
relative to the sample-wise D=2000 prototype. It passed with the same projection
hash and initial mean accuracy, but final mean accuracy was 0.67720105 and there
were 295 updates (cosine: 0.70402592 and 239 updates). Dot-only substitution is
therefore not an improvement under the current normalized-prototype semantics.

## HDZoo-style batch retraining diagnostic (prototype only)

This diagnostic preserves the USER_SPECIFIED class-mean, row-normalized
initialization. It adopts only the current public HDZoo retraining mechanics:
dot-product predictions fixed for each batch, then class-wise aggregation of
all misclassified push-pull vectors before a single batch update.

```bash
PYTHONPATH=src python3 -m horu_artifact smoke \
  --config configs/prototype_ucihar_hdzoo_batch_d2000.yaml \
  --data-root data --output results/prototype_ucihar_hdzoo_batch_d2000 --device cpu
```

At D=2000, the run passed with the same initial mean accuracy (0.97273878), but
final mean accuracy was 0.31722679 with 433 updates. This is markedly worse
than sample-wise dot retraining (0.67720105, 295 updates), confirming that
HDZoo's batch aggregation cannot be transplanted onto normalized mean
prototypes with the same learning rate as a valid comparison.

## No-row-normalization diagnostic (prototype only)

```bash
PYTHONPATH=src python3 -m horu_artifact smoke \
  --config configs/prototype_ucihar_samplewise_d2000_no_row_norm.yaml \
  --data-root data --output results/prototype_ucihar_samplewise_d2000_no_row_norm --device cpu
```

This USER_SPECIFIED diagnostic changed only initial prototype row normalization
from enabled to disabled, retaining cosine similarity and sample-wise updates.
It passed with the same initial mean accuracy (0.97273878), final mean accuracy
0.98291498, and only 17 updates. The unnormalized class-mean row norms were
about 32.15--33.46, so the mean update magnitude of about 1.18 was only about
3.6% of a row norm, rather than exceeding a norm-1 prototype. This setting
violates the TASK 001 row-normalization requirement and is diagnostic only.

## Unit-norm update-target diagnostic (prototype only)

```bash
PYTHONPATH=src python3 -m horu_artifact smoke \
  --config configs/prototype_ucihar_samplewise_d2000_unit_update.yaml \
  --data-root data --output results/prototype_ucihar_samplewise_d2000_unit_update --device cpu
```

This USER_SPECIFIED diagnostic retains row-normalized prototypes, cosine
prediction, and `η=0.035`. It leaves the encoder output unchanged but applies
`h / ||h||₂` only to each push--pull update vector. The run passed with initial
mean accuracy 0.97273878, final mean accuracy 0.98291498, and 17 updates. It
matches the no-row-normalization diagnostic on this run while preserving the
TASK 001 initialization policy; however, it changes the TASK's raw `ηh` update
equation and remains prototype-only.

## D=256 update-scale diagnostic (debug only)

Using the unchanged default smoke config in memory, the observed mean encoded
norm over the three clients was approximately 12.09. Thus a single update had
mean magnitude `0.035 * ||h|| = 0.423`, relative to the initial prototype row
norm of 1.0. Current sample-wise update counts were 28, 28, and 27 by client.
Some prototype rows were touched (push or pull) up to 24 times in one epoch.
The cosine between an initial and final prototype row ranged from about 0.937
to 0.994. This confirms that the update scale is material and should be treated
as an unresolved stability diagnostic, not a tuning conclusion.

## TASK T002: FedHDC bootstrap and federated runner (smoke test only)

The T002 runner is a deterministic, single-process CPU simulation. It reads the
prepared cache and performs no network access. It is not a reproduction or an
accuracy claim: the T002 acceptance configuration is provided separately and
has not been run.

```bash
PYTHONPATH=src python3 -m horu_artifact federated --method fedhdc \
  --config configs/fedhdc_ucihar_smoke.yaml --data-root data \
  --output results/fedhdc_ucihar_smoke --device cpu
```

The subject 1--3, two-round CPU smoke passed. The official performance metric
is the **global model's pooled accuracy over all 297 participating-client test
samples**; per-client accuracies are diagnostic only. It uses the T002 USER_SPECIFIED
dot prediction, unit update target, stale batch semantics, batch size 16,
changed-row normalization, and client-train-sample-weighted full-model
aggregation. The global projection hash was
`d5e11b2a9bb6bca5746068175277c35d89e382518b7600549b55ab92e8d5f087`.

| Metric | Value |
|---|---:|
| bootstrap upload / download | 18,432 / 18,432 bytes |
| bootstrap sequential / parallel estimate | 0.825515 / 0.598920 ms |
| round 1 global pooled test accuracy | 0.92929292 (297 samples) |
| round 2 official global pooled test accuracy | 0.94612795 (297 samples) |
| round 2 client diagnostic mean / P10 / worst | 0.94460487 / 0.91111112 / 0.91111112 |
| deterministic final global hash | `782ae482e8031e1afbe97f56166f7838e704a9f65e21766f1c1111c65d1d4596` |

The bootstrap breakdown and recurring payloads are in
`results/fedhdc_ucihar_smoke/{bootstrap_metrics,round_metrics,communication}.csv`.
The payload is `3 × 6 × 256 × 4 = 18,432` bytes in each direction for bootstrap
and each recurring round. Timings exclude data download, archive extraction,
preprocessing, and nonlinear-cache creation; the parallel figure is not network
latency.

The same config was executed again in
`results/fedhdc_ucihar_smoke_repeat/`; the final global hash matched. Invoking
the original output with `--resume` did not rerun bootstrap and preserved the
same hash. `PYTHONPATH=src python3 -m pytest -q` passed with `17 passed in
4.67s`. The subsequent evaluation-definition update added one test, yielding
`18 passed in 4.67s`.

No failure occurred in this smoke path. Unconfirmed items are paper-level
accuracy reproduction, the 30-client/D=2000/five-round acceptance-scale run,
multi-seed evaluation, and real network latency; none is claimed here.

## TASK T006/T007: shared-cache quick suite (candidate results)

All six prepared dataset caches use ordered client IDs, fixed train/test sample
IDs, and a split hash. WISDM has 51 clients: the upstream transformed archive
omits subject 1614, which was reconstructed from the included raw phone
accelerometer stream and marked in its manifest. The shared copies used by the
suite live under `/home/longnew/data/projects/horu/datasets`.

The following quick run uses the user-specified quick settings (3 rounds, 3
local epochs, batch 32, D=2000, learning rate 0.035), and is a
**VALID_EXPERIMENT_CANDIDATE** for FedHDC and HyperFeel only:

| Dataset | Method | Seed | pooled client-test accuracy |
|---|---|---:|---:|
| UCI-HAR | FedHDC | 0 | 0.92626131 |
| UCI-HAR | HyperFeel | 0 | 0.36578268 |
| ISOLET | FedHDC | 0 | 0.91014117 |
| ISOLET | HyperFeel | 0 | 0.93752670 |

`results/accuracy_quick/summary/validation_report.json` passed all requested
run-presence and common-metric checks; `PYTHONPATH=src python3 -m pytest -q`
passed **31 tests**.

HoRU quick/full results are intentionally not generated yet. The revised task
specifies ranks but does not specify `eta_shared`, `eta_personal`, and
`eta_global`; using a guessed value would violate the experiment provenance
policy. Full 6-dataset × 3-method × 3-seed results likewise remain pending
those three values. No paper-target comparison is claimed because no
dataset/method target values have been verified from a cited source.

## TASK T005: HoRU recurring coefficient-space round (smoke test only)

The T005 CPU smoke completes two full-participation rounds from the T004
bootstrap checkpoint:

```bash
PYTHONPATH=src python3 -m horu_artifact federated --method horu \
  --config configs/horu_ucihar_round_smoke.yaml --data-root data \
  --bootstrap-checkpoint results/horu_ucihar_bootstrap_smoke/checkpoints/bootstrap.pt \
  --output results/horu_ucihar_round_smoke --device cpu
```

It uses the task-specified `eta_shared=eta_personal=eta_global=0.035`, one
local epoch, loader chunk size 16, and has status **SMOKE_TEST_ONLY**. This is
not a paper-matched accuracy result or a comparison result.

Prediction and updates stay in coefficient space. To preserve the requested
reconstructed-cosine semantics when finite-precision personal/shared bases are
not exactly mutually orthogonal, score calculation uses the induced Gram metric
`B^T B`; it does not reconstruct full prototype rows or add normalization,
clipping, or a scale-changing update. Unit coverage directly checks equality to
the reconstructed reference.

| Metric | Result |
|---|---:|
| round-2 personalized pooled accuracy (297 samples) | 0.70033669 |
| mean / P10 / worst client accuracy | 0.70748486 / 0.60194176 / 0.60194176 |
| round upload/download per client | 120 / 120 bytes |
| round upload/download all clients | 360 / 360 bytes |
| deterministic final `C_bar` hash | `70fac0e195004904d9963b30df84b6c06ffb200011a2b63e546cd4c6563588e3` |
| deterministic final `G_bar` hash | `fa316554009c663fb8fd085b3932599bb7726b6ecc956d600fd741c0daf73c18` |

`--resume` preserves the completed state. An independent repeat in
`/tmp/horu_t005_repeat` produced the same final shared-state hashes. The
server state manifest contains only `C_bar,G_bar`; `delta`, `P`, personal
basis, caches, and error statistics are client-local. `PYTHONPATH=src python3
-m pytest -q` passed: **31 passed**.

Unverified: paper-rank/main-experiment settings, multi-seed behavior, six
dataset accuracy, and physical network latency/energy. None is claimed by this
smoke output.

## TASK T003: HyperFeel horizontal personalized retraining (smoke test only)

```bash
PYTHONPATH=src python3 -m horu_artifact federated --method hyperfeel \
  --config configs/hyperfeel_ucihar_smoke.yaml --data-root data \
  --output results/hyperfeel_ucihar_smoke --device cpu
```

This is a functional CPU-only smoke, not a HyperFeel paper-accuracy
reproduction or a comparison with FedHDC/HoRU. It uses the T003-specified
three UCI-HAR subjects, two rounds, `D=256`, raw encoded sample vectors, dot
prediction, no prototype normalization, and server **sum** of client deltas.
The repository's pre-existing deterministic `test_ratio: 0.3` is recorded in
the resolved config. The reference mapping and scope differences are in
[`docs/baselines/hyperfeel_reference.md`](docs/baselines/hyperfeel_reference.md).

| Metric | Value |
|---|---:|
| central-AM bootstrap upload / download | 18,432 / 18,432 bytes |
| round delta upload / download | 18,432 / 18,432 bytes |
| round-2 personalized pooled accuracy (297 samples) | 0.34343433 |
| round-2 personalized mean / P10 / worst client accuracy | 0.34278276 / 0.28155339 / 0.28155339 |
| central AM hash | `d78679f69c3129aa6582732353af259ed1c7b8ac01ce69da2c1cda29a9348f68` |
| final server-delta hash | `063b34dd46739721bfae5031895af3afcfa7cd964a4f6e08a1cb29491738bedd` |

`bootstrap_metrics.csv`, `round_metrics.csv`, and `client_metrics.csv` retain
the bootstrap/round timing, central/delta/personalized-AM hashes, and
per-class count/error/personalization weights. The same configuration produced
the identical final delta hash in `results/hyperfeel_ucihar_smoke_repeat/`;
`--resume` preserved it without rerunning completed rounds. Full CPU test suite:
`PYTHONPATH=src python3 -m pytest -q` → **21 passed**.

### Raw-Q 및 class-scale diagnostic (DEBUG_ONLY)

동일 smoke split에서 encoded `Q`의 client별 평균 L2 norm은 약 12.1이므로,
`lr=0.035` raw-Q update의 평균 norm은 약 0.424다. Central-AM class count와 row
norm의 상관계수는 0.9841이었고, bootstrap train prediction은 693개 모두 최대
row-norm class로 향했다. 따라서 이 split의 dot-product score에는 강한 class
count/row-norm scale bias가 있다.

동일 조건의 prototype-only counterfactual에서 update만 unit-normalize하면
round-2 personalized pooled accuracy는 0.34343433에서 **0.21212122**로
낮아졌다. 반대로 raw-Q update는 유지하고 bootstrap 및 update 뒤 AM rows만
normalize하면 **0.46801347**로 높아졌다. 이 값들은 논문 충실 HyperFeel 결과나
일반화된 성능 결론이 아니라, 현재 smoke split의 scale 영향 진단이다.
