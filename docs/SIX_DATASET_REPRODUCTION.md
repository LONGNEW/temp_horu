# Six-dataset HD reconstruction

## Scope

This surface contains only the canonical HD comparison needed for:

- `uci_har`
- `isolet_raw`
- `femnist`
- `wisdm`
- `synthetic`
- `ninapro_db1`

and the methods:

- `horu_hd`
- `hyperfeel`
- `fedhdc`

The active repository entry points are:

- `src/horu_artifact/experiments/accuracy_suite.py`
- `src/horu_artifact/methods/fedhdc.py`
- `src/horu_artifact/methods/hyperfeel.py`
- `src/horu_artifact/horu/`

## Input forms

- UCI-HAR: official subject federation.
- ISOLET: official files with the repository's eight-client Dirichlet
  `alpha=5` construction.
- FEMNIST: LEAF revision
  `09ec454a5675e32e1f0546b456b77857fdece018`, non-IID writer data, 200
  selected writers.
- WISDM: official phone accelerometer data, 51 native users, raw xyz rows.
- Synthetic: the same pinned LEAF revision, official example generator,
  1,000 tasks, 5 classes, 60 dimensions, and 30 selected users.
- NinaPro DB1: 27 subjects using EMG plus glove features.

Exact preparation controls are in `artifact/manifests/`.

The reviewer-facing wrapper is
`artifact/scripts/prepare_and_run_cuda_reconstruction_suite.py`, which runs
public-input preparation and the benchmark in one command.

## Metric contract

At round 25:

- HoRU: `mean_personalized_accuracy`
- HyperFeel: `mean_personalized_accuracy`
- FedHDC: `global_test_accuracy`

The suite summary is an unweighted mean across the six dataset-level values.
It is not a pooled sample-level mean.

## Reference integrity

`reference_results/cuda_suite/summary.json` contains SHA-256 hashes for
all six reports and for the suite manifest. Run:

```bash
python3 artifact/scripts/verify_reconstruction_suite.py \
  --manifest artifact/manifests/reconstruction_cuda_suite_v1.json \
  --suite-output reference_results/cuda_suite
```

The verifier rejects missing datasets, methods, round-25 metrics, mismatched
metric definitions, changed protocol fields, and modified report files.
