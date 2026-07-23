# HoRU Reproduction Artifact

This repository contains exactly two reproducibility surfaces:

1. the controlled-system reproduction of paper Tables I, II, and III; and
2. the verified six-dataset HD benchmark for HoRU, HyperFeel, and FedHDC.

The two surfaces intentionally share a repository but retain separate metric
contracts. Generated datasets and experiment outputs are not committed.

## Installation

Python 3.11+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

For CUDA runs, install the appropriate CUDA-enabled PyTorch wheel before the
editable install. `nvidia-ml-py` is optional and is used only for GPU energy
measurement.

## 1. Reproduce Tables I, II, and III

The tracked controlled-system fixture is the input for this experiment.

```bash
horu-artifact reproduce-tables \
  --data-root data \
  --output results/table_reproduction \
  --warmup 5 \
  --repeats 30 \
  --threads 1
```

The command writes `table1.csv`, `table2.csv`, `table3.csv`,
`raw_timings.csv`, `environment.json`, and `result.json`.

This surface uses the coefficient-cache HoRU implementation under
`src/horu_artifact/horu/`. Each client caches projections onto the server
common/global bases and its personal basis, then uses and communicates
coefficients during recurring training.

## 2. Verify the six-dataset reference result

The committed reports are the immutable outputs of the verified seed-42 CUDA
screening run:

```bash
python3 artifact/scripts/verify_reconstruction_suite.py \
  --manifest artifact/manifests/reconstruction_cuda_suite_seed42_v1.json \
  --suite-output reference_results/cuda_suite_seed42
```

Expected round-25 accuracies, in percent:

| Dataset | HoRU | HyperFeel | FedHDC |
|---|---:|---:|---:|
| UCI-HAR | 97.66 | 98.08 | 95.25 |
| ISOLET | 88.14 | 90.59 | 93.85 |
| FEMNIST | 68.54 | 67.78 | 57.26 |
| WISDM | 56.69 | 51.11 | 6.95 |
| Synthetic | 78.00 | 73.19 | 71.09 |
| NinaPro DB1 | 75.68 | 61.48 | 30.86 |
| Unweighted mean | **77.45** | **73.71** | **59.21** |

HoRU and HyperFeel use the unweighted mean of personalized client/subject
accuracies. FedHDC uses the sample-weighted accuracy of one global prototype.
Changing this contract changes the reported comparison.

## Prepare public inputs

The preparation manifests pin the public source revision and preprocessing
choices. Use fresh, non-existing output directories.

```bash
python3 artifact/scripts/acquire_uci_har_prototype.py \
  --source-root /data/horu/uci_har \
  --archive /data/downloads/UCI_HAR_Dataset.zip

python3 artifact/scripts/acquire_isolet_prototype.py \
  --source-root /data/horu/isolet \
  --download-dir /data/downloads/isolet

python3 artifact/scripts/prepare_femnist_reconstruction.py \
  --source-root /data/horu/femnist

python3 artifact/scripts/acquire_wisdm_reconstruction.py \
  --source-root /data/horu/wisdm \
  --outer-archive /data/downloads/WISDM.zip

python3 artifact/scripts/prepare_synthetic_reconstruction.py \
  --source-root /data/horu/synthetic

python3 artifact/scripts/acquire_ninapro_db1_reconstruction.py \
  --source-root /data/horu/ninapro \
  --download-dir /data/downloads/ninapro
```

Some acquisition scripts accept an explicit archive or download directory;
run each command with `--help` before a remote deployment. FEMNIST is large
and its complete preparation needs substantial disk space.

## Run the six-dataset CUDA suite

```bash
python3 artifact/scripts/run_cuda_reconstruction_suite.py \
  --uci-har-source-root /data/horu/uci_har \
  --isolet-raw-source-root /data/horu/isolet \
  --femnist-source-root /data/horu/femnist \
  --wisdm-source-root /data/horu/wisdm \
  --synthetic-source-root /data/horu/synthetic \
  --ninapro-db1-source-root /data/horu/ninapro \
  --output-dir results_reconstruction/cuda_seed42
```

The suite fixes seed 42, 25 rounds, full participation, 3 local epochs,
batch size 32, HD dimension 2000, learning rate 0.035, common rank 24,
global-only rank 8, and personal rank 64.

## Provenance boundary

The benchmark implementation was vendored from
`LONGNEW/-26CASES-HoRU` commit
`c6a65d7e442705e5d7b7d2a33c7a68d129d32864`.

The committed six-dataset reports are labeled
`CUDA_RECONSTRUCTION_SCREENING_ONLY`. They are source-traceable public-input
reconstruction results, not proof that the manuscript's unavailable original
multi-seed result files were reproduced.
