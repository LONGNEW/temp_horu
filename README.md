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

## 1. Reproduce Tables I, II, and III

First prepare the tracked controlled-system fixture, then run the table
reproduction command.

```bash
horu-artifact prepare-data controlled-systems \
  --data-root data

horu-artifact reproduce-tables \
  --data-root data \
  --output results/table_reproduction \
  --warmup 5 \
  --repeats 30 \
  --threads 1
```

The command writes `table1.csv`, `table2.csv`, `table3.csv`,
`raw_timings.csv`, `environment.json`, and `result.json`.

## 2. Prepare Public Inputs

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

## 3. Run the six-dataset accuracy comparison

After the public inputs exist, run the complete six-dataset pipeline in one
command. The wrapper builds the pinned caches, runs the shared protocol, and
writes the validation report.

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

## 4. Verify the six-dataset reference result

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
