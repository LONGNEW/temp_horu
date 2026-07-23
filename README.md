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

## Docker

Build the reviewer image:

```bash
docker build -t horu-ae:1.0 .
```

Run the CPU smoke test:

```bash
docker run --rm \
  -v "$PWD/data:/artifact/data" \
  -v "$PWD/results:/artifact/results" \
  horu-ae:1.0 \
  bash scripts/run_smoke.sh
```

The image also provides:

```bash
bash scripts/run_tables.sh
bash scripts/run_benchmark.sh
bash scripts/verify_reference.sh
```

## 1. Reproduce Tables I, II, and III

Run the table wrapper:

```bash
python3 artifact/scripts/run_table_reproduction.py \
  --data-root data \
  --output-dir results/table_reproduction \
  --warmup 5 \
  --repeats 30 \
  --threads 1
```

The command writes `table1.csv`, `table2.csv`, `table3.csv`,
`raw_timings.csv`, `environment.json`, and `result.json`.

## 2. Run the six-dataset accuracy comparison

Run the preparation-and-execution wrapper. Use fresh, non-existing output
directories.

```bash
python3 artifact/scripts/prepare_and_run_cuda_reconstruction_suite.py \
  --uci-har-source-root /data/horu/uci_har \
  --uci-har-archive /data/downloads/UCI_HAR_Dataset.zip \
  --isolet-raw-source-root /data/horu/isolet \
  --isolet-download-dir /data/downloads/isolet \
  --femnist-source-root /data/horu/femnist \
  --wisdm-source-root /data/horu/wisdm \
  --wisdm-outer-archive /data/downloads/WISDM.zip \
  --synthetic-source-root /data/horu/synthetic \
  --ninapro-db1-source-root /data/horu/ninapro \
  --ninapro-download-dir /data/downloads/ninapro \
  --output-dir results_reconstruction/cuda_suite
```

This wrapper acquires the public inputs, builds the pinned caches, runs the
shared protocol, and writes the validation report.

To run only a subset, repeat `--dataset`. Example:

```bash
python3 artifact/scripts/prepare_and_run_cuda_reconstruction_suite.py \
  --dataset wisdm \
  --wisdm-source-root /data/horu/wisdm \
  --wisdm-outer-archive /data/downloads/WISDM.zip \
  --output-dir results_reconstruction/cuda_suite_wisdm
```

## 3. Verify the six-dataset reference result

```bash
python3 artifact/scripts/verify_reconstruction_suite.py \
  --manifest artifact/manifests/reconstruction_cuda_suite_v1.json \
  --suite-output reference_results/cuda_suite
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
