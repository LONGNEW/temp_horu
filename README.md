# HoRU Reproduction Artifact

This repository contains exactly two reproducibility surfaces:

1. the controlled-system reproduction of paper Tables I, II, and III; and
2. the verified six-dataset HD benchmark for HoRU, HyperFeel, and FedHDC.

The two surfaces intentionally share a repository but retain separate metric
contracts. Generated datasets and experiment outputs are not committed.

## Badge Targets

This artifact is prepared for the following ACM artifact badges:

- `Artifacts Available`
- `Artifacts Evaluated – Functional`
- `Results Validated – Reproduced`

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

If a published container image is later attached to the submission release,
reviewers can pull it instead of rebuilding locally:

```bash
docker pull ghcr.io/longnew/horu-artifact:1.0
```

This image is aligned with the CUDA runtime used for the benchmark path:
`pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`.

Run the controlled-systems table reproduction in the container:

```bash
docker run --rm \
  -v "$PWD/data:/artifact/data" \
  -v "$PWD/results:/artifact/results" \
  horu-ae:1.0 \
  bash scripts/run_tables.sh
```

The image also provides:

```bash
bash scripts/run_tables.sh
bash scripts/run_benchmark.sh
bash scripts/run_benchmark_sequential.sh
bash scripts/verify_reference.sh
```

## 1. Reproduce Tables I, II, and III

Run the controlled-systems wrapper:

```bash
python3 artifact/scripts/run_tables123.py \
  --data-root data \
  --output-dir results/table_reproduction \
  --warmup 5 \
  --repeats 30 \
  --threads 1
```

This command prepares the controlled-systems fixture, reproduces Tables I/II/III,
and prints the final table contents to the terminal when it finishes.

Outputs are written under `results/table_reproduction/`:

- `table1.csv`
- `table2.csv`
- `table3.csv`
- `raw_timings.csv`
- `environment.json`
- `result.json`

The Docker wrapper uses the same entrypoint:

```bash
bash scripts/run_tables.sh
```

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

To execute the same benchmark one dataset at a time, use:

```bash
python3 artifact/scripts/prepare_and_run_cuda_reconstruction_sequential.py \
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
  --output-dir results_reconstruction/cuda_suite_sequential
```

The sequential wrapper writes one subdirectory per dataset and a top-level
`sequential_summary.json`.

Inside Docker, the sequential benchmark should be launched on a GPU host:

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/artifact/data" \
  -v "$PWD/results:/artifact/results" \
  horu-ae:1.0 \
  bash scripts/run_benchmark_sequential.sh
```

The full six-dataset wrapper uses the same container image:

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/artifact/data" \
  -v "$PWD/results:/artifact/results" \
  horu-ae:1.0 \
  bash scripts/run_benchmark.sh
```

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
bash scripts/verify_reference.sh
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
