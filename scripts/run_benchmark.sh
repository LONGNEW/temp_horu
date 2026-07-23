#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/artifact/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/artifact/results}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/cuda_suite}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${DATA_ROOT}/downloads}"

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}" "${DOWNLOAD_ROOT}"

cd "${ROOT}"
python3 artifact/scripts/prepare_and_run_cuda_reconstruction_suite.py \
  --uci-har-source-root "${DATA_ROOT}/uci_har" \
  --uci-har-archive "${DOWNLOAD_ROOT}/UCI_HAR_Dataset.zip" \
  --isolet-raw-source-root "${DATA_ROOT}/isolet" \
  --isolet-download-dir "${DOWNLOAD_ROOT}/isolet" \
  --femnist-source-root "${DATA_ROOT}/femnist" \
  --wisdm-source-root "${DATA_ROOT}/wisdm" \
  --wisdm-outer-archive "${DOWNLOAD_ROOT}/WISDM.zip" \
  --synthetic-source-root "${DATA_ROOT}/synthetic" \
  --ninapro-db1-source-root "${DATA_ROOT}/ninapro" \
  --ninapro-download-dir "${DOWNLOAD_ROOT}/ninapro" \
  --output-dir "${OUTPUT_DIR}"
