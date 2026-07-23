#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/artifact/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/artifact/results}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/table_reproduction}"
WARMUP="${WARMUP:-5}"
REPEATS="${REPEATS:-30}"
THREADS="${THREADS:-1}"

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}"

cd "${ROOT}"
python3 artifact/scripts/run_table_reproduction.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --warmup "${WARMUP}" \
  --repeats "${REPEATS}" \
  --threads "${THREADS}"
