#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/artifact/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/artifact/results}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/smoke_ucihar_cpu}"

mkdir -p "${DATA_ROOT}" "${RESULTS_ROOT}"

cd "${ROOT}"
horu-artifact prepare-data ucihar --data-root "${DATA_ROOT}"
horu-artifact smoke \
  --config configs/smoke_ucihar_cpu.yaml \
  --data-root "${DATA_ROOT}" \
  --output "${OUTPUT_DIR}" \
  --device cpu \
  --overwrite
