#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT}"
python3 artifact/scripts/verify_reconstruction_suite.py \
  --manifest artifact/manifests/reconstruction_cuda_suite_v1.json \
  --suite-output reference_results/cuda_suite
