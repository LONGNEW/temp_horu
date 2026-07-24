#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PARENT_ROOT=$(cd "${REPO_ROOT}/.." && pwd)

LOCAL_IMAGE="${1:-horu-ae:1.0}"
REMOTE_IMAGE="${2:-ghcr.io/longnew/horu-artifact:1.0}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found in PATH" >&2
  exit 1
fi

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  else
    echo "docker daemon is not accessible for the current user; rerun with sudo or add the user to the docker group" >&2
    exit 1
  fi
fi

TOKEN="${GITHUB_TOKEN:-}"
if [[ -z "${TOKEN}" && -f "${PARENT_ROOT}/.env" ]]; then
  TOKEN=$(awk -F= '$1=="GIT"{gsub(/"/, "", $2); print $2}' "${PARENT_ROOT}/.env")
fi

if [[ -z "${TOKEN}" ]]; then
  echo "missing GitHub token; set GITHUB_TOKEN or populate ../.env with GIT=..." >&2
  exit 1
fi

if [[ "${REMOTE_IMAGE}" =~ [A-Z] ]]; then
  echo "remote image must be lowercase: ${REMOTE_IMAGE}" >&2
  exit 1
fi

echo "${TOKEN}" | docker login ghcr.io -u longnew --password-stdin
"${DOCKER[@]}" tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"
"${DOCKER[@]}" push "${REMOTE_IMAGE}"
