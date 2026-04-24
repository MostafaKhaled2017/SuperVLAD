#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TMP_DIR="${TMPDIR:-${REPO_ROOT}/.tmp/pip}"
CACHE_DIR="${PIP_CACHE_DIR:-${REPO_ROOT}/.cache/pip}"
DEFAULT_VENV_PYTHON="${REPO_ROOT}/venv/bin/python"

mkdir -p "${TMP_DIR}" "${CACHE_DIR}"

export TMPDIR="${TMP_DIR}"
export TMP="${TMP_DIR}"
export TEMP="${TMP_DIR}"
export PIP_CACHE_DIR="${CACHE_DIR}"

if [[ -n "${VENV_PYTHON:-}" ]]; then
    PYTHON_BIN="${VENV_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
else
    PYTHON_BIN="${DEFAULT_VENV_PYTHON}"
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "Expected Python interpreter not found at ${PYTHON_BIN:-<empty>}" >&2
    echo "Activate your virtual environment first, set VENV_PYTHON=/path/to/python, or create ${DEFAULT_VENV_PYTHON}." >&2
    exit 1
fi

exec "${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt" "$@"
