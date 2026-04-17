#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results/checkpoint_benchmarks}"
COMPARE_ROOT="${COMPARE_ROOT:-${ROOT_DIR}/results/checkpoint_comparisons}"

REFERENCE_MANIFEST="${ROOT_DIR}/configs/benchmark_original_supervlad.json"
CANDIDATE_MANIFEST="${ROOT_DIR}/configs/benchmark_fgsm_finetuned_supervlad_epoch010.json"

find_matching_run_root() {
  local manifest_path="$1"
  python3 - "${RESULTS_ROOT}" "${manifest_path}" <<'PY'
import json
import sys
from pathlib import Path

results_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
target = json.loads(manifest_path.read_text())
matches = []
if results_root.exists():
    for candidate in results_root.rglob("manifest.json"):
        try:
            candidate_manifest = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
        if candidate_manifest == target:
            matches.append(str(candidate.parent.resolve()))
print(matches[-1] if matches else "")
PY
}

default_run_root() {
  local manifest_path="$1"
  local manifest_name
  manifest_name="$(basename "${manifest_path}" .json)"
  printf "%s/%s\n" "${RESULTS_ROOT}" "${manifest_name}"
}

resolve_run_root() {
  local manifest_path="$1"
  local configured_root="${2:-}"
  if [[ -n "${configured_root}" ]]; then
    printf "%s\n" "${configured_root}"
    return
  fi

  local existing_root
  existing_root="$(find_matching_run_root "${manifest_path}")"
  if [[ -n "${existing_root}" ]]; then
    printf "%s\n" "${existing_root}"
    return
  fi

  default_run_root "${manifest_path}"
}

REFERENCE_RUN_ROOT="$(resolve_run_root "${REFERENCE_MANIFEST}" "${REFERENCE_RUN_ROOT:-}")"
CANDIDATE_RUN_ROOT="$(resolve_run_root "${CANDIDATE_MANIFEST}" "${CANDIDATE_RUN_ROOT:-}")"

echo "Running reference benchmark from ${REFERENCE_MANIFEST}"
echo "Reference run root: ${REFERENCE_RUN_ROOT}"
python3 "${ROOT_DIR}/robustness_benchmark.py" \
  --manifest "${REFERENCE_MANIFEST}" \
  --output-root "${RESULTS_ROOT}" \
  --run-root "${REFERENCE_RUN_ROOT}" \
  --skip-completed

echo "Running candidate benchmark from ${CANDIDATE_MANIFEST}"
echo "Candidate run root: ${CANDIDATE_RUN_ROOT}"
python3 "${ROOT_DIR}/robustness_benchmark.py" \
  --manifest "${CANDIDATE_MANIFEST}" \
  --output-root "${RESULTS_ROOT}" \
  --run-root "${CANDIDATE_RUN_ROOT}" \
  --skip-completed

echo "Comparing benchmark outputs"
COMPARISON_OUTPUT_DIR="$(python3 "${ROOT_DIR}/scripts/compare_checkpoint_benchmarks.py" \
  --reference-run-root "${REFERENCE_RUN_ROOT}" \
  --candidate-run-root "${CANDIDATE_RUN_ROOT}" \
  --output-dir "${COMPARE_ROOT}/supervlad_vs_fgsm_finetuned")"

echo "Comparison written to: ${COMPARISON_OUTPUT_DIR}"
