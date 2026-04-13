#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_MANIFEST="${BASE_MANIFEST:-configs/adversarial_benchmark.json}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-robustness_benchmark.py}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-compare_benchmark_results.py}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
EVAL_DATASETS_FOLDER="${EVAL_DATASETS_FOLDER:-datasets}"
BACKBONE="${BACKBONE:-dino}"
SUPERVLAD_CLUSTERS="${SUPERVLAD_CLUSTERS:-4}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TEST_METHOD="${TEST_METHOD:-hard_resize}"
USE_CROSSIMAGE_ENCODER="${USE_CROSSIMAGE_ENCODER:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

NORDLAND_CHECKPOINT="${NORDLAND_CHECKPOINT:-checkpoints/SuperVLAD.pth}"
SPED_CHECKPOINT="${SPED_CHECKPOINT:-checkpoints/SuperVLAD.pth}"

RUN_STAMP="${RUN_STAMP:-}"
TMP_MANIFEST_DIR="$(mktemp -d /tmp/supervlad_adversarial_manifests.XXXXXX)"
trap 'rm -rf "${TMP_MANIFEST_DIR}"' EXIT

EXP_START_INDEX=21
DEFAULT_DATASETS=("nordland" "sped")

DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

progress_bar() {
  local current="$1"
  local total="$2"
  local width=24
  local filled=0
  local i
  if (( total > 0 )); then
    filled=$(( current * width / total ))
  fi
  local empty=$(( width - filled ))
  printf '['
  for ((i = 0; i < filled; i++)); do
    printf '#'
  done
  for ((i = 0; i < empty; i++)); do
    printf '-'
  done
  printf ']'
}

print_progress() {
  local current="$1"
  local total="$2"
  local label="$3"
  printf '[%s] ' "$(date +'%F %T')"
  progress_bar "${current}" "${total}"
  printf ' %d/%d %s\n' "${current}" "${total}" "${label}"
}

checkpoint_for_dataset() {
  local dataset="$1"
  case "${dataset}" in
    nordland)
      printf '%s\n' "${NORDLAND_CHECKPOINT}"
      ;;
    sped)
      printf '%s\n' "${SPED_CHECKPOINT}"
      ;;
    *)
      printf 'Unsupported dataset: %s\n' "${dataset}" >&2
      printf 'Supported datasets: %s\n' "${DEFAULT_DATASETS[*]}" >&2
      exit 1
      ;;
  esac
}

latest_run_root() {
  local dataset="$1"
  "${PYTHON_BIN}" - "$RESULTS_ROOT" "$dataset" <<'PY'
from pathlib import Path
import sys

results_root = Path(sys.argv[1])
dataset = sys.argv[2]
dataset_root = results_root / dataset
if not dataset_root.exists():
    raise SystemExit(f"Missing dataset results directory: {dataset_root}")
candidates = sorted(path for path in dataset_root.iterdir() if path.is_dir())
if not candidates:
    raise SystemExit(f"No benchmark runs found for dataset: {dataset}")
print(candidates[-1])
PY
}

resolve_run_root() {
  local dataset="$1"
  if [[ -n "${RUN_STAMP}" ]]; then
    local candidate="${RESULTS_ROOT}/${dataset}/${RUN_STAMP}"
    if [[ ! -d "${candidate}" ]]; then
      printf 'Missing run root for %s: %s\n' "${dataset}" "${candidate}" >&2
      exit 1
    fi
    printf '%s\n' "${candidate}"
    return
  fi
  latest_run_root "${dataset}"
}

write_manifest() {
  local dataset="$1"
  local checkpoint="$2"
  local output_path="$3"
  "${PYTHON_BIN}" - "$BASE_MANIFEST" "$dataset" "$checkpoint" "$output_path" "$EVAL_DATASETS_FOLDER" "$BACKBONE" "$SUPERVLAD_CLUSTERS" "$INFER_BATCH_SIZE" "$DEVICE" "$NUM_WORKERS" "$TEST_METHOD" "$USE_CROSSIMAGE_ENCODER" <<'PY'
from pathlib import Path
import json
import sys

base_manifest = Path(sys.argv[1])
dataset = sys.argv[2]
checkpoint = sys.argv[3]
output_path = Path(sys.argv[4])
eval_datasets_folder = sys.argv[5]
backbone = sys.argv[6]
supervlad_clusters = int(sys.argv[7])
infer_batch_size = int(sys.argv[8])
device = sys.argv[9]
num_workers = int(sys.argv[10])
test_method = sys.argv[11]
use_crossimage_encoder = sys.argv[12] == "1"

manifest = json.loads(base_manifest.read_text())
manifest.setdefault("eval", {})
manifest["eval"].update({
    "eval_datasets_folder": eval_datasets_folder,
    "eval_dataset_name": dataset,
    "resume": checkpoint,
    "backbone": backbone,
    "supervlad_clusters": supervlad_clusters,
    "infer_batch_size": infer_batch_size,
    "device": device,
    "num_workers": num_workers,
    "test_method": test_method,
})
if use_crossimage_encoder:
    manifest["eval"]["crossimage_encoder"] = True
else:
    manifest["eval"].pop("crossimage_encoder", None)

output_path.write_text(json.dumps(manifest, indent=2))
PY
}

TOTAL_STEPS=${#DATASETS[@]}
if [[ ${#DATASETS[@]} -ge 2 ]]; then
  TOTAL_STEPS=$(( TOTAL_STEPS + 1 ))
fi
CURRENT_STEP=0

run_dataset_benchmark() {
  local dataset="$1"
  local checkpoint="$2"
  local manifest_path="${TMP_MANIFEST_DIR}/${dataset}_benchmark.json"
  local run_root
  run_root="$(resolve_run_root "${dataset}")"

  write_manifest "$dataset" "$checkpoint" "$manifest_path"

  local cmd=(
    "${PYTHON_BIN}" "${BENCHMARK_SCRIPT}"
    "--manifest" "${manifest_path}"
    "--output-root" "${RESULTS_ROOT}"
    "--run-root" "${run_root}"
    "--exp-start-index" "${EXP_START_INDEX}"
  )
  if [[ "${SKIP_COMPLETED}" == "1" ]]; then
    cmd+=("--skip-completed")
  fi

  CURRENT_STEP=$(( CURRENT_STEP + 1 ))
  print_progress "${CURRENT_STEP}" "${TOTAL_STEPS}" "running ${dataset} -> ${run_root} (experiments ${EXP_START_INDEX}+)"
  "${cmd[@]}"
}

RUN_ROOTS=()
for dataset in "${DATASETS[@]}"; do
  run_dataset_benchmark "${dataset}" "$(checkpoint_for_dataset "${dataset}")"
  RUN_ROOTS+=("$(resolve_run_root "${dataset}")")
done

COMPARISON_STAMP="${RUN_STAMP:-$(basename "${RUN_ROOTS[0]}")}"
COMPARISON_OUTPUT_DIR="${RESULTS_ROOT}/comparisons/${COMPARISON_STAMP}"

if [[ ${#RUN_ROOTS[@]} -ge 2 ]]; then
  CURRENT_STEP=$(( CURRENT_STEP + 1 ))
  print_progress "${CURRENT_STEP}" "${TOTAL_STEPS}" "aggregating comparison for ${DATASETS[*]} -> ${COMPARISON_OUTPUT_DIR}"
  "${PYTHON_BIN}" "${COMPARE_SCRIPT}" \
    --results-root "${RESULTS_ROOT}" \
    --run-roots "${RUN_ROOTS[@]}" \
    --output-dir "${COMPARISON_OUTPUT_DIR}"
fi

for i in "${!DATASETS[@]}"; do
  printf '%s run: %s\n' "${DATASETS[i]}" "${RUN_ROOTS[i]}"
done

if [[ ${#RUN_ROOTS[@]} -ge 2 ]]; then
  printf 'Comparison outputs: %s\n' "${COMPARISON_OUTPUT_DIR}"
fi
