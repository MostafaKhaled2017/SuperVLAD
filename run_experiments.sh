#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Config (edit here)
# ==============================
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval.py}"
POSTPROC_SCRIPT="${POSTPROC_SCRIPT:-utils/diagnostics/diagnostics.py}"

# Core evaluation args
EVAL_DATASETS_FOLDER="${EVAL_DATASETS_FOLDER:-datasets}"   # Datasets root
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-nordland}"         # Dataset name
RESUME_CKPT="${RESUME_CKPT:-checkpoints/SuperVLAD.pth}"    # Checkpoint path
BACKBONE="${BACKBONE:-dino}"
SUPERVLAD_CLUSTERS="${SUPERVLAD_CLUSTERS:-4}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Optional model switches (leave empty to disable)
CROSSIMAGE_ENCODER="${CROSSIMAGE_ENCODER:---crossimage_encoder}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"                             # TODO: add extra eval.py args if needed

# Diagnostics switches
ENABLE_RETRIEVAL_DIAGNOSTICS_FLAG="--enable_retrieval_diagnostics"
RETURN_DEBUG_METRICS_FLAG="--return_debug_metrics"
MASKING_MODE="${MASKING_MODE:-random}"                               # allowed: none|random
LOW_MASS_THRESHOLD="${LOW_MASS_THRESHOLD:-1e-3}"
RETRIEVAL_DIAGNOSTICS_OUTPUT_DIR="${RETRIEVAL_DIAGNOSTICS_OUTPUT_DIR:-retrieval_diagnostics}"
DIAG_FLAGS="${DIAG_FLAGS:-}"                                         # TODO: replace with any custom diagnostic flags

# Dropout sweep
KEEP_RATIOS=(1.0 0.75 0.50 0.25)
DROP_SEEDS=(0 1 2)

# Postprocessing options
CASE_STUDY_N_EACH="${CASE_STUDY_N_EACH:-12}"

# Output root
RESULTS_ROOT="${RESULTS_ROOT:-results}"
RUN_STAMP="${RUN_STAMP:-$(date +'%Y-%m-%d_%H-%M-%S')}"
RUN_ROOT="${RESULTS_ROOT}/${EVAL_DATASET_NAME}/${RUN_STAMP}"
TMP_SAVE_PREFIX="tmp_eval_runs/${EVAL_DATASET_NAME}/${RUN_STAMP}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"   # 1: skip completed experiment folders, 0: force rerun

mkdir -p "${RUN_ROOT}"


# ==============================
# Helpers
# ==============================
quote_cmd() {
  local out=()
  local arg
  for arg in "$@"; do
    out+=("$(printf '%q' "${arg}")")
  done
  printf '%s' "${out[*]}"
}

make_exp_dir() {
  local exp_name="$1"
  local exp_dir="${RUN_ROOT}/${exp_name}"
  mkdir -p "${exp_dir}"
  printf '%s\n' "${exp_dir}"
}

write_command_file() {
  local exp_dir="$1"
  shift
  quote_cmd "$@" > "${exp_dir}/command.txt"
}

run_and_log() {
  local exp_dir="$1"
  shift
  local log_file="${exp_dir}/run.log"
  write_command_file "${exp_dir}" "$@"
  printf '[%s] %s\n' "$(date +'%F %T')" "$(cat "${exp_dir}/command.txt")" | tee "${log_file}"
  "$@" 2>&1 | tee -a "${log_file}"
}

is_eval_complete() {
  local exp_dir="$1"
  [[ -s "${exp_dir}/metrics.json" ]] \
    && [[ -s "${exp_dir}/per_query.csv" ]] \
    && [[ -s "${exp_dir}/per_bin.csv" ]] \
    && [[ -s "${exp_dir}/cluster_mass_stats.csv" ]]
}

is_step_complete() {
  local expected_file="$1"
  [[ -s "${expected_file}" ]]
}

keep_ratio_to_tag() {
  local ratio="$1"
  "${PYTHON_BIN}" -c "r=float('${ratio}'); print(f'{int(round(r*100)):03d}')"
}

collect_eval_outputs() {
  local exp_dir="$1"
  local save_dir_key="$2"
  local src_root="test/${save_dir_key}"
  local run_dir
  run_dir="$(find "${src_root}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n1)"
  if [[ -z "${run_dir:-}" ]]; then
    echo "ERROR: Could not locate eval output dir under ${src_root}" >&2
    exit 1
  fi

  local retrieval_diagnostics_dir="${run_dir}/${RETRIEVAL_DIAGNOSTICS_OUTPUT_DIR}"
  if [[ ! -d "${retrieval_diagnostics_dir}" ]]; then
    echo "ERROR: Could not locate retrieval diagnostics dir: ${retrieval_diagnostics_dir}" >&2
    exit 1
  fi

  cp -f "${retrieval_diagnostics_dir}/summary.json" "${exp_dir}/summary.json"
  cp -f "${retrieval_diagnostics_dir}/summary.json" "${exp_dir}/metrics.json"
  cp -f "${retrieval_diagnostics_dir}/per_query.csv" "${exp_dir}/per_query.csv"
  cp -f "${retrieval_diagnostics_dir}/per_bin.csv" "${exp_dir}/per_bin.csv"
  cp -f "${retrieval_diagnostics_dir}/cluster_mass_stats.csv" "${exp_dir}/cluster_mass_stats.csv"
  cp -f "${retrieval_diagnostics_dir}/cluster_mass_stats.csv" "${exp_dir}/cluster_stats.csv"
}

run_eval_experiment() {
  local exp_name="$1"
  local token_keep_ratio="$2"
  local step_seed="$3"

  local exp_dir
  exp_dir="$(make_exp_dir "${exp_name}")"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && is_eval_complete "${exp_dir}"; then
    echo "[SKIP] ${exp_name} already completed at ${exp_dir}"
    return 0
  fi

  local save_dir_key="${TMP_SAVE_PREFIX}/${exp_name}"
  local cmd=(
    "${PYTHON_BIN}" "${EVAL_SCRIPT}"
    "--eval_datasets_folder" "${EVAL_DATASETS_FOLDER}"
    "--eval_dataset_name" "${EVAL_DATASET_NAME}"
    "--resume" "${RESUME_CKPT}"
    "--backbone" "${BACKBONE}"
    "--supervlad_clusters" "${SUPERVLAD_CLUSTERS}"
    "--infer_batch_size" "${INFER_BATCH_SIZE}"
    "--device" "${DEVICE}"
    "--num_workers" "${NUM_WORKERS}"
    "--save_dir" "${save_dir_key}"
    "--retrieval_diagnostics_output_dir" "${RETRIEVAL_DIAGNOSTICS_OUTPUT_DIR}"
    "${ENABLE_RETRIEVAL_DIAGNOSTICS_FLAG}"
    "${RETURN_DEBUG_METRICS_FLAG}"
    "--token_keep_ratio" "${token_keep_ratio}"
    "--token_dropout_seed" "${step_seed}"
    "--masking_mode" "${MASKING_MODE}"
    "--low_mass_threshold" "${LOW_MASS_THRESHOLD}"
  )

  if [[ -n "${CROSSIMAGE_ENCODER}" ]]; then
    cmd+=("${CROSSIMAGE_ENCODER}")
  fi

  if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args=( ${EXTRA_MODEL_ARGS} )
    cmd+=("${extra_args[@]}")
  fi

  if [[ -n "${DIAG_FLAGS}" ]]; then
    # TODO: replace with actual custom diagnostic CLI flags if you added any.
    # shellcheck disable=SC2206
    diag_args=( ${DIAG_FLAGS} )
    cmd+=("${diag_args[@]}")
  fi

  run_and_log "${exp_dir}" "${cmd[@]}"
  collect_eval_outputs "${exp_dir}" "${save_dir_key}"
}


# ==============================
# 1) Baseline reproduction
# ==============================
run_eval_experiment "01_baseline" "1.0" "0"

# ==============================
# 2) Success-vs-failure extraction
# ==============================
exp2_dir="$(make_exp_dir "02_success_failure_analysis")"
if [[ "${SKIP_COMPLETED}" == "1" ]] && is_step_complete "${exp2_dir}/summary.json"; then
  echo "[SKIP] 02_success_failure_analysis already completed at ${exp2_dir}"
else
  run_and_log "${exp2_dir}" \
    "${PYTHON_BIN}" "${POSTPROC_SCRIPT}" success_failure \
    --per-query-csv "${RUN_ROOT}/01_baseline/per_query.csv" \
    --summary-json "${exp2_dir}/summary.json"
fi

# ==============================
# 3) Per-bin baseline analysis
# ==============================
exp3_dir="$(make_exp_dir "03_per_bin_baseline")"
if [[ "${SKIP_COMPLETED}" == "1" ]] && is_step_complete "${exp3_dir}/per_bin.csv"; then
  echo "[SKIP] 03_per_bin_baseline already completed at ${exp3_dir}"
else
  run_and_log "${exp3_dir}" \
    cp "${RUN_ROOT}/01_baseline/per_bin.csv" "${exp3_dir}/per_bin.csv"
fi

# ==============================
# 4) Token dropout robustness sweeps
# ==============================
for keep_ratio in "${KEEP_RATIOS[@]}"; do
  keep_tag="$(keep_ratio_to_tag "${keep_ratio}")"
  for seed in "${DROP_SEEDS[@]}"; do
    exp_name="04_dropout_keep_${keep_tag}_seed_${seed}"
    run_eval_experiment "${exp_name}" "${keep_ratio}" "${seed}"
  done
done

# ==============================
# 5) Per-bin analysis under dropout
# ==============================
exp5_dir="$(make_exp_dir "05_per_bin_dropout")"
if [[ "${SKIP_COMPLETED}" == "1" ]] && is_step_complete "${exp5_dir}/per_bin_dropout_summary.csv"; then
  echo "[SKIP] 05_per_bin_dropout already completed at ${exp5_dir}"
else
  run_and_log "${exp5_dir}" \
    "${PYTHON_BIN}" "${POSTPROC_SCRIPT}" aggregate_per_bin_dropout \
    --dropout-root "${RUN_ROOT}" \
    --output-csv "${exp5_dir}/per_bin_dropout_all.csv" \
    --summary-csv "${exp5_dir}/per_bin_dropout_summary.csv"
fi

# ==============================
# 6) Failure transition analysis
# ==============================
exp6_dir="$(make_exp_dir "06_failure_transition")"
if [[ "${SKIP_COMPLETED}" == "1" ]] && is_step_complete "${exp6_dir}/summary.json"; then
  echo "[SKIP] 06_failure_transition already completed at ${exp6_dir}"
else
  run_and_log "${exp6_dir}" \
    "${PYTHON_BIN}" "${POSTPROC_SCRIPT}" failure_transition \
    --dropout-root "${RUN_ROOT}" \
    --detail-csv "${exp6_dir}/failure_transition_matrix.csv" \
    --summary-json "${exp6_dir}/summary.json"
fi

# ==============================
# 7) Correlation (margin vs cluster stats)
# ==============================
exp7_dir="$(make_exp_dir "07_correlation_analysis")"
if [[ "${SKIP_COMPLETED}" == "1" ]] && is_step_complete "${exp7_dir}/summary.json"; then
  echo "[SKIP] 07_correlation_analysis already completed at ${exp7_dir}"
else
  run_and_log "${exp7_dir}" \
    "${PYTHON_BIN}" "${POSTPROC_SCRIPT}" correlation \
    --per-query-csv "${RUN_ROOT}/01_baseline/per_query.csv" \
    --cluster-csv "${RUN_ROOT}/01_baseline/cluster_mass_stats.csv" \
    --output-csv "${exp7_dir}/correlations.csv" \
    --summary-json "${exp7_dir}/summary.json"
fi

# ==============================
# 8) Qualitative case studies
# ==============================
exp8_dir="$(make_exp_dir "08_case_studies")"
if [[ "${SKIP_COMPLETED}" == "1" ]] && is_step_complete "${exp8_dir}/summary.json"; then
  echo "[SKIP] 08_case_studies already completed at ${exp8_dir}"
else
  run_and_log "${exp8_dir}" \
    "${PYTHON_BIN}" "${POSTPROC_SCRIPT}" case_studies \
    --per-query-csv "${RUN_ROOT}/01_baseline/per_query.csv" \
    --output-csv "${exp8_dir}/case_studies.csv" \
    --summary-json "${exp8_dir}/summary.json" \
    --n-each "${CASE_STUDY_N_EACH}"
fi

printf 'Completed. Results saved under: %s\n' "${RUN_ROOT}"
