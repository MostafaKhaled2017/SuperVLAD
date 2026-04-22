#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_VENV_PYTHON="${REPO_ROOT}/venv/bin/python"

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

cd "${REPO_ROOT}"

EVAL_DATASETS_FOLDER="${EVAL_DATASETS_FOLDER:-datasets}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-msls}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-32}"
BACKBONE="${BACKBONE:-dino}"
SUPERVLAD_CLUSTERS="${SUPERVLAD_CLUSTERS:-4}"
BATCH_ID="${BATCH_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
EXTRA_ARGS=("$@")

EPSILONS=(0.01 0.1 0.2)
FGSM_LOSSES=(positive_distance training_style)
CHECKPOINT_SPECS=(
    "supervlad:checkpoints/SuperVLAD.pth"
    "adv_last:logs/default/2026-04-18_20-05-19/last_model.pth"
)

find_resume_eval_dir() {
    local save_parent="$1"
    local parent_dir="${REPO_ROOT}/test/${save_parent}"
    local existing_dirs=()

    if [[ ! -d "${parent_dir}" ]]; then
        return 1
    fi

    while IFS= read -r existing_dir; do
        existing_dirs+=("${existing_dir}")
    done < <(find "${parent_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

    if (( ${#existing_dirs[@]} == 0 )); then
        return 1
    fi

    if (( ${#existing_dirs[@]} > 1 )); then
        echo "Found multiple candidate resume directories under ${parent_dir}:" >&2
        printf '  %s\n' "${existing_dirs[@]}" >&2
        echo "Set BATCH_ID to a clean batch directory or remove the extra directories." >&2
        exit 1
    fi

    if [[ ! -f "${existing_dirs[0]}/info.log" ]]; then
        echo "Resume directory exists but has no info.log: ${existing_dirs[0]}" >&2
        echo "Delete that partial directory or choose a new BATCH_ID." >&2
        exit 1
    fi

    printf '%s\n' "${existing_dirs[0]}"
}

run_eval() {
    local checkpoint_label="$1"
    local checkpoint_path="$2"
    local fgsm_loss="$3"
    local save_dir="batch_fgsm/${BATCH_ID}/${fgsm_loss}/${checkpoint_label}"
    local resume_eval_dir=""
    local cmd=(
        "${PYTHON_BIN}" "${REPO_ROOT}/fgsm_eval.py"
        --eval_datasets_folder="${EVAL_DATASETS_FOLDER}"
        --eval_dataset_name="${EVAL_DATASET_NAME}"
        --resume="${checkpoint_path}"
        --backbone="${BACKBONE}"
        --supervlad_clusters="${SUPERVLAD_CLUSTERS}"
        --crossimage_encoder
        --infer_batch_size="${INFER_BATCH_SIZE}"
        --save_dir="${save_dir}"
        --epsilons "${EPSILONS[@]}"
        --fgsm_loss "${fgsm_loss}"
        "${EXTRA_ARGS[@]}"
    )

    if resume_eval_dir="$(find_resume_eval_dir "${save_dir}")"; then
        echo "Resuming ${fgsm_loss} evaluation for ${checkpoint_path} from ${resume_eval_dir}"
        cmd+=(--resume_eval_dir "${resume_eval_dir}")
    else
        echo "Running ${fgsm_loss} evaluation for ${checkpoint_path}"
    fi

    "${cmd[@]}"
}

for checkpoint_spec in "${CHECKPOINT_SPECS[@]}"; do
    checkpoint_label="${checkpoint_spec%%:*}"
    checkpoint_path="${checkpoint_spec#*:}"

    if [[ ! -f "${REPO_ROOT}/${checkpoint_path}" ]]; then
        echo "Checkpoint not found: ${REPO_ROOT}/${checkpoint_path}" >&2
        exit 1
    fi

    for fgsm_loss in "${FGSM_LOSSES[@]}"; do
        run_eval "${checkpoint_label}" "${REPO_ROOT}/${checkpoint_path}" "${fgsm_loss}"
    done
done

echo "Finished FGSM evaluations. Results are under test/batch_fgsm/${BATCH_ID}/"
