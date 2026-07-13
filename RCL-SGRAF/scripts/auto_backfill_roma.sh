#!/usr/bin/env bash
# Dispatch RoMa jobs only onto GPUs that have no tracked running job or CUDA work.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${RCL_ROOT}/../../.." && pwd)"
ESA_ROOT="${PROJECT_ROOT}/Comparison/ESA"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/comparison_matrix_logs}"
STATE_DIR="${STATE_DIR:-${LOG_DIR}/rcl_sgraf_backfill_state}"
LOG_SUFFIX="${LOG_SUFFIX:-}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-20}"
GPU_IDS="${GPU_IDS:-1,2,3,4,5,6}"
PYTHON_BIN="${PYTHON_BIN:-/home/ktj/miniconda3/envs/crossmodal/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/RoMa/data}"
VOCAB_PATH="${VOCAB_PATH:-${PROJECT_ROOT}/RoMa/vocab}"
BERT_PATH="${BERT_PATH:-/mnt/newdisk/ktj/pretrained/bert-base-uncased}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/roma}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
RCL_LEARNING_RATE="${RCL_LEARNING_RATE:-0.0005}"
RCL_BERT_LEARNING_RATE="${RCL_BERT_LEARNING_RATE:-3e-5}"
ESA_LEARNING_RATE="${ESA_LEARNING_RATE:-0.0003}"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"
SCHEDULER_LOG="${LOG_DIR}/rcl_sgraf_backfill_scheduler${LOG_SUFFIX}.log"
IFS=, read -r -a GPU_LIST <<<"${GPU_IDS}"

log() {
    printf '%(%F %T %z)T %s\n' -1 "$*" >> "${SCHEDULER_LOG}"
}

job_exit_code() {
    local file="${STATE_DIR}/$1.exit"
    [[ -s "${file}" ]] || return 1
    head -n 1 "${file}"
}

gpu_is_idle() {
    local gpu="$1" uuid processes metrics memory utilization
    uuid="$(nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
    processes="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "${processes}" ]] && rg -Fq "${uuid}" <<<"${processes}"; then
        return 1
    fi
    metrics="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
    IFS=, read -r memory utilization <<<"${metrics}"
    memory="${memory//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    [[ "${memory}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || return 1
    (( memory <= 256 && utilization <= 10 ))
}

start_job() {
    local method="$1" dataset="$2" text_encoder="$3" gpu="$4"
    local method_root session_prefix log_prefix
    case "${method}" in
        rcl-sgraf) method_root="${RCL_ROOT}"; session_prefix="rcl-sgraf"; log_prefix="rcl_sgraf" ;;
        esa) method_root="${ESA_ROOT}"; session_prefix="esa"; log_prefix="esa" ;;
        *) log "unknown queued method: ${method}"; return 1 ;;
    esac
    local job="${method}_${dataset}_${text_encoder}"
    local session="${session_prefix}-${dataset}-${text_encoder}${LOG_SUFFIX}"
    local log_file="${LOG_DIR}/${log_prefix}_${dataset}_${text_encoder}${LOG_SUFFIX}.log"
    local running_file="${STATE_DIR}/${job}.running"
    local exit_file="${STATE_DIR}/${job}.exit"
    local command method_environment

    printf 'gpu=%s\nsession=%s\nlog=%s\nstarted_at=%(%F %T %z)T\n' "${gpu}" "${session}" "${log_file}" -1 > "${running_file}"
    if [[ "${method}" == rcl-sgraf ]]; then
        method_environment="LEARNING_RATE='${RCL_LEARNING_RATE}' BERT_LEARNING_RATE='${RCL_BERT_LEARNING_RATE}'"
    else
        method_environment="LEARNING_RATE='${ESA_LEARNING_RATE}'"
    fi
    command="cd '${method_root}'; DATA_ROOT='${DATA_ROOT}' VOCAB_PATH='${VOCAB_PATH}' BERT_PATH='${BERT_PATH}' PYTHON_BIN='${PYTHON_BIN}' WORKERS=4 GPU_ID='${gpu}' OUTPUT_ROOT='${OUTPUT_ROOT}' NUM_EPOCHS='${NUM_EPOCHS}' ${method_environment} DATASETS='${dataset}' TEXT_ENCODERS='${text_encoder}' bash scripts/run_roma_matrix.sh > '${log_file}' 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > '${exit_file}'; exit \"\$rc\""
    if ! tmux new-session -d -s "${session}" "${command}"; then
        rm -f "${running_file}"
        return 1
    fi
    log "started ${method} ${dataset}/${text_encoder} on GPU ${gpu} (session ${session})"
}

if [[ -n "${JOB_SPECS:-}" ]]; then
    IFS=';' read -r -a JOBS <<<"${JOB_SPECS}"
elif [[ "${RECOVERY_ONLY:-0}" == "1" ]]; then
    declare -a JOBS=(
        "rcl-sgraf scanrefer bert"
        "rcl-sgraf nr3d bigru"
        "esa scenedepict bigru"
        "esa scenedepict bert"
        "esa scanrefer bigru"
        "esa scanrefer bert"
        "esa nr3d bigru"
        "esa nr3d bert"
        "esa 3dllm bigru"
        "esa 3dllm bert"
    )
else
    declare -a JOBS=(
        "rcl-sgraf 3dllm bigru"
        "rcl-sgraf 3dllm bert"
        "esa scenedepict bigru"
        "esa scenedepict bert"
        "esa scanrefer bigru"
        "esa scanrefer bert"
        "esa nr3d bigru"
        "esa nr3d bert"
        "esa 3dllm bigru"
        "esa 3dllm bert"
    )
fi

log "scheduler online; allowed GPUs: ${GPU_IDS}; interval: ${INTERVAL_SECONDS}s"
while :; do
    pending=0
    running=0
    claimed_gpus=()

    for job_spec in "${JOBS[@]}"; do
        read -r method dataset text_encoder <<<"${job_spec}"
        job="${method}_${dataset}_${text_encoder}"
        if job_exit_code "${job}" >/dev/null; then
            continue
        fi
        running_file="${STATE_DIR}/${job}.running"
        if [[ -s "${running_file}" ]]; then
            gpu="$(sed -n 's/^gpu=//p' "${running_file}" | head -n 1)"
            [[ -n "${gpu}" ]] && claimed_gpus+=("${gpu}")
            running=1
            continue
        fi
        pending=1
    done

    for job_spec in "${JOBS[@]}"; do
        read -r method dataset text_encoder <<<"${job_spec}"
        job="${method}_${dataset}_${text_encoder}"
        job_exit_code "${job}" >/dev/null && continue
        [[ -s "${STATE_DIR}/${job}.running" ]] && continue

        for gpu in "${GPU_LIST[@]}"; do
            [[ " ${claimed_gpus[*]} " == *" ${gpu} "* ]] && continue
            if gpu_is_idle "${gpu}"; then
                start_job "${method}" "${dataset}" "${text_encoder}" "${gpu}"
                claimed_gpus+=("${gpu}")
                running=1
                break
            fi
        done
    done

    (( pending == 0 && running == 0 )) && break
    sleep "${INTERVAL_SECONDS}"
done

failed=0
for job_spec in "${JOBS[@]}"; do
    read -r method dataset text_encoder <<<"${job_spec}"
    job="${method}_${dataset}_${text_encoder}"
    rc="$(job_exit_code "${job}" || printf 'missing')"
    if [[ "${rc}" != "0" ]]; then
        log "failed ${method} ${dataset}/${text_encoder}: exit=${rc}"
        failed=1
    fi
done
(( failed == 0 )) && log "all queued jobs completed successfully"
exit "${failed}"
