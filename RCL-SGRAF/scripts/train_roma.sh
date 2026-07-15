#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:?usage: train_roma.sh <dataset> <bigru|bert>}"; TEXT_ENCODER="${2:?usage: train_roma.sh <dataset> <bigru|bert>}"
GPU_ID="${GPU_ID:-0}"; PYTHON_BIN="${PYTHON_BIN:-python}"; DATA_ROOT="${DATA_ROOT:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/data}"; VOCAB_PATH="${VOCAB_PATH:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/vocab}"; BERT_PATH="${BERT_PATH:-/home/ktj/Projects/RoMa/pretrained/bert-base-uncased}"; OUTPUT_ROOT="${OUTPUT_ROOT:-runs/roma}"; EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
cd "$(dirname "$0")/.."; EXTRA=(); DEFAULT_EPOCHS=30
if [[ "${TEXT_ENCODER}" == bert ]]; then
  DEFAULT_EPOCHS=15
  EFFECTIVE_LR="${LEARNING_RATE:-${BERT_LEARNING_RATE:-1e-4}}"; EFFECTIVE_LR_UPDATE="${LR_UPDATE_OVERRIDE:-${BERT_LR_UPDATE:-5}}"
  EXTRA+=(--bert_path "${BERT_PATH}" --learning_rate "${EFFECTIVE_LR}" --lr_update "${EFFECTIVE_LR_UPDATE}" --bert_learning_rate "${BERT_ENCODER_LEARNING_RATE:-1e-5}" --bert_warmup_epochs "${BERT_WARMUP_EPOCHS:-2}" --weight_decay "${WEIGHT_DECAY:-1e-4}")
fi
TRAIN_COMMAND=("${PYTHON_BIN}" train.py --data_name "${DATASET}" --data_path "${DATA_ROOT}" --data_root "${DATA_ROOT}" --vocab_path "${VOCAB_PATH}" --text_enc_type "${TEXT_ENCODER}" --img_dim 1024 --num_regions 200 --embed_size 1024 --num_epochs "${NUM_EPOCHS:-${DEFAULT_EPOCHS}}" --batch_size "${BATCH_SIZE:-8}" --workers "${WORKERS:-4}" --model_name "${OUTPUT_ROOT}/${DATASET}/${TEXT_ENCODER}/checkpoint" --logger_name "${OUTPUT_ROOT}/${DATASET}/${TEXT_ENCODER}/log" "${EXTRA[@]}")
EARLY_STOP_LOG="${EARLY_STOP_LOG:-${OUTPUT_ROOT}/${DATASET}/${TEXT_ENCODER}/early_stop_monitor.log}"; EARLY_STOP_STATE="${EARLY_STOP_STATE:-${OUTPUT_ROOT}/${DATASET}/${TEXT_ENCODER}/early_stop.json}"
if [[ "${EARLY_STOP_PATIENCE}" =~ ^[0-9]+$ ]] && [[ "${EARLY_STOP_PATIENCE}" -gt 0 ]]; then
  mkdir -p "$(dirname "${EARLY_STOP_LOG}")" "$(dirname "${EARLY_STOP_STATE}")"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${PWD}/../../../tools/train_with_early_stop.py" --patience "${EARLY_STOP_PATIENCE}" --log "${EARLY_STOP_LOG}" --state-json "${EARLY_STOP_STATE}" -- "${TRAIN_COMMAND[@]}"
else
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${TRAIN_COMMAND[@]}"
fi
