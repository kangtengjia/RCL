#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:?usage: eval_roma.sh <checkpoint> <dataset> <bigru|bert>}"
DATASET="${2:?usage: eval_roma.sh <checkpoint> <dataset> <bigru|bert>}"
TEXT_ENCODER="${3:?usage: eval_roma.sh <checkpoint> <dataset> <bigru|bert>}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/data}"
VOCAB_PATH="${VOCAB_PATH:-/home/ktj/Projects/Cross-Modality-Learning/RoMa/vocab}"
BERT_PATH="${BERT_PATH:-/home/ktj/Projects/RoMa/pretrained/bert-base-uncased}"

cd "$(dirname "$0")/.."
EXTRA=()
[[ "${TEXT_ENCODER}" == bert ]] && EXTRA+=(--bert_path "${BERT_PATH}")
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" eval_roma.py \
  --checkpoint "${CHECKPOINT}" \
  --data_name "${DATASET}" \
  --data_root "${DATA_ROOT}" \
  --vocab_path "${VOCAB_PATH}" \
  --text_enc_type "${TEXT_ENCODER}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --workers "${WORKERS:-4}" \
  "${EXTRA[@]}"
