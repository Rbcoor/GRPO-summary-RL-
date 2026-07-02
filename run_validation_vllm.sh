#!/usr/bin/env bash
set -euo pipefail

cd /root/yaojiaxin/RL/Summary-RL

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/tmp/repliqa_hf_home}"

PYTHON="${PYTHON:-/root/.conda/envs/summaryRL/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/summary_validation_benchmark_91_vllm_r5}"
GPUS="${GPUS:-1,4}"
MAX_SUMMARY_ROUNDS="${MAX_SUMMARY_ROUNDS:-5}"
SUMMARY_BATCH_SIZE="${SUMMARY_BATCH_SIZE:-4}"
LIMIT_ARGS=()

if [[ "${LIMIT:-}" != "" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

if [[ "${START_INDEX:-}" != "" ]]; then
  LIMIT_ARGS+=(--start-index "${START_INDEX}")
fi

"${PYTHON}" src/summarizer/validation_benchmark.py \
  --gpus "${GPUS}" \
  --summary-runner vllm \
  --summary-gpu-memory-utilization "${SUMMARY_GPU_MEMORY_UTILIZATION:-0.80}" \
  --summary-batch-size "${SUMMARY_BATCH_SIZE}" \
  --max-summary-rounds "${MAX_SUMMARY_ROUNDS}" \
  --output-dir "${OUTPUT_DIR}" \
  "${LIMIT_ARGS[@]}"
