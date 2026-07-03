#!/usr/bin/env bash
set -euo pipefail

cd /root/yaojiaxin/RL/Summary-RL

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/tmp/repliqa_hf_home}"

PYTHON="${PYTHON:-/root/miniforge3/envs/gen-summary/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/summary_validation_benchmark_91_vllm_r5}"
SUMMARY_MODEL_PATH="${SUMMARY_MODEL_PATH:-/tmp/models/Qwen2.5-3B-Instruct}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/tmp/models/Qwen3-14B}"
GPUS="${GPUS:-1,4}"
MAX_SUMMARY_ROUNDS="${MAX_SUMMARY_ROUNDS:-5}"
SUMMARY_BATCH_SIZE=${SUMMARY_BATCH_SIZE:-3}
JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-6}
LIMIT_ARGS=()

if [[ "${LIMIT:-}" != "" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

if [[ "${START_INDEX:-}" != "" ]]; then
  LIMIT_ARGS+=(--start-index "${START_INDEX}")
fi

"${PYTHON}" src/summarizer/validation_benchmark.py \
  --gpus "${GPUS}" \
  --summary-model-path "${SUMMARY_MODEL_PATH}" \
  --judge-model-path "${JUDGE_MODEL_PATH}" \
  --summary-runner vllm \
  --summary-gpu-memory-utilization "${SUMMARY_GPU_MEMORY_UTILIZATION:-0.80}" \
  --judge-runner "${JUDGE_RUNNER:-vllm}" \
  --judge-gpu-memory-utilization "${JUDGE_GPU_MEMORY_UTILIZATION:-0.70}" \
  --judge-max-model-len "${JUDGE_MAX_MODEL_LEN:-4096}" \
  --judge-max-new-tokens "${JUDGE_MAX_NEW_TOKENS:-1024}" \
  --judge-batch-size "${JUDGE_BATCH_SIZE}" \
  --summary-batch-size "${SUMMARY_BATCH_SIZE}" \
  --max-summary-rounds "${MAX_SUMMARY_ROUNDS}" \
  --output-dir "${OUTPUT_DIR}" \
  "${LIMIT_ARGS[@]}"
