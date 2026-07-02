#!/usr/bin/env bash
set -euo pipefail

cd /root/yaojiaxin/RL/Summary-RL

PYTHON_BIN="/root/.conda/envs/summaryRL/bin/python"
JSON_FILE="${1:-/tmp/repliqa_documents_by_file/repliqa_0/kiqpsbuw.json}"
FIELD="${2:-document_extracted}"

"$PYTHON_BIN" src/summarizer/keyword_reader.py "$JSON_FILE" "$FIELD"
