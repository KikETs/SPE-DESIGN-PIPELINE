#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_CSV="${1:-$ROOT_DIR/../../PI1M/PI1M_v2.csv}"
OUTPUT_DIR="${2:-$ROOT_DIR/outputs/pi1m_endpoint_dataset}"

EXTRA_ARGS=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--debug)
fi

(
  cd "$ROOT_DIR"
  python -m data.build_pi1m_dataset \
    --input-csv "$INPUT_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --smiles-col "SMILES" \
    --save-excluded \
    "${EXTRA_ARGS[@]}"
)
