#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/pi1m_endpoint_baseline.yaml}"

EXTRA_ARGS=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--debug)
fi

(
  cd "$ROOT_DIR"
  python -m train.train_endpoint_model \
    --config "$CONFIG_PATH" \
    "${EXTRA_ARGS[@]}"
)
