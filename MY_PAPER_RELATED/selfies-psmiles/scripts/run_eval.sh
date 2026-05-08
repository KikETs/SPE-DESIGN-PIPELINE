#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/pi1m_endpoint_baseline.yaml}"
CHECKPOINT_PATH="${2:-}"
MODE="${3:-auto}"  # auto|constrained|unconstrained

if [[ -z "$CHECKPOINT_PATH" ]]; then
  CHECKPOINT_PATH="$(python - <<'PY' "$CONFIG_PATH"
import sys
from pathlib import Path
import yaml
cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text())
root = Path(cfg.get('experiment', {}).get('output_root', 'outputs/experiments'))
name = str(cfg.get('experiment', {}).get('name', ''))
if not root.exists():
    raise SystemExit('')
candidates = sorted([p for p in root.glob(f"{name}_*") if (p / 'checkpoints' / 'best.pt').exists()])
print(candidates[-1] / 'checkpoints' / 'best.pt' if candidates else '')
PY
)"
fi

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "No checkpoint found. Run training first." >&2
  exit 1
fi

ARGS=(
  --config "$CONFIG_PATH"
  --checkpoint "$CHECKPOINT_PATH"
)

if [[ "$MODE" == "constrained" ]]; then
  ARGS+=(--constrained-decoding)
elif [[ "$MODE" == "unconstrained" ]]; then
  ARGS+=(--unconstrained-decoding)
fi

(
  cd "$ROOT_DIR"
  python -m train.eval_endpoint_model "${ARGS[@]}"
)
