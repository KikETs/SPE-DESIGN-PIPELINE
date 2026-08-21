from __future__ import annotations
from pathlib import Path
import importlib.util
import os
import sys

_PHASE_DIR = Path(__file__).resolve().parent
if str(_PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE_DIR))
_COMMON_PATH = _PHASE_DIR / 'gromacs_new_phase_common.py'
_COMMON_MODULE_NAME = f"{os.environ.get('GROMACS_MODULE_NAME', 'gromacs_phase')}_common"
_common_spec = importlib.util.spec_from_file_location(_COMMON_MODULE_NAME, _COMMON_PATH)
_common = importlib.util.module_from_spec(_common_spec)
sys.modules[_COMMON_MODULE_NAME] = _common
_common_spec.loader.exec_module(_common)
for _k, _v in _common.__dict__.items():
    if _k in {'__name__', '__file__', '__package__', '__loader__', '__spec__', '__builtins__'}:
        continue
    globals()[_k] = _v
print("__STAGEV3__:charge_sanity:entry", flush=True)


# ---- phase body: charge_sanity ----
_BATCH_ROOT = _PHASE_DIR.parent
if str(_BATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BATCH_ROOT))
from batch_utils.charge_sanity import ensure_interphase_charge_sanity

print("__STAGEV3__:charge_sanity:charge-sanity", flush=True)
report = ensure_interphase_charge_sanity(
    ROOT,
    li_target=float(spec.li_charge_scale),
    anion_target=float(spec.anion_charge_scale if spec.anion_charge_scale is not None else spec.li_charge_scale),
    attempt_fix=True,
)
log(
    f"[charge-sanity] status={report.get('status')} "
    f"reason={report.get('reason')} "
    f"polymer_q_chain={report.get('polymer_q_chain')} "
    f"system_q={report.get('system_q')}"
)


print("__PHASE_DONE__:charge_sanity", flush=True)
