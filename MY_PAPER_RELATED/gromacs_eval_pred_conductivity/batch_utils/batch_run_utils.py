from __future__ import annotations
import time
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from .charge_sanity import charge_sanity_report_ok

def _load_batch_tqdm(mode: Any | None = None):
    raw_mode = os.environ.get('GROMACS_BATCH_TQDM_MODE') if mode in ('', None) else mode
    mode_s = str(raw_mode or 'text').strip().lower()
    if mode_s in ('0', 'false', 'no', 'off', 'none', 'disable', 'disabled'):
        return None

    try:
        if mode_s in ('widget', 'widgets', 'notebook', 'ipywidgets'):
            from tqdm.notebook import tqdm as _tqdm
        else:
            # tqdm.auto selects ipywidgets in Jupyter. Those widget models can
            # remain referenced across long reruns, so the batch notebook uses
            # the lightweight text backend by default.
            from tqdm.std import tqdm as _tqdm
        return _tqdm
    except Exception:  # pragma: no cover
        try:
            from tqdm import tqdm as _tqdm
            return _tqdm
        except Exception:
            return None


tqdm = _load_batch_tqdm()


_RUN_INTERRUPTED = False


DEFAULT_CFG = {
    'batch_python': '',
    'top_k': 10,
    'bottom_k': 10,
    'stratified_n': 100,
    'strata_bins': 10,
    'random_seed': 20260214,
    'max_attempts': 3,
    'force_rerun': False,
    'force_rebuild_pipeline': False,
    'max_traj_to_run': None,
    'tqdm_mode': 'text',
    'tqdm_leave': False,
    'physical_cores': None,
    'max_parallel_traj': 1,
    'base_ntomp': None,
    'gromacs_ntomp': None,
    'fast_pysoftk': True,
    'pysoftk_uff_iters': 600,
    'pysoftk_localopt_steps': 150,
    'phase_cpu_workers': 4,
    'pysoftk_phase_workers': 8,
    'packmol_phase_workers': None,
    'atomtyping_phase_workers': None,
    'charge_sanity_phase_workers': None,
    'analysis_phase_workers': None,
    'pysoftk_internal_threads': None,
    'pysoftk_num_confs': None,
    'pysoftk_ob_workers': None,
    'pysoftk_skip_final_localopt': None,
    'local_pysoftk_root': None,
    'shared_cache_root': None,
    'start_phase': 'pysoftk',
    'production_total_ns': 70.0,
    'production_replicas': 1,
    'production_ntomp': 12,
    'production_tcoupl': 'v-rescale',
    'production_tau_t': 5.0,
    'production_bonded_gpu': True,
    'gk_output_enabled': False,
    'gk_frame_interval_ps': 1.0,
    'gk_save_velocities': False,
    'run_gk_analysis': False,
    'gk_analysis_mode': 'eh',
    'gk_analysis_group': 'System',
    'gk_analysis_begin_ns': 0.0,
    'gk_analysis_end_ns': None,
    'gk_analysis_sample_dt_ps': 1.0,
    'gk_analysis_temperature_k': 353.0,
    'gk_sigma_unit': 's_per_m',
    'analysis_begin_ns': 40.0,
    'analysis_end_ns': 70.0,
    'n_chains': None,
    'li_tfsi_pairs': 100,
    'auto_update_n_chains': True,
    'molality_basis': 'mixture',
    'tfsi_charge_model': 'lammps_fq07',
    'li_charge_scale': 0.7,
    'anion_charge_scale': None,
    'analysis_li_charge_scale': 1.0,
    'analysis_anion_charge_scale': None,
    'cluster_cutoff_auto': False,
    'htpmd_strict_match': True,
    'analysis_cne_diffusion_mode': 'legacy',
    'analysis_cne_cluster_drag_exponent': 0.0,
    'nvt1_variant': 'short',
    'nvt1_short_ps': 200.0,
    'nvt1_split_vrescale_ps': 100.0,
    'nvt1_split_nosehoover_ps': 100.0,
    'md_stop_after_stage': None,
    'force_restart': False,
    'force_rerun_from_start_phase': False,
    'resume_existing': True,
}


@dataclass(frozen=True)
class GromacsBatchConfig:
    work_root: Path
    gromacs_dir: Path
    base_notebook: Path
    ref_csv: Path
    out_dir: Path
    runs_dir: Path
    results_dir: Path
    pipeline_py: Path
    python_executable: str
    physical_cores: int
    max_parallel_traj: int
    base_ntomp: int
    gromacs_ntomp: int
    fast_pysoftk: bool
    pysoftk_uff_iters: int
    pysoftk_localopt_steps: int
    sigma_ref_col: str = 'CONDUCTIVITY'
    tplus_ref_col: str = 'Transference Number'
    diff_li_ref_col: str = 'Li Diffusivity'
    diff_an_ref_col: str = 'TFSI Diffusivity'
    sigma_pred_col: str = 'sigma_cNE_htpmd_S_cm'
    cfg: dict[str, Any] = field(default_factory=dict)


def _python_has_modules(py_exe: str, modules: list[str]) -> bool:
    try:
        mod_code = '; '.join([f"import {m}" for m in modules])
        r = subprocess.run([py_exe, '-c', mod_code], capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


def _physical_core_count() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if isinstance(n, int) and n > 0:
            return int(n)
    except Exception:
        pass

    try:
        pairs = set()
        phys = None
        core = None
        with open('/proc/cpuinfo', 'r', encoding='utf-8', errors='ignore') as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = None
                    core = None
                    continue
                if s.startswith('physical id'):
                    phys = s.split(':', 1)[1].strip()
                elif s.startswith('core id'):
                    core = s.split(':', 1)[1].strip()
        if phys is not None and core is not None:
            pairs.add((phys, core))
        if pairs:
            return len(pairs)
    except Exception:
        pass

    return max(1, int(os.cpu_count() or 1))


def _select_python(cfg: dict[str, Any]) -> str:
    batch_python = str(cfg.get('batch_python', '') or '').strip()
    if batch_python:
        return batch_python

    candidates = [
        sys.executable,
        os.environ.get('GROMACS_BATCH_PYTHON', ''),
        shutil.which('python') or '',
    ]
    python = sys.executable
    for cand in candidates:
        if cand and Path(cand).exists() and _python_has_modules(cand, ['rdkit', 'pysoftk']):
            python = cand
            break
    return python


def config_from_cfg(cfg: dict[str, Any] | None = None, *, work_root: Path | str | None = None) -> GromacsBatchConfig:
    merged = dict(DEFAULT_CFG)
    if cfg:
        merged.update(cfg)

    base_work_root = Path(work_root or Path.cwd()).expanduser().resolve()
    if (
        (base_work_root / 'gromacs_new.ipynb').exists()
        and (base_work_root / 'gromacs_new_batch_eval_top_bottom_stratified.ipynb').exists()
    ):
        gromacs_dir = base_work_root
        out_dir = base_work_root
    else:
        gromacs_dir = base_work_root / 'DL' / 'gromacs'
        out_dir = gromacs_dir / 'eval_top10_bottom10_stratified100'
    runs_dir = out_dir / 'runs'
    results_dir = out_dir / 'results'
    pipeline_py = out_dir / 'gromacs_new_pipeline_importable.py'
    for path in (out_dir, runs_dir, results_dir):
        path.mkdir(parents=True, exist_ok=True)

    if merged.get('physical_cores') in ('', None):
        merged['physical_cores'] = _physical_core_count()
    physical_cores = max(1, int(merged['physical_cores']))

    max_parallel_traj = max(1, min(int(merged.get('max_parallel_traj', 1)), physical_cores))

    base_ntomp_raw = merged.get('base_ntomp')
    if base_ntomp_raw in ('', None):
        base_ntomp = physical_cores
    else:
        base_ntomp = max(1, min(int(base_ntomp_raw), physical_cores))

    gromacs_ntomp_raw = merged.get('gromacs_ntomp')
    if gromacs_ntomp_raw in ('', None):
        gromacs_ntomp = max(1, base_ntomp // max(1, max_parallel_traj))
    else:
        gromacs_ntomp = max(1, min(int(gromacs_ntomp_raw), physical_cores))

    if merged.get('local_pysoftk_root') in ('', None):
        merged['local_pysoftk_root'] = str(gromacs_dir)
    if merged.get('shared_cache_root') in ('', None):
        merged['shared_cache_root'] = str(out_dir / 'shared_cache')

    python = _select_python(merged)

    return GromacsBatchConfig(
        work_root=base_work_root,
        gromacs_dir=gromacs_dir,
        base_notebook=gromacs_dir / 'gromacs_new.ipynb',
        ref_csv=gromacs_dir / 'simulation-trajectory-aggregate.csv',
        out_dir=out_dir,
        runs_dir=runs_dir,
        results_dir=results_dir,
        pipeline_py=pipeline_py,
        python_executable=python,
        physical_cores=physical_cores,
        max_parallel_traj=max_parallel_traj,
        base_ntomp=base_ntomp,
        gromacs_ntomp=gromacs_ntomp,
        fast_pysoftk=bool(merged.get('fast_pysoftk', True)),
        pysoftk_uff_iters=int(merged.get('pysoftk_uff_iters', 600)),
        pysoftk_localopt_steps=int(merged.get('pysoftk_localopt_steps', 150)),
        cfg=merged,
    )


def print_batch_environment(config: GromacsBatchConfig) -> None:
    print('BASE_NOTEBOOK =', config.base_notebook)
    print('REF_CSV =', config.ref_csv)
    print('OUT_DIR =', config.out_dir)
    print('PIPELINE_PY =', config.pipeline_py)
    print('PYTHON =', config.python_executable)
    print('PHYSICAL_CORES =', config.physical_cores)
    print('MAX_PARALLEL_TRAJ =', config.max_parallel_traj)
    print('GROMACS_NTOMP =', config.gromacs_ntomp)
    print('FAST_PYSOFTK =', config.fast_pysoftk)
    print('PYSOFTK_UFF_ITERS =', config.pysoftk_uff_iters)
    print('PYSOFTK_LOCALOPT_STEPS =', config.pysoftk_localopt_steps)


def _infer_stage_from_source(src: str) -> str:
    s = src.lower()

    # prefer pysoftk for build cells that contain both utility and packmol text
    if 'from pysoftk' in s or 'linear_polymer' in s or ' lp(' in s or ' fmt(' in s:
        return 'pysoftk'
    if 'acpype' in s or 'write_monatomic_itp' in s or 'locate_gmx_itp' in s:
        return 'atomtyping'
    if 'packmol' in s and ('run(' in s or 'packmol ok' in s or 'packmol <' in s):
        return 'packmol'
    if 'gmx mdrun' in s or 'gmx grompp' in s or 'mdrun(' in s or 'run_stage(' in s:
        return 'md'
    if 'conductivity' in s or 'msd' in s or 'pop_mat' in s or 'analysis_dir' in s:
        return 'conductivity-analysis'
    return 'setup'


def _infer_stage_from_output_line(line: str):
    if not line:
        return None
    s = line.lower()

    if 'acpype' in s:
        return 'atomtyping'
    if 'packmol' in s:
        return 'packmol'
    if 'charge-sanity' in s or 'charge_sanity' in s:
        return 'charge_sanity'

    if 'gmx mdrun' in s or 'gmx grompp' in s:
        m = re.search(r'-deffnm\s+([a-z0-9_\-\.]+)', s)
        if m:
            deffnm = m.group(1)
            if deffnm.startswith('em'):
                return 'md-em'
            if deffnm.startswith('nvt'):
                return 'md-nvt'
            if deffnm.startswith('npt'):
                return 'md-npt'
            if deffnm.startswith('prod'):
                return 'md-prod'
        if 'production' in s:
            return 'md-prod'
        return 'md'

    if 'conductivity' in s or 'msd' in s or 'pop_mat' in s or '/analysis/' in s:
        return 'conductivity-analysis'
    if 'chain mw' in s or 'chain_fix' in s or 'linear_polymer' in s or '[pysoftk-fast]' in s:
        return 'pysoftk'
    return None


def notebook_to_single_py(base_nb: Path, out_py: Path):
    nb = json.loads(base_nb.read_text())
    chunks = []

    for idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        if not src.strip():
            continue

        stage = _infer_stage_from_source(src)
        marker_line = f'print("__STAGEV3__:{stage}:cell{idx}", flush=True)\n'

        # Keep future-import validity: never place runtime statements before
        # `from __future__ import ...` lines.
        lines = src.rstrip().splitlines(keepends=True)
        fut_idx = None
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith('from __future__ import '):
                fut_idx = i
                break

        chunks.append(f'\n# ===== Notebook Cell {idx} [{stage}] =====\n')

        if fut_idx is None:
            chunks.append(marker_line)
            chunks.extend(lines)
            chunks.append('\n')
        else:
            chunks.extend(lines[:fut_idx + 1])
            chunks.append(marker_line)
            chunks.extend(lines[fut_idx + 1:])
            chunks.append('\n')

    script = ''.join(chunks)

    marker = 'spec = SystemSpec()\n'
    inject = (
        "spec = SystemSpec()\n"
        "_spec_name_override = os.environ.get('GROMACS_SPEC_NAME')\n"
        "if _spec_name_override:\n"
        "    spec.name = _spec_name_override\n"
        "_FAST_PYSOFTK = os.environ.get('GROMACS_FAST_PYSOFTK', '1') != '0'\n"
        "_PYSOFTK_UFF_ITERS = int(os.environ.get('GROMACS_PYSOFTK_UFF_ITERS', '600' if _FAST_PYSOFTK else '2000'))\n"
        "_PYSOFTK_LOCALOPT_STEPS = int(os.environ.get('GROMACS_PYSOFTK_LOCALOPT_STEPS', '150' if _FAST_PYSOFTK else '500'))\n"
        "_PYSOFTK_SKIP_FINAL_LOCALOPT = os.environ.get('GROMACS_PYSOFTK_SKIP_FINAL_LOCALOPT', '0').strip().lower() not in ('0', 'false', 'no', 'off')\n"
        "_molality_basis = os.environ.get('GROMACS_MOLALITY_BASIS')\n"
        "if _molality_basis:\n"
        "    spec.molality_basis = _molality_basis.strip().lower()\n"
    )

    if marker not in script:
        raise RuntimeError('Could not find `spec = SystemSpec()` marker in converted script.')

    script = script.replace(marker, inject, 1)
    # Keep the analysis defaults stable even when gromacs_new.ipynb is regenerated
    # from an older copy. The screening workflow currently compares cNE with a
    # 0.28 nm structural cutoff and max_cluster=101.
    script = script.replace(
        'cluster_cutoff_nm: float = 0.34      # 3.4 Å',
        'cluster_cutoff_nm: float = 0.28      # 2.8 Å',
    )
    script = script.replace(
        'cluster_stride_ps: float = 0.0\n    cluster_max_cluster: int = 101',
        'cluster_stride_ps: float = 0.0\n    cluster_persistence_threshold_ps: float = 20.0\n    cluster_max_cluster: int = 101',
    )
    script = script.replace(
        'htpmd_max_cluster = int(getattr(spec, "htpmd_max_cluster", 10))',
        'htpmd_max_cluster = int(getattr(spec, "htpmd_max_cluster", getattr(spec, "cluster_max_cluster", 101)))',
    )

    # ---- pysoftk speed-up patch (leave other pipeline logic unchanged) ----
    script = script.replace(
        'import os, re, shlex, shutil, subprocess, warnings, math, random\n',
        'import os, re, shlex, shutil, subprocess, warnings, math, random, sys\n',
        1,
    )
    script = script.replace(
        'RDLogger.DisableLog("rdApp.*")\n\nfrom pysoftk.linear_polymer.linear_polymer import Lp\nfrom pysoftk.format_printers.format_mol import Fmt\n',
        'RDLogger.DisableLog("rdApp.*")\n\n_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()\n_LOCAL_PYSOFTK_ROOT = os.environ.get("GROMACS_LOCAL_PYSOFTK_ROOT", "").strip()\n_pysoftk_candidates = []\nif _LOCAL_PYSOFTK_ROOT:\n    _pysoftk_candidates.append(Path(_LOCAL_PYSOFTK_ROOT).expanduser().resolve())\n_pysoftk_candidates.extend([_THIS_DIR.parent, Path.cwd().resolve()])\n_seen_pysoftk_candidates = set()\nfor _cand in _pysoftk_candidates:\n    try:\n        _cand_resolved = _cand.resolve()\n    except Exception:\n        continue\n    if _cand_resolved in _seen_pysoftk_candidates:\n        continue\n    _seen_pysoftk_candidates.add(_cand_resolved)\n    if (_cand_resolved / "pysoftk" / "__init__.py").exists():\n        if str(_cand_resolved) not in sys.path:\n            sys.path.insert(0, str(_cand_resolved))\n        print("[pysoftk-local] using vendored pysoftk from", _cand_resolved / "pysoftk", flush=True)\n        break\n\nfrom pysoftk.linear_polymer.linear_polymer import Lp\nfrom pysoftk.format_printers.format_mol import Fmt\n',
        1,
    )
    script = script.replace(
        'AllChem.UFFOptimizeMolecule(mol, maxIters=2000)',
        'AllChem.UFFOptimizeMolecule(mol, maxIters=_PYSOFTK_UFF_ITERS)'
    )
    script = script.replace(
        'try:\n        chain.localopt(forcefield=\"uff\", steps=500)\n    except Exception:\n        pass',
        'if not _PYSOFTK_SKIP_FINAL_LOCALOPT:\n        try:\n            chain.localopt(forcefield=\"uff\", steps=_PYSOFTK_LOCALOPT_STEPS)\n        except Exception:\n            pass'
    )

    script = script.replace(
        'extra = ["-ntmpi","1","-ntomp","16","-pin","on"]',
        '_ntomp = int(os.environ.get("GROMACS_NTOMP", "16"))\n    extra = ["-ntmpi","1","-ntomp",str(_ntomp),"-pin","on"]'
    )

    # ---- cNE htp-md alignment: legacy mode + no drag by default ----
    script = script.replace(
        'cne_diffusion_mode: str = "cluster_weighted"',
        'cne_diffusion_mode: str = "legacy"',
        1,
    )
    script = script.replace(
        'cne_cluster_drag_exponent: float = 0.5',
        'cne_cluster_drag_exponent: float = 0.0',
        1,
    )

    script = script.replace(
        '_cutoff_auto = os.environ.get("GROMACS_CLUSTER_CUTOFF_AUTO")\nif _cutoff_auto:\n    spec.cluster_cutoff_auto = _cutoff_auto.strip().lower() not in ("0", "false", "no", "off")\n',
        '_cutoff_auto = os.environ.get("GROMACS_CLUSTER_CUTOFF_AUTO")\nif _cutoff_auto:\n    spec.cluster_cutoff_auto = _cutoff_auto.strip().lower() not in ("0", "false", "no", "off")\n_cne_mode = os.environ.get("GROMACS_CNE_DIFFUSION_MODE")\nif _cne_mode:\n    spec.cne_diffusion_mode = _cne_mode.strip().lower()\n_cne_drag = os.environ.get("GROMACS_CNE_CLUSTER_DRAG_EXPONENT")\nif _cne_drag:\n    spec.cne_cluster_drag_exponent = float(_cne_drag)\n',
        1,
    )

    poly_def = 'def build_polymer_chain(spec: SystemSpec, out_dir: Path) -> Tuple[Path, float]:\n'
    poly_inject = (
        poly_def +
        '    print("__STAGEV3__:pysoftk", flush=True)\n'
        '    pdb_fix = out_dir / f"{spec.name}_chain_fix.pdb"\n'
        '    mol_path = out_dir / "chain.mol"\n'
        '    if _FAST_PYSOFTK and pdb_fix.exists() and mol_path.exists():\n'
        '        try:\n'
        '            poly_cached = Chem.MolFromMolBlock(mol_path.read_text(), sanitize=False, removeHs=False)\n'
        '            if poly_cached is not None:\n'
        '                try:\n'
        '                    Chem.SanitizeMol(poly_cached)\n'
        '                except Exception:\n'
        '                    pass\n'
        '                chain_mw_cached = float(Descriptors.MolWt(poly_cached))\n'
        '                log(f"[pysoftk-fast] reuse cached polymer: {pdb_fix}")\n'
        '                return pdb_fix, chain_mw_cached\n'
        '        except Exception:\n'
        '            pass\n'
    )
    if poly_def in script:
        script = script.replace(poly_def, poly_inject, 1)

    tfsi_def = 'def build_tfsi(spec: SystemSpec, out_dir: Path) -> Tuple[Path, float]:\n'
    tfsi_inject = (
        tfsi_def +
        '    print("__STAGEV3__:pysoftk", flush=True)\n'
        '    out_pdb = out_dir / "tfsi.pdb"\n'
        '    if _FAST_PYSOFTK and out_pdb.exists():\n'
        '        tfsi_cached = Chem.MolFromSmiles(spec.anion_smiles)\n'
        '        tfsi_mw_cached = float(Descriptors.MolWt(tfsi_cached)) if tfsi_cached is not None else float("nan")\n'
        '        log(f"[pysoftk-fast] reuse cached tfsi: {out_pdb}")\n'
        '        return out_pdb, tfsi_mw_cached\n'
    )
    if tfsi_def in script:
        script = script.replace(tfsi_def, tfsi_inject, 1)

    # clear runtime stage markers for tqdm postfix
    script = script.replace(
        'seeds = [int(spec.packmol_seed) + i * 1000 for i in range(n_seeds)]\n',
        'seeds = [int(spec.packmol_seed) + i * 1000 for i in range(n_seeds)]\n\nprint("__STAGEV3__:packmol", flush=True)\n'
    )
    script = script.replace(
        'res = run(["bash","-lc", f"packmol < {shlex.quote(inp_path.name)}"], cwd=PACKMOL_DIR, check=False, capture_output=True)\n',
        'print("__STAGEV3__:packmol", flush=True)\n    res = run(["bash","-lc", f"packmol < {shlex.quote(inp_path.name)}"], cwd=PACKMOL_DIR, check=False, capture_output=True)\n'
    )

    run_stage_sig = (
        'def run_stage(stage: str, mdp: Path, in_gro: Path, *, use_posres_ref: bool=False,\n'
        '              prev_stage_dir: Optional[Path]=None, mdrun_extra: Optional[List[str]]=None,\n'
        '              force_cpu: bool=False) -> Path:\n'
        '    sd = MD_DIR / stage\n'
    )
    run_stage_inject = (
        'def run_stage(stage: str, mdp: Path, in_gro: Path, *, use_posres_ref: bool=False,\n'
        '              prev_stage_dir: Optional[Path]=None, mdrun_extra: Optional[List[str]]=None,\n'
        '              force_cpu: bool=False) -> Path:\n'
        '    stl = stage.lower()\n'
        '    if stl.startswith("em"):\n'
        '        md_stage = "md-em"\n'
        '    elif stl.startswith("nvt"):\n'
        '        md_stage = "md-nvt"\n'
        '    elif stl.startswith("npt"):\n'
        '        md_stage = "md-npt"\n'
        '    elif stl.startswith("prod"):\n'
        '        md_stage = "md-prod"\n'
        '    else:\n'
        '        md_stage = "md"\n'
        '    print(f"__STAGEV3__:{md_stage}:{stage}", flush=True)\n'
        '    sd = MD_DIR / stage\n'
        '    sd.mkdir(parents=True, exist_ok=True)\n'
        '    out_gro = sd / f"{stage}.gro"\n'
        '    if out_gro.exists() and out_gro.stat().st_size > 0:\n'
        '        print(f"__STAGEV3__:{md_stage}:{stage}:skip-existing", flush=True)\n'
        '        return out_gro\n'
    )
    script = script.replace(run_stage_sig, run_stage_inject, 1)

    # route all trajectory artifacts to run directory when requested
    script = script.replace(
        'spec.workspace = Path(spec.name).resolve()\n',
        '_workspace_override = os.environ.get("GROMACS_TRAJ_ROOT")\n'
        'if _workspace_override:\n'
        '    spec.workspace = Path(_workspace_override).resolve()\n'
        'else:\n'
        '    spec.workspace = Path(spec.name).resolve()\n',
        1
    )

    # GPU policy: EM only on CPU, other MD stages use GPU first
    script = script.replace('"mdrun_extra": GPU_SAFE_EXTRA, "force_cpu": True,', '"mdrun_extra": GPU_SAFE_EXTRA, "force_cpu": False,', 1)
    script = script.replace('em_gro   = run_stage("em",   em_mdp,   start, use_posres_ref=False)',
                            'em_gro   = run_stage("em",   em_mdp,   start, use_posres_ref=False, force_cpu=True)', 1)
    script = script.replace('em2_gro = run_stage("em2", em_mdp, densfix_gro, use_posres_ref=False)',
                            'em2_gro = run_stage("em2", em_mdp, densfix_gro, use_posres_ref=False, force_cpu=True)', 1)

    # atomtyping robustness: keep packmol n_chains in sync + safer RDKit/ACPYPE pre-fix
    script = script.replace(
        'def fix_close_contacts_rdkit(mol, min_dist=0.80, max_iter=2000, seed=0):',
        'def fix_close_contacts_rdkit(mol, min_dist=1.00, max_iter=4000, seed=0):',
        1,
    )
    script = script.replace(
        'mol = Chem.MolFromPDBFile(str(pdb_in), removeHs=False, sanitize=True, proximityBonding=True)\n    if mol is None:\n        raise RuntimeError(f"RDKit failed to read PDB: {pdb_in}")\n',
        'mol = Chem.MolFromPDBFile(str(pdb_in), removeHs=False, sanitize=True, proximityBonding=True)\n    if mol is None:\n        mol = Chem.MolFromPDBFile(str(pdb_in), removeHs=False, sanitize=False, proximityBonding=True)\n        if mol is None:\n            raise RuntimeError(f"RDKit failed to read PDB: {pdb_in}")\n        try:\n            Chem.SanitizeMol(mol)\n        except Exception:\n            warnings.warn(f"RDKit sanitize fallback failed for {pdb_in}; proceeding with sanitize=False molecule.")\n',
        1,
    )
    script = script.replace(
        'polymer_pdb = repair_polymer_pdb_for_acpype(polymer_pdb, polymer_pdb_fixed, min_dist=0.80, seed=123)',
        'polymer_pdb = repair_polymer_pdb_for_acpype(polymer_pdb, polymer_pdb_fixed, min_dist=1.00, seed=123)',
        1,
    )

    script = script.replace(
        'order = [spec.polymer_resname_coords, spec.cation_resname_coords, spec.anion_resname_coords]\n\nn_mols = {\n',
        'order = [spec.polymer_resname_coords, spec.cation_resname_coords, spec.anion_resname_coords]\n\n# Sync n_chains to what packmol actually used in this trajectory.\ndef infer_polymer_count_from_packmol(packmol_inp: Path, polymer_pdb: Path) -> Optional[int]:\n    if not packmol_inp.exists():\n        return None\n    lines = packmol_inp.read_text().splitlines()\n    target_name = polymer_pdb.name\n    target_abs = str(polymer_pdb.resolve())\n    for i, ln in enumerate(lines):\n        s = ln.strip()\n        if not s.lower().startswith("structure "):\n            continue\n        path = s.split(None, 1)[1].strip().strip(chr(34)).strip(chr(39))\n        if path == target_abs or Path(path).name == target_name:\n            for j in range(i + 1, min(i + 20, len(lines))):\n                m = re.match(r"\\s*number\\s+(\\d+)", lines[j], flags=re.I)\n                if m:\n                    return int(m.group(1))\n            break\n    return None\n\npackmol_polymer_ref = STRUCT_DIR / f"{spec.name}_chain_fix.pdb"\npackmol_used_n = infer_polymer_count_from_packmol(PACKMOL_DIR / "packmol.inp", packmol_polymer_ref)\nif packmol_used_n is not None and packmol_used_n != int(spec.n_chains):\n    log(f"[atomtyping] n_chains override from packmol.inp: {spec.n_chains} -> {packmol_used_n}")\n    spec.n_chains = int(packmol_used_n)\n\n# Keep topology molecule counts synced with packmol-updated n_chains.\nwrite_topology_ordered(TOPOL_TOP, all_atomtypes, pol_clean, tfsi_clean, li_clean, pol_mt, li_mt, tfsi_mt, spec)\n\nn_mols = {\n',
        1,
    )

    out_py.parent.mkdir(parents=True, exist_ok=True)
    out_py.write_text(script)


def ensure_pipeline_script(config: GromacsBatchConfig, *, force_rebuild: bool = False) -> Path:
    pipeline_py = config.pipeline_py
    needs_build = force_rebuild or (not pipeline_py.exists())
    if (not needs_build) and pipeline_py.exists():
        txt = pipeline_py.read_text(errors='ignore')
        required_tokens = [
            '__STAGEV3__:',
            'conductivity-analysis',
            '_FAST_PYSOFTK',
            'md-em',
            'md-prod',
            'skip-existing',
            'infer_polymer_count_from_packmol',
            'min_dist=1.00',
            'write_topology_ordered(TOPOL_TOP, all_atomtypes, pol_clean, tfsi_clean, li_clean, pol_mt, li_mt, tfsi_mt, spec)',
            'GROMACS_CNE_DIFFUSION_MODE',
            'cne_diffusion_mode: str = "legacy"',
            'cne_cluster_drag_exponent: float = 0.0',
            'GROMACS_HTPMD_STRICT_MATCH',
            'htpmd_strict_match: bool = False',
            'GROMACS_N_CHAINS',
            'GROMACS_LI_TFSI_PAIRS',
            'GROMACS_TFSI_CHARGE_MODEL',
            'tfsi_charge_model: str = "lammps_fq07"',
            'LAMMPS_TFSI_FQ07_CHARGES',
            'apply_itp_atomwise_charges(tfsi_clean',
            'canonicalize_tfsi_clean_itp(',
            'charge_diagnosis_atomtyping.json',
            'return Chem.RemoveHs(mol)',
            'GROMACS_LOCAL_PYSOFTK_ROOT',
            '_PYSOFTK_SKIP_FINAL_LOCALOPT',
            '_polymer_shared_cache_dir',
            'polymer_shared_cache_dir = _polymer_shared_cache_dir',
            '_generate_bonded_terms_from_graph',
            '_rebalance_fallback_polymer_charges',
            'polymer_trimer_fallback_full_GMX.itp',
        ]
        if any(token not in txt for token in required_tokens):
            needs_build = True
        else:
            try:
                compile(txt, str(pipeline_py), 'exec')
            except SyntaxError:
                needs_build = True
    if needs_build:
        notebook_to_single_py(config.base_notebook, pipeline_py)
    return pipeline_py


def read_prediction_from_analysis(config: GromacsBatchConfig, traj_id: int) -> dict[str, Any]:
    candidates = [
        config.runs_dir / f'Traj_{traj_id}' / 'analysis' / 'conductivity_summary_htpmd_ref.csv',
        config.gromacs_dir / f'Traj_{traj_id}' / 'analysis' / 'conductivity_summary_htpmd_ref.csv',
    ]
    src = next((path for path in candidates if path.exists()), None)
    if src is None:
        raise FileNotFoundError(f'Missing analysis output in candidates: {candidates}')

    row = pd.read_csv(src).iloc[0]

    sigma_cne = pd.to_numeric(row.get(config.sigma_pred_col, np.nan), errors='coerce')
    sigma_ne = pd.to_numeric(row.get('sigma_NE_htpmd_S_cm', np.nan), errors='coerce')
    c_tn = pd.to_numeric(row.get('c_tn_htpmd', np.nan), errors='coerce')

    d_li_raw = pd.to_numeric(row.get('D_Li_cm2s', np.nan), errors='coerce')
    d_an_raw = pd.to_numeric(row.get('D_an_cm2s', np.nan), errors='coerce')
    d_li = float(d_li_raw) if np.isfinite(d_li_raw) else float('nan')
    d_an = float(d_an_raw) if np.isfinite(d_an_raw) else float('nan')

    den = d_li + d_an
    tplus_ne = float(d_li / den) if np.isfinite(den) and den > 0 else float('nan')

    sigma_cne_pred = float(sigma_cne) if np.isfinite(sigma_cne) and sigma_cne > 0 else float('nan')
    sigma_ne_pred = float(sigma_ne) if np.isfinite(sigma_ne) and sigma_ne > 0 else float('nan')
    sigma_eval_mode = str(row.get('sigma_eval_mode', 'pure_cNE')).strip() or 'pure_cNE'

    return {
        'sigma_cNE_htpmd_S_cm_pred': sigma_cne_pred,
        'sigma_NE_htpmd_S_cm_pred': sigma_ne_pred,
        'used_sigma_ne_fallback': 0,
        'sigma_pred_source': 'cNE' if np.isfinite(sigma_cne_pred) else 'missing',
        'sigma_eval_mode_pred': sigma_eval_mode,
        'D_Li_cm2s_pred': d_li,
        'D_an_cm2s_pred': d_an,
        'tplus_NE_pred': tplus_ne,
        'c_tn_htpmd_pred': float(c_tn) if np.isfinite(c_tn) else float('nan'),
        'analysis_csv': str(src),
    }


def _extract_stage_marker(line: str):
    if not line:
        return None
    s = line.strip()
    token = None
    if s.startswith('__STAGEV3__:'):
        token = '__STAGEV3__:'
    elif s.startswith('__STAGEV2__:'):
        token = '__STAGEV2__:'
    elif s.startswith('__STAGE__:'):
        token = '__STAGE__:'
    if token is None:
        return None
    rest = s[len(token):]
    parts = rest.split(':')
    if len(parts) >= 1:
        return parts[0]
    return None


def select_batch_candidates(config: GromacsBatchConfig) -> pd.DataFrame:
    if not config.base_notebook.exists():
        raise FileNotFoundError(f'Base notebook missing: {config.base_notebook}')
    if not config.ref_csv.exists():
        raise FileNotFoundError(f'Reference CSV missing: {config.ref_csv}')

    top_k = int(config.cfg.get('top_k', 10))
    bottom_k = int(config.cfg.get('bottom_k', 10))
    stratified_n = int(config.cfg.get('stratified_n', 100))
    strata_bins = int(config.cfg.get('strata_bins', 10))
    random_seed = int(config.cfg.get('random_seed', 20260214))
    selection_group_col = config.cfg.get('selection_group_col') or config.cfg.get('group_col')

    df = pd.read_csv(config.ref_csv)
    required = [
        'Trajectory ID',
        config.sigma_ref_col,
        config.tplus_ref_col,
        config.diff_li_ref_col,
        config.diff_an_ref_col,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f'Missing reference columns: {missing}')
    if selection_group_col and selection_group_col not in df.columns:
        raise KeyError(f'Missing selection group column: {selection_group_col}')

    df = df.copy()
    df = df[np.isfinite(df[config.sigma_ref_col]) & np.isfinite(df[config.tplus_ref_col])]
    df = df[df[config.sigma_ref_col] > 0.0]

    def _stratified_middle_sample(df_middle: pd.DataFrame, n: int, bins: int, seed: int) -> pd.DataFrame:
        if n <= 0 or len(df_middle) == 0:
            return df_middle.iloc[0:0].copy()

        bins = max(1, min(int(bins), len(df_middle)))
        ranked = df_middle.copy()
        ranked['_rank'] = ranked[config.sigma_ref_col].rank(method='first')
        ranked['_bin'] = pd.qcut(ranked['_rank'], q=bins, labels=False, duplicates='drop')

        out_parts = []
        per_bin = n // bins
        rem = n % bins
        for b in range(bins):
            grp = ranked[ranked['_bin'] == b]
            if grp.empty:
                continue
            want = per_bin + (1 if b < rem else 0)
            take = min(want, len(grp))
            if take > 0:
                out_parts.append(grp.sample(n=take, random_state=seed + b))

        sampled = pd.concat(out_parts, ignore_index=False) if out_parts else ranked.iloc[0:0].copy()
        deficit = n - len(sampled)
        if deficit > 0:
            remain = ranked.drop(index=sampled.index, errors='ignore')
            if len(remain) > 0:
                extra = remain.sample(n=min(deficit, len(remain)), random_state=seed + 999)
                sampled = pd.concat([sampled, extra], ignore_index=False)

        sampled = sampled.drop(columns=['_rank', '_bin'], errors='ignore')
        sampled = sampled.drop_duplicates(subset=['Trajectory ID'])
        return sampled

    if selection_group_col:
        raw_groups = config.cfg.get('selection_groups')
        if raw_groups in ('', None):
            group_values = sorted(df[selection_group_col].dropna().astype(str).unique().tolist())
        elif isinstance(raw_groups, str):
            group_values = [x.strip() for x in raw_groups.split(',') if x.strip()]
        else:
            group_values = [str(x) for x in raw_groups]

        top_k_group = int(config.cfg.get('top_k_per_group', top_k))
        bottom_k_group = int(config.cfg.get('bottom_k_per_group', bottom_k))
        stratified_n_group = int(config.cfg.get('stratified_n_per_group', 0))
        total_need = len(group_values) * (top_k_group + bottom_k_group + stratified_n_group)
        sample_parts = []

        for gi, group_value in enumerate(group_values):
            gdf = df[df[selection_group_col].astype(str) == str(group_value)].copy()
            gdf = gdf.sort_values(config.sigma_ref_col, ascending=True).reset_index(drop=True)
            group_need = top_k_group + bottom_k_group + stratified_n_group
            if len(gdf) < group_need:
                raise RuntimeError(
                    f'Not enough rows for group {group_value}: have={len(gdf)}, need at least {group_need}'
                )

            safe_group = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(group_value)).strip('_') or f'group{gi}'
            bottom = gdf.head(bottom_k_group).copy()
            bottom['sample_group'] = f'{safe_group}_bottom'
            top = gdf.tail(top_k_group).copy()
            top['sample_group'] = f'{safe_group}_top'

            middle_pool = gdf.iloc[bottom_k_group: len(gdf) - top_k_group].copy()
            middle = _stratified_middle_sample(
                middle_pool,
                stratified_n_group,
                strata_bins,
                random_seed + gi * 1009,
            )
            middle['sample_group'] = f'{safe_group}_middle_stratified'
            sample_parts.extend([top, bottom, middle])

        sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else df.iloc[0:0].copy()
    else:
        df = df.sort_values(config.sigma_ref_col, ascending=True).reset_index(drop=True)
        total_need = top_k + bottom_k + stratified_n
        if len(df) < total_need:
            raise RuntimeError(f'Not enough rows for sampling: have={len(df)}, need at least {total_need}')

        bottom = df.head(bottom_k).copy()
        bottom['sample_group'] = 'bottom'
        top = df.tail(top_k).copy()
        top['sample_group'] = 'top'

        middle_pool = df.iloc[bottom_k: len(df) - top_k].copy()
        middle = _stratified_middle_sample(middle_pool, stratified_n, strata_bins, random_seed)
        middle['sample_group'] = 'middle_stratified'

        sample = pd.concat([top, bottom, middle], ignore_index=True)
        if len(sample) < total_need:
            used = set(sample['Trajectory ID'].tolist())
            remain = middle_pool[~middle_pool['Trajectory ID'].astype(int).isin(used)]
            need = total_need - len(sample)
            if need > 0 and len(remain) > 0:
                extra = remain.sample(n=min(need, len(remain)), random_state=random_seed + 2026).copy()
                extra['sample_group'] = 'middle_fill'
                sample = pd.concat([sample, extra], ignore_index=True)

    sample['Trajectory ID'] = sample['Trajectory ID'].astype(int)
    sample = sample.drop_duplicates(subset=['Trajectory ID']).reset_index(drop=True)

    sample = sample.sort_values(
        ['sample_group', config.sigma_ref_col, 'Trajectory ID'],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    manifest_cols = [
        'Trajectory ID',
        'sample_group',
        config.sigma_ref_col,
        config.tplus_ref_col,
        config.diff_li_ref_col,
        config.diff_an_ref_col,
        'SMILES',
        'Degree of Polymerization',
        'Density',
        'Molality',
        'design_condition',
        'PSMILES',
        'pred_log10_cond',
        'pred_cond',
        'prediction_source',
        'polybert_exact_available',
        'candidate_rank_by_pred_cond',
        'repeat_heavy_atoms',
        'target_polymer_heavy_atoms',
        'estimated_polymer_heavy_atoms',
    ]
    keep_cols = []
    for col in manifest_cols:
        if col in sample.columns and col not in keep_cols:
            keep_cols.append(col)
    manifest = sample[keep_cols].copy()

    manifest_path = config.results_dir / 'sample_manifest.csv'
    manifest.to_csv(manifest_path, index=False)

    print(f'Saved: {manifest_path}')
    print(manifest['sample_group'].value_counts(dropna=False))
    print('total sampled =', len(manifest))
    return manifest


def run_batch_pipeline(manifest_df: pd.DataFrame, *, config: GromacsBatchConfig) -> pd.DataFrame:
    global _RUN_INTERRUPTED
    _RUN_INTERRUPTED = False

    CFG = dict(config.cfg)
    WORK_ROOT = config.work_root
    GROMACS_DIR = config.gromacs_dir
    BASE_NOTEBOOK = config.base_notebook
    REF_CSV = config.ref_csv
    OUT_DIR = config.out_dir
    RUNS_DIR = config.runs_dir
    RESULTS_DIR = config.results_dir
    PIPELINE_PY = config.pipeline_py
    PYTHON = config.python_executable
    PHYSICAL_CORES = int(config.physical_cores)
    MAX_PARALLEL_TRAJ = int(config.max_parallel_traj)
    BASE_NTOMP = int(config.base_ntomp)
    GROMACS_NTOMP = int(config.gromacs_ntomp)
    FAST_PYSOFTK = bool(config.fast_pysoftk)
    PYSOFTK_UFF_ITERS = int(config.pysoftk_uff_iters)
    PYSOFTK_LOCALOPT_STEPS = int(config.pysoftk_localopt_steps)
    SIGMA_REF_COL = config.sigma_ref_col
    TPLUS_REF_COL = config.tplus_ref_col
    DIFF_LI_REF_COL = config.diff_li_ref_col
    DIFF_AN_REF_COL = config.diff_an_ref_col
    SIGMA_PRED_COL = config.sigma_pred_col
    MAX_ATTEMPTS = int(CFG.get('max_attempts', 3))
    FORCE_RERUN = bool(CFG.get('force_rerun', False))
    FORCE_REBUILD_PIPELINE = bool(CFG.get('force_rebuild_pipeline', False))
    MAX_TRAJ_TO_RUN = CFG.get('max_traj_to_run')

    def ensure_pipeline_script(*, force_rebuild: bool = False) -> Path:
        return globals()['ensure_pipeline_script'](config, force_rebuild=force_rebuild)

    def read_prediction_from_analysis(traj_id: int) -> dict[str, Any]:
        return globals()['read_prediction_from_analysis'](config, traj_id)

    manifest_df = manifest_df.copy()

    # Run batch by global phases:
    # pysoftk/packmol/atomtyping/analysis each use their own phase workers;
    # md remains sequential at batch level and each traj uses GROMACS_NTOMP for full-core mdrun.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import gc
    import os
    import shutil

    tqdm = _load_batch_tqdm(CFG.get('tqdm_mode'))


    def _iter_tqdm_classes():
        out = []
        if tqdm is not None:
            out.append(tqdm)

        try:
            from tqdm.std import tqdm as _tqdm_std
            out.append(_tqdm_std)
        except Exception:
            pass

        try:
            from tqdm.notebook import tqdm as _tqdm_nb
            out.append(_tqdm_nb)
        except Exception:
            pass

        uniq = []
        seen = set()
        for cls in out:
            if cls is None:
                continue
            key = id(cls)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(cls)
        return uniq


    def _close_tqdm_bar(bar) -> None:
        if bar is None:
            return

        try:
            bar.clear(nolock=True)
        except TypeError:
            try:
                bar.clear()
            except Exception:
                pass
        except Exception:
            pass

        try:
            bar.close()
        except Exception:
            pass

        try:
            bar.disable = True
        except Exception:
            pass

        for attr in ('container', 'disp', 'displayed'):
            try:
                obj = getattr(bar, attr, None)
            except Exception:
                obj = None
            if obj is None:
                continue
            try:
                if hasattr(obj, 'close'):
                    obj.close()
            except Exception:
                pass


    def _cleanup_tqdm_instances(tag: str = '') -> int:
        seen_bars = set()
        closed = 0

        for cls in _iter_tqdm_classes():
            inst = getattr(cls, '_instances', None)
            if inst is None:
                continue

            try:
                bars = list(inst)
            except Exception:
                bars = []

            for bar in bars:
                key = id(bar)
                if key in seen_bars:
                    continue
                seen_bars.add(key)
                try:
                    _close_tqdm_bar(bar)
                    closed += 1
                except Exception:
                    pass

            try:
                inst.clear()
            except Exception:
                pass

            for mon_attr in ('monitor', '_monitor'):
                mon = getattr(cls, mon_attr, None)
                if mon is None:
                    continue
                try:
                    if hasattr(mon, 'exit'):
                        mon.exit()
                except Exception:
                    pass
                try:
                    setattr(cls, mon_attr, None)
                except Exception:
                    pass

        gc.collect()
        if closed > 0:
            suffix = f' ({tag})' if tag else ''
            print(f'[tqdm-cleanup] closed stale bars: {closed}{suffix}')
        return closed


    def _as_bool(raw, default: bool = False) -> bool:
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() not in ('0', 'false', 'no', 'off')


    TQDM_LEAVE = _as_bool(CFG.get('tqdm_leave', False), False)


    _cleanup_tqdm_instances('cell-init')

    _RUN_INTERRUPTED = False

    _phase_cap = int(PHYSICAL_CORES)
    _phase_default = int(CFG.get('phase_cpu_workers', 4) or 4)
    PHASE_CPU_WORKERS = max(1, min(_phase_default, _phase_cap))
    print('PHASE_CPU_WORKERS =', PHASE_CPU_WORKERS, f'(legacy fallback/default, physical cap={_phase_cap}); MD keeps GROMACS_NTOMP core allocation')


    def _phase_workers_from_cfg(name: str, default: int) -> int:
        raw = CFG.get(name, default)
        try:
            value = int(default if raw in ('', None) else raw)
        except Exception:
            value = int(default)
        return max(1, min(value, _phase_cap))


    PYSOFTK_PHASE_WORKERS = _phase_workers_from_cfg('pysoftk_phase_workers', PHASE_CPU_WORKERS)
    PACKMOL_PHASE_WORKERS = _phase_workers_from_cfg('packmol_phase_workers', PHASE_CPU_WORKERS)
    ATOMTYPING_PHASE_WORKERS = _phase_workers_from_cfg('atomtyping_phase_workers', PHASE_CPU_WORKERS)
    CHARGE_SANITY_PHASE_WORKERS = _phase_workers_from_cfg('charge_sanity_phase_workers', PHASE_CPU_WORKERS)
    ANALYSIS_PHASE_WORKERS = _phase_workers_from_cfg('analysis_phase_workers', PHASE_CPU_WORKERS)
    PHASE_WORKERS = {
        'pysoftk': PYSOFTK_PHASE_WORKERS,
        'packmol': PACKMOL_PHASE_WORKERS,
        'atomtyping': ATOMTYPING_PHASE_WORKERS,
        'charge_sanity': CHARGE_SANITY_PHASE_WORKERS,
        'analysis': ANALYSIS_PHASE_WORKERS,
    }
    print('PHASE_WORKERS =', PHASE_WORKERS, '; md uses sequential batch scheduling and GROMACS_NTOMP for full-core mdrun')

    _pys_thread_default = max(1, _phase_cap // max(1, PYSOFTK_PHASE_WORKERS))
    _pys_internal_cfg = CFG.get('pysoftk_internal_threads')
    PYSOFTK_INTERNAL_THREADS = int(_pys_thread_default if _pys_internal_cfg in ('', None) else _pys_internal_cfg)
    PYSOFTK_INTERNAL_THREADS = max(1, min(PYSOFTK_INTERNAL_THREADS, _phase_cap))
    _pys_num_confs_cfg = CFG.get('pysoftk_num_confs')
    PYSOFTK_NUM_CONFS = int(max(2, min(8, PYSOFTK_INTERNAL_THREADS)) if _pys_num_confs_cfg in ('', None) else _pys_num_confs_cfg)
    PYSOFTK_NUM_CONFS = max(1, PYSOFTK_NUM_CONFS)
    _pys_ob_workers_cfg = CFG.get('pysoftk_ob_workers')
    PYSOFTK_OB_WORKERS = int(max(1, min(PYSOFTK_INTERNAL_THREADS, PYSOFTK_NUM_CONFS)) if _pys_ob_workers_cfg in ('', None) else _pys_ob_workers_cfg)
    PYSOFTK_OB_WORKERS = max(1, min(PYSOFTK_OB_WORKERS, PYSOFTK_NUM_CONFS, _phase_cap))
    _pys_skip_cfg = CFG.get('pysoftk_skip_final_localopt')
    PYSOFTK_SKIP_FINAL_LOCALOPT = (PYSOFTK_INTERNAL_THREADS > 1) if _pys_skip_cfg in ('', None) else _as_bool(_pys_skip_cfg, PYSOFTK_INTERNAL_THREADS > 1)
    LOCAL_PYSOFTK_ROOT = str(Path(CFG.get('local_pysoftk_root', str(GROMACS_DIR))).resolve())
    SHARED_CACHE_ROOT = str(Path(CFG.get('shared_cache_root', str(OUT_DIR / 'shared_cache'))).resolve())
    print('LOCAL_PYSOFTK_ROOT =', LOCAL_PYSOFTK_ROOT)
    print('SHARED_CACHE_ROOT =', SHARED_CACHE_ROOT)
    print('PYSOFTK_INTERNAL_THREADS =', PYSOFTK_INTERNAL_THREADS)
    print('PYSOFTK_NUM_CONFS =', PYSOFTK_NUM_CONFS)
    print('PYSOFTK_OB_WORKERS =', PYSOFTK_OB_WORKERS)
    print('PYSOFTK_SKIP_FINAL_LOCALOPT =', PYSOFTK_SKIP_FINAL_LOCALOPT)


    # Batch defaults (CFG에서만 관리)
    BATCH_START_PHASE_DEFAULT = str(CFG.get('start_phase', 'pysoftk')).strip().lower()
    BATCH_PRODUCTION_TOTAL_NS = float(CFG.get('production_total_ns', 70))
    BATCH_PRODUCTION_REPLICAS = max(1, int(CFG.get('production_replicas', 1)))
    _production_ntomp_cfg = CFG.get('production_ntomp')
    BATCH_PRODUCTION_NTOMP = None if _production_ntomp_cfg in ('', None) else max(1, int(_production_ntomp_cfg))
    BATCH_PRODUCTION_TCOUPL = str(CFG.get('production_tcoupl', 'v-rescale')).strip().lower()
    if BATCH_PRODUCTION_TCOUPL not in ('nose-hoover', 'v-rescale', 'berendsen', 'andersen', 'no'):
        raise ValueError(f"Invalid BATCH_PRODUCTION_TCOUPL={BATCH_PRODUCTION_TCOUPL}")
    BATCH_PRODUCTION_TAU_T = float(CFG.get('production_tau_t', 5.0))
    if BATCH_PRODUCTION_TCOUPL != 'no' and BATCH_PRODUCTION_TAU_T <= 0:
        raise ValueError(f"Invalid BATCH_PRODUCTION_TAU_T={BATCH_PRODUCTION_TAU_T}")
    BATCH_PRODUCTION_BONDED_GPU = '1' if _as_bool(CFG.get('production_bonded_gpu', True), True) else '0'
    BATCH_GK_OUTPUT_ENABLED = '1' if _as_bool(CFG.get('gk_output_enabled', False), False) else '0'
    BATCH_GK_FRAME_INTERVAL_PS = max(0.001, float(CFG.get('gk_frame_interval_ps', 1.0)))
    BATCH_GK_SAVE_VELOCITIES = '1' if _as_bool(CFG.get('gk_save_velocities', False), False) else '0'
    BATCH_RUN_GK_ANALYSIS = _as_bool(CFG.get('run_gk_analysis', False), False)
    BATCH_GK_ANALYSIS_MODE = str(CFG.get('gk_analysis_mode', 'eh')).strip().lower()
    if BATCH_GK_ANALYSIS_MODE not in ('eh', 'acf', 'both'):
        raise ValueError(f"Invalid BATCH_GK_ANALYSIS_MODE={BATCH_GK_ANALYSIS_MODE}")
    BATCH_GK_ANALYSIS_GROUP = str(CFG.get('gk_analysis_group', 'System')).strip() or 'System'
    BATCH_GK_ANALYSIS_BEGIN_NS = float(CFG.get('gk_analysis_begin_ns', 0.0))
    _gk_end_cfg = CFG.get('gk_analysis_end_ns')
    BATCH_GK_ANALYSIS_END_NS = None if _gk_end_cfg in ('', None) else float(_gk_end_cfg)
    BATCH_GK_ANALYSIS_SAMPLE_DT_PS = max(0.001, float(CFG.get('gk_analysis_sample_dt_ps', 1.0)))
    BATCH_GK_ANALYSIS_TEMPERATURE_K = float(CFG.get('gk_analysis_temperature_k', 353.0))
    BATCH_GK_SIGMA_UNIT = str(CFG.get('gk_sigma_unit', 's_per_m')).strip().lower()
    if BATCH_GK_SIGMA_UNIT not in ('s_per_m', 's_per_cm'):
        raise ValueError(f"Invalid BATCH_GK_SIGMA_UNIT={BATCH_GK_SIGMA_UNIT}")
    BATCH_ANALYSIS_BEGIN_NS = float(CFG.get('analysis_begin_ns', 40))
    BATCH_ANALYSIS_END_NS = float(CFG.get('analysis_end_ns', 70))
    if BATCH_ANALYSIS_END_NS <= BATCH_ANALYSIS_BEGIN_NS:
        raise ValueError(f'Invalid analysis window: begin={BATCH_ANALYSIS_BEGIN_NS}, end={BATCH_ANALYSIS_END_NS}')

    # htp-md alignment defaults (cluster 개수는 파이프라인 기존 방식 유지)
    _batch_n_chains_raw = CFG.get('n_chains')
    if _batch_n_chains_raw in ('', None):
        BATCH_N_CHAINS = None
    else:
        BATCH_N_CHAINS = max(1, int(_batch_n_chains_raw))
    BATCH_LI_TFSI_PAIRS = max(1, int(CFG.get('li_tfsi_pairs', 100)))
    BATCH_AUTO_UPDATE_N_CHAINS = '1' if _as_bool(CFG.get('auto_update_n_chains', True), True) else '0'
    _molality_basis_cfg = str(CFG.get('molality_basis', 'mixture')).strip().lower()
    if _molality_basis_cfg in ('polymer', 'polymer-basis', 'solvent', 'solvent-basis'):
        BATCH_MOLALITY_BASIS = 'polymer'
    elif _molality_basis_cfg in ('mixture', 'mixture-basis', 'solution', 'solution-basis'):
        BATCH_MOLALITY_BASIS = 'mixture'
    else:
        raise ValueError(f"Invalid molality_basis={_molality_basis_cfg!r}; expected 'mixture' or 'polymer'")
    if BATCH_N_CHAINS is None and BATCH_AUTO_UPDATE_N_CHAINS != '1':
        BATCH_N_CHAINS = 31
    BATCH_TFSI_CHARGE_MODEL = str(CFG.get('tfsi_charge_model', 'lammps_fq07')).strip().lower()
    BATCH_LI_CHARGE_SCALE = float(CFG.get('li_charge_scale', 0.7))
    _anion_scale_cfg = CFG.get('anion_charge_scale')
    BATCH_ANION_CHARGE_SCALE = BATCH_LI_CHARGE_SCALE if _anion_scale_cfg in ('', None) else float(_anion_scale_cfg)
    # reference-style analysis는 z=1 사용 (analysis phase에서만 적용)
    BATCH_ANALYSIS_LI_CHARGE_SCALE = float(CFG.get('analysis_li_charge_scale', 1.0))
    _analysis_anion_scale_cfg = CFG.get('analysis_anion_charge_scale')
    BATCH_ANALYSIS_ANION_CHARGE_SCALE = BATCH_ANALYSIS_LI_CHARGE_SCALE if _analysis_anion_scale_cfg in ('', None) else float(_analysis_anion_scale_cfg)
    BATCH_CLUSTER_CUTOFF_AUTO = '1' if _as_bool(CFG.get('cluster_cutoff_auto', False), False) else '0'
    BATCH_HTPMD_STRICT_MATCH = '1' if _as_bool(CFG.get('htpmd_strict_match', True), True) else '0'
    BATCH_ANALYSIS_CNE_DIFFUSION_MODE = str(CFG.get('analysis_cne_diffusion_mode', 'legacy')).strip().lower()
    if BATCH_ANALYSIS_CNE_DIFFUSION_MODE not in ('legacy', 'cluster_weighted', 'harmonic'):
        raise ValueError(f"Invalid BATCH_ANALYSIS_CNE_DIFFUSION_MODE={BATCH_ANALYSIS_CNE_DIFFUSION_MODE}")
    BATCH_ANALYSIS_CNE_CLUSTER_DRAG_EXPONENT = float(CFG.get('analysis_cne_cluster_drag_exponent', 0.0))
    BATCH_NVT1_VARIANT = str(CFG.get('nvt1_variant', 'short')).strip().lower()
    if BATCH_NVT1_VARIANT not in ('baseline', 'short', 'split'):
        raise ValueError(f"Invalid BATCH_NVT1_VARIANT={BATCH_NVT1_VARIANT}")
    BATCH_NVT1_SHORT_PS = float(CFG.get('nvt1_short_ps', 200.0))
    BATCH_NVT1_SPLIT_VRESCALE_PS = float(CFG.get('nvt1_split_vrescale_ps', 100.0))
    BATCH_NVT1_SPLIT_NOSEHOOVER_PS = float(CFG.get('nvt1_split_nosehoover_ps', 100.0))
    _md_stop_after_cfg = CFG.get('md_stop_after_stage')
    BATCH_MD_STOP_AFTER_STAGE = None if _md_stop_after_cfg in ('', None) else str(_md_stop_after_cfg).strip().lower()

    # restart/resume policy도 CFG에서만 관리
    FORCE_RESTART = _as_bool(CFG.get('force_restart', False), False)
    FORCE_RERUN_FROM_START_PHASE = _as_bool(CFG.get('force_rerun_from_start_phase', False), False)
    if FORCE_RESTART:
        FORCE_RERUN_FROM_START_PHASE = True


    def build_phase_scripts(full_pipeline_py: Path) -> dict:
        txt = full_pipeline_py.read_text()

        k2 = txt.find('\n# ===== Notebook Cell 2 ')
        k3 = txt.find('\n# ===== Notebook Cell 3 ')
        k4 = txt.find('\n# ===== Notebook Cell 4 ')
        k5 = txt.find('\n# ===== Notebook Cell 5 ')
        if min(k2, k3, k4, k5) < 0:
            raise RuntimeError('Failed to locate notebook cell markers in pipeline script.')

        cell1 = txt[:k2]
        # Keep shared setup in one importable module instead of duplicating it into every phase file.
        # The setup marker belongs to the old copied-file layout; phase launchers now emit their own marker.
        common_setup = cell1.replace('print("__STAGEV3__:pysoftk:cell1", flush=True)\n', '', 1)
        common_setup = common_setup.replace('# ===== Notebook Cell 1 [pysoftk] =====', '# ===== Phase Common Setup =====', 1)
        if 'def write_pdb_strict_from_rdkit' not in common_setup:
            common_setup += textwrap.dedent(r'''

            # ---- common phase helper ----
            def write_pdb_strict_from_rdkit(mol, path, resname="POL", chain_id="A", resseq=1):
                conf = mol.GetConformer()
                resname3 = (resname or "MOL").upper()[:3]
                lines = []
                serial = 1
                for i, atom in enumerate(mol.GetAtoms(), start=1):
                    el = atom.GetSymbol().upper()
                    pos = conf.GetAtomPosition(i - 1)
                    name = f"{el}{i % 100:02d}"
                    if len(el) == 1:
                        name = f"{name:>4s}"[:4]
                    else:
                        name = f"{name:<4s}"[:4]
                    line = (
                        f"HETATM{serial:5d} {name}"
                        f" {resname3:>3s} {chain_id:1s}{resseq:4d}    "
                        f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}"
                        f"{1.00:6.2f}{0.00:6.2f}          {el:>2s}"
                    )
                    lines.append(line)
                    serial += 1
                lines += ["TER", "END", ""]
                path.write_text("\n".join(lines))
                return path
            ''')
        if 'def _placeholder_smiles' not in common_setup:
            common_setup += textwrap.dedent(r'''

            # ---- common polymer builder helpers used by atomtyping fallback ----
            def _placeholder_smiles(psmiles: str, placeholder: str) -> str:
                return re.sub(r"\[\*\]|\*", placeholder, psmiles)

            def ensure_3d_conformer(mol: Chem.Mol, seed: int = 1) -> Chem.Mol:
                mol = Chem.AddHs(mol)
                ps = AllChem.ETKDGv3()
                ps.randomSeed = int(seed)
                ps.useSmallRingTorsions = True
                ps.useMacrocycleTorsions = True
                ok = AllChem.EmbedMolecule(mol, ps)
                if ok != 0:
                    ok = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                    if ok != 0:
                        raise RuntimeError("RDKit 3D embedding failed")
                AllChem.UFFOptimizeMolecule(mol, maxIters=_PYSOFTK_UFF_ITERS)
                return Chem.RemoveHs(mol)
            ''')
        cell2 = txt[k2:k3]
        cell3 = txt[k3:k4]
        cell4 = txt[k4:k5]
        cell5_plus = txt[k5:]

        md_anchor = cell4.find('\n# ---- MDP writing ----')
        if md_anchor < 0:
            raise RuntimeError('Failed to locate md anchor in cell4 block.')
        atomtyping_part = cell4[:md_anchor]
        md_part = cell4[md_anchor:]

        phase_dir = OUT_DIR / 'phase_scripts'
        phase_dir.mkdir(parents=True, exist_ok=True)

        atomtyping_prelude = textwrap.dedent('''
    # ---- phase prelude: atomtyping ----
    polymer_pdb = STRUCT_DIR / f"{spec.name}_chain_fix.pdb"
    tfsi_pdb = STRUCT_DIR / "tfsi.pdb"
    li_pdb = STRUCT_DIR / "li.pdb"
    for _p in [polymer_pdb, tfsi_pdb, li_pdb, MD_DIR / "conf_initial.gro"]:
        if not _p.exists():
            raise FileNotFoundError(f"[atomtyping phase] missing prerequisite: {_p}")
    ''')

        packmol_prelude = textwrap.dedent('''
    # ---- phase prelude: packmol ----
    polymer_pdb = STRUCT_DIR / f"{spec.name}_chain_fix.pdb"
    tfsi_pdb = STRUCT_DIR / "tfsi.pdb"
    li_pdb = STRUCT_DIR / "li.pdb"
    for _p in [polymer_pdb, tfsi_pdb, li_pdb, STRUCT_DIR / "chain.mol"]:
        if not _p.exists():
            raise FileNotFoundError(f"[packmol phase] missing prerequisite: {_p}")

    _chain_mol = STRUCT_DIR / "chain.mol"
    _poly = Chem.MolFromMolBlock(_chain_mol.read_text(), sanitize=False, removeHs=False)
    if _poly is None:
        raise RuntimeError(f"[packmol phase] failed to parse chain.mol: {_chain_mol}")
    try:
        Chem.SanitizeMol(_poly)
    except Exception:
        pass
    polymer_mw_g_mol = float(Descriptors.MolWt(_poly))

    li_mw_g_mol = 6.941
    _tf = Chem.MolFromSmiles(spec.anion_smiles)
    if _tf is None:
        raise RuntimeError(f"[packmol phase] invalid anion smiles: {spec.anion_smiles}")
    tfsi_mw_g_mol = float(Descriptors.MolWt(_tf))

    def _normalized_molality_basis() -> str:
        basis = str(getattr(spec, "molality_basis", "mixture")).strip().lower()
        if basis in ("polymer", "polymer-basis", "solvent", "solvent-basis"):
            return "polymer"
        if basis in ("mixture", "mixture-basis", "solution", "solution-basis"):
            return "mixture"
        raise ValueError(f"Invalid molality_basis={basis!r}; expected 'mixture' or 'polymer'")

    def molality_from_counts(n_salt: int, n_chains: int, chain_mw_g_per_mol: float,
                             salt_mw_g_per_mol: Optional[float]=None, basis: Optional[str]=None) -> float:
        basis = _normalized_molality_basis() if basis is None else str(basis).strip().lower()
        denom_g_per_mol = n_chains * chain_mw_g_per_mol
        if basis in ("mixture", "mixture-basis", "solution", "solution-basis"):
            if salt_mw_g_per_mol is None:
                raise ValueError("mixture-basis molality requires salt_mw_g_per_mol")
            denom_g_per_mol += n_salt * salt_mw_g_per_mol
        return 1000.0 * n_salt / denom_g_per_mol

    def n_chains_for_target_molality(n_salt: int, target_molality: float,
                                     chain_mw_g_per_mol: float, salt_mw_g_per_mol: float) -> int:
        basis = _normalized_molality_basis()
        mass_budget_g_per_mol = 1000.0 * n_salt / float(target_molality)
        if basis == "mixture":
            mass_budget_g_per_mol -= n_salt * salt_mw_g_per_mol
        if mass_budget_g_per_mol <= 0:
            raise ValueError(
                f"target_molality={target_molality} is too high for mixture-basis salt mass "
                f"(n_salt={n_salt}, salt_mw={salt_mw_g_per_mol:.3f})"
            )
        return max(1, int(round(mass_budget_g_per_mol / chain_mw_g_per_mol)))

    if spec.auto_update_n_chains and spec.target_molality:
        salt_mw_g_mol = li_mw_g_mol + tfsi_mw_g_mol
        n_new = n_chains_for_target_molality(
            spec.li_tfsi_pairs,
            float(spec.target_molality),
            polymer_mw_g_mol,
            salt_mw_g_mol,
        )
        spec.n_chains = max(1, n_new)
        log(f"[auto:{_normalized_molality_basis()}] n_chains -> {spec.n_chains} for target molality={spec.target_molality}")

    salt_mw_g_mol = li_mw_g_mol + tfsi_mw_g_mol
    log(f"[molality:{_normalized_molality_basis()}] ≈ {molality_from_counts(spec.li_tfsi_pairs, spec.n_chains, polymer_mw_g_mol, salt_mw_g_mol):.4f} mol/kg")
    ''')

        md_prelude = textwrap.dedent('''
    # ---- phase prelude: md ----
    TOPOL_TOP = TOPO_DIR / "topol.top"
    STRUCT_FIXED = MD_DIR / "conf_initial_fixed.gro"
    if not TOPOL_TOP.exists():
        raise FileNotFoundError(f"[md phase] missing topology: {TOPOL_TOP}")
    if not STRUCT_FIXED.exists():
        raise FileNotFoundError(f"[md phase] missing structure: {STRUCT_FIXED}")

    _chain_mol = STRUCT_DIR / "chain.mol"
    if not _chain_mol.exists():
        raise FileNotFoundError(f"[md phase] missing chain.mol: {_chain_mol}")
    _poly = Chem.MolFromMolBlock(_chain_mol.read_text(), sanitize=False, removeHs=False)
    if _poly is None:
        raise RuntimeError(f"[md phase] failed to parse chain.mol: {_chain_mol}")
    try:
        Chem.SanitizeMol(_poly)
    except Exception:
        pass
    polymer_mw_g_mol = float(Descriptors.MolWt(_poly))

    li_mw_g_mol = 6.941
    _tf = Chem.MolFromSmiles(spec.anion_smiles)
    if _tf is None:
        raise RuntimeError(f"[md phase] invalid anion smiles: {spec.anion_smiles}")
    tfsi_mw_g_mol = float(Descriptors.MolWt(_tf))
    ''')

        analysis_prelude = textwrap.dedent('''
    # ---- phase prelude: analysis ----
    PROD_DIR = MD_DIR / "production"
    for _p in [PROD_DIR / "production.tpr", PROD_DIR / "production.xtc", PROD_DIR / "production.gro"]:
        if not _p.exists():
            raise FileNotFoundError(f"[analysis phase] missing prerequisite: {_p}")

    _chain_mol = STRUCT_DIR / "chain.mol"
    if not _chain_mol.exists():
        raise FileNotFoundError(f"[analysis phase] missing chain.mol: {_chain_mol}")
    _poly = Chem.MolFromMolBlock(_chain_mol.read_text(), sanitize=False, removeHs=False)
    if _poly is None:
        raise RuntimeError(f"[analysis phase] failed to parse chain.mol: {_chain_mol}")
    try:
        Chem.SanitizeMol(_poly)
    except Exception:
        pass
    polymer_mw_g_mol = float(Descriptors.MolWt(_poly))

    li_mw_g_mol = 6.941
    _tf = Chem.MolFromSmiles(spec.anion_smiles)
    if _tf is None:
        raise RuntimeError(f"[analysis phase] invalid anion smiles: {spec.anion_smiles}")
    tfsi_mw_g_mol = float(Descriptors.MolWt(_tf))

    def _normalized_molality_basis() -> str:
        basis = str(getattr(spec, "molality_basis", "mixture")).strip().lower()
        if basis in ("polymer", "polymer-basis", "solvent", "solvent-basis"):
            return "polymer"
        if basis in ("mixture", "mixture-basis", "solution", "solution-basis"):
            return "mixture"
        raise ValueError(f"Invalid molality_basis={basis!r}; expected 'mixture' or 'polymer'")

    def molality_from_counts(n_salt: int, n_chains: int, chain_mw_g_per_mol: float,
                             salt_mw_g_per_mol: Optional[float]=None, basis: Optional[str]=None) -> float:
        basis = _normalized_molality_basis() if basis is None else str(basis).strip().lower()
        denom_g_per_mol = n_chains * chain_mw_g_per_mol
        if basis in ("mixture", "mixture-basis", "solution", "solution-basis"):
            if salt_mw_g_per_mol is None:
                raise ValueError("mixture-basis molality requires salt_mw_g_per_mol")
            denom_g_per_mol += n_salt * salt_mw_g_per_mol
        return 1000.0 * n_salt / denom_g_per_mol

    _agg_expected = ROOT.parent / "simulation-trajectory-aggregate.csv"
    if not _agg_expected.exists():
        for _cand in [Path("simulation-trajectory-aggregate.csv")]:
            if _cand.exists():
                _agg_expected.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_cand, _agg_expected)
                break
    ''')

        charge_sanity_body = textwrap.dedent('''
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
    ''')

        def _phase_launcher(phase: str, body: str) -> str:
            # Import the common setup without star-import so underscore-prefixed runtime knobs
            # (_ACPYPE_FORCE, _FAST_PYSOFTK, etc.) are preserved in the phase namespace.
            return (
                "from __future__ import annotations\n"
                "from pathlib import Path\n"
                "import importlib.util\n"
                "import os\n"
                "import sys\n\n"
                "_PHASE_DIR = Path(__file__).resolve().parent\n"
                "if str(_PHASE_DIR) not in sys.path:\n"
                "    sys.path.insert(0, str(_PHASE_DIR))\n"
                "_COMMON_PATH = _PHASE_DIR / 'gromacs_new_phase_common.py'\n"
                "_COMMON_MODULE_NAME = f\"{os.environ.get('GROMACS_MODULE_NAME', 'gromacs_phase')}_common\"\n"
                "_common_spec = importlib.util.spec_from_file_location(_COMMON_MODULE_NAME, _COMMON_PATH)\n"
                "_common = importlib.util.module_from_spec(_common_spec)\n"
                "sys.modules[_COMMON_MODULE_NAME] = _common\n"
                "_common_spec.loader.exec_module(_common)\n"
                "for _k, _v in _common.__dict__.items():\n"
                "    if _k in {'__name__', '__file__', '__package__', '__loader__', '__spec__', '__builtins__'}:\n"
                "        continue\n"
                "    globals()[_k] = _v\n"
                f"print(\"__STAGEV3__:{phase}:entry\", flush=True)\n\n"
                f"{body}\n"
                f"\nprint(\"__PHASE_DONE__:{phase}\", flush=True)\n"
            )

        atomtyping_part = atomtyping_part.replace('print("__STAGEV3__:pysoftk:cell4", flush=True)\n', '', 1)
        atomtyping_part = atomtyping_part.replace('# ===== Notebook Cell 4 [pysoftk] =====\n', '', 1)

        scripts = {
            'pysoftk': _phase_launcher('pysoftk', cell2),
            'packmol': _phase_launcher('packmol', packmol_prelude + cell3),
            'atomtyping': _phase_launcher('atomtyping', atomtyping_prelude + atomtyping_part),
            'charge_sanity': _phase_launcher('charge_sanity', charge_sanity_body),
            'md': _phase_launcher('md', md_prelude + md_part),
            'analysis': _phase_launcher('analysis', analysis_prelude + cell5_plus),
        }

        out = {}
        common_path = phase_dir / 'gromacs_new_phase_common.py'
        compile(common_setup, str(common_path), 'exec')
        common_path.write_text(common_setup)
        for phase, body in scripts.items():
            path = phase_dir / f'gromacs_new_phase_{phase}.py'
            compile(body, str(path), 'exec')
            path.write_text(body)
            out[phase] = path

        return out


    def _stage_allowed_for_phase(phase: str, stage: str) -> bool:
        if not stage:
            return False
        p = (phase or '').lower()
        s = (stage or '').lower()

        # Hide shared setup/upstream stage markers in later phases so tqdm postfix shows the active phase only.
        allowed = {
            'pysoftk': {'setup', 'pysoftk', 'done'},
            'packmol': {'setup', 'packmol', 'done'},
            'atomtyping': {'setup', 'atomtyping', 'done'},
            'charge_sanity': {'setup', 'charge_sanity', 'done'},
            'md': {'md', 'md-em', 'md-nvt', 'md-npt', 'md-prod', 'done'},
            'analysis': {'conductivity-analysis', 'done'},
        }
        return s in allowed.get(p, {s})


    def _classify_phase_failure(phase: str, stdout: str, last_stage: str = '') -> str:
        text = (stdout or '').lower()
        if phase == 'atomtyping':
            m = re.search(r'__atomtyping_error__[:\\s]+([a-z0-9_\\-]+)', text)
            if m:
                return m.group(1)
            m = re.search(r'\[atomtyping:([a-z0-9_-]+)\]', text)
            if m:
                return m.group(1)
            if 'atoms too close' in text:
                return 'acpype_atoms_too_close'
            if 'atoms too scattered' in text:
                return 'acpype_atoms_too_scattered'
            if 'no gasteiger parameter for atom' in text or 'type: du' in text or 'antechamber failed' in text or 'maxbond' in text:
                return 'antechamber_typing_failure'
            if 'polymer_nonneutral' in text or 'polymer_clean.itp net charge is too large' in text:
                return 'polymer_nonneutral'
            if 'polymer_nonneutral_fallback_failed' in text:
                return 'polymer_nonneutral_fallback_failed'
            if 'charge_sanity_failed' in text or 'charge-sanity' in text:
                return 'charge_sanity_failed'
            if 'rdkit failed to read pdb' in text:
                return 'rdkit_read_pdb_failure'
            if 'sanitize fallback failed' in text:
                return 'rdkit_repair_failure'
            if '[natoms mismatch]' in text:
                return 'natoms_mismatch'
            if 'command failed: acpype' in text:
                return 'acpype_other_failure'
        if phase == 'charge_sanity':
            if 'charge_sanity_failed' in text or 'charge-sanity' in text:
                return 'charge_sanity_failed'
            return 'charge_sanity_failed'
        if phase == 'md' and (last_stage or '').startswith('md'):
            if '[gk-output-check]' in text or 'production.trr missing after mdrun' in text:
                return 'gk_output_mismatch'
            return 'md_runtime_failure'
        if phase == 'analysis' and 'traceback' in text:
            return 'analysis_runtime_failure'
        return 'unknown'


    def _atomtyping_recovery_env(error_type: str, attempt: int) -> dict:
        env = {}
        et = (error_type or '').strip().lower()
        if et == 'acpype_atoms_too_close':
            env['GROMACS_ACPYPE_MIN_DIST'] = f"{min(1.55, 1.20 + 0.15 * max(0, attempt - 1)):.2f}"
            env['GROMACS_ACPYPE_FORCE'] = '1'
            env['GROMACS_REBUILD_FROM_CHAIN_MOL'] = '0'
        elif et == 'acpype_atoms_too_scattered':
            env['GROMACS_ACPYPE_MIN_DIST'] = '1.20'
            env['GROMACS_ACPYPE_FORCE'] = '1'
            env['GROMACS_REBUILD_FROM_CHAIN_MOL'] = '1'
            env['GROMACS_ACPYPE_SKIP_REPAIR'] = '1'
            env['GROMACS_ACPYPE_TIMEOUT_SEC'] = '240'
        elif et == 'antechamber_typing_failure':
            env['GROMACS_ACPYPE_CHARGE_METHOD'] = 'gas'
            env['GROMACS_ACPYPE_ATOM_TYPE'] = 'gaff'
            env['GROMACS_ACPYPE_FORCE'] = '1'
            env['GROMACS_REBUILD_FROM_CHAIN_MOL'] = '1'
            env['GROMACS_ACPYPE_MIN_DIST'] = '1.30'
        elif et in ('rdkit_read_pdb_failure', 'rdkit_repair_failure'):
            env['GROMACS_REBUILD_FROM_CHAIN_MOL'] = '1'
            env['GROMACS_ACPYPE_FORCE'] = '1'
            env['GROMACS_ACPYPE_MIN_DIST'] = '1.35'
        elif et == 'natoms_mismatch':
            env['GROMACS_REBUILD_FROM_CHAIN_MOL'] = '1'
            env['GROMACS_ACPYPE_FORCE'] = '1'
            env['GROMACS_ACPYPE_MIN_DIST'] = '1.40'
            env['GROMACS_ACPYPE_CHARGE_METHOD'] = 'gas'
            env['GROMACS_ACPYPE_ATOM_TYPE'] = 'gaff'
        return env


    def _is_nonretryable_atomtyping_error(error_type: str) -> bool:
        et = (error_type or '').strip().lower()
        return et in {
            'polymer_trimer_fallback_incomplete',
            'polymer_trimer_fallback_failed',
            'polymer_nonneutral',
            'polymer_nonneutral_fallback_failed',
            'charge_sanity_failed',
        }


    GK_ANALYSIS_PY = OUT_DIR / 'tools' / 'gk_analysis.py'
    GK_COLLECT_PY = OUT_DIR / 'tools' / 'gk_collect_results.py'


    def _run_gk_postprocess(traj_id: int) -> dict:
        if not BATCH_RUN_GK_ANALYSIS:
            return {}

        run_dir = RUNS_DIR / f'Traj_{traj_id}'
        log_path = run_dir / 'analysis' / 'gk_attempt1.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if not GK_ANALYSIS_PY.exists():
            msg = f'missing script: {GK_ANALYSIS_PY}'
            log_path.write_text(msg + '\n')
            return {
                'gk_status': 'failed',
                'gk_returncode': -1,
                'gk_mode': BATCH_GK_ANALYSIS_MODE,
                'gk_error_tail': msg,
                'gk_summary_json': '',
            }

        cmd = [
            PYTHON,
            str(GK_ANALYSIS_PY),
            '--run-dir',
            str(run_dir),
            '--group',
            BATCH_GK_ANALYSIS_GROUP,
            '--begin-ns',
            f'{BATCH_GK_ANALYSIS_BEGIN_NS:g}',
            '--sample-dt-ps',
            f'{BATCH_GK_ANALYSIS_SAMPLE_DT_PS:g}',
            '--temperature-k',
            f'{BATCH_GK_ANALYSIS_TEMPERATURE_K:g}',
            '--mode',
            BATCH_GK_ANALYSIS_MODE,
        ]
        if BATCH_GK_ANALYSIS_END_NS is not None:
            cmd.extend(['--end-ns', f'{BATCH_GK_ANALYSIS_END_NS:g}'])

        proc = subprocess.run(
            cmd,
            cwd=OUT_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout = proc.stdout or ''
        stderr = proc.stderr or ''
        log_path.write_text(
            f'CMD: {" ".join(cmd)}\n'
            f'RETURN_CODE: {proc.returncode}\n\n'
            f'--- STDOUT ---\n{stdout}\n'
            f'--- STDERR ---\n{stderr}\n'
        )
        summary_json = run_dir / 'analysis_gk' / 'gk_summary.json'
        err_tail = '\n'.join((stdout + '\n' + stderr).strip().splitlines()[-20:]).strip()
        return {
            'gk_status': 'ok' if proc.returncode == 0 else 'failed',
            'gk_returncode': int(proc.returncode),
            'gk_mode': BATCH_GK_ANALYSIS_MODE,
            'gk_error_tail': err_tail,
            'gk_summary_json': str(summary_json) if summary_json.exists() else '',
        }


    def execute_phase_traj(
        traj_id: int,
        phase: str,
        phase_py: Path,
        max_attempts: int = MAX_ATTEMPTS,
        stage_cb=None,
        attempt_cb=None,
    ) -> dict:
        run_dir = RUNS_DIR / f'Traj_{traj_id}'
        run_dir.mkdir(parents=True, exist_ok=True)

        last_err = ''
        last_code = None
        last_stage = 'setup'
        last_error_type = ''

        runner_code = (
            "import importlib.util, os, sys\n"
            "module_path = os.environ['GROMACS_PIPELINE_PY']\n"
            "module_name = os.environ['GROMACS_MODULE_NAME']\n"
            "spec = importlib.util.spec_from_file_location(module_name, module_path)\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "sys.modules[module_name] = mod\n"
            "spec.loader.exec_module(mod)\n"
        )

        for attempt in range(1, max_attempts + 1):
            if attempt > 1 and attempt_cb is not None:
                attempt_cb(attempt, last_code, last_stage, last_error_type)

            log_path = run_dir / f'{phase}_attempt{attempt}.log'
            module_name = f'gromacs_new_{phase}_traj{traj_id}_a{attempt}_{int(time.time())}'

            env = os.environ.copy()
            env['GROMACS_SPEC_NAME'] = f'Traj_{traj_id}'
            env['GROMACS_PIPELINE_PY'] = str(phase_py)
            env['GROMACS_MODULE_NAME'] = module_name
            env['GROMACS_NTOMP'] = str(GROMACS_NTOMP)
            env['GROMACS_TRAJ_ROOT'] = str(run_dir)
            env['GROMACS_FAST_PYSOFTK'] = '1' if FAST_PYSOFTK else '0'
            env['GROMACS_PYSOFTK_UFF_ITERS'] = str(PYSOFTK_UFF_ITERS)
            env['GROMACS_PYSOFTK_LOCALOPT_STEPS'] = str(PYSOFTK_LOCALOPT_STEPS)
            env['GROMACS_LOCAL_PYSOFTK_ROOT'] = LOCAL_PYSOFTK_ROOT
            env['GROMACS_SHARED_CACHE_ROOT'] = SHARED_CACHE_ROOT
            env['GROMACS_PYSOFTK_MULTICORE'] = '1'
            env['GROMACS_PYSOFTK_INTERNAL_THREADS'] = str(PYSOFTK_INTERNAL_THREADS)
            env['GROMACS_PYSOFTK_NUM_CONFS'] = str(PYSOFTK_NUM_CONFS)
            env['GROMACS_PYSOFTK_OB_WORKERS'] = str(PYSOFTK_OB_WORKERS)
            env['GROMACS_PYSOFTK_SKIP_FINAL_LOCALOPT'] = '1' if PYSOFTK_SKIP_FINAL_LOCALOPT else '0'

            # Per-run overrides for md/analysis and htp-md alignment
            env['GROMACS_PRODUCTION_NS'] = f'{BATCH_PRODUCTION_TOTAL_NS:g}'
            env['GROMACS_PRODUCTION_REPLICAS'] = str(BATCH_PRODUCTION_REPLICAS)
            if BATCH_PRODUCTION_NTOMP is None:
                env.pop('GROMACS_NTOMP_PRODUCTION', None)
            else:
                env['GROMACS_NTOMP_PRODUCTION'] = str(BATCH_PRODUCTION_NTOMP)
            env['GROMACS_PRODUCTION_TCOUPL'] = BATCH_PRODUCTION_TCOUPL
            env['GROMACS_PRODUCTION_TAU_T'] = f'{BATCH_PRODUCTION_TAU_T:g}'
            env['GROMACS_PRODUCTION_BONDED_GPU'] = BATCH_PRODUCTION_BONDED_GPU
            env['GROMACS_GK_OUTPUT_ENABLED'] = BATCH_GK_OUTPUT_ENABLED
            env['GROMACS_GK_FRAME_INTERVAL_PS'] = f'{BATCH_GK_FRAME_INTERVAL_PS:g}'
            env['GROMACS_GK_SAVE_VELOCITIES'] = BATCH_GK_SAVE_VELOCITIES
            env['GROMACS_ANALYSIS_BEGIN_NS'] = f'{BATCH_ANALYSIS_BEGIN_NS:g}'
            env['GROMACS_ANALYSIS_END_NS'] = f'{BATCH_ANALYSIS_END_NS:g}'
            if BATCH_N_CHAINS is None:
                env.pop('GROMACS_N_CHAINS', None)
            else:
                env['GROMACS_N_CHAINS'] = str(BATCH_N_CHAINS)
            env['GROMACS_LI_TFSI_PAIRS'] = str(BATCH_LI_TFSI_PAIRS)
            env['GROMACS_AUTO_UPDATE_N_CHAINS'] = BATCH_AUTO_UPDATE_N_CHAINS
            env['GROMACS_MOLALITY_BASIS'] = BATCH_MOLALITY_BASIS
            env['GROMACS_TFSI_CHARGE_MODEL'] = BATCH_TFSI_CHARGE_MODEL
            if phase == 'analysis':
                env['GROMACS_LI_CHARGE_SCALE'] = f'{BATCH_ANALYSIS_LI_CHARGE_SCALE:g}'
                env['GROMACS_ANION_CHARGE_SCALE'] = f'{BATCH_ANALYSIS_ANION_CHARGE_SCALE:g}'
            else:
                env['GROMACS_LI_CHARGE_SCALE'] = f'{BATCH_LI_CHARGE_SCALE:g}'
                env['GROMACS_ANION_CHARGE_SCALE'] = f'{BATCH_ANION_CHARGE_SCALE:g}'
            env['GROMACS_CLUSTER_CUTOFF_AUTO'] = BATCH_CLUSTER_CUTOFF_AUTO
            env['GROMACS_NVT1_VARIANT'] = BATCH_NVT1_VARIANT
            env['GROMACS_NVT1_SHORT_PS'] = f'{BATCH_NVT1_SHORT_PS:g}'
            env['GROMACS_NVT1_SPLIT_VRESCALE_PS'] = f'{BATCH_NVT1_SPLIT_VRESCALE_PS:g}'
            env['GROMACS_NVT1_SPLIT_NOSEHOOVER_PS'] = f'{BATCH_NVT1_SPLIT_NOSEHOOVER_PS:g}'
            if BATCH_MD_STOP_AFTER_STAGE:
                env['GROMACS_MD_STOP_AFTER_STAGE'] = BATCH_MD_STOP_AFTER_STAGE
            else:
                env.pop('GROMACS_MD_STOP_AFTER_STAGE', None)
            env.pop('GROMACS_FORCE_PRODUCTION_RERUN', None)
            if phase == 'analysis':
                env['GROMACS_CNE_DIFFUSION_MODE'] = BATCH_ANALYSIS_CNE_DIFFUSION_MODE
                env['GROMACS_CNE_CLUSTER_DRAG_EXPONENT'] = f'{BATCH_ANALYSIS_CNE_CLUSTER_DRAG_EXPONENT:g}'
                env['GROMACS_HTPMD_STRICT_MATCH'] = BATCH_HTPMD_STRICT_MATCH

            if phase == 'atomtyping' and attempt > 1:
                recov = _atomtyping_recovery_env(last_error_type, attempt)
                env.update(recov)
                if recov:
                    print(f"[RECOVERY] atomtyping Traj_{traj_id} attempt {attempt}: {recov}")
                topo_dir = run_dir / 'topology'
                shutil.rmtree(topo_dir, ignore_errors=True)
                topo_dir.mkdir(parents=True, exist_ok=True)
                fixed_gro = run_dir / 'md' / 'conf_initial_fixed.gro'
                if fixed_gro.exists():
                    fixed_gro.unlink()

            if phase in ('pysoftk', 'packmol', 'atomtyping', 'analysis'):
                env.setdefault('OMP_NUM_THREADS', '1')
                env.setdefault('OPENBLAS_NUM_THREADS', '1')
                env.setdefault('MKL_NUM_THREADS', '1')
                env.setdefault('NUMEXPR_NUM_THREADS', '1')
                env.setdefault('VECLIB_MAXIMUM_THREADS', '1')

            cmd = [PYTHON, '-u', '-c', runner_code]

            if stage_cb is not None and phase not in ('md', 'analysis'):
                stage_cb('setup')

            t0 = time.time()
            proc = subprocess.Popen(
                cmd,
                cwd=GROMACS_DIR,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )

            out_lines = []
            if proc.stdout is not None:
                for line in proc.stdout:
                    out_lines.append(line)

                    st = _extract_stage_marker(line)
                    if st is None:
                        st = _infer_stage_from_output_line(line)

                    if st is not None and _stage_allowed_for_phase(phase, st) and st != last_stage:
                        last_stage = st
                        if stage_cb is not None:
                            stage_cb(st)

            proc.wait()
            dt = time.time() - t0

            stdout = ''.join(out_lines)
            cmd_str = ' '.join(cmd)
            log_path.write_text(
                f'PHASE: {phase}\n'
                f'CMD: {cmd_str}\n'
                f'RETURN_CODE: {proc.returncode}\n'
                f'ELAPSED_SEC: {dt:.1f}\n'
                f'LAST_STAGE: {last_stage}\n\n'
                f'--- STDOUT ---\n{stdout}\n\n--- STDERR ---\n\n'
            )

            last_code = int(proc.returncode)
            if proc.returncode == 0:
                rec = {
                    'Trajectory ID': int(traj_id),
                    'phase': phase,
                    'status': 'ok',
                    'attempts_used': attempt,
                    'last_return_code': 0,
                    'elapsed_sec': round(dt, 1),
                    'last_stage': last_stage,
                    'error_type': '',
                    'error_tail': '',
                }
                if phase == 'analysis':
                    rec.update(_run_gk_postprocess(traj_id))
                return rec

            tail = (stdout.strip().splitlines()[-20:] if stdout.strip() else ['(no stdout)'])
            last_err = '\n'.join(tail)
            try:
                last_error_type = _classify_phase_failure(phase, stdout, last_stage)
            except Exception as e:
                last_error_type = 'phase_failure_unclassified'
                with log_path.open('a') as f:
                    f.write(f'CLASSIFIER_ERROR: {type(e).__name__}: {e}\n')
            with log_path.open('a') as f:
                f.write(f'ERROR_TYPE: {last_error_type}\n')

            if phase == 'atomtyping' and _is_nonretryable_atomtyping_error(last_error_type):
                break

        return {
            'Trajectory ID': int(traj_id),
            'phase': phase,
            'status': 'failed',
            'attempts_used': attempt,
            'last_return_code': last_code,
            'last_stage': last_stage,
            'error_type': last_error_type,
            'error_tail': last_err,
        }


    RESUME_EXISTING = _as_bool(CFG.get('resume_existing', True), True)
    if FORCE_RESTART:
        RESUME_EXISTING = False


    _MD_STAGE_DIR_CANDIDATES = {
        'md-em': ('em2', 'em'),
        'md-nvt': ('nvt',),
        'md-npt': ('npt',),
        'md-prod': ('production', 'prod'),
    }

    _MDP_KV_RE = re.compile(r'^\s*([A-Za-z0-9_\-]+)\s*=\s*([^;]+)')
    _LOG_STEP_TIME_RE = re.compile(
        r'^\s*Step\s+Time\s*$\r?\n^\s*([0-9]+)\s+([0-9]+(?:\.[0-9]+)?)\s*$',
        re.MULTILINE,
    )


    def _read_text_tail(path: Path, max_bytes: int = 262144) -> str:
        try:
            if not path.exists():
                return ''
            size = path.stat().st_size
            with path.open('rb') as f:
                if size > max_bytes:
                    f.seek(-max_bytes, os.SEEK_END)
                data = f.read()
            return data.decode(errors='ignore')
        except Exception:
            return ''


    def _mdp_time_window_ns(mdp_path: Path):
        txt = _read_text_tail(mdp_path, max_bytes=524288)
        if not txt:
            return (None, 0.0)

        nsteps = None
        dt_ps = None
        tinit_ps = 0.0
        init_step = 0.0

        for raw in txt.splitlines():
            line = raw.split(';', 1)[0].strip()
            if not line:
                continue
            m = _MDP_KV_RE.match(line)
            if not m:
                continue

            key = m.group(1).strip().lower()
            val = m.group(2).strip().split()[0]

            if key == 'nsteps':
                try:
                    nsteps = float(val)
                except Exception:
                    pass
            elif key == 'dt':
                try:
                    dt_ps = float(val)
                except Exception:
                    pass
            elif key == 'tinit':
                try:
                    tinit_ps = float(val)
                except Exception:
                    pass
            elif key in ('init-step', 'init_step'):
                try:
                    init_step = float(val)
                except Exception:
                    pass

        start_ps = tinit_ps
        if dt_ps is not None:
            start_ps = tinit_ps + (init_step * dt_ps)

        total_ns = None
        if nsteps is not None and dt_ps is not None and nsteps > 0 and dt_ps > 0:
            total_ns = float(nsteps * dt_ps / 1000.0)

        return (total_ns, float(max(0.0, start_ps / 1000.0)))


    def _md_log_current_ns(log_path: Path):
        txt = _read_text_tail(log_path)
        if not txt:
            return None

        matches = list(_LOG_STEP_TIME_RE.finditer(txt))
        if not matches:
            return None

        try:
            time_ps = float(matches[-1].group(2))
        except Exception:
            return None

        if time_ps < 0:
            return None
        return float(time_ps / 1000.0)


    def _md_log_current_step(log_path: Path):
        txt = _read_text_tail(log_path)
        if not txt:
            return None

        matches = list(_LOG_STEP_TIME_RE.finditer(txt))
        if not matches:
            return None

        try:
            step = int(float(matches[-1].group(1)))
        except Exception:
            return None

        if step < 0:
            return None
        return step


    def _mdp_nsteps(mdp_path: Path):
        txt = _read_text_tail(mdp_path, max_bytes=524288)
        if not txt:
            return None

        for raw in txt.splitlines():
            line = raw.split(';', 1)[0].strip()
            if not line:
                continue
            m = _MDP_KV_RE.match(line)
            if not m:
                continue
            key = m.group(1).strip().lower()
            if key != 'nsteps':
                continue
            try:
                val = int(float(m.group(2).strip().split()[0]))
                if val > 0:
                    return val
            except Exception:
                pass
        return None


    def _md_stage_dirs(run_dir: Path, stage: str):
        stage_key = (stage or '').strip().lower()
        md_root = run_dir / 'md'
        if stage_key == 'md-nvt':
            dirs = [p for p in md_root.iterdir() if p.is_dir() and p.name.lower().startswith('nvt')] if md_root.exists() else []
            exact = [run_dir / 'md' / name for name in _MD_STAGE_DIR_CANDIDATES.get(stage_key, ())]
            merged = []
            seen = set()
            for p in dirs + exact:
                key = str(p)
                if key not in seen:
                    seen.add(key)
                    merged.append(p)
            return merged
        if stage_key == 'md-npt':
            dirs = [p for p in md_root.iterdir() if p.is_dir() and p.name.lower().startswith('npt')] if md_root.exists() else []
            exact = [run_dir / 'md' / name for name in _MD_STAGE_DIR_CANDIDATES.get(stage_key, ())]
            merged = []
            seen = set()
            for p in dirs + exact:
                key = str(p)
                if key not in seen:
                    seen.add(key)
                    merged.append(p)
            return merged
        names = _MD_STAGE_DIR_CANDIDATES.get(stage_key, ())
        return [run_dir / 'md' / name for name in names]


    def _md_stage_sort_mtime(stage_dir: Path):
        try:
            best = float(stage_dir.stat().st_mtime)
        except Exception:
            best = -1.0
        log_path = stage_dir / f'{stage_dir.name}.log'
        if log_path.exists():
            try:
                best = max(best, float(log_path.stat().st_mtime))
            except Exception:
                pass
        return best


    def _md_stage_scan_dirs(run_dir: Path, stage: str):
        cands = _md_stage_dirs(run_dir, stage)
        if not cands:
            return []

        existing = [sd for sd in cands if sd.exists()]
        if not existing:
            return cands

        existing.sort(key=_md_stage_sort_mtime, reverse=True)
        seen = set(existing)
        return existing + [sd for sd in cands if sd not in seen]


    def _md_stage_active_dir(run_dir: Path, stage: str):
        scan_dirs = _md_stage_scan_dirs(run_dir, stage)
        if not scan_dirs:
            return None
        return scan_dirs[0]


    def _md_stage_timing_ns(stage_dir: Path):
        for mdp_name in (f'{stage_dir.name}.mdp', 'mdout.mdp'):
            mdp_path = stage_dir / mdp_name
            total_ns, start_ns = _mdp_time_window_ns(mdp_path)
            if total_ns is not None and total_ns > 0:
                return total_ns, start_ns
        return None, 0.0


    def _md_stage_total_ns(run_dir: Path, stage: str):
        sd = _md_stage_active_dir(run_dir, stage)
        if sd is None:
            return None
        total_ns, _ = _md_stage_timing_ns(sd)
        if total_ns is not None and total_ns > 0:
            return total_ns
        return None


    def _md_stage_current_ns(run_dir: Path, stage: str):
        sd = _md_stage_active_dir(run_dir, stage)
        if sd is None:
            return None
        log_path = sd / f'{sd.name}.log'
        cur_abs_ns = _md_log_current_ns(log_path)
        if cur_abs_ns is not None:
            return float(cur_abs_ns)
        return None


    def _md_stage_current_rel_ns(run_dir: Path, stage: str):
        sd = _md_stage_active_dir(run_dir, stage)
        if sd is None:
            return None
        total_ns, start_ns = _md_stage_timing_ns(sd)
        log_path = sd / f'{sd.name}.log'
        cur_abs_ns = _md_log_current_ns(log_path)
        if cur_abs_ns is None:
            return None
        try:
            cur_rel_ns = float(cur_abs_ns) - float(start_ns or 0.0)
        except Exception:
            cur_rel_ns = float(cur_abs_ns)
        if cur_rel_ns < 0 and abs(cur_rel_ns) < 1e-6:
            cur_rel_ns = 0.0
        return max(0.0, float(cur_rel_ns))


    def _md_stage_total_steps(run_dir: Path, stage: str):
        sd = _md_stage_active_dir(run_dir, stage)
        if sd is None:
            return None
        for mdp_name in (f'{sd.name}.mdp', 'mdout.mdp'):
            nsteps = _mdp_nsteps(sd / mdp_name)
            if nsteps is not None and nsteps > 0:
                return int(nsteps)
        return None


    def _md_stage_current_step(run_dir: Path, stage: str):
        sd = _md_stage_active_dir(run_dir, stage)
        if sd is None:
            return None
        log_path = sd / f'{sd.name}.log'
        step = _md_log_current_step(log_path)
        if step is not None:
            return int(step)
        return None


    def _md_stage_from_latest_log(run_dir: Path):
        best_stage = None
        best_mtime = -1.0
        for st in ('md-em', 'md-nvt', 'md-npt', 'md-prod'):
            for sd in _md_stage_scan_dirs(run_dir, st):
                log_path = sd / f'{sd.name}.log'
                if not log_path.exists():
                    continue
                try:
                    mt = float(log_path.stat().st_mtime)
                except Exception:
                    continue
                if mt > best_mtime:
                    best_mtime = mt
                    best_stage = st
        return best_stage


    def _production_required_files(run_dir: Path) -> list[Path]:
        req = []
        for replica_idx in range(1, BATCH_PRODUCTION_REPLICAS + 1):
            stage = 'production' if replica_idx == 1 else f'production_rep{replica_idx}'
            stage_dir = run_dir / 'md' / stage
            req.extend([
                stage_dir / f'{stage}.tpr',
                stage_dir / f'{stage}.xtc',
                stage_dir / f'{stage}.gro',
            ])
        return req

    def _phase_complete_by_files(traj_id: int, phase: str) -> bool:
        run_dir = RUNS_DIR / f'Traj_{traj_id}'
        if phase == 'pysoftk':
            req = [
                run_dir / 'structures' / f'Traj_{traj_id}_chain_fix.pdb',
                run_dir / 'structures' / 'tfsi.pdb',
                run_dir / 'structures' / 'li.pdb',
            ]
        elif phase == 'packmol':
            req = [run_dir / 'md' / 'conf_initial.gro']
        elif phase == 'atomtyping':
            req = [run_dir / 'md' / 'conf_initial_fixed.gro', run_dir / 'topology' / 'topol.top']
        elif phase == 'charge_sanity':
            report_path = run_dir / 'topology' / 'charge_sanity_interphase.json'
            return charge_sanity_report_ok(report_path)
        elif phase == 'md':
            req = _production_required_files(run_dir)
        elif phase == 'analysis':
            req = _production_required_files(run_dir) + [
                run_dir / 'analysis' / 'conductivity_summary_htpmd_ref.csv',
            ]
        else:
            return False

        for p in req:
            if (not p.exists()) or p.stat().st_size <= 0:
                return False
        return True


    def _filter_start_phase_prerequisites(tids: list[int], start_phase: str) -> list[int]:
        """Keep only trajectories with the files needed to enter start_phase.

        This avoids sending atomtyping failures into MD/analysis where they
        would fail immediately and clutter the logs.
        """
        phase_order = ['pysoftk', 'packmol', 'atomtyping', 'charge_sanity', 'md', 'analysis']
        if start_phase not in phase_order:
            return tids
        start_idx = phase_order.index(start_phase)
        if start_idx == 0:
            return tids
        prereq_phase = phase_order[start_idx - 1]
        kept = [tid for tid in tids if _phase_complete_by_files(tid, prereq_phase)]
        dropped = len(tids) - len(kept)
        if dropped:
            print(
                f'[{start_phase}] prerequisite filter: keep {len(kept)} / {len(tids)} '
                f'with completed {prereq_phase}; drop {dropped}'
            )
        return kept


    def _make_skipped_rec(traj_id: int, phase: str) -> dict:
        return {
            'Trajectory ID': int(traj_id),
            'phase': phase,
            'status': 'ok',
            'attempts_used': 0,
            'last_return_code': 0,
            'elapsed_sec': 0.0,
            'last_stage': 'skipped-existing',
            'error_tail': '',
        }


    def _read_bytes_if_exists(path: Path):
        try:
            return path.read_bytes() if path.exists() else None
        except Exception:
            return None


    def _write_bytes_if_not_none(path: Path, data):
        if data is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)





    def _clear_outputs_from_phase(traj_id: int, start_phase: str):
        run_dir = RUNS_DIR / f'Traj_{traj_id}'
        if not run_dir.exists():
            return

        phase = (start_phase or '').strip().lower()

        if phase == 'pysoftk':
            shutil.rmtree(run_dir, ignore_errors=True)
            return

        if phase == 'packmol':
            for p in [run_dir / 'packmol', run_dir / 'md', run_dir / 'topology', run_dir / 'analysis']:
                shutil.rmtree(p, ignore_errors=True)
            return

        if phase == 'atomtyping':
            conf_initial = _read_bytes_if_exists(run_dir / 'md' / 'conf_initial.gro')
            shutil.rmtree(run_dir / 'topology', ignore_errors=True)
            shutil.rmtree(run_dir / 'analysis', ignore_errors=True)
            shutil.rmtree(run_dir / 'md', ignore_errors=True)
            _write_bytes_if_not_none(run_dir / 'md' / 'conf_initial.gro', conf_initial)
            return

        if phase == 'charge_sanity':
            conf_initial = _read_bytes_if_exists(run_dir / 'md' / 'conf_initial.gro')
            conf_fixed = _read_bytes_if_exists(run_dir / 'md' / 'conf_initial_fixed.gro')
            try:
                (run_dir / 'topology' / 'charge_sanity_interphase.json').unlink()
            except FileNotFoundError:
                pass
            shutil.rmtree(run_dir / 'analysis', ignore_errors=True)
            shutil.rmtree(run_dir / 'md', ignore_errors=True)
            _write_bytes_if_not_none(run_dir / 'md' / 'conf_initial.gro', conf_initial)
            _write_bytes_if_not_none(run_dir / 'md' / 'conf_initial_fixed.gro', conf_fixed)
            return

        if phase == 'md':
            conf_initial = _read_bytes_if_exists(run_dir / 'md' / 'conf_initial.gro')
            conf_fixed = _read_bytes_if_exists(run_dir / 'md' / 'conf_initial_fixed.gro')
            shutil.rmtree(run_dir / 'analysis', ignore_errors=True)
            shutil.rmtree(run_dir / 'analysis_gk', ignore_errors=True)
            shutil.rmtree(run_dir / 'md', ignore_errors=True)
            _write_bytes_if_not_none(run_dir / 'md' / 'conf_initial.gro', conf_initial)
            _write_bytes_if_not_none(run_dir / 'md' / 'conf_initial_fixed.gro', conf_fixed)
            return

        if phase == 'analysis':
            shutil.rmtree(run_dir / 'analysis', ignore_errors=True)
            return


    def _run_phase_parallel(phase: str, rows_phase, phase_py: Path, workers=None):
        global _RUN_INTERRUPTED

        out = {}
        if not rows_phase:
            return out

        _cleanup_tqdm_instances(f'phase-start:{phase}')
        pbar = tqdm(
            total=len(rows_phase),
            desc=f'Phase {phase}',
            unit='traj',
            leave=TQDM_LEAVE,
            dynamic_ncols=True,
        ) if tqdm is not None else None

        ex = ThreadPoolExecutor(max_workers=workers)
        futs = {}
        try:
            for row in rows_phase:
                tid = int(row['Trajectory ID'])
                grp = str(row['sample_group'])

                def _attempt_cb(attempt: int, prev_rc, prev_stage, prev_err_type='', _tid=tid, _grp=grp):
                    if attempt == 2:
                        print(
                            f"[RETRY] {phase} Traj_{_tid} ({_grp}) entering attempt 2/{MAX_ATTEMPTS} "
                            f"after rc={prev_rc}, stage={prev_stage}, error_type={prev_err_type}"
                        )

                fut = ex.submit(
                    execute_phase_traj,
                    tid,
                    phase,
                    phase_py,
                    MAX_ATTEMPTS,
                    None,
                    _attempt_cb,
                )
                futs[fut] = (tid, grp)

            for fut in as_completed(futs):
                tid, grp = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {
                        'Trajectory ID': int(tid),
                        'phase': phase,
                        'status': 'failed',
                        'attempts_used': 0,
                        'last_return_code': None,
                        'elapsed_sec': 0.0,
                        'last_stage': 'internal',
                        'error_type': 'runner_internal_exception',
                        'error_tail': f'{type(e).__name__}: {e}',
                    }
                out[tid] = rec

                if pbar is not None:
                    pbar.update(1)
                    fails = sum(1 for r in out.values() if r.get('status') != 'ok')
                    pbar.set_postfix_str(f'done={len(out)} fail={fails}')

                if rec.get('status') != 'ok':
                    print(
                        f"[ERROR] {phase} Traj_{tid} ({grp}) "
                        f"return_code={rec.get('last_return_code')} attempts={rec.get('attempts_used')} "
                        f"stage={rec.get('last_stage')} error_type={rec.get('error_type')}"
                    )

        except KeyboardInterrupt:
            _RUN_INTERRUPTED = True
            print(f'[INTERRUPTED] phase {phase}: user interrupt received; stopping phase gracefully.')
            for fut in futs:
                try:
                    fut.cancel()
                except Exception:
                    pass
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                ex.shutdown(wait=False)
            except Exception:
                pass
            return out

        finally:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                try:
                    ex.shutdown(wait=False)
                except Exception:
                    pass
            except Exception:
                pass

            if pbar is not None:
                _close_tqdm_bar(pbar)

            _cleanup_tqdm_instances(f'phase-exit:{phase}')

        return out

    def _run_phase_sequential(phase: str, rows_phase, phase_py: Path):
        global _RUN_INTERRUPTED

        out = {}
        if not rows_phase:
            return out

        _cleanup_tqdm_instances(f'phase-start:{phase}')
        pbar = tqdm(
            total=len(rows_phase),
            desc=f'Phase {phase}',
            unit='traj',
            leave=TQDM_LEAVE,
            dynamic_ncols=True,
        ) if tqdm is not None else None

        try:
            for row in rows_phase:
                tid = int(row['Trajectory ID'])
                grp = str(row['sample_group'])
                run_dir = RUNS_DIR / f'Traj_{tid}'
                use_md_ns_progress = (phase == 'md' and pbar is not None)
                stage_state = {'stage': 'setup'}

                def _set_main_postfix(text: str):
                    if pbar is not None and hasattr(pbar, 'set_postfix_str'):
                        pbar.set_postfix_str(text)

                def _stage_cb(stage: str, _tid=tid, _grp=grp):
                    stage_state['stage'] = stage
                    if not use_md_ns_progress:
                        _set_main_postfix(f'Traj_{_tid} {_grp} | {stage}')

                def _attempt_cb(attempt: int, prev_rc, prev_stage, prev_err_type='', _tid=tid, _grp=grp):
                    if attempt == 2:
                        print(
                            f"[RETRY] {phase} Traj_{_tid} ({_grp}) entering attempt 2/{MAX_ATTEMPTS} "
                            f"after rc={prev_rc}, stage={prev_stage}, error_type={prev_err_type}"
                        )

                if use_md_ns_progress:
                    md_stage_bar = None
                    md_stage_name = None
                    md_stage_subname = None
                    md_stage_mode = None
                    md_total_cache = {}
                    md_stage_offset = {}
                    md_main_fraction = 0.0

                    def _close_md_stage_bar():
                        nonlocal md_stage_bar, md_stage_name, md_stage_subname, md_stage_mode
                        if md_stage_bar is not None:
                            _close_tqdm_bar(md_stage_bar)
                            md_stage_bar = None
                            md_stage_name = None
                            md_stage_subname = None
                            md_stage_mode = None

                    def _refresh_main_md_progress(
                        stage: str,
                        progress_mode: str = '',
                        cur_val=None,
                        total_val=None,
                        unit: str = '',
                        display_stage: str = '',
                    ):
                        nonlocal md_main_fraction
                        if pbar is None:
                            return

                        st = str(stage or '').strip().lower()
                        disp = str(display_stage or st or 'setup').strip()
                        label = f'Traj_{tid} {grp} | {disp}'
                        ratio = None

                        if cur_val is not None and total_val is not None:
                            try:
                                total_f = float(total_val)
                                cur_f = max(0.0, float(cur_val))
                            except Exception:
                                total_f = 0.0
                                cur_f = 0.0

                            if total_f > 0:
                                ratio = min(1.0, max(0.0, cur_f / total_f))
                                pct = 100.0 * ratio
                                if progress_mode == 'ns':
                                    label = f'Traj_{tid} {grp} | {disp} {cur_f:.2f}/{total_f:.2f} {unit} ({pct:.1f}%)'
                                else:
                                    label = (
                                        f'Traj_{tid} {grp} | {disp} '
                                        f'{int(round(cur_f))}/{int(round(total_f))} {unit} ({pct:.1f}%)'
                                    )

                        if ratio is not None:
                            if st == 'md-prod':
                                frac = ratio
                            elif st == 'md-npt':
                                frac = 0.02 + 0.03 * ratio
                            elif st == 'md-nvt':
                                frac = 0.01 + 0.01 * ratio
                            elif st == 'md-em':
                                frac = 0.01 * ratio
                            else:
                                frac = 0.0
                            md_main_fraction = max(md_main_fraction, min(0.99, max(0.0, frac)))
                        elif st == 'md-prod':
                            md_main_fraction = max(md_main_fraction, 0.05)

                        target_n = float(len(out)) + min(0.99, max(0.0, md_main_fraction))
                        if pbar.total is not None:
                            target_n = min(float(pbar.total), target_n)
                        try:
                            if target_n > float(getattr(pbar, 'n', 0.0)) + 1e-9:
                                pbar.n = target_n
                                pbar.refresh()
                        except Exception:
                            pass
                        _set_main_postfix(label)

                    def _update_md_stage_bar(stage: str, *, allow_complete: bool = False):
                        nonlocal md_stage_bar, md_stage_name, md_stage_subname, md_stage_mode
                        st = str(stage or '').strip().lower()
                        if not st.startswith('md-'):
                            _refresh_main_md_progress(st)
                            _close_md_stage_bar()
                            return

                        active_dir = _md_stage_active_dir(run_dir, st)
                        active_name = active_dir.name if active_dir is not None else ''
                        display_stage = f'{st}:{active_name}' if active_name else st

                        total_ns = _md_stage_total_ns(run_dir, st)
                        cur_ns = _md_stage_current_rel_ns(run_dir, st)
                        if total_ns is not None and float(total_ns) > 0:
                            progress_mode = 'ns'
                            total_val = float(total_ns)
                            cur_val = 0.0 if cur_ns is None else float(cur_ns)
                            bar_unit = 'ns'
                        else:
                            progress_mode = 'step'
                            total_val = md_total_cache.get(f'{st}:step')
                            if total_val is None:
                                total_val = _md_stage_total_steps(run_dir, st)
                                md_total_cache[f'{st}:step'] = total_val
                            cur_val = _md_stage_current_step(run_dir, st)
                            bar_unit = 'step'

                        cache_key = f'{st}:{progress_mode}'
                        if progress_mode == 'ns':
                            md_total_cache[cache_key] = total_val
                        else:
                            total_val = md_total_cache.get(cache_key)

                        if st != md_stage_name or active_name != md_stage_subname or progress_mode != md_stage_mode:
                            _close_md_stage_bar()
                            md_stage_offset.pop(st, None)
                            total_for_bar = total_val if (total_val is not None and float(total_val) > 0) else None
                            md_stage_bar = tqdm(
                                total=total_for_bar,
                                desc=f'Traj_{tid} {grp} | {display_stage}',
                                unit=bar_unit,
                                leave=False,
                                dynamic_ncols=True,
                                mininterval=0.2,
                                position=1,
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]',
                            )
                            md_stage_name = st
                            md_stage_subname = active_name
                            md_stage_mode = progress_mode

                        if cur_val is None or md_stage_bar is None:
                            _refresh_main_md_progress(st, display_stage=display_stage)
                            return

                        if progress_mode == 'ns':
                            target = max(0.0, float(cur_val))
                        else:
                            if st not in md_stage_offset:
                                md_stage_offset[st] = float(cur_val)
                            target = max(0.0, float(cur_val) - float(md_stage_offset[st]))

                        if md_stage_bar.total is not None:
                            total = float(md_stage_bar.total)
                            if not allow_complete and total > 0:
                                total = max(0.0, total * 0.99)
                            target = min(target, total)

                        delta = target - float(md_stage_bar.n)
                        if delta > 0:
                            md_stage_bar.update(delta)

                        _refresh_main_md_progress(
                            st,
                            progress_mode,
                            target,
                            total_val,
                            bar_unit,
                            display_stage=display_stage,
                        )

                        if md_stage_bar.total is not None and md_stage_bar.total > 0:
                            total_nominal = float(md_stage_bar.total)
                            pct = (100.0 * float(md_stage_bar.n) / total_nominal) if total_nominal > 0 else 0.0
                            if not allow_complete:
                                pct = min(pct, 99.0)
                            suffix = bar_unit
                            if not allow_complete and pct >= 99.0:
                                suffix = f'{bar_unit} | finalizing'

                            if progress_mode == 'ns':
                                n_disp = float(md_stage_bar.n)
                                t_disp = float(total_nominal)
                                elapsed = float(getattr(md_stage_bar, 'format_dict', {}).get('elapsed') or 0.0)
                                ns_per_day = (n_disp / elapsed * 86400.0) if elapsed > 0 and n_disp > 0 else None
                                speed = f' | {ns_per_day:.1f} ns/day' if ns_per_day is not None else ''
                                md_stage_bar.set_postfix_str(f'{n_disp:.2f}/{t_disp:.2f} {suffix} ({pct:.1f}%){speed}')
                            else:
                                n_disp = int(round(float(md_stage_bar.n)))
                                t_disp = int(round(total_nominal))
                                md_stage_bar.set_postfix_str(f'{n_disp}/{t_disp} {suffix} ({pct:.1f}%)')
                        else:
                            if progress_mode == 'ns':
                                n_disp = float(md_stage_bar.n)
                                elapsed = float(getattr(md_stage_bar, 'format_dict', {}).get('elapsed') or 0.0)
                                ns_per_day = (n_disp / elapsed * 86400.0) if elapsed > 0 and n_disp > 0 else None
                                speed = f' | {ns_per_day:.1f} ns/day' if ns_per_day is not None else ''
                                md_stage_bar.set_postfix_str(f'{n_disp:.2f} ns{speed}')
                            else:
                                md_stage_bar.set_postfix_str(f'{int(round(float(md_stage_bar.n)))} step')

                    ex_one = ThreadPoolExecutor(max_workers=1)
                    fut = ex_one.submit(
                        execute_phase_traj,
                        tid,
                        phase,
                        phase_py,
                        MAX_ATTEMPTS,
                        _stage_cb,
                        _attempt_cb,
                    )

                    def _active_stage_now(final_stage: str = ''):
                        st = str(final_stage or stage_state.get('stage') or 'setup').strip().lower()
                        if st in ('done', 'failed'):
                            return st
                        if st.startswith('md-'):
                            return st
                        inferred = _md_stage_from_latest_log(run_dir)
                        if inferred:
                            return inferred
                        return st

                    try:
                        while not fut.done():
                            stage_now = _active_stage_now()
                            _set_main_postfix(f'Traj_{tid} {grp} | {stage_now}')
                            _update_md_stage_bar(stage_now, allow_complete=False)
                            time.sleep(1.0)

                        rec = fut.result()
                        stage_now = _active_stage_now(str(rec.get('last_stage') or 'done'))
                        _set_main_postfix(f'Traj_{tid} {grp} | {stage_now}')
                        _update_md_stage_bar(stage_now, allow_complete=True)

                    except KeyboardInterrupt:
                        _RUN_INTERRUPTED = True
                        print(f'[INTERRUPTED] phase {phase} traj={tid}: user interrupt received; cleaning up progress bars.')
                        try:
                            fut.cancel()
                        except Exception:
                            pass
                        try:
                            ex_one.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            ex_one.shutdown(wait=False)
                        except Exception:
                            pass
                        raise

                    finally:
                        _close_md_stage_bar()
                        try:
                            ex_one.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            try:
                                ex_one.shutdown(wait=False)
                            except Exception:
                                pass
                        except Exception:
                            pass
                else:
                    rec = execute_phase_traj(
                        tid,
                        phase,
                        phase_py,
                        MAX_ATTEMPTS,
                        _stage_cb if tqdm is not None else None,
                        _attempt_cb,
                    )

                out[tid] = rec

                if pbar is not None:
                    if use_md_ns_progress:
                        try:
                            pbar.n = min(float(pbar.total), float(len(out))) if pbar.total is not None else float(len(out))
                            pbar.refresh()
                        except Exception:
                            pbar.update(1)
                    else:
                        pbar.update(1)
                    fails = sum(1 for r in out.values() if r.get('status') != 'ok')
                    pbar.set_postfix_str(f'done={len(out)} fail={fails}')

                if rec.get('status') != 'ok':
                    print(
                        f"[ERROR] {phase} Traj_{tid} ({grp}) "
                        f"return_code={rec.get('last_return_code')} attempts={rec.get('attempts_used')} "
                        f"stage={rec.get('last_stage')} error_type={rec.get('error_type')}"
                    )

        except KeyboardInterrupt:
            _RUN_INTERRUPTED = True
            print(f'[INTERRUPTED] phase {phase}: user interrupt received; stopping phase gracefully.')
            return out

        finally:
            if pbar is not None:
                _close_tqdm_bar(pbar)
            _cleanup_tqdm_instances(f'phase-exit:{phase}')

        return out

    # ---- prepare ----
    pipeline_built = ensure_pipeline_script(force_rebuild=FORCE_REBUILD_PIPELINE)
    print('Prepared full pipeline script:', pipeline_built)
    phase_scripts = build_phase_scripts(pipeline_built)
    for k, v in phase_scripts.items():
        print(f'Phase script [{k}] = {v}')

    if MAX_TRAJ_TO_RUN is not None:
        work_manifest = manifest_df.head(int(MAX_TRAJ_TO_RUN)).copy()
    else:
        work_manifest = manifest_df.copy()

    rows = [row for _, row in work_manifest.iterrows()]
    print('to run:', len(rows))

    records = {}
    for row in rows:
        tid = int(row['Trajectory ID'])
        grp = str(row['sample_group'])
        records[tid] = {
            'Trajectory ID': tid,
            'sample_group': grp,
            SIGMA_REF_COL: float(row[SIGMA_REF_COL]),
            TPLUS_REF_COL: float(row[TPLUS_REF_COL]),
            DIFF_LI_REF_COL: float(row[DIFF_LI_REF_COL]) if pd.notna(row[DIFF_LI_REF_COL]) else np.nan,
            DIFF_AN_REF_COL: float(row[DIFF_AN_REF_COL]) if pd.notna(row[DIFF_AN_REF_COL]) else np.nan,
        }


    def _rows_from_tids(tids):
        tid_set = set(int(x) for x in tids)
        return [r for r in rows if int(r['Trajectory ID']) in tid_set]


    def _merge_phase_result(phase: str, phase_out: dict):
        for tid, rec in phase_out.items():
            records[tid][f'{phase}_status'] = rec.get('status')
            records[tid][f'{phase}_attempts'] = int(rec.get('attempts_used', 0) or 0)
            records[tid][f'{phase}_last_stage'] = rec.get('last_stage')
            records[tid][f'{phase}_last_return_code'] = rec.get('last_return_code')
            records[tid][f'{phase}_error_type'] = rec.get('error_type')
            records[tid][f'{phase}_error_tail'] = rec.get('error_tail')
            if phase == 'analysis':
                for key in ('gk_status', 'gk_returncode', 'gk_mode', 'gk_error_tail', 'gk_summary_json'):
                    if key in rec:
                        records[tid][key] = rec.get(key)


    PHASE_ORDER = ['pysoftk', 'packmol', 'atomtyping', 'charge_sanity', 'md', 'analysis']
    PHASE_MODE = {
        'pysoftk': 'parallel',
        'packmol': 'parallel',
        'atomtyping': 'parallel',
        'charge_sanity': 'parallel',
        'md': 'sequential',
        'analysis': 'parallel',
    }
    _start_phase_cfg = CFG.get('start_phase', BATCH_START_PHASE_DEFAULT)
    START_PHASE = str(_start_phase_cfg if _start_phase_cfg is not None else BATCH_START_PHASE_DEFAULT).strip().lower()
    if START_PHASE not in PHASE_ORDER:
        raise ValueError(f"Invalid START_PHASE={START_PHASE}. valid={PHASE_ORDER}")

    if START_PHASE == 'analysis' and (not FORCE_RERUN_FROM_START_PHASE):
        FORCE_RERUN_FROM_START_PHASE = True
        print('[analysis-only] auto-set FORCE_RERUN_FROM_START_PHASE=True to avoid resume-skip on existing analysis outputs')

    resume_existing_effective = RESUME_EXISTING and (not FORCE_RERUN_FROM_START_PHASE)

    print('START_PHASE =', START_PHASE, '(from CFG)')
    _run_n_chains_label = 'auto' if BATCH_N_CHAINS is None else str(BATCH_N_CHAINS)

    print(
        'run_overrides:',
        f'n_chains={_run_n_chains_label}',
        f'li_tfsi_pairs={BATCH_LI_TFSI_PAIRS}',
        f'auto_update_n_chains={BATCH_AUTO_UPDATE_N_CHAINS}',
        f'molality_basis={BATCH_MOLALITY_BASIS}',
        f'tfsi_charge_model={BATCH_TFSI_CHARGE_MODEL}',
        f'shared_cache_root={SHARED_CACHE_ROOT}',
        f'production_total_ns={BATCH_PRODUCTION_TOTAL_NS:g}',
        f'production_replicas={BATCH_PRODUCTION_REPLICAS}',
        f'production_ntomp={BATCH_PRODUCTION_NTOMP if BATCH_PRODUCTION_NTOMP is not None else "auto"}',
        f'production_tcoupl={BATCH_PRODUCTION_TCOUPL}',
        f'production_tau_t={BATCH_PRODUCTION_TAU_T:g}',
        f'production_bonded_gpu={BATCH_PRODUCTION_BONDED_GPU}',
        f'gk_output_enabled={BATCH_GK_OUTPUT_ENABLED}',
        f'gk_frame_interval_ps={BATCH_GK_FRAME_INTERVAL_PS:g}',
        f'gk_save_velocities={BATCH_GK_SAVE_VELOCITIES}',
        f'run_gk_analysis={int(BATCH_RUN_GK_ANALYSIS)}',
        f'gk_analysis_mode={BATCH_GK_ANALYSIS_MODE}',
        f'gk_analysis_group={BATCH_GK_ANALYSIS_GROUP}',
        f'gk_analysis_begin_ns={BATCH_GK_ANALYSIS_BEGIN_NS:g}',
        f'gk_analysis_end_ns={BATCH_GK_ANALYSIS_END_NS if BATCH_GK_ANALYSIS_END_NS is not None else "auto"}',
        f'analysis_window_ns={BATCH_ANALYSIS_BEGIN_NS:g}~{BATCH_ANALYSIS_END_NS:g}',
        f'li_scale={BATCH_LI_CHARGE_SCALE:g}',
        f'anion_scale={BATCH_ANION_CHARGE_SCALE:g}',
        f'analysis_li_scale={BATCH_ANALYSIS_LI_CHARGE_SCALE:g}',
        f'analysis_anion_scale={BATCH_ANALYSIS_ANION_CHARGE_SCALE:g}',
        f'cluster_cutoff_auto={BATCH_CLUSTER_CUTOFF_AUTO}',
        f'htpmd_strict_match={BATCH_HTPMD_STRICT_MATCH}',
        f'analysis_cne_mode={BATCH_ANALYSIS_CNE_DIFFUSION_MODE}',
        f'analysis_cne_drag_exp={BATCH_ANALYSIS_CNE_CLUSTER_DRAG_EXPONENT:g}',
        f'nvt1_variant={BATCH_NVT1_VARIANT}',
        f'nvt1_short_ps={BATCH_NVT1_SHORT_PS:g}',
        f'nvt1_split_vrescale_ps={BATCH_NVT1_SPLIT_VRESCALE_PS:g}',
        f'nvt1_split_nosehoover_ps={BATCH_NVT1_SPLIT_NOSEHOOVER_PS:g}',
        f'md_stop_after_stage={BATCH_MD_STOP_AFTER_STAGE or "none"}',
        f'pysoftk_workers={PYSOFTK_PHASE_WORKERS}',
        f'packmol_workers={PACKMOL_PHASE_WORKERS}',
        f'atomtyping_workers={ATOMTYPING_PHASE_WORKERS}',
        f'charge_sanity_workers={CHARGE_SANITY_PHASE_WORKERS}',
        f'analysis_workers={ANALYSIS_PHASE_WORKERS}',
        f'pysoftk_threads={PYSOFTK_INTERNAL_THREADS}',
        f'pysoftk_num_confs={PYSOFTK_NUM_CONFS}',
        f'pysoftk_ob_workers={PYSOFTK_OB_WORKERS}',
        f'pysoftk_skip_final_localopt={int(PYSOFTK_SKIP_FINAL_LOCALOPT)}',
    )
    print(
        'resume_policy:',
        f'FORCE_RESTART={FORCE_RESTART}',
        f'RESUME_EXISTING={RESUME_EXISTING}',
        f'FORCE_RERUN_FROM_START_PHASE={FORCE_RERUN_FROM_START_PHASE}',
        f'effective_resume={resume_existing_effective}',
        f'cfg_force_rerun_from_start_phase={FORCE_RERUN_FROM_START_PHASE}',
    )

    eligible_tids = [int(r['Trajectory ID']) for r in rows]
    eligible_tids = _filter_start_phase_prerequisites(eligible_tids, START_PHASE)
    if FORCE_RERUN_FROM_START_PHASE:
        for tid in eligible_tids:
            _clear_outputs_from_phase(tid, START_PHASE)
        print(f'force-rerun: cleared outputs from {START_PHASE} for {len(eligible_tids)} trajectories')

    start_idx = PHASE_ORDER.index(START_PHASE)
    for phase in PHASE_ORDER[start_idx:]:
        phase_rows_all = _rows_from_tids(eligible_tids)
        out_prefill = {}
        if resume_existing_effective:
            phase_rows = []
            for row in phase_rows_all:
                tid = int(row['Trajectory ID'])
                if _phase_complete_by_files(tid, phase):
                    out_prefill[tid] = _make_skipped_rec(tid, phase)
                else:
                    phase_rows.append(row)
        else:
            phase_rows = phase_rows_all

        if out_prefill:
            print(f'[{phase}] resume-skip {len(out_prefill)} already complete traj')

        if PHASE_MODE[phase] == 'parallel':
            phase_workers = PHASE_WORKERS.get(phase, PHASE_CPU_WORKERS)
            out_run = _run_phase_parallel(phase, phase_rows, phase_scripts[phase], workers=phase_workers)
        else:
            out_run = _run_phase_sequential(phase, phase_rows, phase_scripts[phase])

        if _RUN_INTERRUPTED:
            print(f'[{phase}] interrupted by user; stopping remaining phases.')
            break

        out = {**out_prefill, **out_run}
        _merge_phase_result(phase, out)
        eligible_tids = [tid for tid in eligible_tids if out.get(tid, {}).get('status') == 'ok']
        print(f'[{phase}] success {len(eligible_tids)} / {len(phase_rows_all)} (run={len(phase_rows)} skip={len(out_prefill)})')

    # finalize records
    for tid, rec in records.items():
        phases = ['pysoftk', 'packmol', 'atomtyping', 'charge_sanity', 'md', 'analysis']
        rec['attempts_used'] = int(sum(int(rec.get(f'{p}_attempts', 0) or 0) for p in phases))
        rec['error_phase'] = np.nan
        rec['error_type'] = np.nan
        rec['error_tail'] = np.nan

        if rec.get('analysis_status') == 'ok':
            rec['status'] = 'ok'
            try:
                pred = read_prediction_from_analysis(int(tid))
            except Exception:
                pred = {
                    'sigma_cNE_htpmd_S_cm_pred': np.nan,
                    'sigma_NE_htpmd_S_cm_pred': np.nan,
                    'used_sigma_ne_fallback': 0,
                    'sigma_pred_source': 'missing',
                    'sigma_eval_mode_pred': 'missing',
                    'D_Li_cm2s_pred': np.nan,
                    'D_an_cm2s_pred': np.nan,
                    'tplus_NE_pred': np.nan,
                    'c_tn_htpmd_pred': np.nan,
                    'analysis_csv': np.nan,
                }
            rec.update(pred)
        else:
            rec['status'] = 'failed'
            rec['sigma_cNE_htpmd_S_cm_pred'] = np.nan
            rec['sigma_NE_htpmd_S_cm_pred'] = np.nan
            rec['used_sigma_ne_fallback'] = 0
            rec['sigma_pred_source'] = 'missing'
            rec['sigma_eval_mode_pred'] = 'missing'
            rec['D_Li_cm2s_pred'] = np.nan
            rec['D_an_cm2s_pred'] = np.nan
            rec['tplus_NE_pred'] = np.nan
            rec['c_tn_htpmd_pred'] = np.nan
            rec['analysis_csv'] = np.nan

            failed_phase = None
            for p in phases:
                st = rec.get(f'{p}_status')
                if pd.isna(st):
                    failed_phase = p
                    break
                if str(st).lower() not in ('ok', 'cached'):
                    failed_phase = p
                    break
            if failed_phase is None:
                failed_phase = 'analysis'
            rec['error_phase'] = failed_phase
            rec['error_type'] = rec.get(f'{failed_phase}_error_type')
            rec['error_tail'] = rec.get(f'{failed_phase}_error_tail')

        rec.setdefault('gk_status', np.nan)
        rec.setdefault('gk_returncode', np.nan)
        rec.setdefault('gk_mode', np.nan)
        rec.setdefault('gk_error_tail', np.nan)
        rec.setdefault('gk_summary_json', np.nan)

    if BATCH_RUN_GK_ANALYSIS and GK_COLLECT_PY.exists():
        collect_cmd = [
            PYTHON,
            str(GK_COLLECT_PY),
            '--root',
            str(OUT_DIR),
            '--sigma-unit',
            BATCH_GK_SIGMA_UNIT,
        ]
        collect = subprocess.run(
            collect_cmd,
            cwd=OUT_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        gk_collect_log = RESULTS_DIR / 'gk_collect.log'
        gk_collect_log.write_text(
            f'CMD: {" ".join(collect_cmd)}\n'
            f'RETURN_CODE: {collect.returncode}\n\n'
            f'--- STDOUT ---\n{collect.stdout or ""}\n'
            f'--- STDERR ---\n{collect.stderr or ""}\n'
        )
        print(f'[gk-collect] rc={collect.returncode} log={gk_collect_log}')

    run_df = pd.DataFrame([records[k] for k in sorted(records.keys())]).reset_index(drop=True)
    run_csv = RESULTS_DIR / 'run_results.csv'
    if run_df.empty:
        print(f'[dry-run] no trajectories selected; not overwriting {run_csv}')
    else:
        run_df.to_csv(run_csv, index=False)
        print(f'Saved: {run_csv}')
    run_df.head(20)

    return run_df


def summarize_batch_results(run_df: pd.DataFrame | None = None, *, config: GromacsBatchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR = config.results_dir
    SIGMA_REF_COL = config.sigma_ref_col
    TPLUS_REF_COL = config.tplus_ref_col
    DIFF_LI_REF_COL = config.diff_li_ref_col
    DIFF_AN_REF_COL = config.diff_an_ref_col

    if run_df is None:
        run_csv = RESULTS_DIR / 'run_results.csv'
        if not run_csv.exists():
            raise FileNotFoundError(f'run_results.csv not found: {run_csv}')
        run_df = pd.read_csv(run_csv)
    else:
        run_df = run_df.copy()

    # Metrics: conductivity MA(log)E + transference MAE + diffusivity MA(log)E (pure cNE only)
    run_csv = RESULTS_DIR / 'run_results.csv'
    if not run_csv.exists():
        raise FileNotFoundError('run_results.csv not found. Run previous cell first.')
    run_df = pd.read_csv(run_csv)

    metric_df = run_df.copy()
    if 'sigma_eval_mode_pred' in metric_df.columns:
        metric_df['sigma_eval_mode'] = metric_df['sigma_eval_mode_pred'].fillna('pure_cNE').astype(str)
        metric_df.loc[metric_df['sigma_eval_mode'].str.len() == 0, 'sigma_eval_mode'] = 'pure_cNE'
    elif 'sigma_eval_mode' not in metric_df.columns:
        metric_df['sigma_eval_mode'] = 'pure_cNE'

    _required_cols = [
        'Trajectory ID', 'sample_group', 'status', 'attempts_used',
        'sigma_cNE_htpmd_S_cm_pred', 'sigma_NE_htpmd_S_cm_pred', 'used_sigma_ne_fallback', 'sigma_pred_source', 'sigma_eval_mode',
        'tplus_NE_pred', 'c_tn_htpmd_pred', 'D_Li_cm2s_pred', 'D_an_cm2s_pred',
        SIGMA_REF_COL, TPLUS_REF_COL, DIFF_LI_REF_COL, DIFF_AN_REF_COL,
    ]
    for _c in _required_cols:
        if _c not in metric_df.columns:
            metric_df[_c] = np.nan

    for _c in [
        'sigma_cNE_htpmd_S_cm_pred', 'sigma_NE_htpmd_S_cm_pred', 'used_sigma_ne_fallback',
        'tplus_NE_pred', 'c_tn_htpmd_pred', 'D_Li_cm2s_pred', 'D_an_cm2s_pred',
        SIGMA_REF_COL, TPLUS_REF_COL, DIFF_LI_REF_COL, DIFF_AN_REF_COL,
    ]:
        metric_df[_c] = pd.to_numeric(metric_df[_c], errors='coerce')

    if metric_df['tplus_NE_pred'].notna().sum() == 0 and metric_df['c_tn_htpmd_pred'].notna().sum() > 0:
        metric_df['tplus_NE_pred'] = metric_df['c_tn_htpmd_pred']

    metric_df['ma_loge_sigma'] = np.nan
    metric_df['mae_tplus'] = np.nan
    metric_df['ma_loge_d_li'] = np.nan
    metric_df['ma_loge_d_tfsi'] = np.nan

    m_sigma = (
        np.isfinite(metric_df['sigma_cNE_htpmd_S_cm_pred'])
        & np.isfinite(metric_df[SIGMA_REF_COL])
        & (metric_df['sigma_cNE_htpmd_S_cm_pred'] > 0)
        & (metric_df[SIGMA_REF_COL] > 0)
    )
    metric_df.loc[m_sigma, 'ma_loge_sigma'] = np.abs(
        np.log10(metric_df.loc[m_sigma, 'sigma_cNE_htpmd_S_cm_pred'])
        - np.log10(metric_df.loc[m_sigma, SIGMA_REF_COL])
    )

    m_tn = np.isfinite(metric_df['tplus_NE_pred']) & np.isfinite(metric_df[TPLUS_REF_COL])
    metric_df.loc[m_tn, 'mae_tplus'] = np.abs(
        metric_df.loc[m_tn, 'tplus_NE_pred'] - metric_df.loc[m_tn, TPLUS_REF_COL]
    )

    m_dli = (
        np.isfinite(metric_df['D_Li_cm2s_pred'])
        & np.isfinite(metric_df[DIFF_LI_REF_COL])
        & (metric_df['D_Li_cm2s_pred'] > 0)
        & (metric_df[DIFF_LI_REF_COL] > 0)
    )
    metric_df.loc[m_dli, 'ma_loge_d_li'] = np.abs(
        np.log10(metric_df.loc[m_dli, 'D_Li_cm2s_pred'])
        - np.log10(metric_df.loc[m_dli, DIFF_LI_REF_COL])
    )

    m_dan = (
        np.isfinite(metric_df['D_an_cm2s_pred'])
        & np.isfinite(metric_df[DIFF_AN_REF_COL])
        & (metric_df['D_an_cm2s_pred'] > 0)
        & (metric_df[DIFF_AN_REF_COL] > 0)
    )
    metric_df.loc[m_dan, 'ma_loge_d_tfsi'] = np.abs(
        np.log10(metric_df.loc[m_dan, 'D_an_cm2s_pred'])
        - np.log10(metric_df.loc[m_dan, DIFF_AN_REF_COL])
    )

    per_traj_cols = [
        'Trajectory ID', 'sample_group', 'status', 'attempts_used', 'sigma_eval_mode',
        SIGMA_REF_COL, 'sigma_cNE_htpmd_S_cm_pred', 'sigma_NE_htpmd_S_cm_pred', 'used_sigma_ne_fallback', 'sigma_pred_source', 'ma_loge_sigma',
        TPLUS_REF_COL, 'tplus_NE_pred', 'c_tn_htpmd_pred', 'mae_tplus',
        DIFF_LI_REF_COL, 'D_Li_cm2s_pred', 'ma_loge_d_li',
        DIFF_AN_REF_COL, 'D_an_cm2s_pred', 'ma_loge_d_tfsi',
    ]
    per_traj_cols = [c for c in per_traj_cols if c in metric_df.columns]
    per_traj = metric_df[per_traj_cols].sort_values(['sample_group', 'Trajectory ID'])


    def _summary_row(name: str, sub: pd.DataFrame) -> dict:
        mode_series = sub.loc[sub['ma_loge_sigma'].notna(), 'sigma_eval_mode'] if 'sigma_eval_mode' in sub.columns else pd.Series(dtype=object)
        modes = [str(x) for x in mode_series.dropna().unique() if str(x)]
        sigma_eval_mode = modes[0] if len(modes) == 1 else ('mixed' if modes else 'unknown')
        return {
            'group': name,
            'sigma_eval_mode': sigma_eval_mode,
            'n_total': int(len(sub)),
            'n_sigma': int(sub['ma_loge_sigma'].notna().sum()),
            'n_tplus': int(sub['mae_tplus'].notna().sum()),
            'n_d_li': int(sub['ma_loge_d_li'].notna().sum()),
            'n_d_tfsi': int(sub['ma_loge_d_tfsi'].notna().sum()),
            'MA_logE_sigma': float(sub['ma_loge_sigma'].mean(skipna=True)),
            'MedAE_logE_sigma': float(sub['ma_loge_sigma'].median(skipna=True)),
            'MAE_tplus_NE': float(sub['mae_tplus'].mean(skipna=True)),
            'MedAE_tplus_NE': float(sub['mae_tplus'].median(skipna=True)),
            'MA_logE_D_Li': float(sub['ma_loge_d_li'].mean(skipna=True)),
            'MedAE_logE_D_Li': float(sub['ma_loge_d_li'].median(skipna=True)),
            'MA_logE_D_TFSI': float(sub['ma_loge_d_tfsi'].mean(skipna=True)),
            'MedAE_logE_D_TFSI': float(sub['ma_loge_d_tfsi'].median(skipna=True)),
        }


    summary_rows = []
    for grp, sub in metric_df.groupby('sample_group'):
        summary_rows.append(_summary_row(grp, sub))
    summary_rows.append(_summary_row('ALL', metric_df))

    summary_df = pd.DataFrame(summary_rows)

    per_traj_csv = RESULTS_DIR / 'per_traj_eval.csv'
    summary_csv = RESULTS_DIR / 'metrics_summary.csv'
    per_traj.to_csv(per_traj_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f'Saved: {per_traj_csv}')
    print(f'Saved: {summary_csv}')

    return per_traj, summary_df
