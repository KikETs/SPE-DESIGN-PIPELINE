
# ===== Phase Common Setup =====
# =========================
# Cell 1) Spec & Utils
# =========================
from __future__ import annotations

import os, re, shlex, shutil, subprocess, warnings, math, random, sys
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional, Iterable, Tuple, List, Dict
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# RDKit / PySoftK
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
_LOCAL_PYSOFTK_ROOT = os.environ.get("GROMACS_LOCAL_PYSOFTK_ROOT", "").strip()
_pysoftk_candidates = []
if _LOCAL_PYSOFTK_ROOT:
    _pysoftk_candidates.append(Path(_LOCAL_PYSOFTK_ROOT).expanduser().resolve())
_pysoftk_candidates.extend([_THIS_DIR.parent, Path.cwd().resolve()])
_seen_pysoftk_candidates = set()
for _cand in _pysoftk_candidates:
    try:
        _cand_resolved = _cand.resolve()
    except Exception:
        continue
    if _cand_resolved in _seen_pysoftk_candidates:
        continue
    _seen_pysoftk_candidates.add(_cand_resolved)
    if (_cand_resolved / "pysoftk" / "__init__.py").exists():
        if str(_cand_resolved) not in sys.path:
            sys.path.insert(0, str(_cand_resolved))
        print(f"[pysoftk-local] using vendored pysoftk from {_cand_resolved / 'pysoftk'}")
        break

from pysoftk.linear_polymer.linear_polymer import Lp
from pysoftk.format_printers.format_mol import Fmt

# --- constants ---
NA    = 6.02214076e23
KB    = 1.380649e-23
E_CHG = 1.602176634e-19

GMX = os.environ.get("GMX", "gmx")
GPU_AVAILABLE = shutil.which("nvidia-smi") is not None

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def run(cmd: Iterable[str], *, cwd: Optional[Path]=None, input_text: Optional[str]=None,
        check: bool=True, capture_output: bool=False) -> subprocess.CompletedProcess:
    cmd = [str(x) for x in cmd]
    log("$ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, input=input_text, text=True,
                         capture_output=capture_output, check=False)
    if check and res.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    return res

def gmx_cmd(sub: str, args: Iterable[str], cwd: Path, input_text: Optional[str]=None,
            check: bool=True, capture_output: bool=False) -> subprocess.CompletedProcess:
    return run([GMX, sub, *map(str, args)], cwd=cwd, input_text=input_text, check=check, capture_output=capture_output)

def check_dependency(name: str, exe: str) -> bool:
    path = shutil.which(exe)
    if path is None:
        warnings.warn(f"{name}({exe}) 미탐지. 해당 단계는 수동 처리 필요.")
        return False
    log(f"{name}: {path}")
    return True

@dataclass
class SystemSpec:
    name: str = "Traj_27710"
    psmiles: str = ""
    placeholder: str = "Br"
    n_repeat: int = 0
    n_chains: int = 31
    li_tfsi_pairs: int = 100

    # optional: molality control
    target_molality: Optional[float] = 1.50
    auto_update_n_chains: bool = True
    molality_basis: str = "mixture"  # mixture | polymer

    # cluster/cNE analysis
    cluster_cutoff_nm: float = 0.28      # 2.8 Å
    cluster_stride_ps: float = 0.0
    cluster_persistence_threshold_ps: float = 20.0
    cluster_max_cluster: int = 101

    # charge scaling
    li_charge_scale: float = 0.7
    anion_charge_scale: Optional[float] = None
    tfsi_charge_model: str = "lammps_fq07"

    # packing/box
    density_guess: float = 0.6
    target_density_g_cm3: float = 1.2
    box_length_nm: Optional[float] = 50  # if None -> estimated
    packmol_tolerance: float = 3.5
    packmol_maxit: int = 40000
    packmol_nloop: int = 2000
    packmol_movefrac: float = 0.5
    packmol_movebadrandom: bool = True
    packmol_seed: int = 456789
    packmol_n_seeds: int = 3
    packmol_seed_list: Optional[List[int]] = None
    packmol_min_li_tfsi_dist_A: float = 3.4
    packmol_score_atom_prefixes: Optional[List[str]] = None
    packmol_split_ions: bool = True
    packmol_split_axis: str = "z"
    packmol_li_sphere_frac: float = 0.35
    packmol_tfsi_sphere_frac: float = 0.45

    polymer_resname_coords: str = "POL"
    cation_resname_coords: str = "LI"
    anion_resname_coords: str = "TFSI"

    anion_smiles: str = "[N-](S(=O)(=O)C(F)(F)F)(S(=O)(=O)C(F)(F)F)"

    # MD
    temperature_equil: float = 353.0
    temperature_high: float = 453.0
    temperature_prod: float = 353.0
    pressure_bar: float = 1.0
    dt_fs: float = 2.0
    nvt_ns: float = 5.0
    npt0_ns: float = 5.0
    npt1_ns: float = 5.0
    npt2_ns: float = 40.0
    nvteq_ns: float = 5.0
    production_ns: float = 20.0
    nstxout_compressed: int = 500
    nstvout: int = 0
    nstenergy: int = 10000
    gk_output_enabled: bool = False
    gk_frame_interval_ps: float = 1.0
    gk_save_velocities: bool = False

    # NVT1 A/B options
    nvt1_variant: str = "baseline"  # baseline | short | split
    nvt1_short_ps: float = 200.0
    nvt1_split_vrescale_ps: float = 100.0
    nvt1_split_nosehoover_ps: float = 100.0

    analysis_begin_ns: float = 10.0
    analysis_end_ns: float = 20.0

    gpu_id: str = "0"
    workspace: Path = field(default_factory=lambda: Path("peo_litfsi_gmx"))

    def dt_ps(self) -> float: return self.dt_fs * 1e-3
    def ns_to_steps(self, ns: float) -> int: return int(round((ns*1_000_000)/self.dt_fs))

spec = SystemSpec()
_spec_name_override = os.environ.get('GROMACS_SPEC_NAME')
if _spec_name_override:
    spec.name = _spec_name_override
_FAST_PYSOFTK = os.environ.get('GROMACS_FAST_PYSOFTK', '1') != '0'
_PYSOFTK_UFF_ITERS = int(os.environ.get('GROMACS_PYSOFTK_UFF_ITERS', '600' if _FAST_PYSOFTK else '2000'))
_PYSOFTK_LOCALOPT_STEPS = int(os.environ.get('GROMACS_PYSOFTK_LOCALOPT_STEPS', '150' if _FAST_PYSOFTK else '500'))
_PYSOFTK_SKIP_FINAL_LOCALOPT = os.environ.get('GROMACS_PYSOFTK_SKIP_FINAL_LOCALOPT', '0').strip().lower() not in ('0', 'false', 'no', 'off')
_molality_basis = os.environ.get('GROMACS_MOLALITY_BASIS')
if _molality_basis:
    spec.molality_basis = _molality_basis.strip().lower()
_FAST_PYSOFTK = os.environ.get('GROMACS_FAST_PYSOFTK', '1') != '0'
_PYSOFTK_UFF_ITERS = int(os.environ.get('GROMACS_PYSOFTK_UFF_ITERS', '600' if _FAST_PYSOFTK else '2000'))
_PYSOFTK_LOCALOPT_STEPS = int(os.environ.get('GROMACS_PYSOFTK_LOCALOPT_STEPS', '150' if _FAST_PYSOFTK else '500'))
_PYSOFTK_SKIP_FINAL_LOCALOPT = os.environ.get('GROMACS_PYSOFTK_SKIP_FINAL_LOCALOPT', '0').strip().lower() not in ('0', 'false', 'no', 'off')
_n_chains = os.environ.get("GROMACS_N_CHAINS")
if _n_chains:
    spec.n_chains = max(1, int(_n_chains))
_li_tfsi_pairs = os.environ.get("GROMACS_LI_TFSI_PAIRS")
if _li_tfsi_pairs:
    spec.li_tfsi_pairs = max(1, int(_li_tfsi_pairs))
_auto_update_n_chains = os.environ.get("GROMACS_AUTO_UPDATE_N_CHAINS")
if _auto_update_n_chains:
    spec.auto_update_n_chains = _auto_update_n_chains.strip().lower() not in ("0", "false", "no", "off")
_molality_basis = os.environ.get("GROMACS_MOLALITY_BASIS")
if _molality_basis:
    spec.molality_basis = _molality_basis.strip().lower()
_li_scale = os.environ.get("GROMACS_LI_CHARGE_SCALE")
if _li_scale:
    spec.li_charge_scale = float(_li_scale)
_an_scale = os.environ.get("GROMACS_ANION_CHARGE_SCALE")
if _an_scale:
    spec.anion_charge_scale = float(_an_scale)
_tfsi_charge_model = os.environ.get("GROMACS_TFSI_CHARGE_MODEL")
if _tfsi_charge_model:
    spec.tfsi_charge_model = _tfsi_charge_model.strip().lower()
_production_ns = os.environ.get("GROMACS_PRODUCTION_NS")
if _production_ns:
    spec.production_ns = float(_production_ns)
_gk_output_enabled = os.environ.get("GROMACS_GK_OUTPUT_ENABLED")
if _gk_output_enabled:
    spec.gk_output_enabled = _gk_output_enabled.strip().lower() not in ("0", "false", "no", "off")
_gk_frame_interval_ps = os.environ.get("GROMACS_GK_FRAME_INTERVAL_PS")
if _gk_frame_interval_ps:
    spec.gk_frame_interval_ps = float(_gk_frame_interval_ps)
_gk_save_velocities = os.environ.get("GROMACS_GK_SAVE_VELOCITIES")
if _gk_save_velocities:
    spec.gk_save_velocities = _gk_save_velocities.strip().lower() not in ("0", "false", "no", "off")
_analysis_begin_ns = os.environ.get("GROMACS_ANALYSIS_BEGIN_NS")
if _analysis_begin_ns:
    spec.analysis_begin_ns = float(_analysis_begin_ns)
_analysis_end_ns = os.environ.get("GROMACS_ANALYSIS_END_NS")
if _analysis_end_ns:
    spec.analysis_end_ns = float(_analysis_end_ns)
_nvt1_variant = os.environ.get("GROMACS_NVT1_VARIANT")
if _nvt1_variant:
    spec.nvt1_variant = _nvt1_variant.strip().lower()
_nvt1_short_ps = os.environ.get("GROMACS_NVT1_SHORT_PS")
if _nvt1_short_ps:
    spec.nvt1_short_ps = float(_nvt1_short_ps)
_nvt1_split_vrescale_ps = os.environ.get("GROMACS_NVT1_SPLIT_VRESCALE_PS")
if _nvt1_split_vrescale_ps:
    spec.nvt1_split_vrescale_ps = float(_nvt1_split_vrescale_ps)
_nvt1_split_nosehoover_ps = os.environ.get("GROMACS_NVT1_SPLIT_NOSEHOOVER_PS")
if _nvt1_split_nosehoover_ps:
    spec.nvt1_split_nosehoover_ps = float(_nvt1_split_nosehoover_ps)
# --- optional input table update ---
# This workflow may use simulation-trajectory-aggregate.csv as a candidate/input
# table. It is not required reference data for analysis after MD has completed.
INPUT_CSV_ENV = os.environ.get("GROMACS_INPUT_CSV", os.environ.get("GROMACS_REF_CSV", "")).strip()
_input_csv_candidates = []
if INPUT_CSV_ENV:
    _input_csv_candidates.append(Path(INPUT_CSV_ENV).expanduser())
_input_csv_candidates.append(Path("simulation-trajectory-aggregate.csv"))
_traj_root_for_input = os.environ.get("GROMACS_TRAJ_ROOT", "").strip()
if _traj_root_for_input:
    _traj_root_path = Path(_traj_root_for_input).expanduser().resolve()
    _input_csv_candidates.extend([
        _traj_root_path.parent / "simulation-trajectory-aggregate.csv",
        _traj_root_path.parent.parent / "simulation-trajectory-aggregate.csv",
    ])
_input_csv_candidates.append(_THIS_DIR.parent / "simulation-trajectory-aggregate.csv")

_seen_input_csv = set()
input_csv_candidates = []
for _cand in _input_csv_candidates:
    try:
        _resolved = _cand.resolve()
    except Exception:
        _resolved = _cand
    if _resolved in _seen_input_csv:
        continue
    _seen_input_csv.add(_resolved)
    input_csv_candidates.append(_cand)

_input_csv_loaded = False
_input_csv_existing = [p for p in input_csv_candidates if p.is_file()]
if not _input_csv_existing:
    warnings.warn(
        "[spec] input candidate CSV not found; continuing with environment/default spec values. "
        f"tried: {[str(p) for p in input_csv_candidates]}"
    )
else:
    m = re.search(r"(\d+)", spec.name)
    if not m:
        warnings.warn(f"[spec] cannot parse Trajectory ID from spec.name='{spec.name}'")
    else:
        tid = m.group(1)
        for input_csv_path in _input_csv_existing:
            df_input = pd.read_csv(input_csv_path)
            if "Trajectory ID" not in df_input.columns:
                warnings.warn(f"[spec] input CSV missing 'Trajectory ID' column: {input_csv_path}")
                continue
            row = df_input.loc[df_input["Trajectory ID"].astype(str) == tid]
            if row.empty:
                continue
            if "SMILES" in row.columns and not pd.isna(row.iloc[0]["SMILES"]):
                spec.psmiles = str(row.iloc[0]["SMILES"])
            if "Degree of Polymerization" in row.columns and not pd.isna(row.iloc[0]["Degree of Polymerization"]):
                spec.n_repeat = int(float(row.iloc[0]["Degree of Polymerization"]))
            if "Molality" in row.columns and not pd.isna(row.iloc[0]["Molality"]):
                spec.target_molality = float(row.iloc[0]["Molality"])
            if "Density" in row.columns and not pd.isna(row.iloc[0]["Density"]):
                spec.target_density_g_cm3 = float(row.iloc[0]["Density"])
            _input_csv_loaded = True
            log(
                f"[spec] loaded input CSV: {input_csv_path} "
                f"PSMILES={spec.psmiles}, n_repeat={spec.n_repeat}, "
                f"target_molality={spec.target_molality}, "
                f"target_density_g_cm3={spec.target_density_g_cm3}, "
                f"molality_basis={getattr(spec, 'molality_basis', 'mixture')}"
            )
            break
        if not _input_csv_loaded:
            warnings.warn(
                f"[spec] Trajectory ID {tid} not found in input CSV candidates; "
                "continuing with environment/default spec values."
            )

log(
    f"[spec-gk] output_enabled={spec.gk_output_enabled} "
    f"frame_interval_ps={spec.gk_frame_interval_ps:g} "
    f"save_velocities={spec.gk_save_velocities} "
    f"production_ns={spec.production_ns:g} "
    f"analysis_window={spec.analysis_begin_ns:g}~{spec.analysis_end_ns:g} ns"
)

_workspace_override = os.environ.get("GROMACS_TRAJ_ROOT")
if _workspace_override:
    spec.workspace = Path(_workspace_override).resolve()
else:
    spec.workspace = Path(spec.name).resolve()
ROOT = spec.workspace.resolve()
STRUCT_DIR   = ROOT / "structures"
PACKMOL_DIR  = ROOT / "packmol"
TOPO_DIR     = ROOT / "topology"
MDP_DIR      = ROOT / "mdp"
MD_DIR       = ROOT / "md"
ANALYSIS_DIR = ROOT / "analysis"
_default_shared_cache_root = (
    ROOT.parent.parent / "shared_cache"
    if ROOT.parent.name == "runs"
    else ROOT.parent / "shared_cache"
)
SHARED_CACHE_ROOT = Path(
    os.environ.get("GROMACS_SHARED_CACHE_ROOT", str(_default_shared_cache_root))
).resolve()

for d in (STRUCT_DIR, PACKMOL_DIR, TOPO_DIR, MDP_DIR, MD_DIR, ANALYSIS_DIR, SHARED_CACHE_ROOT):
    d.mkdir(parents=True, exist_ok=True)

dependencies_ok = {
    "GROMACS": check_dependency("GROMACS", GMX),
    "PACKMOL": check_dependency("PACKMOL", "packmol"),
    "ACPYPE":  check_dependency("ACPYPE", "acpype"),
}

def _read_topology_molecule_counts(top_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not top_path.exists():
        return counts
    in_molecules = False
    for raw in top_path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_molecules = line.strip("[]").strip().lower() == "molecules"
            continue
        if not in_molecules:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            counts[parts[0]] = int(float(parts[1]))
        except ValueError:
            continue
    return counts

def _read_moleculetype_name(itp_path: Path) -> Optional[str]:
    if not itp_path.exists():
        return None
    in_moleculetype = False
    for raw in itp_path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_moleculetype = line.strip("[]").strip().lower() == "moleculetype"
            continue
        if in_moleculetype:
            return line.split()[0]
    return None

def _sync_n_chains_from_topology_for_density(top_path: Optional[Path]=None) -> None:
    top_path = Path(top_path) if top_path is not None else TOPO_DIR / "topol.top"
    counts = _read_topology_molecule_counts(top_path)
    if not counts:
        warnings.warn(f"[density] could not read [ molecules ] from {top_path}; using spec.n_chains={spec.n_chains}")
        return

    candidates: List[str] = []
    pol_mt = _read_moleculetype_name(TOPO_DIR / "polymer_clean.itp")
    for name in (pol_mt, spec.polymer_resname_coords, "polymer", "POL"):
        if name and name not in candidates:
            candidates.append(name)

    actual_n = None
    actual_name = None
    for name in candidates:
        if name in counts:
            actual_n = counts[name]
            actual_name = name
            break

    if actual_n is None:
        salt_names = {spec.cation_resname_coords.lower(), spec.anion_resname_coords.lower(), "li", "tfsi"}
        non_salt = [(name, n) for name, n in counts.items() if name.lower() not in salt_names]
        if len(non_salt) == 1:
            actual_name, actual_n = non_salt[0]

    if actual_n is None:
        warnings.warn(f"[density] could not infer polymer count from topology molecules={counts}; using spec.n_chains={spec.n_chains}")
        return

    if int(actual_n) != int(spec.n_chains):
        log(f"[density] n_chains override from topology [molecules] {actual_name}: {spec.n_chains} -> {int(actual_n)}")
        spec.n_chains = int(actual_n)
    else:
        log(f"[density] n_chains={spec.n_chains} from topology [molecules] {actual_name}")

# ---- references (URLs in code comments only) ----
# GROMACS msd: https://manual.gromacs.org/documentation/current/onlinehelp/gmx-msd.html
# TRI-AMDD htp_md: https://github.com/TRI-AMDD/htp_md
# MDAnalysis FastNS: https://docs.mdanalysis.org/


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
