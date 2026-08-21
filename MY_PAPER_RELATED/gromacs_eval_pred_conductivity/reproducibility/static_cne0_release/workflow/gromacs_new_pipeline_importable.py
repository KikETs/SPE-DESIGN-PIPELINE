
# ===== Notebook Cell 1 [pysoftk] =====
# =========================
# Cell 1) Spec & Utils
# =========================
from __future__ import annotations
print("__STAGEV3__:pysoftk:cell1", flush=True)

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
# --- auto-update from reference CSV (toggle) ---
USE_REF_CSV = True
REF_CSV_PATH = Path("simulation-trajectory-aggregate.csv")
if USE_REF_CSV:
    ref_candidates = [REF_CSV_PATH]
    ref_path = next((p for p in ref_candidates if p.is_file()), None)
    if ref_path is None:
        raise FileNotFoundError(
            f"reference CSV not found. tried: {[str(p) for p in ref_candidates]}"
        )
    df_ref = pd.read_csv(ref_path)
    if "Trajectory ID" not in df_ref.columns:
        raise KeyError("reference CSV missing 'Trajectory ID' column")
    m = re.search(r"(\d+)", spec.name)
    if not m:
        raise ValueError(f"cannot parse Trajectory ID from spec.name='{spec.name}'")
    tid = m.group(1)
    row = df_ref.loc[df_ref["Trajectory ID"].astype(str) == tid]
    if row.empty:
        raise KeyError(f"Trajectory ID {tid} not found in reference CSV")
    spec.psmiles = str(row.iloc[0]["SMILES"])
    spec.n_repeat = int(float(row.iloc[0]["Degree of Polymerization"]))
    if "Molality" in row.columns and not pd.isna(row.iloc[0]["Molality"]):
        spec.target_molality = float(row.iloc[0]["Molality"])
    if "Density" in row.columns and not pd.isna(row.iloc[0]["Density"]):
        spec.target_density_g_cm3 = float(row.iloc[0]["Density"])
    log(
        f"[spec] override from reference CSV: PSMILES={spec.psmiles}, "
        f"n_repeat={spec.n_repeat}, target_molality={spec.target_molality}, "
        f"target_density_g_cm3={spec.target_density_g_cm3}, "
        f"molality_basis={getattr(spec, 'molality_basis', 'mixture')}"
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

# ===== Notebook Cell 2 [pysoftk] =====
print("__STAGEV3__:pysoftk:cell2", flush=True)
# =========================
# Cell 2) Build structures (pysoftk polymer + RDKit TFSI + Li)
# =========================
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

def _find_exact_overlap_groups(coords, eps=1e-6):
    mapping = {}
    for i, (x,y,z) in enumerate(coords):
        key = (round(x/eps)*eps, round(y/eps)*eps, round(z/eps)*eps)
        mapping.setdefault(key, []).append(i)
    return [idxs for idxs in mapping.values() if len(idxs) > 1]

def _jitter_overlaps_inplace(chain_pybel, groups, step=0.2, seed=123):
    rng = np.random.default_rng(seed)
    for g in groups:
        base = g[0]
        bx, by, bz = chain_pybel.atoms[base].coords
        for k, idx in enumerate(g[1:], start=1):
            direction = rng.normal(size=3)
            direction /= (np.linalg.norm(direction) + 1e-12)
            dx,dy,dz = direction * (step * k)
            chain_pybel.atoms[idx].OBAtom.SetVector(bx+dx, by+dy, bz+dz)

def _min_pair_distance_estimate(chain_pybel, max_pairs=200000):
    coords = np.array([a.coords for a in chain_pybel.atoms], dtype=float)
    n = len(coords)
    if n < 2:
        return float('inf')
    total_pairs = n*(n-1)//2
    best = float('inf')
    if total_pairs > max_pairs:
        for _ in range(max_pairs):
            i = random.randrange(n)
            j = random.randrange(n-1)
            if j >= i: j += 1
            d = np.linalg.norm(coords[i]-coords[j])
            best = min(best, d)
            if best == 0.0: break
        return best
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i]-coords[j])
            best = min(best, d)
            if best == 0.0:
                return 0.0
    return best

def _fmt_atom_name(name: str, element: str) -> str:
    if len(element.strip()) == 1:
        return f"{name:>4s}"[:4]
    else:
        return f"{name:<4s}"[:4]

def write_pdb_strict_from_rdkit(mol, path, resname="POL", chain_id="A", resseq=1):
    from rdkit import Chem
    conf = mol.GetConformer()
    resname3 = (resname or "MOL").upper()[:3]
    lines = []
    serial = 1
    for i, atom in enumerate(mol.GetAtoms(), start=1):
        el = atom.GetSymbol().upper()
        pos = conf.GetAtomPosition(i-1)

        # atom name (<=4 chars). H64 같은 것도 OK
        name = f"{el}{i%100:02d}"
        # PDB atom name alignment rule (element 1char right-justified)
        if len(el) == 1:
            name = f"{name:>4s}"[:4]
        else:
            name = f"{name:<4s}"[:4]

        # PDB 고정폭(좌표 컬럼 포함)으로 작성  :contentReference[oaicite:3]{index=3}
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


def build_polymer_chain(spec: SystemSpec, out_dir: Path) -> Tuple[Path, float]:
    print("__STAGEV3__:pysoftk", flush=True)
    pdb_fix = out_dir / f"{spec.name}_chain_fix.pdb"
    mol_path = out_dir / "chain.mol"
    if _FAST_PYSOFTK and pdb_fix.exists() and mol_path.exists():
        try:
            poly_cached = Chem.MolFromMolBlock(mol_path.read_text(), sanitize=False, removeHs=False)
            if poly_cached is not None:
                try:
                    Chem.SanitizeMol(poly_cached)
                except Exception:
                    pass
                chain_mw_cached = float(Descriptors.MolWt(poly_cached))
                log(f"[pysoftk-fast] reuse cached polymer: {pdb_fix}")
                return pdb_fix, chain_mw_cached
        except Exception:
            pass
    monomer_smi = _placeholder_smiles(spec.psmiles, spec.placeholder)
    monomer = Chem.MolFromSmiles(monomer_smi)
    if monomer is None:
        raise ValueError(f"Invalid monomer SMILES after replacement: {monomer_smi}")
    monomer3d = ensure_3d_conformer(monomer, seed=spec.packmol_seed)

    chain = Lp(mol=monomer3d, atom=spec.placeholder, n_copies=spec.n_repeat, shift=1.0).linear_polymer(force_field="UFF")
    if not _PYSOFTK_SKIP_FINAL_LOCALOPT:
        try:
            chain.localopt(forcefield="uff", steps=_PYSOFTK_LOCALOPT_STEPS)
        except Exception:
            pass

    coords = np.array([a.coords for a in chain.atoms], dtype=float)
    groups = _find_exact_overlap_groups(coords)
    if groups:
        log(f"[overlap-fix] found {len(groups)} exact-overlap groups → jitter")
        _jitter_overlaps_inplace(chain, groups, step=0.2, seed=spec.packmol_seed)

    min_d = _min_pair_distance_estimate(chain)
    log(f"[overlap-fix] min inter-atomic distance (est) = {min_d}")
    if min_d == 0.0:
        raise RuntimeError("Exact r=0 pair remains → will cause NaN. Abort.")

    # Save with pysoftk
    mol_path = out_dir / "chain.mol"
    pdb_path = out_dir / "chain_pysoftk.pdb"
    Fmt(chain).mol_print(str(mol_path))
    Fmt(chain).pdb_print(str(pdb_path))

    # Parse MOL with RDKit for MW and strict PDB
    mol_block = mol_path.read_text()
    poly = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if poly is None:
        raise RuntimeError("RDKit failed to parse chain.mol from pysoftk output")
    try:
        Chem.SanitizeMol(poly)
    except Exception:
        warnings.warn("RDKit sanitize failed for polymer (MW may be approximate).")

    chain_mw_g_mol = float(Descriptors.MolWt(poly))
    pdb_fix = out_dir / f"{spec.name}_chain_fix.pdb"
    write_pdb_strict_from_rdkit(poly, pdb_fix, resname=spec.polymer_resname_coords)
    log(f"[polymer] chain MW ≈ {chain_mw_g_mol:.3f} g/mol")
    return pdb_fix, chain_mw_g_mol

def build_tfsi(spec: SystemSpec, out_dir: Path) -> Tuple[Path, float]:
    print("__STAGEV3__:pysoftk", flush=True)
    out_pdb = out_dir / "tfsi.pdb"
    if _FAST_PYSOFTK and out_pdb.exists():
        tfsi_cached = Chem.MolFromSmiles(spec.anion_smiles)
        tfsi_mw_cached = float(Descriptors.MolWt(tfsi_cached)) if tfsi_cached is not None else float("nan")
        log(f"[pysoftk-fast] reuse cached tfsi: {out_pdb}")
        return out_pdb, tfsi_mw_cached
    tfsi = Chem.MolFromSmiles(spec.anion_smiles)
    if tfsi is None:
        raise ValueError(f"Invalid TFSI SMILES: {spec.anion_smiles}")
    tfsi3d = ensure_3d_conformer(tfsi, seed=spec.packmol_seed+7)
    tfsi_mw_g_mol = float(Descriptors.MolWt(tfsi3d))
    out_pdb = out_dir / "tfsi.pdb"
    write_pdb_strict_from_rdkit(tfsi3d, out_pdb, resname=spec.anion_resname_coords)
    log(f"[tfsi] MW ≈ {tfsi_mw_g_mol:.3f} g/mol")
    return out_pdb, tfsi_mw_g_mol

def build_li(spec: SystemSpec, out_dir: Path) -> Tuple[Path, float]:
    out_pdb = out_dir / "li.pdb"
    resname3 = spec.cation_resname_coords[:3].upper()
    # (버그 수정) 문자열 깨짐 없이 정상 PDB
    out_pdb.write_text(
        "\n".join([
            f"HETATM{1:5d} {'LI':>4s} {resname3:>3s} C{1:4d}    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.00:6.2f}{0.00:6.2f}          {'LI':>2s}",
            "END",
            ""
        ])
    )
    li_mw_g_mol = 6.941
    return out_pdb, li_mw_g_mol

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

polymer_pdb, polymer_mw_g_mol = build_polymer_chain(spec, STRUCT_DIR)
tfsi_pdb, tfsi_mw_g_mol       = build_tfsi(spec, STRUCT_DIR)
li_pdb, li_mw_g_mol           = build_li(spec, STRUCT_DIR)

# optional auto-update n_chains
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
(polymer_pdb, tfsi_pdb, li_pdb)

# ===== Notebook Cell 3 [packmol] =====
print("__STAGEV3__:packmol:cell3", flush=True)
# =========================
# Cell 3) Packmol → editconf (conf_initial.gro)
# =========================
def estimate_box_length_nm(spec: SystemSpec, polymer_mw_g_mol: float, li_mw_g_mol: float, tfsi_mw_g_mol: float) -> float:
    total_polymer_mass = polymer_mw_g_mol * spec.n_chains
    total_ion_mass     = (li_mw_g_mol + tfsi_mw_g_mol) * spec.li_tfsi_pairs
    total_mass_g_mol   = total_polymer_mass + total_ion_mass
    mass_g = total_mass_g_mol / NA
    vol_cm3 = mass_g / spec.density_guess
    vol_nm3 = vol_cm3 * 1e21
    return vol_nm3 ** (1/3)

if spec.box_length_nm is None:
    spec.box_length_nm = estimate_box_length_nm(spec, polymer_mw_g_mol, li_mw_g_mol, tfsi_mw_g_mol)

log(f"box length (nm): {spec.box_length_nm:.3f}")

def _ion_constraints(L_A: float, spec: SystemSpec) -> Tuple[str, str]:
    if not getattr(spec, "packmol_split_ions", False):
        return "", ""

    axis = str(getattr(spec, "packmol_split_axis", "z")).lower()
    if axis not in ("x", "y", "z"):
        axis = "z"

    # Use sphere-based separation centered in the box to avoid PBC aliasing
    cx = cy = cz = L_A / 2.0
    r_li = float(getattr(spec, "packmol_li_sphere_frac", 0.35)) * L_A
    r_tf = float(getattr(spec, "packmol_tfsi_sphere_frac", 0.45)) * L_A

    # Clamp to safe range
    r_li = min(r_li, 0.49 * L_A)
    r_tf = min(r_tf, 0.49 * L_A)
    if r_tf <= r_li:
        r_tf = min(0.49 * L_A, r_li + 2.0)  # ensure at least 2 Å gap

    li_extra = f"  inside sphere {cx:.3f} {cy:.3f} {cz:.3f} {r_li:.3f}\n"
    tf_extra = f"  outside sphere {cx:.3f} {cy:.3f} {cz:.3f} {r_tf:.3f}\n"
    return li_extra, tf_extra


def build_packmol_input(spec: SystemSpec, polymer_pdb: Path, li_pdb: Path, tfsi_pdb: Path,
                        *, seed: int, output_pdb: str) -> str:
    L_A = spec.box_length_nm * 10.0
    movebad = "yes" if spec.packmol_movebadrandom else "no"
    li_extra, tf_extra = _ion_constraints(L_A, spec)

    return f"""
    tolerance {spec.packmol_tolerance}
    seed {seed}
    maxit {spec.packmol_maxit}
    nloop {spec.packmol_nloop}
    movefrac {spec.packmol_movefrac}
    movebadrandom {movebad}
    filetype pdb
    output {output_pdb}

    structure {polymer_pdb}
      number {spec.n_chains}
      inside box 0. 0. 0. {L_A:.3f} {L_A:.3f} {L_A:.3f}
    end structure

    structure {li_pdb}
      number {spec.li_tfsi_pairs}
      inside box 0. 0. 0. {L_A:.3f} {L_A:.3f} {L_A:.3f}
    {li_extra.strip() if li_extra else ""}
    end structure

    structure {tfsi_pdb}
      number {spec.li_tfsi_pairs}
      inside box 0. 0. 0. {L_A:.3f} {L_A:.3f} {L_A:.3f}
    {tf_extra.strip() if tf_extra else ""}
    end structure
    """.strip() + "\n"


def _pdb_first_resname(pdb_path: Path) -> Optional[str]:
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith(("ATOM", "HETATM")):
                return ln[17:20].strip()
    return None


def _pdb_resname_set(pdb_path: Path) -> List[str]:
    names = set()
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith(("ATOM", "HETATM")):
                names.add(ln[17:20].strip())
    return sorted(names)


def _load_pdb_coords(pdb_path: Path, resname: str, atom_prefixes: Optional[List[str]] = None):
    coords = []
    with pdb_path.open() as f:
        for ln in f:
            if not ln.startswith(("ATOM", "HETATM")):
                continue
            rname = ln[17:20].strip()
            if rname != resname:
                continue
            aname = ln[12:16].strip()
            if atom_prefixes and not any(aname.startswith(p) for p in atom_prefixes):
                continue
            try:
                x = float(ln[30:38])
                y = float(ln[38:46])
                z = float(ln[46:54])
            except ValueError:
                continue
            coords.append([x, y, z])
    return np.array(coords, dtype=float)


def _try_resnames(pdb_path: Path, resnames: List[str], atom_prefixes: Optional[List[str]] = None):
    for rn in resnames:
        coords = _load_pdb_coords(pdb_path, rn, atom_prefixes=atom_prefixes)
        if coords.size:
            return coords
    return np.array([], dtype=float)


def score_packmol_pdb(pdb_path: Path, li_res: str, an_res: str, cutoff_A: float):
    prefixes = getattr(spec, "packmol_score_atom_prefixes", None)
    if prefixes is None:
        prefixes = ["O", "S", "N"]

    li_resnames = [li_res]
    try:
        li_pdb_res = _pdb_first_resname(li_pdb)
        if li_pdb_res:
            li_resnames.append(li_pdb_res)
    except Exception:
        pass
    if len(li_res) > 3:
        li_resnames.append(li_res[:3])
    li_resnames = list(dict.fromkeys([r for r in li_resnames if r]))
    li = _try_resnames(pdb_path, li_resnames)

    an_resnames = [an_res]
    try:
        an_pdb_res = _pdb_first_resname(tfsi_pdb)
        if an_pdb_res:
            an_resnames.append(an_pdb_res)
    except Exception:
        pass
    if len(an_res) > 3:
        an_resnames.append(an_res[:3])
    an_resnames = list(dict.fromkeys([r for r in an_resnames if r]))
    an = _try_resnames(pdb_path, an_resnames, atom_prefixes=prefixes)

    if li.size == 0 or an.size == 0:
        resnames = _pdb_resname_set(pdb_path)
        log(f"Packmol score failed for {pdb_path.name}: resnames={resnames}")
        return None
    diff = li[:, None, :] - an[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    d = np.sqrt(d2)
    contact = int(np.sum(d < cutoff_A))
    min_d = float(np.min(d))
    return contact, min_d


if not dependencies_ok["PACKMOL"]:
    raise RuntimeError("PACKMOL 미설치")

# multi-seed sampling
seed_list = getattr(spec, "packmol_seed_list", None)
if seed_list:
    seeds = list(seed_list)
else:
    n_seeds = int(getattr(spec, "packmol_n_seeds", 1))
    if n_seeds < 1:
        n_seeds = 1
    seeds = [int(spec.packmol_seed) + i * 1000 for i in range(n_seeds)]

print("__STAGEV3__:packmol", flush=True)

scores = []
for seed in seeds:
    out_pdb = PACKMOL_DIR / f"{spec.name}_packmol_seed{seed}.pdb"
    inp_path = PACKMOL_DIR / f"packmol_seed{seed}.inp"
    inp_path.write_text(build_packmol_input(spec, polymer_pdb, li_pdb, tfsi_pdb, seed=seed, output_pdb=out_pdb.name))

    print("__STAGEV3__:packmol", flush=True)
    res = run(["bash","-lc", f"packmol < {shlex.quote(inp_path.name)}"], cwd=PACKMOL_DIR, check=False, capture_output=True)
    (PACKMOL_DIR / f"packmol_seed{seed}.log").write_text((res.stdout or "") + "\n" + (res.stderr or ""))
    if res.returncode != 0:
        log(f"Packmol failed for seed {seed}. See packmol_seed{seed}.log")
        continue

    score = score_packmol_pdb(out_pdb, spec.cation_resname_coords, spec.anion_resname_coords,
                              float(getattr(spec, "packmol_min_li_tfsi_dist_A", 3.4)))
    if score is None:
        log(f"Packmol score failed for seed {seed}: missing LI/N coords")
        continue

    contact_count, min_dist = score
    scores.append({
        "seed": seed,
        "output": out_pdb.name,
        "contact_count": contact_count,
        "min_li_tfsi_dist_A": min_dist,
    })

if not scores:
    raise RuntimeError("Packmol failed for all seeds. Check packmol_seed*.log")

scores_df = pd.DataFrame(scores).sort_values(["contact_count", "min_li_tfsi_dist_A"], ascending=[True, False])
(scores_df).to_csv(PACKMOL_DIR / "packmol_seed_scores.csv", index=False)

best = scores_df.iloc[0]
PACKMOL_OUTPUT = PACKMOL_DIR / f"{spec.name}_packmol.pdb"
shutil.copy2(PACKMOL_DIR / best["output"], PACKMOL_OUTPUT)

# also keep the best input for reference
best_inp = PACKMOL_DIR / f"packmol_seed{int(best['seed'])}.inp"
if best_inp.exists():
    shutil.copy2(best_inp, PACKMOL_DIR / "packmol.inp")

log(f"Packmol OK: {PACKMOL_OUTPUT} | best seed={int(best['seed'])}, contact={int(best['contact_count'])}, min_d={best['min_li_tfsi_dist_A']:.2f} Å")

if not dependencies_ok["GROMACS"]:
    raise RuntimeError("GROMACS 미설치")

STRUCT_START = MD_DIR / "conf_initial.gro"
gmx_cmd("editconf", [
    "-f", str(PACKMOL_OUTPUT),
    "-o", str(STRUCT_START),
    "-box", f"{spec.box_length_nm:.5f}", f"{spec.box_length_nm:.5f}", f"{spec.box_length_nm:.5f}"
], cwd=MD_DIR)

log(f"GRO start: {STRUCT_START}")
STRUCT_START

# ===== Notebook Cell 4 [pysoftk] =====
print("__STAGEV3__:pysoftk:cell4", flush=True)
# ===== Notebook Cell 4 [atomtyping] =====
print("__STAGEV3__:atomtyping:cell4", flush=True)
# =========================
# Cell 4) ACPYPE & Topology (atomtypes merge, GRO renaming, charge scaling, MD, cNE)
#   - 이 셀은 길지만 "지금 네가 붙인 전체"를 변수/파일 정합된 형태로 한 번에 묶은 것이다.
# =========================
import hashlib
import json
from difflib import SequenceMatcher

# ---- ACPYPE wrappers ----
LI_AMBER99SB_PARAMS = {
    "atomtype": "Li", "resname": "LI", "atom_name": "LI",
    "mass": 6.94100, "charge": 1.0, "sigma": 0.202590, "epsilon": 0.0765672,
}

def write_monatomic_itp(out_dir: Path, basename: str, params: Dict[str,float], note: Optional[str]=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"; Auto-generated fallback for {basename}"]
    if note: lines.append(f"; {note}")
    lines += [
        "",
        "[ atomtypes ]",
        "; name mass charge ptype sigma epsilon",
        f"{params['atomtype']:<6s} {params['mass']:10.5f} {params['charge']:10.5f} A {params['sigma']:.6f} {params['epsilon']:.6f}",
        "",
        "[ moleculetype ]",
        f"{params.get('resname','LI')} 1",
        "",
        "[ atoms ]",
        f"1 {params['atomtype']:<6s} 1 {params.get('resname','LI'):<6s} {params.get('atom_name','LI'):<6s} 1 {params['charge']:.6f}",
    ]
    (out_dir / f"{basename}_GMX.itp").write_text("\n".join(lines) + "\n")
    (out_dir / f"{basename}_GMX.top").write_text(f'#include "{basename}_GMX.itp"\n')

def _has_gmx_acpype_outputs(path: Path) -> bool:
    return path.exists() and any(path.glob("*_GMX.itp")) and any(path.glob("*_GMX.top"))

def _copytree_into(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

def _tfsi_shared_cache_dir(input_path: Path, charge_method: str, atom_type: Optional[str], charge: Optional[int]) -> Path:
    payload = [
        input_path.read_bytes(),
        f"charge_method={charge_method}".encode(),
        f"atom_type={atom_type or ''}".encode(),
        f"charge={charge if charge is not None else ''}".encode(),
    ]
    digest = hashlib.sha1(b"\0".join(payload)).hexdigest()[:16]
    key = f"{input_path.stem}_{digest}"
    return SHARED_CACHE_ROOT / f"tfsi_acpype_{key}"


def _polymer_shared_cache_dir(spec: SystemSpec, charge_method: str, atom_type: Optional[str], charge: Optional[int]) -> Path:
    payload = [
        f"psmiles={spec.psmiles}".encode(),
        f"placeholder={spec.placeholder}".encode(),
        f"n_repeat={int(spec.n_repeat)}".encode(),
        f"charge_method={charge_method}".encode(),
        f"atom_type={atom_type or ''}".encode(),
        f"charge={charge if charge is not None else ''}".encode(),
    ]
    digest = hashlib.sha1(b"\0".join(payload)).hexdigest()[:16]
    return SHARED_CACHE_ROOT / f"polymer_acpype_dp{int(spec.n_repeat)}_{digest}"

def _try_publish_shared_cache(src_dir: Path, shared_dir: Path):
    if not _has_gmx_acpype_outputs(src_dir):
        return
    shared_dir.parent.mkdir(parents=True, exist_ok=True)
    if shared_dir.exists() and _has_gmx_acpype_outputs(shared_dir):
        return
    tmp_dir = shared_dir.parent / f".{shared_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.copytree(src_dir, tmp_dir, dirs_exist_ok=True)
    try:
        tmp_dir.replace(shared_dir)
    except Exception:
        if not shared_dir.exists():
            shutil.copytree(tmp_dir, shared_dir, dirs_exist_ok=True)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

def _acpype_log_path(workdir: Path, basename: str) -> Path:
    return workdir / f"{basename}_acpype.log"

def run_acpype(input_path: Path, basename: str, workdir: Path,
               charge: Optional[int]=None, charge_method: str="gas",
               atom_type: Optional[str]=None, fallback: Optional[Dict[str,float]]=None,
               force_recompute: bool = True,
               shared_cache_dir: Optional[Path]=None) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / f"{basename}.acpype"
    log_path = _acpype_log_path(workdir, basename)

    if (not force_recompute) and _has_gmx_acpype_outputs(out_dir):
        log(f"[acpype-cache] reuse local {basename}: {out_dir}")
        return out_dir

    if shared_cache_dir is not None and _has_gmx_acpype_outputs(shared_cache_dir):
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        _copytree_into(shared_cache_dir, out_dir)
        log(f"[acpype-cache] reuse shared {basename}: {shared_cache_dir}")
        return out_dir

    if force_recompute and out_dir.exists():
        shutil.rmtree(out_dir)
    if force_recompute and log_path.exists():
        log_path.unlink()

    cmd = ["acpype", "-i", str(input_path), "-b", basename, "-c", charge_method, "-o", "gmx"]
    if _ACPYPE_FORCE:
        cmd += ["-f"]
    if atom_type:
        cmd += ["-a", atom_type]
    if charge is not None:
        cmd += ["-n", str(charge)]

    log("$ " + " ".join(str(x) for x in cmd))
    acpype_timeout = _ACPYPE_TIMEOUT_SEC if _ACPYPE_TIMEOUT_SEC > 0 else None
    try:
        proc = subprocess.run(cmd, cwd=workdir, text=True, capture_output=True, check=False, timeout=acpype_timeout)
    except subprocess.TimeoutExpired as exc:
        combined_log = (
            f"CMD: {' '.join(str(x) for x in cmd)}\n"
            f"RETURN_CODE: TIMEOUT\n"
            f"TIMEOUT_SEC: {acpype_timeout}\n\n"
            f"--- STDOUT ---\n{exc.stdout or ''}\n"
            f"--- STDERR ---\n{exc.stderr or ''}\n"
        )
        log_path.write_text(combined_log)
        raise RuntimeError(f"ACPYPE timeout ({basename}) after {acpype_timeout:.1f}s log={log_path}") from exc
    combined_log = (
        f"CMD: {' '.join(str(x) for x in cmd)}\n"
        f"RETURN_CODE: {proc.returncode}\n\n"
        f"--- STDOUT ---\n{proc.stdout or ''}\n"
        f"--- STDERR ---\n{proc.stderr or ''}\n"
    )
    log_path.write_text(combined_log)

    if proc.returncode != 0:
        if fallback is None:
            raise RuntimeError(
                f"ACPYPE failed ({basename}) rc={proc.returncode} log={log_path}\n"
                f"STDERR:\n{proc.stderr or ''}"
            )
        warnings.warn(f"ACPYPE 실패({basename}) → fallback 사용", RuntimeWarning)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        write_monatomic_itp(out_dir, basename, fallback, note="forced fallback")

    if not out_dir.exists():
        raise FileNotFoundError(f"ACPYPE output missing: {out_dir}")

    if shared_cache_dir is not None and _has_gmx_acpype_outputs(out_dir):
        _try_publish_shared_cache(out_dir, shared_cache_dir)
    return out_dir

def locate_gmx_itp(acpype_dir: Path) -> Tuple[Path, Path]:
    itp = next(acpype_dir.glob("*_GMX.itp"))
    top = next(acpype_dir.glob("*_GMX.top"))
    return itp, top

if not dependencies_ok["ACPYPE"]:
    raise RuntimeError("ACPYPE 미설치")

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in ("0", "false", "no", "off")

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(raw)

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)

def _env_int_list(name: str, default: List[int]) -> List[int]:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return list(default)
    out = []
    for part in re.split(r"[,\s]+", str(raw).strip()):
        if part:
            out.append(int(part))
    return out or list(default)

_ACPYPE_MIN_DIST = _env_float("GROMACS_ACPYPE_MIN_DIST", 1.00)
_ACPYPE_FORCE = _env_bool("GROMACS_ACPYPE_FORCE", False)
_REBUILD_FROM_CHAIN_MOL = _env_bool("GROMACS_REBUILD_FROM_CHAIN_MOL", False)
_ACPYPE_SKIP_REPAIR = _env_bool("GROMACS_ACPYPE_SKIP_REPAIR", False)
_ACPYPE_TIMEOUT_SEC = _env_float("GROMACS_ACPYPE_TIMEOUT_SEC", 0.0)
_ACPYPE_CHARGE_METHOD = str(os.environ.get("GROMACS_ACPYPE_CHARGE_METHOD", "gas") or "gas").strip().lower()
_acpype_atom_type_raw = str(os.environ.get("GROMACS_ACPYPE_ATOM_TYPE", "") or "").strip().lower()
_ACPYPE_ATOM_TYPE = _acpype_atom_type_raw or None
_TRIMER_FALLBACK_ENABLED = _env_bool("GROMACS_TRIMER_FALLBACK_ENABLED", True)
_TRIMER_FALLBACK_WORKSPACE_NAME = str(
    os.environ.get("GROMACS_TRIMER_FALLBACK_WORKSPACE", "polymer_trimer_fallback")
    or "polymer_trimer_fallback"
).strip()
_TRIMER_FALLBACK_MAX_ATTEMPTS = max(1, _env_int("GROMACS_TRIMER_FALLBACK_MAX_ATTEMPTS", 6))
_TRIMER_FALLBACK_MIN_DIST = _env_float("GROMACS_TRIMER_FALLBACK_MIN_DIST", max(0.80, _ACPYPE_MIN_DIST))
_TRIMER_FALLBACK_STRICT_REBUILD = _env_bool("GROMACS_TRIMER_FALLBACK_STRICT_REBUILD", False)
_TRIMER_FALLBACK_SEED_OFFSETS = _env_int_list(
    "GROMACS_TRIMER_FALLBACK_SEED_OFFSETS",
    [0, 1009, 2027, 4099, 7919, 15401],
)

def _set_rdkit_coords(mol, coords):
    from rdkit.Geometry import Point3D
    conf = mol.GetConformer()
    for k in range(coords.shape[0]):
        conf.SetAtomPosition(k, Point3D(float(coords[k, 0]), float(coords[k, 1]), float(coords[k, 2])))

def _separate_nonbonded_coords(coords, bonded_pairs, min_dist=0.80, max_iter=2000, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.array(coords, dtype=np.float64, copy=True)
    for _ in range(max_iter):
        worst = None
        worst_d = float("inf")
        n = coords.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in bonded_pairs:
                    continue
                d = float(np.linalg.norm(coords[i] - coords[j]))
                if d < worst_d:
                    worst_d = d
                    worst = (i, j)
        if worst is None or worst_d >= min_dist:
            break
        i, j = worst
        vec = coords[i] - coords[j]
        dist = float(np.linalg.norm(vec))
        if dist < 1e-8:
            vec = rng.normal(size=3)
            dist = float(np.linalg.norm(vec))
        unit = vec / dist
        delta = 0.5 * (min_dist - worst_d)
        coords[i] += unit * delta
        coords[j] -= unit * delta
    return coords

def _separate_all_close_coords(coords, min_dist=0.55, max_iter=2000, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.array(coords, dtype=np.float64, copy=True)
    for _ in range(max_iter):
        worst = None
        worst_d = float("inf")
        n = coords.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(coords[i] - coords[j]))
                if d < worst_d:
                    worst_d = d
                    worst = (i, j)
        if worst is None or worst_d >= min_dist:
            break
        i, j = worst
        vec = coords[i] - coords[j]
        dist = float(np.linalg.norm(vec))
        if dist < 1e-8:
            vec = rng.normal(size=3)
            dist = float(np.linalg.norm(vec))
        unit = vec / dist
        delta = 0.5 * (min_dist - worst_d)
        coords[i] += unit * delta
        coords[j] -= unit * delta
    return coords

def _nonbond_min_distance(mol, coords, bonded_pairs):
    n = coords.shape[0]
    min_d = 1e9
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in bonded_pairs:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < min_d:
                min_d = d
    return min_d

def fix_close_contacts_rdkit(mol, min_dist=1.00, max_iter=4000, seed=0):
    """
    - 결합(1-2) 원자쌍은 제외
    - min_dist 미만인 비결합 원자쌍을 발견하면 서로 밀어냄(push)
    - 최종적으로 UFF 최적화로 한 번 정리
    """
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=int(seed))

    conf = mol.GetConformer()
    coords = np.array(conf.GetPositions(), dtype=np.float64)

    bonded_pairs = set()
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        if i < j:
            bonded_pairs.add((i, j))
        else:
            bonded_pairs.add((j, i))

    coords = _separate_nonbonded_coords(coords, bonded_pairs, min_dist=min_dist, max_iter=max_iter, seed=seed)
    _set_rdkit_coords(mol, coords)

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=_PYSOFTK_UFF_ITERS)
    except Exception:
        pass

    coords2 = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
    final_min = _nonbond_min_distance(mol, coords2, bonded_pairs)
    if final_min < min_dist:
        coords2 = _separate_nonbonded_coords(coords2, bonded_pairs, min_dist=min_dist, max_iter=max_iter, seed=seed + 1)
        _set_rdkit_coords(mol, coords2)
        final_min = _nonbond_min_distance(mol, coords2, bonded_pairs)

    return mol, final_min

def repair_polymer_pdb_for_acpype(pdb_in, pdb_out, min_dist=0.80, seed=0):
    # proximityBonding=True: PDB에 CONECT 없어도 거리로 bond 추정
    mol = Chem.MolFromPDBFile(str(pdb_in), removeHs=False, sanitize=True, proximityBonding=True)
    if mol is None:
        mol = Chem.MolFromPDBFile(str(pdb_in), removeHs=False, sanitize=False, proximityBonding=True)
        if mol is None:
            raise RuntimeError(f"RDKit failed to read PDB: {pdb_in}")
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            warnings.warn(f"RDKit sanitize fallback failed for {pdb_in}; proceeding with sanitize=False molecule.")

    mol, mind = fix_close_contacts_rdkit(mol, min_dist=min_dist, seed=seed)
    Chem.MolToPDBFile(mol, str(pdb_out))
    print(f"[repair_polymer_pdb_for_acpype] wrote: {pdb_out} | nonbond min dist ~= {mind:.3f} Å")
    return Path(pdb_out), float(mind)

def rebuild_polymer_pdb_from_chain_mol(mol_in: Path, pdb_out: Path, resname: str = "POL") -> Path:
    mol = Chem.MolFromMolBlock(mol_in.read_text(), sanitize=False, removeHs=False)
    if mol is None:
        raise RuntimeError(f"RDKit failed to parse MOL block: {mol_in}")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        warnings.warn(f"RDKit sanitize failed for {mol_in}; writing fallback PDB from unsanitized mol.")
    write_pdb_strict_from_rdkit(mol, pdb_out, resname=resname)
    print(f"[atomtyping] rebuilt polymer PDB from chain.mol: {pdb_out}")
    return Path(pdb_out)

def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""

def _acpype_failure_text(workdir: Path, basename: str, err: Exception) -> str:
    parts = [str(err)]
    log_path = _acpype_log_path(workdir, basename)
    if log_path.exists():
        parts.append(_safe_read_text(log_path))
    out_dir = workdir / f"{basename}.acpype"
    if out_dir.exists():
        for cand in sorted(out_dir.glob("*.log")):
            parts.append(_safe_read_text(cand))
    return "\n".join(p for p in parts if p)

def _looks_like_antechamber_typing_failure(text: str) -> bool:
    if not text:
        return False
    pats = [
        r"antechamber failed",
        r"maxbond",
        r"type:\s*du",
        r"no gasteiger parameter",
        r"cannot properly run",
        r"_gaff2\.mol2",
        r"parmchk.*failed",
        r"tleap.*failed",
        r"_ac\.prmtop",
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in pats)

def _atom_signature(mol: Chem.Mol):
    return [
        (
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            tuple(sorted(n.GetAtomicNum() for n in atom.GetNeighbors())),
        )
        for atom in mol.GetAtoms()
    ]

def _rdkit_coord_quality(mol: Chem.Mol) -> Dict[str, float]:
    if mol.GetNumConformers() == 0:
        return {
            "min_any_dist": float("nan"),
            "min_nonbond_dist": float("nan"),
            "duplicate_coord_pairs": 0,
            "num_atoms": int(mol.GetNumAtoms()),
        }
    conf = mol.GetConformer()
    coords = np.array(conf.GetPositions(), dtype=np.float64)
    bonded_pairs = set()
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        bonded_pairs.add((min(i, j), max(i, j)))

    min_any = float("inf")
    min_nonbond = float("inf")
    duplicate_pairs = 0
    for i in range(coords.shape[0]):
        for j in range(i + 1, coords.shape[0]):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < min_any:
                min_any = d
            if d < 1e-4:
                duplicate_pairs += 1
            if (i, j) not in bonded_pairs and d < min_nonbond:
                min_nonbond = d
    if not np.isfinite(min_any):
        min_any = float("nan")
    if not np.isfinite(min_nonbond):
        min_nonbond = float("nan")
    return {
        "min_any_dist": float(min_any),
        "min_nonbond_dist": float(min_nonbond),
        "duplicate_coord_pairs": int(duplicate_pairs),
        "num_atoms": int(mol.GetNumAtoms()),
    }

def _fallback_attempt_configs():
    seed_offsets = list(_TRIMER_FALLBACK_SEED_OFFSETS)
    while len(seed_offsets) < _TRIMER_FALLBACK_MAX_ATTEMPTS:
        seed_offsets.append(seed_offsets[-1] + 7919)
    shifts = [1.0, 1.15, 1.35, 1.60, 1.90, 2.20]
    localopt_steps = [
        _PYSOFTK_LOCALOPT_STEPS,
        max(_PYSOFTK_LOCALOPT_STEPS, 500),
        max(_PYSOFTK_LOCALOPT_STEPS, 1000),
        max(_PYSOFTK_LOCALOPT_STEPS, 1500),
        max(_PYSOFTK_LOCALOPT_STEPS, 2000),
        max(_PYSOFTK_LOCALOPT_STEPS, 3000),
    ]
    for idx in range(_TRIMER_FALLBACK_MAX_ATTEMPTS):
        yield {
            "attempt": idx + 1,
            "seed_offset": int(seed_offsets[idx]),
            "shift": float(shifts[min(idx, len(shifts) - 1)]),
            "force_localopt": bool(_TRIMER_FALLBACK_STRICT_REBUILD or idx > 0),
            "localopt_steps": int(localopt_steps[min(idx, len(localopt_steps) - 1)]),
            "repair_min_dist": float(max(_TRIMER_FALLBACK_MIN_DIST, 0.80 + 0.05 * min(idx, 6))),
        }

def _build_typing_probe_oligomer(
    spec: SystemSpec,
    n_repeat: int,
    out_dir: Path,
    *,
    seed_offset: int = 0,
    shift: float = 1.0,
    force_localopt: bool = False,
    localopt_steps: Optional[int] = None,
    repair_min_dist: Optional[float] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    monomer_smi = _placeholder_smiles(spec.psmiles, spec.placeholder)
    monomer = Chem.MolFromSmiles(monomer_smi)
    if monomer is None:
        raise ValueError(f"Invalid monomer SMILES after replacement: {monomer_smi}")
    seed = int(spec.packmol_seed + 97 * n_repeat + int(seed_offset))
    steps = int(localopt_steps if localopt_steps is not None else _PYSOFTK_LOCALOPT_STEPS)
    monomer3d = ensure_3d_conformer(monomer, seed=seed)
    chain = Lp(mol=monomer3d, atom=spec.placeholder, n_copies=n_repeat, shift=float(shift)).linear_polymer(force_field="UFF")
    if force_localopt or not _PYSOFTK_SKIP_FINAL_LOCALOPT:
        try:
            chain.localopt(forcefield="uff", steps=steps)
        except Exception as localopt_exc:
            log(f"[trimer-fallback] pysoftk localopt warning n={n_repeat}: {type(localopt_exc).__name__}: {localopt_exc}")

    mol_path = out_dir / f"chain_n{n_repeat}.mol"
    pdb_path = out_dir / f"chain_n{n_repeat}.pdb"
    Fmt(chain).mol_print(str(mol_path))

    mol = Chem.MolFromMolBlock(mol_path.read_text(), sanitize=False, removeHs=False)
    if mol is None:
        raise RuntimeError(f"failed to parse oligomer MOL: {mol_path}")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    pre_quality = _rdkit_coord_quality(mol)
    post_repair_min = float("nan")
    if repair_min_dist is not None:
        needs_repair = (
            int(pre_quality.get("duplicate_coord_pairs", 0)) > 0
            or float(pre_quality.get("min_any_dist", 999.0)) < 0.50
            or float(pre_quality.get("min_nonbond_dist", 999.0)) < float(repair_min_dist)
        )
        if needs_repair:
            mol, post_repair_min = fix_close_contacts_rdkit(
                mol,
                min_dist=float(repair_min_dist),
                max_iter=8000,
                seed=seed + 17,
            )
            Chem.MolToMolFile(mol, str(mol_path))
    post_quality = _rdkit_coord_quality(mol)
    if int(post_quality.get("duplicate_coord_pairs", 0)) > 0 or float(post_quality.get("min_any_dist", 999.0)) < 0.50:
        coords = np.array(mol.GetConformer().GetPositions(), dtype=np.float64)
        coords = _separate_all_close_coords(coords, min_dist=0.60, max_iter=8000, seed=seed + 29)
        _set_rdkit_coords(mol, coords)
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=max(_PYSOFTK_UFF_ITERS, 2000))
        except Exception as uff_exc:
            log(f"[trimer-fallback] post-jitter UFF warning n={n_repeat}: {type(uff_exc).__name__}: {uff_exc}")
        Chem.MolToMolFile(mol, str(mol_path))
        post_quality = _rdkit_coord_quality(mol)
    write_pdb_strict_from_rdkit(mol, pdb_path, resname=spec.polymer_resname_coords)
    quality = {
        "n_repeat": int(n_repeat),
        "seed": int(seed),
        "seed_offset": int(seed_offset),
        "shift": float(shift),
        "force_localopt": bool(force_localopt),
        "localopt_steps": int(steps),
        "repair_min_dist": None if repair_min_dist is None else float(repair_min_dist),
        "pre_quality": pre_quality,
        "post_repair_min_nonbond_dist": float(post_repair_min),
        "post_quality": post_quality,
        "mol_path": str(mol_path),
        "pdb_path": str(pdb_path),
    }
    (out_dir / f"chain_n{n_repeat}_build_quality.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False))
    return mol_path, pdb_path, _atom_signature(mol), quality

def _parse_itp_atoms(itp_path: Path):
    atoms = []
    in_atoms = False
    for ln in itp_path.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        if s.startswith("["):
            in_atoms = s.lower().startswith("[ atoms")
            continue
        if not in_atoms:
            continue
        core = ln.split(";", 1)[0].split()
        if len(core) < 8:
            continue
        atoms.append({
            "nr": int(core[0]),
            "type": core[1],
            "resnr": int(core[2]),
            "residue": core[3],
            "atom": core[4],
            "cgnr": int(core[5]),
            "charge": float(core[6]),
            "mass": float(core[7]),
        })
    if not atoms:
        raise ValueError(f"no [ atoms ] parsed from {itp_path}")
    return atoms

def _infer_insert_scheme(sig3, sig4):
    sm = SequenceMatcher(a=list(sig3), b=list(sig4), autojunk=False)
    inserts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag != "insert":
            raise ValueError(f"unsupported diff opcode: {(tag, i1, i2, j1, j2)}")
        inserts.append({"pos": i1, "at_end": i1 == len(sig3), "j1": j1, "j2": j2})
    if not inserts:
        raise ValueError("no insert scheme inferred from trimer/tetramer")
    return inserts

def _apply_insert_scheme(base_atoms, insert_specs, extra_repeats: int):
    atoms = []
    for atom in base_atoms:
        rec = dict(atom)
        rec["_fallback_region"] = "base"
        atoms.append(rec)
    for _ in range(extra_repeats):
        out = []
        cursor = 0
        for block_idx, spec in enumerate(insert_specs):
            pos = len(atoms) if spec["at_end"] else int(spec["pos"])
            pos = max(cursor, min(pos, len(atoms)))
            out.extend(atoms[cursor:pos])
            for atom in spec["block"]:
                rec = dict(atom)
                rec["_fallback_region"] = "insert"
                rec["_fallback_block_idx"] = int(block_idx)
                out.append(rec)
            cursor = pos
        out.extend(atoms[cursor:])
        atoms = out

    renum = []
    for idx, atom in enumerate(atoms, start=1):
        rec = dict(atom)
        rec["nr"] = idx
        rec["cgnr"] = idx
        renum.append(rec)
    return renum

def _sum_atom_charge(atoms):
    return float(sum(float(atom.get("charge", 0.0)) for atom in atoms))

def _rebalance_fallback_polymer_charges(
    atoms,
    *,
    target_total: float = 0.0,
    max_preferred_per_atom_shift: float = 1e-3,
):
    q_before = _sum_atom_charge(atoms)
    correction = float(target_total) - q_before
    if abs(correction) <= 1e-12:
        return {
            "applied": False,
            "charge_before": q_before,
            "charge_after": q_before,
            "scope": "none",
            "n_atoms_touched": 0,
            "max_per_atom_shift": 0.0,
        }

    def _non_h(atom):
        return not str(atom.get("atom", "")).upper().startswith("H")

    preferred_idx = [i for i, atom in enumerate(atoms) if atom.get("_fallback_region") == "insert" and _non_h(atom)]
    heavy_idx = [i for i, atom in enumerate(atoms) if _non_h(atom)]
    all_idx = list(range(len(atoms)))

    candidate_scopes = [
        ("insert_heavy", preferred_idx),
        ("all_heavy", heavy_idx),
        ("all_atoms", all_idx),
    ]

    chosen_scope = "all_atoms"
    chosen_idx = all_idx
    for scope_name, idxs in candidate_scopes:
        if not idxs:
            continue
        per_atom = correction / float(len(idxs))
        chosen_scope = scope_name
        chosen_idx = idxs
        if abs(per_atom) <= float(max_preferred_per_atom_shift) or scope_name != "insert_heavy":
            break

    per_atom = correction / float(len(chosen_idx))
    for idx in chosen_idx[:-1]:
        atoms[idx]["charge"] = float(atoms[idx]["charge"]) + per_atom
    residual = float(target_total) - _sum_atom_charge(atoms)
    atoms[chosen_idx[-1]]["charge"] = float(atoms[chosen_idx[-1]]["charge"]) + residual

    return {
        "applied": True,
        "charge_before": q_before,
        "charge_after": _sum_atom_charge(atoms),
        "scope": chosen_scope,
        "n_atoms_touched": len(chosen_idx),
        "max_per_atom_shift": max(abs(per_atom), abs(residual)),
    }

def _write_atoms_only_itp(dst: Path, atoms, molname: str = "POL"):
    lines = [
        "; atom-block-only fallback preview",
        "; WARNING: bonded terms are intentionally missing here.",
        "",
        "[ moleculetype ]",
        f"{molname} 3",
        "",
        "[ atoms ]",
        "; nr type resnr residue atom cgnr charge mass",
    ]
    for atom in atoms:
        lines.append(
            f"{atom['nr']:6d} {atom['type']:<10s} {atom['resnr']:4d} "
            f"{atom['residue']:<6s} {atom['atom']:<8s} {atom['cgnr']:6d} "
            f"{atom['charge']:11.6f} {atom['mass']:10.5f}"
        )
    dst.write_text("\n".join(lines) + "\n")
    return dst


def _itp_section_rows(itp_path: Path):
    current = None
    for ln in itp_path.read_text().splitlines():
        s = ln.strip()
        if s.startswith("["):
            current = s
            continue
        if current is None or not s or s.startswith(";"):
            continue
        yield current, ln

def _itp_atomtypes_block(itp_path: Path):
    lines = itp_path.read_text().splitlines()
    out = []
    in_section = False
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("[ atomtypes"):
            in_section = True
            out.append(ln)
            continue
        if in_section and s.startswith("["):
            break
        if in_section:
            out.append(ln)
    return out

def _row_core_tokens(row: str):
    return row.split(";", 1)[0].split()

def _parse_indexed_rows(itp_path: Path, atoms, section_prefix: str, nidx: int, proper_kind: Optional[str] = None):
    atom_types = {int(a["nr"]): a["type"] for a in atoms}
    out = []
    for header, row in _itp_section_rows(itp_path):
        h = header.lower()
        if not h.startswith(section_prefix.lower()):
            continue
        if proper_kind is not None:
            if proper_kind == "propers" and "propers" not in h:
                continue
            if proper_kind == "impropers" and "impropers" not in h:
                continue
        toks = _row_core_tokens(row)
        if len(toks) <= nidx:
            continue
        try:
            idx = tuple(int(x) for x in toks[:nidx])
        except ValueError:
            continue
        if any(i not in atom_types for i in idx):
            continue
        out.append({"idx": idx, "types": tuple(atom_types[i] for i in idx), "params": toks[nidx:]})
    return out

def _add_lookup(lookup: dict, key, params, multi: bool = False):
    if multi:
        lookup.setdefault(key, [])
        if params not in lookup[key]:
            lookup[key].append(params)
    elif key not in lookup:
        lookup[key] = params

def _build_bonded_param_lookups(template_itps: List[Path], template_atoms: List[list]):
    lookups = {"bonds": {}, "angles": {}, "propers": {}, "impropers": {}}
    for itp_path, atoms in zip(template_itps, template_atoms):
        for rec in _parse_indexed_rows(itp_path, atoms, "[ bonds", 2):
            key = tuple(sorted(rec["types"]))
            _add_lookup(lookups["bonds"], key, rec["params"])
        for rec in _parse_indexed_rows(itp_path, atoms, "[ angles", 3):
            key = rec["types"]
            rev = tuple(reversed(key))
            _add_lookup(lookups["angles"], key, rec["params"])
            _add_lookup(lookups["angles"], rev, rec["params"])
        for rec in _parse_indexed_rows(itp_path, atoms, "[ dihedrals", 4, proper_kind="propers"):
            key = rec["types"]
            rev = tuple(reversed(key))
            _add_lookup(lookups["propers"], key, rec["params"], multi=True)
            _add_lookup(lookups["propers"], rev, rec["params"], multi=True)
        # ACPYPE/GROMACS Amber impropers use the third atom as the central atom in these templates.
        for rec in _parse_indexed_rows(itp_path, atoms, "[ dihedrals", 4, proper_kind="impropers"):
            t = rec["types"]
            key = (t[2], tuple(sorted((t[0], t[1], t[3]))))
            lookups["impropers"].setdefault(key, {"neighbor_types": (t[0], t[1], t[3]), "params": rec["params"]})
    return lookups

def _full_chain_mol_for_fallback(mol_path: Path):
    mol = Chem.MolFromMolBlock(mol_path.read_text(), sanitize=False, removeHs=False)
    if mol is None:
        raise RuntimeError(f"failed to parse full chain MOL: {mol_path}")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol

def _mol_adjacency(mol: Chem.Mol):
    adj = {i + 1: set() for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx() + 1
        b = bond.GetEndAtomIdx() + 1
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _lookup_params_with_fallback(lookup: dict, key, default=None):
    if key in lookup:
        return lookup[key]
    rev = tuple(reversed(key)) if isinstance(key, tuple) else key
    if rev in lookup:
        return lookup[rev]
    if isinstance(key, tuple):
        # Trimer/tetramer templates can assign terminal aliphatic H as h1 while
        # the expanded full chain sees the analogous local environment as hc.
        # Keep this intentionally narrow; broader GAFF type aliasing would hide
        # real template-coverage problems.
        alt_keys = [()]
        for typ in key:
            aliases = [typ]
            if typ == "h1":
                aliases.append("hc")
            elif typ == "hc":
                aliases.append("h1")
            aliases = list(dict.fromkeys(aliases))
            alt_keys = [prefix + (alias,) for prefix in alt_keys for alias in aliases]
        for alt in alt_keys:
            if alt == key:
                continue
            if alt in lookup:
                return lookup[alt]
            alt_rev = tuple(reversed(alt))
            if alt_rev in lookup:
                return lookup[alt_rev]
    return default

def _match_neighbor_order(neighbors, neighbor_types, wanted_types, atom_types):
    remaining = list(neighbors)
    ordered = []
    for wt in wanted_types:
        found = None
        for n in remaining:
            if atom_types[n] == wt:
                found = n
                break
        if found is None:
            return None
        ordered.append(found)
        remaining.remove(found)
    return ordered

def _generate_bonded_terms_from_graph(full_mol, expanded_atoms, lookups):
    atom_types = {int(a["nr"]): a["type"] for a in expanded_atoms}
    adj = _mol_adjacency(full_mol)
    terms = {"bonds": [], "pairs": [], "angles": [], "propers": [], "impropers": []}
    missing = {"bonds": 0, "angles": 0, "propers": 0, "impropers": 0}

    for i in sorted(adj):
        for j in sorted(adj[i]):
            if i >= j:
                continue
            params = lookups["bonds"].get(tuple(sorted((atom_types[i], atom_types[j]))))
            if params is None:
                missing["bonds"] += 1
                continue
            terms["bonds"].append((i, j, params))

    for j in sorted(adj):
        neigh = sorted(adj[j])
        for a_pos in range(len(neigh)):
            for c_pos in range(a_pos + 1, len(neigh)):
                i, k = neigh[a_pos], neigh[c_pos]
                key = (atom_types[i], atom_types[j], atom_types[k])
                params = _lookup_params_with_fallback(lookups["angles"], key)
                if params is None:
                    missing["angles"] += 1
                    continue
                terms["angles"].append((i, j, k, params))

    seen_pairs = set()
    seen_propers = set()
    for j in sorted(adj):
        for k in sorted(adj[j]):
            if j >= k:
                continue
            for i in sorted(adj[j] - {k}):
                for l in sorted(adj[k] - {j}):
                    path = (i, j, k, l)
                    rev = tuple(reversed(path))
                    canon = min(path, rev)
                    if canon in seen_propers:
                        continue
                    seen_propers.add(canon)
                    pair = tuple(sorted((i, l)))
                    seen_pairs.add(pair)
                    key = (atom_types[i], atom_types[j], atom_types[k], atom_types[l])
                    param_list = _lookup_params_with_fallback(lookups["propers"], key, default=[])
                    if not param_list:
                        missing["propers"] += 1
                        continue
                    for params in param_list:
                        terms["propers"].append((i, j, k, l, params))
    terms["pairs"] = sorted((i, j, ["1"]) for i, j in seen_pairs)

    for center in sorted(adj):
        neigh = sorted(adj[center])
        if len(neigh) != 3:
            continue
        key = (atom_types[center], tuple(sorted(atom_types[n] for n in neigh)))
        rec = lookups["impropers"].get(key)
        if rec is None:
            continue
        ordered = _match_neighbor_order(neigh, [atom_types[n] for n in neigh], rec["neighbor_types"], atom_types)
        if ordered is None:
            missing["impropers"] += 1
            continue
        terms["impropers"].append((ordered[0], ordered[1], center, ordered[2], rec["params"]))

    return terms, missing

def _write_full_fallback_itp(dst: Path, atomtypes_block, atoms, terms, molname: str = "polymer"):
    lines = []
    if atomtypes_block:
        lines.extend(atomtypes_block)
        lines.append("")
    lines.extend([
        "[ moleculetype ]",
        ";name            nrexcl",
        f" {molname:<15s} 3",
        "",
        "[ atoms ]",
        ";   nr  type  resi  res  atom  cgnr     charge      mass",
    ])
    for atom in atoms:
        lines.append(
            f"{atom['nr']:6d} {atom['type']:>4s} {atom['resnr']:5d} {atom['residue']:>5s} "
            f"{atom['atom']:>5s} {atom['cgnr']:5d} {atom['charge']:12.6f} {atom['mass']:12.5f}"
        )
    lines.extend(["", "[ bonds ]", ";   ai     aj funct   length       force.c."])
    for i, j, params in terms["bonds"]:
        lines.append(f"{i:6d} {j:6d} " + " ".join(params))
    lines.extend(["", "[ pairs ]", ";   ai     aj funct"])
    for i, j, params in terms["pairs"]:
        lines.append(f"{i:6d} {j:6d} " + " ".join(params))
    lines.extend(["", "[ angles ]", ";   ai     aj     ak funct   angle       force.c."])
    for i, j, k, params in terms["angles"]:
        lines.append(f"{i:6d} {j:6d} {k:6d} " + " ".join(params))
    lines.extend(["", "[ dihedrals ] ; propers", ";   ai     aj     ak     al funct"])
    for i, j, k, l, params in terms["propers"]:
        lines.append(f"{i:6d} {j:6d} {k:6d} {l:6d} " + " ".join(params))
    if terms["impropers"]:
        lines.extend(["", "[ dihedrals ] ; impropers", ";   ai     aj     ak     al funct"])
        for i, j, k, l, params in terms["impropers"]:
            lines.append(f"{i:6d} {j:6d} {k:6d} {l:6d} " + " ".join(params))
    dst.write_text("\n".join(lines).rstrip() + "\n")
    return dst

def _attempt_polymer_trimer_fallback(spec: SystemSpec, workdir: Path, charge_method: str, atom_type: Optional[str]):
    if spec.n_repeat < 4:
        raise ValueError(f"trimer/tetramer fallback requires n_repeat >= 4, got {spec.n_repeat}")

    workspace = workdir / _TRIMER_FALLBACK_WORKSPACE_NAME
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    summary = {
        "traj_name": spec.name,
        "n_repeat": int(spec.n_repeat),
        "charge_method": charge_method,
        "atom_type": atom_type or "",
        "workspace": str(workspace),
        "status": "started",
        "max_attempts": int(_TRIMER_FALLBACK_MAX_ATTEMPTS),
        "strict_rebuild": bool(_TRIMER_FALLBACK_STRICT_REBUILD),
        "attempts": [],
    }
    tri_ac_dir = tet_ac_dir = None
    tri_itp = tet_itp = None
    tri_atoms = tet_atoms = None
    scheme = None
    selected_attempt = None
    last_exc = None

    try:
        for cfg in _fallback_attempt_configs():
            attempt_no = int(cfg["attempt"])
            attempt_dir = workspace / f"attempt_{attempt_no:02d}"
            attempt_rec = dict(cfg)
            attempt_rec["status"] = "started"
            try:
                log(
                    "[trimer-fallback] "
                    f"attempt {attempt_no}/{_TRIMER_FALLBACK_MAX_ATTEMPTS}: "
                    f"seed_offset={cfg['seed_offset']} shift={cfg['shift']:.2f} "
                    f"localopt_steps={cfg['localopt_steps']} repair_min_dist={cfg['repair_min_dist']:.2f}"
                )
                tri_mol, _, sig3, tri_quality = _build_typing_probe_oligomer(
                    spec,
                    3,
                    attempt_dir / "trimer",
                    seed_offset=int(cfg["seed_offset"]),
                    shift=float(cfg["shift"]),
                    force_localopt=bool(cfg["force_localopt"]),
                    localopt_steps=int(cfg["localopt_steps"]),
                    repair_min_dist=float(cfg["repair_min_dist"]),
                )
                tet_mol, _, sig4, tet_quality = _build_typing_probe_oligomer(
                    spec,
                    4,
                    attempt_dir / "tetramer",
                    seed_offset=int(cfg["seed_offset"]),
                    shift=float(cfg["shift"]),
                    force_localopt=bool(cfg["force_localopt"]),
                    localopt_steps=int(cfg["localopt_steps"]),
                    repair_min_dist=float(cfg["repair_min_dist"]),
                )
                scheme = _infer_insert_scheme(sig3, sig4)
                attempt_rec["probe_build"] = {
                    "trimer": tri_quality,
                    "tetramer": tet_quality,
                }
                attempt_rec["insert_positions"] = ["end" if s["at_end"] else int(s["pos"]) for s in scheme]

                tri_ac_dir = run_acpype(
                    tri_mol,
                    "polymer_trimer",
                    attempt_dir / "trimer_topology",
                    charge=0,
                    charge_method=charge_method,
                    atom_type=atom_type,
                    force_recompute=True,
                )
                tet_ac_dir = run_acpype(
                    tet_mol,
                    "polymer_tetramer",
                    attempt_dir / "tetramer_topology",
                    charge=0,
                    charge_method=charge_method,
                    atom_type=atom_type,
                    force_recompute=True,
                )
                tri_itp, _ = locate_gmx_itp(tri_ac_dir)
                tet_itp, _ = locate_gmx_itp(tet_ac_dir)
                tri_atoms = _parse_itp_atoms(tri_itp)
                tet_atoms = _parse_itp_atoms(tet_itp)
                attempt_rec.update({
                    "status": "acpype_ok",
                    "trimer_acpype_dir": str(tri_ac_dir),
                    "tetramer_acpype_dir": str(tet_ac_dir),
                    "trimer_itp": str(tri_itp),
                    "tetramer_itp": str(tet_itp),
                    "trimer_atoms": len(tri_atoms),
                    "tetramer_atoms": len(tet_atoms),
                })
                summary["attempts"].append(attempt_rec)
                selected_attempt = attempt_rec
                log(f"[trimer-fallback] attempt {attempt_no} succeeded")
                break
            except Exception as attempt_exc:
                last_exc = attempt_exc
                failure_text = "\n".join([
                    str(attempt_exc),
                    _acpype_failure_text(attempt_dir / "trimer_topology", "polymer_trimer", attempt_exc),
                    _acpype_failure_text(attempt_dir / "tetramer_topology", "polymer_tetramer", attempt_exc),
                ])
                tail = "\n".join(failure_text.strip().splitlines()[-30:])
                attempt_rec.update({
                    "status": "failed",
                    "error_type": type(attempt_exc).__name__,
                    "error": str(attempt_exc),
                    "error_tail": tail,
                })
                summary["attempts"].append(attempt_rec)
                summary.update({
                    "status": "retrying" if attempt_no < _TRIMER_FALLBACK_MAX_ATTEMPTS else "failed",
                    "last_error_type": type(attempt_exc).__name__,
                    "last_error": str(attempt_exc),
                })
                (workspace / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
                log(f"[trimer-fallback] attempt {attempt_no} failed: {type(attempt_exc).__name__}: {attempt_exc}")

        if selected_attempt is None or tri_itp is None or tet_itp is None or tri_atoms is None or tet_atoms is None or scheme is None:
            raise RuntimeError(
                f"all trimer/tetramer fallback attempts failed; summary={workspace / 'summary.json'}; "
                f"last={type(last_exc).__name__ if last_exc else 'unknown'}: {last_exc}"
            )

        insert_specs = []
        for spec_insert in scheme:
            insert_specs.append({
                "pos": int(spec_insert["pos"]),
                "at_end": bool(spec_insert["at_end"]),
                "block": tet_atoms[spec_insert["j1"]:spec_insert["j2"]],
            })
        expanded_atoms = _apply_insert_scheme(tri_atoms, insert_specs, max(0, spec.n_repeat - 3))
        charge_rebalance = _rebalance_fallback_polymer_charges(expanded_atoms, target_total=0.0)

        full_mol = _full_chain_mol_for_fallback(STRUCT_DIR / "chain.mol")
        if full_mol.GetNumAtoms() != len(expanded_atoms):
            raise ValueError(
                f"full-chain atom count mismatch for fallback: mol={full_mol.GetNumAtoms()} expanded={len(expanded_atoms)}"
            )
        lookups = _build_bonded_param_lookups([tri_itp, tet_itp], [tri_atoms, tet_atoms])
        terms, missing = _generate_bonded_terms_from_graph(full_mol, expanded_atoms, lookups)
        full_itp = workspace / "polymer_trimer_fallback_full_GMX.itp"
        _write_full_fallback_itp(
            full_itp,
            _itp_atomtypes_block(tet_itp) or _itp_atomtypes_block(tri_itp),
            expanded_atoms,
            terms,
            molname="polymer",
        )

        summary.update({
            "status": "full_bonded_generated",
            "selected_attempt": selected_attempt,
            "trimer_atoms": len(tri_atoms),
            "tetramer_atoms": len(tet_atoms),
            "expanded_atoms": len(expanded_atoms),
            "full_chain_atoms": int(full_mol.GetNumAtoms()),
            "insert_positions": ["end" if s["at_end"] else int(s["pos"]) for s in insert_specs],
            "insert_lengths": [len(s["block"]) for s in insert_specs],
            "fallback_itp": str(full_itp),
            "trimer_total_charge": _sum_atom_charge(tri_atoms),
            "tetramer_total_charge": _sum_atom_charge(tet_atoms),
            "insert_block_total_charges": [_sum_atom_charge(s["block"]) for s in insert_specs],
            "charge_rebalance": charge_rebalance,
            "n_bonds": len(terms["bonds"]),
            "n_pairs": len(terms["pairs"]),
            "n_angles": len(terms["angles"]),
            "n_propers": len(terms["propers"]),
            "n_impropers": len(terms["impropers"]),
            "missing_terms": missing,
        })

        pd.DataFrame(expanded_atoms).to_csv(workspace / "expanded_atoms_preview.csv", index=False)
        _write_atoms_only_itp(
            workspace / "polymer_trimer_fallback_atoms_only.itp",
            expanded_atoms,
            molname=tri_atoms[0]["residue"] if tri_atoms else "POL",
        )
        (workspace / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
    except Exception as exc:
        summary.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        (workspace / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        raise

# 사용 예시:
polymer_pdb = STRUCT_DIR/f"{spec.name}_chain_fix.pdb"
polymer_shared_cache_dir = _polymer_shared_cache_dir(spec, _ACPYPE_CHARGE_METHOD, _ACPYPE_ATOM_TYPE, 0)
# Cache hit이면 polymer PDB repair/ACPYPE를 통째로 건너뛴다. 같은 PSMILES+DP에서는 topology가 동일하다.
if _has_gmx_acpype_outputs(polymer_shared_cache_dir):
    log(f"[acpype-cache] polymer shared cache available before repair: {polymer_shared_cache_dir}")
    polymer_input = polymer_pdb
    polymer_min_dist = float("nan")
else:
    if _REBUILD_FROM_CHAIN_MOL:
        polymer_pdb_rebuilt = polymer_pdb.with_name(polymer_pdb.stem + "_frommol.pdb")
        polymer_pdb = rebuild_polymer_pdb_from_chain_mol(STRUCT_DIR / "chain.mol", polymer_pdb_rebuilt, resname=spec.polymer_resname_coords)
    if _ACPYPE_SKIP_REPAIR:
        polymer_min_dist = float("nan")
        polymer_input = polymer_pdb
        log(f"[atomtyping] skipping RDKit repair before ACPYPE; using {polymer_input}")
    else:
        polymer_pdb_fixed = polymer_pdb.with_name(polymer_pdb.stem + "_acpypefix.pdb")
        polymer_pdb, polymer_min_dist = repair_polymer_pdb_for_acpype(polymer_pdb, polymer_pdb_fixed, min_dist=_ACPYPE_MIN_DIST, seed=123)
        if polymer_min_dist < 0.50 and not _ACPYPE_FORCE:
            print("__atomtyping_error__:acpype_atoms_too_close", flush=True)
            raise RuntimeError(f"[atomtyping:acpype_atoms_too_close] repaired polymer still too close ({polymer_min_dist:.3f} A)")
        polymer_input = polymer_pdb

# polymer: PDB로 ACPYPE (네 코드의 핵심 패치 유지)
try:
    polymer_acpype = run_acpype(
        polymer_input,
        "polymer",
        TOPO_DIR,
        charge=0,
        charge_method=_ACPYPE_CHARGE_METHOD,
        atom_type=_ACPYPE_ATOM_TYPE,
        force_recompute=_ACPYPE_FORCE,
        shared_cache_dir=polymer_shared_cache_dir,
    )
    polymer_itp, _ = locate_gmx_itp(polymer_acpype)
except Exception as exc:
    failure_text = _acpype_failure_text(TOPO_DIR, "polymer", exc)
    handled_by_fallback = False
    if _TRIMER_FALLBACK_ENABLED and _looks_like_antechamber_typing_failure(failure_text):
        log("[atomtyping] polymer ACPYPE looks like antechamber typing failure; trying trimer/tetramer fallback")
        try:
            fb = _attempt_polymer_trimer_fallback(spec, TOPO_DIR, _ACPYPE_CHARGE_METHOD, _ACPYPE_ATOM_TYPE)
        except Exception as fb_exc:
            print("__atomtyping_error__:polymer_trimer_fallback_failed", flush=True)
            raise RuntimeError(
                f"[atomtyping:polymer_trimer_fallback_failed] {type(fb_exc).__name__}: {fb_exc}"
            ) from exc
        if fb.get("status") == "full_bonded_generated" and fb.get("fallback_itp"):
            polymer_itp = Path(fb["fallback_itp"])
            handled_by_fallback = True
            log(f"[atomtyping] using trimer/tetramer bonded fallback polymer ITP: {polymer_itp}")
        else:
            print("__atomtyping_error__:polymer_trimer_fallback_incomplete", flush=True)
            raise RuntimeError(
                "[atomtyping:polymer_trimer_fallback_incomplete] "
                f"fallback prepared at {fb['workspace']} but bonded-term reconstruction is incomplete"
            ) from exc
    if not handled_by_fallback:
        print("__atomtyping_error__:antechamber_typing_failure", flush=True)
        raise RuntimeError(
            f"[atomtyping:antechamber_typing_failure] {type(exc).__name__}: {exc}"
        ) from exc

# tfsi: (네 코드의 tfsi.mol 요구를 제거) tfsi.pdb로 ACPYPE
tfsi_input = tfsi_pdb
tfsi_shared_cache_dir = _tfsi_shared_cache_dir(tfsi_input, charge_method="gas", atom_type=None, charge=-1)
tfsi_acpype = run_acpype(
    tfsi_input,
    "tfsi",
    TOPO_DIR,
    charge=-1,
    charge_method="gas",
    force_recompute=False,
    shared_cache_dir=tfsi_shared_cache_dir,
)
tfsi_itp, _ = locate_gmx_itp(tfsi_acpype)

# li: ACPYPE 우회, 강제 fallback
li_acpype_dir = TOPO_DIR / "li.acpype"
li_acpype_dir.mkdir(parents=True, exist_ok=True)
write_monatomic_itp(li_acpype_dir, "li", LI_AMBER99SB_PARAMS, note="forced amber99sb-ildn Li fallback")
li_itp = li_acpype_dir / "li_GMX.itp"

# ---- atomtypes merge & sanitize itp ----
def split_atomtypes_and_body(itp_path: Path):
    lines = itp_path.read_text().splitlines()
    at, body = [], []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.lower().startswith("[ atomtypes"):
            at.append(lines[i]); i += 1
            while i < len(lines):
                t = lines[i].strip()
                if t.startswith("[") and not t.lower().startswith("[ atomtypes"):
                    break
                at.append(lines[i]); i += 1
            continue
        body.append(lines[i]); i += 1
    return at, body

def write_merged_atomtypes(out_path: Path, itps: List[Path]) -> Path:
    seen = set()
    merged = ["[ atomtypes ]", "; name  mass  charge  ptype  sigma  epsilon"]
    for p in itps:
        at, _ = split_atomtypes_and_body(p)
        for ln in at:
            u = ln.strip()
            if not u or u.startswith(";") or u.startswith("["):
                continue
            key = u.split()[0]
            if key not in seen:
                merged.append(ln)
                seen.add(key)
    out_path.write_text("\n".join(merged) + "\n")
    return out_path

def write_sanitized_itp(src: Path, dst: Path) -> Path:
    _, body = split_atomtypes_and_body(src)
    dst.write_text("\n".join(body).rstrip() + "\n")
    return dst


def rewrite_itp_resname(itp_path: Path, resname: str) -> Path:
    lines = itp_path.read_text().splitlines()
    out = []
    in_atoms = False
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("[ atoms ]"):
            in_atoms = True
            out.append(ln)
            continue
        if in_atoms and s.startswith("["):
            in_atoms = False
            out.append(ln)
            continue
        if in_atoms:
            if not s or s.startswith(";"):
                out.append(ln)
                continue
            core, *rest = ln.split(";", 1)
            parts = core.split()
            if len(parts) >= 4:
                parts[3] = resname
                core_new = " ".join(parts)
                ln = core_new + (" ;" + rest[0] if rest else "")
            out.append(ln)
        else:
            out.append(ln)
    itp_path.write_text("\n".join(out).rstrip() + "\n")
    return itp_path

def extract_moleculetype_name(itp_path: Path) -> str:
    in_mt = False
    for s in itp_path.read_text().splitlines():
        t = s.strip()
        if not t or t.startswith(";"):
            continue
        if t.startswith("["):
            in_mt = t.lower().startswith("[ moleculetype")
            continue
        if in_mt:
            return t.split()[0]
    raise ValueError(f"moleculetype name not found in {itp_path}")

pol_src  = TOPO_DIR / polymer_itp.name
tfsi_src = TOPO_DIR / tfsi_itp.name
li_src   = TOPO_DIR / li_itp.name
shutil.copy2(polymer_itp, pol_src)
shutil.copy2(tfsi_itp, tfsi_src)
shutil.copy2(li_itp, li_src)

all_atomtypes = write_merged_atomtypes(TOPO_DIR / "all_atomtypes.itp", [pol_src, tfsi_src, li_src])
pol_clean  = write_sanitized_itp(pol_src,  TOPO_DIR / "polymer_clean.itp")
tfsi_clean = write_sanitized_itp(tfsi_src, TOPO_DIR / "tfsi_clean.itp")
rewrite_itp_resname(tfsi_clean, spec.anion_resname_coords)
li_clean   = write_sanitized_itp(li_src,   TOPO_DIR / "li_clean.itp")

pol_mt  = extract_moleculetype_name(pol_clean)
tfsi_mt = extract_moleculetype_name(tfsi_clean)
li_mt   = extract_moleculetype_name(li_clean)
log(f"moleculetype: polymer={pol_mt}, li={li_mt}, tfsi={tfsi_mt}")

def write_topology_ordered(top_path: Path, atomtypes_itp: Path,
                           pol_itp: Path, tfsi_itp: Path, li_itp: Path,
                           molname_poly: str, molname_li: str, molname_tfsi: str, spec: SystemSpec):
    lines = [
        "[ defaults ]",
        "; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ",
        "1         2          yes        0.5      0.8333333333",
        "",
        f'#include "{atomtypes_itp.name}"',
        "",
        f'#include "{pol_itp.name}"',
        "#ifdef POSRES",
        '#include "posre_POL.itp"',
        "#endif",
        f'#include "{tfsi_itp.name}"',
        f'#include "{li_itp.name}"',
        "",
        "[ system ]",
        f"{spec.name}",
        "",
        "[ molecules ]",
        f"{molname_poly} {spec.n_chains}",
        f"{molname_li}   {spec.li_tfsi_pairs}",
        f"{molname_tfsi} {spec.li_tfsi_pairs}",
        ""
    ]
    top_path.write_text("\n".join(lines))

TOPOL_TOP = TOPO_DIR / "topol.top"
write_topology_ordered(TOPOL_TOP, all_atomtypes, pol_clean, tfsi_clean, li_clean, pol_mt, li_mt, tfsi_mt, spec)

# ---- GRO residue renumber + atom name sync ----
def itp_atom_names(itp_path: Path) -> List[str]:
    names, in_atoms = [], False
    for ln in itp_path.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        if s.startswith("["):
            in_atoms = s.lower().startswith("[ atoms")
            continue
        if in_atoms:
            core = ln.split(";",1)[0].split()
            if len(core) >= 5:
                names.append(core[4])
    if not names:
        raise ValueError(f"[ atoms ] empty: {itp_path}")
    return names

def count_pdb_atoms(pdb_path: Path) -> int:
    n = 0
    with pdb_path.open() as f:
        for ln in f:
            if ln.startswith(("ATOM","HETATM")):
                n += 1
    return n

def sanity_check_polymer_atoms(polymer_pdb: Path, polymer_itp: Path):
    n_pdb = count_pdb_atoms(polymer_pdb)
    n_itp = len(itp_atom_names(polymer_itp))
    if n_pdb != n_itp:
        raise RuntimeError(
            f"[polymer atom mismatch] PDB(chain)={n_pdb}, ITP[atoms]={n_itp}. "
            "pSMILES/DP 변경 후 이전 ITP가 섞였을 가능성이 큼. TOPO_DIR를 비우고 다시 ACPYPE 하세요."
        )
    log(f"[check] polymer atoms OK: {n_pdb}")

sanity_check_polymer_atoms(polymer_pdb, pol_clean)

def load_itp_atomnames(itp_path: Path) -> List[str]:
    names: List[str] = []
    in_atoms = False
    for ln in itp_path.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        if s.startswith("["):
            in_atoms = s.lower().startswith("[ atoms")
            continue
        if in_atoms:
            core = ln.split(";", 1)[0].split()
            if len(core) >= 5:
                names.append(core[4])
    if not names:
        raise ValueError(f"[ atoms ] 섹션에서 이름을 찾지 못함: {itp_path}")
    return names

def _read_gro(gro: Path):
    lines = gro.read_text().splitlines()
    title = lines[0]
    natoms = int(lines[1].strip())
    body = lines[2:2+natoms]
    box = lines[2+natoms].rstrip()
    if len(body) != natoms:
        raise ValueError(f"natoms mismatch: header={natoms}, body={len(body)}")
    return title, natoms, body, box

def _set_resnr(ln: str, resnr: int) -> str:
    return f"{resnr:5d}" + ln[5:]

def _set_resname(ln: str, resname: str) -> str:
    # GRO resname field: columns 6-10 (0-based 5:10), width=5
    res5 = f"{resname:<5s}"[:5]
    return ln[:5] + res5 + ln[10:]

def _set_atomname(ln: str, atomname: str) -> str:
    # GRO atomname field: columns 11-15 (0-based 10:15), width=5
    return ln[:10] + f"{atomname:>5s}"[:5] + ln[15:]

def _set_atomnr(ln: str, atomnr: int) -> str:
    # GRO atomnr field: columns 16-20 (0-based 15:20), width=5
    return ln[:15] + f"{atomnr:5d}" + ln[20:]

def fix_gro_by_counts(
    gro_in: Path,
    gro_out: Path,
    *,
    order: List[str],                  # 예: ["POL","LI","TFS"] (너의 spec에 맞춰)
    n_mols: Dict[str,int],             # resname -> 분자 개수
    itp_atomnames: Dict[str,List[str]] # resname -> [atoms] name list (1 molecule 기준)
):
    title, natoms, body, box = _read_gro(gro_in)

    # 기대 natoms 계산
    expected_natoms = 0
    for r in order:
        expected_natoms += n_mols[r] * len(itp_atomnames[r])

    if natoms != expected_natoms:
        # 여기서 바로 “TFSI가 아예 안 들어갔는지 / resname이 깨졌는지” 원인을 좁힐 수 있음
        raise RuntimeError(
            f"[natoms mismatch] GRO natoms={natoms}, expected={expected_natoms}.\n"
            f"  expected breakdown: " +
            ", ".join([f"{r}:{n_mols[r]}*{len(itp_atomnames[r])}={n_mols[r]*len(itp_atomnames[r])}" for r in order]) + "\n"
            f"이 경우 (1) packmol output에 어떤 종이 빠졌거나, (2) ACPYPE/ITP 원자수가 packmol 구조와 다릅니다."
        )

    out_body = []
    cursor = 0
    new_resnr = 0
    new_atomnr = 0

    for r in order:
        names = itp_atomnames[r]
        per = len(names)
        for _ in range(n_mols[r]):
            new_resnr += 1
            chunk = body[cursor:cursor+per]
            if len(chunk) != per:
                raise RuntimeError(f"Unexpected EOF while slicing {r}: need {per}, got {len(chunk)}")
            cursor += per

            for i, ln in enumerate(chunk):
                ln2 = ln
                ln2 = _set_resnr(ln2, new_resnr)
                ln2 = _set_resname(ln2, r)
                ln2 = _set_atomname(ln2, names[i])
                new_atomnr += 1
                ln2 = _set_atomnr(ln2, new_atomnr)
                out_body.append(ln2)

    if cursor != natoms:
        raise RuntimeError(f"cursor mismatch: cursor={cursor}, natoms={natoms}")

    gro_out.write_text("\n".join([title, f"{natoms:5d}", *out_body, box]) + "\n")
    return gro_out

# ---- 사용부 (너의 변수 그대로) ----
GRO_IN = MD_DIR / "conf_initial.gro"

poly_names = load_itp_atomnames(pol_clean)
li_names   = load_itp_atomnames(li_clean)
tfsi_names = load_itp_atomnames(tfsi_clean)

# 실제 GRO에서 TFSI resname이 'TFSI'가 아니라 'TFS'일 수도 있음 (너 코드에서 [:3] 씀)
# spec.anion_resname_coords 값 그대로 쓰는 게 안전
order = [spec.polymer_resname_coords, spec.cation_resname_coords, spec.anion_resname_coords]

# Sync n_chains to what packmol actually used in this trajectory.
def infer_polymer_count_from_packmol(packmol_inp: Path, polymer_pdb: Path) -> Optional[int]:
    if not packmol_inp.exists():
        return None
    lines = packmol_inp.read_text().splitlines()
    target_name = polymer_pdb.name
    target_abs = str(polymer_pdb.resolve())
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s.lower().startswith("structure "):
            continue
        path = s.split(None, 1)[1].strip().strip(chr(34)).strip(chr(39))
        if path == target_abs or Path(path).name == target_name:
            for j in range(i + 1, min(i + 20, len(lines))):
                m = re.match(r"\s*number\s+(\d+)", lines[j], flags=re.I)
                if m:
                    return int(m.group(1))
            break
    return None

packmol_polymer_ref = STRUCT_DIR / f"{spec.name}_chain_fix.pdb"
packmol_used_n = infer_polymer_count_from_packmol(PACKMOL_DIR / "packmol.inp", packmol_polymer_ref)
if packmol_used_n is not None and packmol_used_n != int(spec.n_chains):
    log(f"[atomtyping] n_chains override from packmol.inp: {spec.n_chains} -> {packmol_used_n}")
    spec.n_chains = int(packmol_used_n)

# Keep topology molecule counts synced with packmol-updated n_chains.
write_topology_ordered(TOPOL_TOP, all_atomtypes, pol_clean, tfsi_clean, li_clean, pol_mt, li_mt, tfsi_mt, spec)

n_mols = {
    spec.polymer_resname_coords: spec.n_chains,
    spec.cation_resname_coords:  spec.li_tfsi_pairs,
    spec.anion_resname_coords:   spec.li_tfsi_pairs,
}

itp_map = {
    spec.polymer_resname_coords: poly_names,
    spec.cation_resname_coords:  li_names,
    spec.anion_resname_coords:   tfsi_names,
}

STRUCT_FIXED = MD_DIR / "conf_initial_fixed.gro"
fix_gro_by_counts(GRO_IN, STRUCT_FIXED, order=order, n_mols=n_mols, itp_atomnames=itp_map)
print("Wrote:", STRUCT_FIXED)


# GRO natoms
_, natoms, _, _ = _read_gro(MD_DIR/"conf_initial.gro")
print("GRO natoms =", natoms)

# 기대 natoms
exp = (
    spec.n_chains * len(load_itp_atomnames(pol_clean))
    + spec.li_tfsi_pairs * len(load_itp_atomnames(li_clean))
    + spec.li_tfsi_pairs * len(load_itp_atomnames(tfsi_clean))
)
print("expected natoms =", exp)
print("diff =", natoms - exp)


# ---- charge scaling + system neutrality fix ----
def rescale_itp_total_charge(itp_path: Path, *, scale=None, target_total=None, fix_on="maxabs"):
    lines = itp_path.read_text().splitlines()
    in_atoms=False; idx=[]
    for i, ln in enumerate(lines):
        s=ln.strip()
        if s.startswith("["):
            in_atoms = s.lower().startswith("[ atoms")
            continue
        if in_atoms and s and not s.startswith(";"):
            core = ln.split(";",1)[0].split()
            if len(core) >= 7:
                idx.append(i)
    if not idx:
        raise ValueError(f"[ atoms ] not found: {itp_path}")

    charges=[]
    for i in idx:
        core = lines[i].split(";",1)[0].split()
        charges.append(float(core[6]))

    new = [q*scale for q in charges] if scale is not None else charges[:]
    if target_total is not None:
        delta = target_total - sum(new)
        if abs(delta) > 1e-12:
            j = max(range(len(new)), key=lambda k: abs(new[k])) if fix_on=="maxabs" else len(new)-1
            new[j] += delta

    for i, q in zip(idx, new):
        left, *comm = lines[i].split(";",1)
        parts = left.split()
        parts[6] = f"{q:.6f}"
        line = " ".join(parts)
        if comm: line += " ;" + comm[0]
        lines[i] = line

    itp_path.write_text("\n".join(lines) + "\n")
    return sum(charges), sum(new)

LAMMPS_TFSI_FQ07_CHARGES = {
    "N01": -0.2982,
    "S02": +0.3395,
    "O03": -0.2513,
    "O04": -0.2513,
    "C05": +0.2100,
    "F06": -0.0826,
    "F07": -0.0826,
    "F08": -0.0826,
    "S09": +0.3395,
    "O10": -0.2513,
    "O11": -0.2513,
    "C12": +0.2100,
    "F13": -0.0826,
    "F14": -0.0826,
    "F15": -0.0826,
}

def apply_itp_atomwise_charges(
    itp_path: Path,
    charge_map: Dict[str, float],
    *,
    scale: float = 1.0,
    target_total: Optional[float] = None,
    fix_on: str = "maxabs",
):
    lines = itp_path.read_text().splitlines()
    in_atoms = False
    idx = []
    atom_names = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("["):
            in_atoms = s.lower().startswith("[ atoms")
            continue
        if in_atoms and s and not s.startswith(";"):
            core = ln.split(";", 1)[0].split()
            if len(core) >= 7:
                idx.append(i)
                atom_names.append(core[4])
    if not idx:
        raise ValueError(f"[ atoms ] not found: {itp_path}")

    missing = [name for name in atom_names if name not in charge_map]
    if missing:
        raise KeyError(f"missing atom-wise charges for {itp_path.name}: {sorted(set(missing))}")

    new = [float(charge_map[name]) * float(scale) for name in atom_names]
    if target_total is not None:
        delta = float(target_total) - sum(new)
        if abs(delta) > 1e-12:
            j = max(range(len(new)), key=lambda k: abs(new[k])) if fix_on == "maxabs" else len(new) - 1
            new[j] += delta

    for i, q in zip(idx, new):
        left, *comm = lines[i].split(";", 1)
        parts = left.split()
        parts[6] = f"{q:.6f}"
        line = " ".join(parts)
        if comm:
            line += " ;" + comm[0]
        lines[i] = line

    itp_path.write_text("\n".join(lines) + "\n")
    return sum(new)

def canonicalize_tfsi_clean_itp(
    itp_path: Path,
    *,
    scale: float = 1.0,
    target_total: Optional[float] = None,
    resname: str = "TFSI",
):
    lines = itp_path.read_text().splitlines()
    in_atoms = False
    idx = []
    atom_names = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("["):
            in_atoms = s.lower().startswith("[ atoms")
            continue
        if in_atoms and s and not s.startswith(";"):
            core = ln.split(";", 1)[0].split()
            if len(core) >= 7:
                idx.append(i)
                atom_names.append(core[4])
    if not idx:
        raise ValueError(f"[ atoms ] not found: {itp_path}")

    missing = [name for name in atom_names if name not in LAMMPS_TFSI_FQ07_CHARGES]
    if missing:
        raise KeyError(f"missing canonical TFSI charges for {itp_path.name}: {sorted(set(missing))}")

    new = [float(LAMMPS_TFSI_FQ07_CHARGES[name]) * float(scale) for name in atom_names]
    if target_total is not None:
        delta = float(target_total) - sum(new)
        if abs(delta) > 1e-12:
            j = max(range(len(new)), key=lambda k: abs(new[k]))
            new[j] += delta

    for i, q in zip(idx, new):
        left, *comm = lines[i].split(";", 1)
        parts = left.split()
        if len(parts) >= 4:
            parts[3] = resname
        parts[6] = f"{q:.6f}"
        line = " ".join(parts)
        if comm:
            line += " ;" + comm[0]
        lines[i] = line

    itp_path.write_text("\n".join(lines) + "\n")
    return sum(new)

def check_system_charge(topol_top: Path, topo_dir: Path):
    txt = topol_top.read_text()
    incs = re.findall(r'#include\s+"([^"]+)"', txt)

    type_charge = {}
    for inc in incs:
        p = topo_dir / inc
        if not p.exists():
            continue
        try:
            mt = extract_moleculetype_name(p)
        except ValueError:
            continue
        total = 0.0
        in_atoms = False
        for s in p.read_text().splitlines():
            u = s.strip()
            if not u or u.startswith(";"):
                continue
            if u.startswith("["):
                in_atoms = u.lower().startswith("[ atoms")
                continue
            if in_atoms and len(u.split()) >= 7:
                total += float(u.split()[6])
        type_charge[mt] = total

    pairs = []
    m = re.search(r"\[ molecules \](.*)", txt, flags=re.S|re.I)
    if m:
        for ln in m.group(1).splitlines():
            u = ln.strip()
            if u and not u.startswith(";"):
                pr = u.split()
                if len(pr)>=2 and pr[1].isdigit():
                    pairs.append((pr[0], int(pr[1])))

    total_charge = sum(type_charge.get(mt,0.0)*n for mt,n in pairs)
    return total_charge, pairs, type_charge

z_li = float(spec.li_charge_scale)
z_an = float(spec.anion_charge_scale) if spec.anion_charge_scale is not None else z_li
log(f"[charge-scale] z_li={z_li:.4f}, z_an={z_an:.4f}, tfsi_model={spec.tfsi_charge_model}")

# (1) Li/TFSI를 각각 목표 전하(±z)로 강제
rescale_itp_total_charge(li_clean, scale=z_li, target_total=+z_li)
tfsi_charge_model = str(getattr(spec, "tfsi_charge_model", "acpype") or "acpype").strip().lower()
if tfsi_charge_model in ("lammps", "lammps_fq07", "fq07", "iff"):
    tfsi_scale = float(z_an) / 0.7
    tfsi_total = canonicalize_tfsi_clean_itp(
        tfsi_clean,
        scale=tfsi_scale,
        target_total=-z_an,
        resname=spec.anion_resname_coords,
    )
else:
    _, tfsi_total = rescale_itp_total_charge(tfsi_clean, scale=z_an, target_total=-z_an)

# (2) 여기서는 TFSI/Li에 잔여 총전하를 떠넘기지 않는다.
#     system total charge는 이후 inter-phase sanity pass에서 polymer/TFSI 원인을 명확히 진단한다.
total_q2, pairs2, type_charge2 = check_system_charge(TOPOL_TOP, TOPO_DIR)
poly_total_q = type_charge2.get(pol_mt, 0.0) * spec.n_chains
li_total_q = type_charge2.get(li_mt, 0.0) * spec.li_tfsi_pairs
tfsi_total_q = type_charge2.get(tfsi_mt, 0.0) * spec.li_tfsi_pairs
tfsi_mol_q = type_charge2.get(tfsi_mt, 0.0)
li_mol_q = type_charge2.get(li_mt, 0.0)
charge_diagnosis = {
    "system_q": float(total_q2),
    "polymer_q_chain": float(type_charge2.get(pol_mt, 0.0)),
    "polymer_total_q": float(poly_total_q),
    "li_q_mol": float(li_mol_q),
    "li_total_q": float(li_total_q),
    "tfsi_q_mol": float(tfsi_mol_q),
    "tfsi_total_q": float(tfsi_total_q),
    "tfsi_model": tfsi_charge_model,
    "tfsi_canonical_total": float(tfsi_total),
}
(TOPO_DIR / "charge_diagnosis_atomtyping.json").write_text(json.dumps(charge_diagnosis, ensure_ascii=False, indent=2))
log(
    f"system total charge(after li/tfsi set) = {total_q2:.3e} | "
    f"polymer_total={poly_total_q:.3e} li_total={li_total_q:.3e} tfsi_total={tfsi_total_q:.3e}"
)

# ---- posres (polymer only, POSRES define) ----
def itp_atoms_table(itp_path: Path):
    rows, in_atoms = [], False
    for ln in itp_path.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        if s.startswith("["):
            in_atoms = s.lower().startswith("[ atoms")
            continue
        if in_atoms:
            core = ln.split(";", 1)[0].split()
            if len(core) >= 7:
                rows.append({"id": int(core[0]), "name": core[4]})
    if not rows:
        raise ValueError(f"[ atoms ] parse failed: {itp_path}")
    return rows

def write_posre_for_itp(src_itp: Path, out_itp: Path, fc: float=1000.0, heavy_only: bool=True):
    rows = itp_atoms_table(src_itp)
    lines = ["[ position_restraints ]", "; ai  funct  fcx   fcy   fcz"]
    for r in rows:
        nm = r["name"]
        if heavy_only and nm.upper().startswith("H"):
            continue
        lines.append(f"{r['id']:5d}    1    {fc:.1f} {fc:.1f} {fc:.1f}")
    out_itp.write_text("\n".join(lines) + "\n")

write_posre_for_itp(pol_clean, TOPO_DIR / "posre_POL.itp", fc=1000.0, heavy_only=True)

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

def _sync_n_chains_from_topology_for_density() -> None:
    counts = _read_topology_molecule_counts(TOPOL_TOP)
    if not counts:
        warnings.warn(f"[density] could not read [ molecules ] from {TOPOL_TOP}; using spec.n_chains={spec.n_chains}")
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

# ---- MDP writing ----
def write_mdp(path: Path, params: dict):
    lines = []
    for k, v in params.items():
        if v is None:
            continue
        lines.append(f"{k:<24s} = {v}")
    path.write_text("\n".join(lines) + "\n")

# --- Option A (LAMMPS-like NH/PR damping) ---
OPT_A = {
    "tcoupl": "nose-hoover",
    "tau_t": 2.0,
    "tau_t_big": 5.0,
    "pcoupl": "Parrinello-Rahman",
    "tau_p": 20.0,
    "tau_p_big": 100.0,
}

# Safer GPU mode: force nonbonded/PME onto GPU, keep update/bonded on CPU.
# CPU fallback is disabled by default so a failed GPU run is not silently
# replaced by a long CPU-only production.
GPU_SAFE_EXTRA = ["-nb", "gpu", "-pme", "gpu", "-bonded", "cpu", "-update", "cpu"]
ALLOW_CPU_FALLBACK = os.environ.get("GROMACS_ALLOW_CPU_FALLBACK", "0").strip().lower() in ("1", "true", "yes", "on")

def _production_bonded_gpu_enabled() -> bool:
    return os.environ.get("GROMACS_PRODUCTION_BONDED_GPU", "1").strip().lower() in ("1", "true", "yes", "on")

def _production_gpu_extra() -> List[str]:
    bonded = "gpu" if _production_bonded_gpu_enabled() else "cpu"
    return ["-nb", "gpu", "-pme", "gpu", "-bonded", bonded, "-update", "cpu"]

def _production_tcoupl() -> str:
    val = os.environ.get("GROMACS_PRODUCTION_TCOUPL", "").strip().lower()
    return val or OPT_A["tcoupl"]

def _production_tau_t() -> float:
    raw = os.environ.get("GROMACS_PRODUCTION_TAU_T", "").strip()
    return float(raw) if raw else float(OPT_A["tau_t"])

def _production_dt_fs() -> float:
    raw = os.environ.get("GROMACS_PRODUCTION_DT_FS", "").strip()
    return float(raw) if raw else 1.0

def _production_nsteps() -> int:
    return int(round((float(spec.production_ns) * 1_000_000.0) / _production_dt_fs()))

def _production_lincs_iter() -> int:
    raw = os.environ.get("GROMACS_PRODUCTION_LINCS_ITER", "").strip()
    return int(raw) if raw else 6

def _production_lincs_order() -> int:
    raw = os.environ.get("GROMACS_PRODUCTION_LINCS_ORDER", "").strip()
    return int(raw) if raw else 12

def _production_lincs_warnangle() -> int:
    raw = os.environ.get("GROMACS_PRODUCTION_LINCS_WARNANGLE", "").strip()
    return int(raw) if raw else 20

def _lincs_abort_warnings() -> int:
    raw = os.environ.get("GROMACS_LINCS_ABORT_WARNINGS", "").strip()
    return int(raw) if raw else 25

def mdp_common(spec: SystemSpec, *, dt_fs: Optional[float]=None,
               tcoupl: Optional[str]=None, tau_t: Optional[float]=None):
    dt_fs = spec.dt_fs if dt_fs is None else float(dt_fs)
    tcoupl = tcoupl or OPT_A["tcoupl"]
    tau_t = OPT_A["tau_t"] if tau_t is None else float(tau_t)
    return {
        "integrator":"md",
        "dt": f"{dt_fs*1e-3:.6f}",
        "cutoff-scheme":"Verlet",
        "nstlist":"20",
        "coulombtype":"PME",
        "rcoulomb":"1.0",
        "vdwtype":"Cut-off",
        "rvdw":"1.0",
        "DispCorr":"EnerPres",
        "fourierspacing":"0.14",
        "pbc":"xyz",
        "constraints":"h-bonds",
        "constraint-algorithm":"lincs",
        "lincs_iter":"3",
        "lincs_order":"8",
        "lincs-warnangle":"30",
        "tcoupl": tcoupl,
        "tc-grps":"System",
        "tau-t": f"{tau_t:.2f}",
        "nstxout-compressed": str(spec.nstxout_compressed),
        "nstvout": str(spec.nstvout),
        "nstenergy": str(spec.nstenergy),
        "gen-vel":"no",
    }

def ps_to_steps(ps: float, dt_fs: float) -> int:
    return int(round(ps * 1000.0 / dt_fs))

em_mdp    = MDP_DIR / "em.mdp"
pro_mdp   = MDP_DIR / "production.mdp"

write_mdp(em_mdp, {
    "integrator":"steep",
    "emtol":"100",
    "emstep":"0.01",
    "nsteps":"200000",
    "cutoff-scheme":"Verlet",
    "coulombtype":"PME",
    "rcoulomb":"1.2",
    "vdwtype":"Cut-off",
    "rvdw":"1.2",
    "pbc":"xyz",
})

ATM_TO_BAR = 1.01325
P1_BAR  = float(spec.pressure_bar)
P1K_BAR = 1000.0 * ATM_TO_BAR
P2K_BAR = 2000.0 * ATM_TO_BAR
P4K_BAR = 4000.0 * ATM_TO_BAR

def build_nvt1_schedule(spec: SystemSpec):
    variant = str(getattr(spec, "nvt1_variant", "baseline") or "baseline").strip().lower()
    if variant == "short":
        return [
            {"name":"nvt1", "kind":"nvt", "ps":float(spec.nvt1_short_ps), "temp_start":spec.temperature_equil,
             "dt_fs":1.0, "tcoupl":"v-rescale", "tau_t": OPT_A["tau_t_big"],
             "mdrun_extra": GPU_SAFE_EXTRA},
        ]
    if variant == "split":
        return [
            {"name":"nvt1_pre", "kind":"nvt", "ps":float(spec.nvt1_split_vrescale_ps), "temp_start":spec.temperature_equil,
             "dt_fs":1.0, "tcoupl":"v-rescale", "tau_t": OPT_A["tau_t_big"],
             "mdrun_extra": GPU_SAFE_EXTRA},
            {"name":"nvt1", "kind":"nvt", "ps":float(spec.nvt1_split_nosehoover_ps), "temp_start":spec.temperature_equil,
             "dt_fs":1.0, "mdrun_extra": GPU_SAFE_EXTRA},
        ]
    return [
        {"name":"nvt1", "kind":"nvt", "ps":800.0, "temp_start":spec.temperature_equil,
         "mdrun_extra": GPU_SAFE_EXTRA},
    ]

NVT1_SCHEDULE = build_nvt1_schedule(spec)

EQ_SCHEDULE = [
    {"name":"nvt0_pre", "kind":"nvt", "ps":5.0, "temp_start":50.0,
     "temp_end": 100.0, "dt_fs":0.5, "tcoupl":"v-rescale",
     "posres": True, "gen_vel": True, "tau_t": OPT_A["tau_t_big"],
     "mdrun_extra": GPU_SAFE_EXTRA, "force_cpu": False,
     "mdp_overrides": {"constraints":"none"}},
    {"name":"nvt0", "kind":"nvt", "ps":100.0, "temp_start":100.0,
     "temp_end": spec.temperature_equil, "dt_fs":1.0, "tcoupl":"v-rescale",
     "posres": True, "tau_t": OPT_A["tau_t_big"],
     "mdrun_extra": GPU_SAFE_EXTRA, "force_cpu": False,
     "mdp_overrides": {"lincs_iter":4, "lincs_order":12, "lincs-warnangle":20}},

    # conservative compression ramp (Berendsen + smaller dt)
    {"name":"npt01_p50", "kind":"npt", "ps":100.0, "temp_start":spec.temperature_equil,
     "ref_p": 50.0, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen", "posres": True},
    {"name":"npt02_p200", "kind":"npt", "ps":100.0, "temp_start":spec.temperature_equil,
     "ref_p": 200.0, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen", "posres": True},
    {"name":"npt03_p500", "kind":"npt", "ps":100.0, "temp_start":spec.temperature_equil,
     "ref_p": 500.0, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen", "posres": True},
    {"name":"npt04_p1k", "kind":"npt", "ps":200.0, "temp_start":spec.temperature_equil,
     "ref_p": P1K_BAR, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen", "posres": True},

    # high-pressure stages (PR, smaller dt)
    {"name":"npt05_p2k", "kind":"npt", "ps":200.0, "temp_start":spec.temperature_equil,
     "ref_p": P2K_BAR, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": OPT_A["tau_p_big"]},
    {"name":"npt06_p4k", "kind":"npt", "ps":300.0, "temp_start":spec.temperature_equil,
     "ref_p": P4K_BAR, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": OPT_A["tau_p_big"]},
    {"name":"npt07_p4k_hold", "kind":"npt", "ps":400.0, "temp_start":spec.temperature_equil,
     "ref_p": P4K_BAR, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": OPT_A["tau_p_big"]},

    # conservative decompression ramp (Berendsen + smaller dt)
    {"name":"npt08_p2k_down", "kind":"npt", "ps":200.0, "temp_start":spec.temperature_equil,
     "ref_p": P2K_BAR, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen"},
    {"name":"npt09_p1k_down", "kind":"npt", "ps":200.0, "temp_start":spec.temperature_equil,
     "ref_p": P1K_BAR, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen"},
    {"name":"npt10_p200_down", "kind":"npt", "ps":200.0, "temp_start":spec.temperature_equil,
     "ref_p": 200.0, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen"},
    {"name":"npt11_p1_down", "kind":"npt", "ps":200.0, "temp_start":spec.temperature_equil,
     "ref_p": P1_BAR, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen"},

    # heating/cooling at 1 bar
    {"name":"npt12_heat", "kind":"npt", "ps":400.0, "temp_start":spec.temperature_equil,
     "temp_end": spec.temperature_high, "ref_p": P1_BAR, "dt_fs":1.0},
    {"name":"npt13_cool", "kind":"npt", "ps":400.0, "temp_start":spec.temperature_high,
     "temp_end": spec.temperature_equil, "ref_p": P1_BAR, "dt_fs":1.0},

    # second compression/decompression (conservative)
    {"name":"npt14_p200_up", "kind":"npt", "ps":100.0, "temp_start":spec.temperature_equil,
     "ref_p": 200.0, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen"},
    {"name":"npt15_p1k_up", "kind":"npt", "ps":100.0, "temp_start":spec.temperature_equil,
     "ref_p": P1K_BAR, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen"},
    {"name":"npt16_p2k_up", "kind":"npt", "ps":150.0, "temp_start":spec.temperature_equil,
     "ref_p": P2K_BAR, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen"},
    {"name":"npt17_p4k_up", "kind":"npt", "ps":150.0, "temp_start":spec.temperature_equil,
     "ref_p": P4K_BAR, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen"},
    {"name":"npt18_p1_down", "kind":"npt", "ps":300.0, "temp_start":spec.temperature_equil,
     "ref_p": P1_BAR, "dt_fs":1.0, "tau_t": OPT_A["tau_t_big"], "tau_p": 200.0,
     "pcoupl":"Berendsen"},

    *NVT1_SCHEDULE,
    {"name":"npt19_p1", "kind":"npt", "ps":1200.0, "temp_start":spec.temperature_equil,
     "ref_p": P1_BAR, "dt_fs":1.0},
    # safe NVT tail: pre-stage w/ posres + 1 fs, then release
    {"name":"nvt2_pre", "kind":"nvt", "ps":200.0, "temp_start":spec.temperature_equil,
     "dt_fs":1.0, "tcoupl":"v-rescale", "tau_t": OPT_A["tau_t_big"], "posres": True,
     "mdrun_extra": GPU_SAFE_EXTRA},
    {"name":"nvt2", "kind":"nvt", "ps":800.0, "temp_start":spec.temperature_equil,
     "dt_fs":1.0, "tcoupl":"v-rescale", "tau_t": OPT_A["tau_t_big"],
     "mdrun_extra": GPU_SAFE_EXTRA},
]

DENSFIX_AFTER = "npt19_p1"


def build_stage_mdp(stage: dict) -> dict:
    dt_fs = float(stage.get("dt_fs", spec.dt_fs))
    tau_t = float(stage.get("tau_t", OPT_A["tau_t"]))
    temp_start = float(stage["temp_start"])
    temp_end = stage.get("temp_end")

    tcoupl = stage.get("tcoupl")
    p = mdp_common(spec, dt_fs=dt_fs, tcoupl=tcoupl, tau_t=tau_t)
    p.update({
        "nsteps": str(ps_to_steps(float(stage["ps"]), dt_fs)),
        "ref-t": f"{temp_start:.1f}",
    })

    if stage["kind"] == "nvt":
        p["pcoupl"] = "no"
        if stage.get("gen_vel"):
            p.update({
                "gen-vel": "yes",
                "gen-temp": f"{temp_start:.1f}",
                "gen-seed": "-1",
            })
    else:
        tau_p = float(stage.get("tau_p", OPT_A["tau_p"]))
        pcoupl = stage.get("pcoupl", OPT_A["pcoupl"])
        p.update({
            "pcoupl": pcoupl,
            "tau-p": f"{tau_p:.1f}",
            "ref-p": f"{float(stage['ref_p']):.2f}",
            "compressibility": "4.5e-5",
            "nstpcouple": "50",
            "refcoord_scaling": "com",
        })

    if stage.get("posres"):
        p["define"] = "-DPOSRES"

    if temp_end is not None:
        temp_end = float(temp_end)
        p.update({
            "annealing": "single",
            "annealing-npoints": "2",
            "annealing-time": f"0 {float(stage['ps']):.1f}",
            "annealing-temp": f"{temp_start:.1f} {temp_end:.1f}",
        })

    mdp_overrides = stage.get("mdp_overrides")
    if mdp_overrides:
        p.update(mdp_overrides)

    return p

EQ_STAGES = []
for st in EQ_SCHEDULE:
    mdp_path = MDP_DIR / f"{st['name']}.mdp"
    write_mdp(mdp_path, build_stage_mdp(st))
    st = dict(st)
    st["mdp"] = mdp_path
    EQ_STAGES.append(st)

def _production_output_overrides(spec: SystemSpec) -> dict:
    out = {
        "nstxout-compressed": str(spec.nstxout_compressed),
        "nstvout": str(spec.nstvout),
        "nstenergy": str(spec.nstenergy),
        "nstxout": "0",
    }
    if not getattr(spec, "gk_output_enabled", False):
        return out

    frame_interval_ps = max(float(getattr(spec, "gk_frame_interval_ps", 1.0)), float(spec.dt_ps()))
    interval_steps = max(1, int(round(frame_interval_ps / float(spec.dt_ps()))))
    out["nstxout-compressed"] = str(interval_steps)
    out["nstenergy"] = str(min(int(spec.nstenergy), interval_steps))
    if getattr(spec, "gk_save_velocities", False):
        out["nstxout"] = str(interval_steps)
        out["nstvout"] = str(interval_steps)
    else:
        out["nstvout"] = "0"
    log(
        f"[gk-output] enabled interval={frame_interval_ps:.3f} ps "
        f"steps={interval_steps} velocities={bool(getattr(spec, 'gk_save_velocities', False))}"
    )
    return out


def _read_mdp_kv(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        line = raw.split(';', 1)[0].strip()
        if not line or '=' not in line:
            continue
        k, v = line.split('=', 1)
        out[k.strip().lower()] = v.strip()
    return out


def _assert_gk_output_stage_ok(stage_dir: Path, spec: SystemSpec):
    if not getattr(spec, "gk_output_enabled", False):
        return

    interval_ps = max(float(getattr(spec, "gk_frame_interval_ps", 1.0)), float(spec.dt_ps()))
    interval_steps = max(1, int(round(interval_ps / float(spec.dt_ps()))))
    expected_energy = min(int(spec.nstenergy), interval_steps)

    mdp_src = stage_dir / "mdout.mdp"
    if not mdp_src.exists():
        mdp_src = MDP_DIR / "production.mdp"
    kv = _read_mdp_kv(mdp_src)

    def _get_int(key: str):
        try:
            return int(float(kv.get(key, "").split()[0]))
        except Exception:
            return None

    got_xtc = _get_int("nstxout-compressed")
    got_energy = _get_int("nstenergy")
    got_x = _get_int("nstxout")
    got_v = _get_int("nstvout")

    problems = []
    if got_xtc != interval_steps:
        problems.append(f"nstxout-compressed expected {interval_steps}, got {got_xtc}")
    if got_energy is None or got_energy > interval_steps or got_energy != expected_energy:
        problems.append(f"nstenergy expected {expected_energy}, got {got_energy}")

    want_vel = bool(getattr(spec, "gk_save_velocities", False))
    if want_vel:
        if got_x != interval_steps:
            problems.append(f"nstxout expected {interval_steps}, got {got_x}")
        if got_v != interval_steps:
            problems.append(f"nstvout expected {interval_steps}, got {got_v}")
    else:
        if got_v not in (0, None):
            problems.append(f"nstvout expected 0, got {got_v}")

    if problems:
        raise RuntimeError("[gk-output-check] " + "; ".join(problems))

    if want_vel:
        trr = stage_dir / "production.trr"
        if trr.exists() and trr.stat().st_size <= 0:
            raise RuntimeError("[gk-output-check] production.trr exists but is empty")

prod = mdp_common(
    spec,
    dt_fs=_production_dt_fs(),
    tcoupl=_production_tcoupl(),
    tau_t=_production_tau_t(),
)
prod.update({
    "nsteps": str(_production_nsteps()),
    "ref-t": f"{spec.temperature_prod:.1f}",
    "pcoupl":"no",
    "lincs_iter": str(_production_lincs_iter()),
    "lincs_order": str(_production_lincs_order()),
    "lincs-warnangle": str(_production_lincs_warnangle()),
})
prod.update(_production_output_overrides(spec))
write_mdp(pro_mdp, prod)
# ---- target density -> box length ----
_sync_n_chains_from_topology_for_density()
spec.target_density_g_cm3 = getattr(spec, "target_density_g_cm3", 1.2)

mass_poly_g = polymer_mw_g_mol * spec.n_chains
mass_salt_g = (li_mw_g_mol + tfsi_mw_g_mol) * spec.li_tfsi_pairs
total_mass_g_mol = mass_poly_g + mass_salt_g
mass_box_kg = total_mass_g_mol / NA / 1000.0
rho_kg_m3 = spec.target_density_g_cm3 * 1000.0
V_target_m3 = mass_box_kg / rho_kg_m3
V_target_nm3 = V_target_m3 * 1e27
spec.box_length_target_nm = V_target_nm3 ** (1/3)
log(f"[target density] n_chains={spec.n_chains}, rho={spec.target_density_g_cm3:.3f} g/cm^3 -> L_target={spec.box_length_target_nm:.3f} nm")

# ---- helpers for run (continuation) ----
def read_box_from_gro(gro_path: Path) -> Tuple[float,float,float]:
    lines = gro_path.read_text().splitlines()
    vals = [float(x) for x in lines[-1].split()]
    if len(vals) < 3:
        raise ValueError("GRO box parse failed")
    return vals[0], vals[1], vals[2]

def scale_box_to_target(prev_gro: Path, target_L_nm: float, stage_name: str="densfix") -> Path:
    Lx,Ly,Lz = read_box_from_gro(prev_gro)
    L_cur = (Lx*Ly*Lz) ** (1/3)
    s = target_L_nm / L_cur
    log(f"[{stage_name}] L_cur={L_cur:.3f} → L_target={target_L_nm:.3f}, scale={s:.5f}")
    stage_dir = MD_DIR / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    out_gro = stage_dir / f"{stage_name}.gro"
    run([GMX,"editconf","-f",str(prev_gro),"-o",str(out_gro),"-scale",f"{s:.6f}",f"{s:.6f}",f"{s:.6f}"], cwd=stage_dir)
    return out_gro

def grompp(mdp: Path, gro: Path, top: Path, tpr: Path, *,
           ref: Optional[Path]=None, prev_cpt: Optional[Path]=None, maxwarn: int=2, cwd: Optional[Path]=None):
    args = ["-f", str(mdp), "-c", str(gro), "-p", str(top), "-o", str(tpr), "-maxwarn", str(maxwarn)]
    if ref is not None:
        args += ["-r", str(ref)]
    if prev_cpt is not None and prev_cpt.exists():
        args += ["-t", str(prev_cpt)]
    gmx_cmd("grompp", args, cwd=cwd or tpr.parent)

def _with_required_gpu_options(args: List[str]) -> List[str]:
    out = list(args)
    if "-nb" not in out:
        out += ["-nb", "gpu"]
    if "-pme" not in out:
        out += ["-pme", "gpu"]
    return out

def _mdrun_log_confirms_gpu(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="ignore")
    gpu_markers = [
        "Using GPU",
        "PP tasks will do",
        "PME tasks will do all aspects on the GPU",
        "GPU 8x4 nonbonded",
    ]
    cpu_only_markers = [
        "PP tasks will do (non-perturbed) short-ranged interactions on the CPU",
        "PME tasks will do all aspects on the CPU",
    ]
    return any(marker in text for marker in gpu_markers) and not any(marker in text for marker in cpu_only_markers)

def _run_mdrun_guarded(cmd: List[str], cwd: Path, deffnm: str) -> Tuple[int, str, str]:
    log("$ " + " ".join(str(x) for x in cmd))
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    out_lines = []
    lincs_count = 0
    abort_reason = ""
    if proc.stdout is not None:
        for line in proc.stdout:
            out_lines.append(line)
            if "LINCS WARNING" in line:
                lincs_count += 1
                if lincs_count >= _lincs_abort_warnings():
                    abort_reason = f"lincs_instability:{lincs_count}_warnings"
                    out_lines.append(
                        f"\n[mdrun-guard] terminating {deffnm}: "
                        f"{lincs_count} LINCS warnings reached threshold {_lincs_abort_warnings()}\n"
                    )
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
    return_code = proc.wait()
    return int(return_code), "".join(out_lines), abort_reason

def mdrun(deffnm: str, cwd: Path, extra: Optional[List[str]]=None, *, force_cpu: bool=False):
    extra = extra or []
    cmd = [GMX,"mdrun","-deffnm",deffnm] + extra
    # GPU try, with optional CPU fallback only when explicitly enabled.
    if GPU_AVAILABLE and not force_cpu:
        gpu_cmd = _with_required_gpu_options(cmd) + ["-gpu_id", spec.gpu_id]
        return_code, captured, abort_reason = _run_mdrun_guarded(gpu_cmd, cwd, deffnm)
        if return_code == 0 and not abort_reason and _mdrun_log_confirms_gpu(cwd / f"{deffnm}.log"):
            return
        (cwd / f"{deffnm}_gpu_fail.log").write_text(
            f"CMD: {' '.join(str(x) for x in gpu_cmd)}\n"
            f"RETURN_CODE: {return_code}\n"
            f"ABORT_REASON: {abort_reason}\n\n"
            f"{captured}"
        )
        if abort_reason:
            raise RuntimeError(
                f"[mdrun:lincs_instability] stopped {deffnm} before CUDA memory violation; "
                f"reason={abort_reason}; see {cwd / f'{deffnm}_gpu_fail.log'}"
            )
        if not ALLOW_CPU_FALLBACK:
            raise RuntimeError(
                f"[mdrun:gpu_failed] GPU mdrun failed or did not confirm GPU tasks for {deffnm}. "
                f"See {cwd / f'{deffnm}_gpu_fail.log'}"
            )
        warnings.warn("GPU failed; falling back to CPU because GROMACS_ALLOW_CPU_FALLBACK=1", RuntimeWarning)
    elif (not GPU_AVAILABLE) and (not force_cpu) and (not ALLOW_CPU_FALLBACK):
        raise RuntimeError("[mdrun:gpu_unavailable] nvidia-smi not found; refusing CPU-only mdrun")
    # CPU fallback / forced CPU
    run(cmd + ["-nb","cpu","-pme","cpu"], cwd=cwd, check=True)

def _stage_ntomp(stage: str) -> int:
    stl = str(stage or "").strip().lower()
    if stl.startswith("prod"):
        return int(os.environ.get("GROMACS_NTOMP_PRODUCTION", os.environ.get("GROMACS_NTOMP", "12")))
    return int(os.environ.get("GROMACS_NTOMP", "16"))

def run_stage(stage: str, mdp: Path, in_gro: Path, *, use_posres_ref: bool=False,
              prev_stage_dir: Optional[Path]=None, mdrun_extra: Optional[List[str]]=None,
              force_cpu: bool=False) -> Path:
    stl = stage.lower()
    if stl.startswith("em"):
        md_stage = "md-em"
    elif stl.startswith("nvt"):
        md_stage = "md-nvt"
    elif stl.startswith("npt"):
        md_stage = "md-npt"
    elif stl.startswith("prod"):
        md_stage = "md-prod"
    else:
        md_stage = "md"
    print(f"__STAGEV3__:{md_stage}:{stage}", flush=True)
    sd = MD_DIR / stage
    sd.mkdir(parents=True, exist_ok=True)
    out_gro = sd / f"{stage}.gro"
    force_production_rerun = (
        stl.startswith("prod")
        and os.environ.get("GROMACS_FORCE_PRODUCTION_RERUN", "").strip().lower()
        in ("1", "true", "yes", "on")
    )
    if out_gro.exists() and out_gro.stat().st_size > 0:
        if force_production_rerun:
            backup = sd.with_name(f"{sd.name}_prev_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            suffix = 1
            while backup.exists():
                backup = sd.with_name(f"{sd.name}_prev_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}")
                suffix += 1
            shutil.move(str(sd), str(backup))
            log(f"[production-rerun] backed up existing production -> {backup}")
            sd.mkdir(parents=True, exist_ok=True)
            out_gro = sd / f"{stage}.gro"
        else:
            if stl.startswith("prod"):
                _assert_gk_output_stage_ok(sd, spec)
                if getattr(spec, "gk_save_velocities", False):
                    trr = sd / f"{stage}.trr"
                    if not trr.exists() or trr.stat().st_size <= 0:
                        raise RuntimeError("[gk-output-check] production stage was skipped but production.trr is missing")
            print(f"__STAGEV3__:{md_stage}:{stage}:skip-existing", flush=True)
            return out_gro
    sd.mkdir(parents=True, exist_ok=True)
    tpr = sd / f"{stage}.tpr"
    ref = in_gro if use_posres_ref else None
    prev_cpt = (prev_stage_dir / f"{prev_stage_dir.name}.cpt") if prev_stage_dir else None

    grompp(mdp, in_gro, TOPOL_TOP, tpr, ref=ref, prev_cpt=prev_cpt, cwd=sd)
    if stl.startswith("prod"):
        _assert_gk_output_stage_ok(sd, spec)
    # production/NPT GPU 옵션(원하면 강화), 실패하면 CPU로 떨어짐
    _ntomp = _stage_ntomp(stage)
    extra = ["-ntmpi","1","-ntomp",str(_ntomp),"-pin","on"]
    if mdrun_extra:
        extra += list(mdrun_extra)
    mdrun(stage, sd, extra=extra, force_cpu=force_cpu)
    if stl.startswith("prod") and getattr(spec, "gk_save_velocities", False):
        trr = sd / f"{stage}.trr"
        if not trr.exists() or trr.stat().st_size <= 0:
            raise RuntimeError("[gk-output-check] production.trr missing after mdrun while gk_save_velocities=True")
    return sd / f"{stage}.gro"

def _equil_final_stage_name() -> str:
    return str(EQ_STAGES[-1]["name"])

def _equil_final_gro() -> Path:
    stage = _equil_final_stage_name()
    return MD_DIR / stage / f"{stage}.gro"

# ---- run MD pipeline ----
start = STRUCT_FIXED.resolve()

resume_from_production = _equil_final_gro().exists() and _equil_final_gro().stat().st_size > 0
print(f"[md-auto-resume] mode={'production' if resume_from_production else 'equil'}", flush=True)

if resume_from_production:
    last_gro = _equil_final_gro()
    prev_dir = MD_DIR / _equil_final_stage_name()
else:
    em_gro   = run_stage("em",   em_mdp,   start, use_posres_ref=False, force_cpu=True)
    prev_dir = MD_DIR / "em"
    last_gro = em_gro

    densfix_gro = None
    em2_gro = None

    for st in EQ_STAGES:
        last_gro = run_stage(st["name"], st["mdp"], last_gro,
                             use_posres_ref=st.get("posres", False),
                             prev_stage_dir=prev_dir,
                             mdrun_extra=st.get("mdrun_extra"),
                             force_cpu=st.get("force_cpu", False))
        prev_dir = MD_DIR / st["name"]

        if st["name"] == DENSFIX_AFTER:
            densfix_gro = scale_box_to_target(last_gro, spec.box_length_target_nm, "densfix")
            em2_gro = run_stage("em2", em_mdp, densfix_gro, use_posres_ref=False, force_cpu=True)
            last_gro = em2_gro
            prev_dir = MD_DIR / "em2"

def _production_replica_count() -> int:
    raw = os.environ.get("GROMACS_PRODUCTION_REPLICAS", "1")
    try:
        return max(1, int(raw))
    except Exception:
        warnings.warn(f"Invalid GROMACS_PRODUCTION_REPLICAS={raw!r}; using 1", RuntimeWarning)
        return 1

def _production_replica_mdp(replica_idx: int) -> Path:
    if replica_idx <= 1:
        return pro_mdp

    rep_mdp = MDP_DIR / f"production_rep{replica_idx}.mdp"
    rep_prod = dict(prod)
    rep_prod.update({
        "continuation": "no",
        "gen-vel": "yes",
        "gen-temp": f"{spec.temperature_prod:.1f}",
        "gen-seed": str(730000 + replica_idx),
    })
    write_mdp(rep_mdp, rep_prod)
    return rep_mdp

production_replicas = _production_replica_count()
production_gros = []
prod_gro = None
for replica_idx in range(1, production_replicas + 1):
    stage_name = "production" if replica_idx == 1 else f"production_rep{replica_idx}"
    mdp_path = _production_replica_mdp(replica_idx)
    replica_prev_dir = prev_dir if replica_idx == 1 else None
    replica_gro = run_stage(stage_name, mdp_path, last_gro,
                            use_posres_ref=False,
                            prev_stage_dir=replica_prev_dir,
                            mdrun_extra=_production_gpu_extra())
    production_gros.append(replica_gro)
    if replica_idx == 1:
        prod_gro = replica_gro

print(f"[production-replicas] requested={production_replicas} completed={len(production_gros)}", flush=True)

# ===== Notebook Cell 5 [conductivity-analysis] =====
print("__STAGEV3__:conductivity-analysis:cell5", flush=True)
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

def _sync_n_chains_from_topology_for_density() -> None:
    top_path = TOPO_DIR / "topol.top"
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

_sync_n_chains_from_topology_for_density()

# ---- cNE analysis (GROMACS xtc/tpr required) ----
# production.tpr/xtc는 stage_dir에 생성됨
def _production_replica_count_for_analysis() -> int:
    raw = os.environ.get("GROMACS_PRODUCTION_REPLICAS", "1")
    try:
        return max(1, int(raw))
    except Exception:
        return 1

def _production_replica_stages_for_analysis():
    count = _production_replica_count_for_analysis()
    return [("production", MD_DIR / "production")] + [
        (f"production_rep{i}", MD_DIR / f"production_rep{i}") for i in range(2, count + 1)
    ]

_missing_production_outputs = []
for _stage, _stage_dir in _production_replica_stages_for_analysis():
    for _ext in ("tpr", "xtc", "gro"):
        _p = _stage_dir / f"{_stage}.{_ext}"
        if not _p.exists():
            _missing_production_outputs.append(_p)
if _missing_production_outputs:
    _shown = "; ".join(str(_p) for _p in _missing_production_outputs[:12])
    if len(_missing_production_outputs) > 12:
        _shown += f"; ... ({len(_missing_production_outputs)} missing total)"
    raise FileNotFoundError(f"[analysis phase] missing production replica outputs: {_shown}")
print(f"[production-replicas] analysis inputs present={len(_production_replica_stages_for_analysis())}", flush=True)

PROD_DIR = MD_DIR / "production"
TPR = PROD_DIR / "production.tpr"
XTC = PROD_DIR / "production.xtc"
GRO = PROD_DIR / "production.gro"
assert TPR.exists() and XTC.exists() and GRO.exists()

# spec.analysis_begin_ns = 1.0
# spec.cluster_cutoff_nm = 0.30

# (A) make_ndx: LI + anion N
INDEX = MD_DIR / "index.ndx"
NDX_CMD = MD_DIR / "index_cmd.txt"

# GRO에서 anion N atomname 하나 추정(첫 anion residue에서 N* 찾기)
with GRO.open() as f:
    _ = f.readline()
    nat = int(f.readline().strip())
    gro_lines = [f.readline().rstrip("\n") for _ in range(nat)]
anion_res = spec.anion_resname_coords
N_NAME = None
for ln in gro_lines:
    if ln[5:10].strip() == anion_res:
        nm = ln[10:15].strip()
        if nm.upper().startswith("N"):
            N_NAME = nm
            break
if N_NAME is None:
    N_NAME = "N1"  # 최후 fallback

NDX_CMD.write_text("\n".join([
    "keep 0",
    "a LI",
    f"a {N_NAME} & r {anion_res}",
    "name 1 LI",
    f"name 2 {anion_res}_N",
    "q",
]) + "\n")

run([GMX,"make_ndx","-f",str(GRO),"-o",str(INDEX)], input_text=NDX_CMD.read_text(), cwd=MD_DIR)

# (B) diffusivity (LAMMPS-style: first-last displacement, no time averaging)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

import MDAnalysis as mda
from MDAnalysis.lib.nsgrid import FastNS

u = mda.Universe(str(GRO), str(XTC))
cation_res = spec.cation_resname_coords

sel_li = u.select_atoms(f"resname {cation_res}")
sel_o = u.select_atoms(f"resname {anion_res} and name O*")
sel_s = u.select_atoms(f"resname {anion_res} and name S*")
sel_anionN = u.select_atoms(f"resname {anion_res} and name N*")
sel_core = u.select_atoms(f"resname {anion_res} and (name O* or name S* or name N*)")

if len(sel_li)==0 or len(sel_o)==0 or len(sel_anionN)==0 or len(sel_core)==0:
    raise RuntimeError(
        f"Selection empty. LI={len(sel_li)}, O={len(sel_o)}, S={len(sel_s)}, "
        f"N={len(sel_anionN)}, core={len(sel_core)}. resname/atomname 확인."
    )

def _center_of_mass(coords, masses):
    if masses is None:
        return coords.mean(axis=0)
    return (coords * masses[:, None]).sum(axis=0) / masses.sum()

def _unwrap_step(pos_unwrap, pos_prev, pos_cur, box_lengths):
    delta = pos_cur - pos_prev
    delta -= box_lengths * np.round(delta / box_lengths)
    pos_unwrap += delta
    return pos_unwrap

def lammps_like_diffusivities(u, sel_indices_list, masses=None):
    n_frames = len(u.trajectory)
    if n_frames < 2:
        raise RuntimeError("Need at least 2 frames for diffusivity.")
    u.trajectory[0]
    if not np.allclose(u.trajectory.ts.dimensions[3:], [90.0, 90.0, 90.0], atol=1e-3):
        raise RuntimeError("LAMMPS-style unwrapping assumes orthorhombic box.")
    pos_prev = u.atoms.positions.copy()
    pos_unwrap = pos_prev.copy()
    com0 = _center_of_mass(pos_unwrap, masses)
    sel_start = [pos_unwrap[idx] - com0 for idx in sel_indices_list]
    for ts in u.trajectory[1:]:
        box = ts.dimensions[:3].astype(np.float64)
        _unwrap_step(pos_unwrap, pos_prev, u.atoms.positions, box)
        pos_prev = u.atoms.positions.copy()
    com_end = _center_of_mass(pos_unwrap, masses)
    total_time_ps = float(u.trajectory.dt) * (n_frames - 1)
    if total_time_ps <= 0:
        raise RuntimeError("Invalid trajectory dt for diffusivity.")
    out = []
    for idx, start in zip(sel_indices_list, sel_start):
        end = pos_unwrap[idx] - com_end
        disp = end - start
        msd_A2 = np.mean(np.sum(disp**2, axis=1))
        D_cm2_s = (msd_A2 / (6.0 * total_time_ps)) * 1e-4
        out.append(float(D_cm2_s))
    u.trajectory[0]
    return out

masses = None
try:
    masses = u.atoms.masses
    if masses is not None and np.allclose(masses, 0):
        masses = None
except Exception:
    masses = None

D_li_cm2s_fl, D_an_cm2s_fl = lammps_like_diffusivities(
    u,
    [sel_li.indices, sel_anionN.indices],
    masses=masses,
)

# (B2) time-averaged MSD via gmx msd (reference method)
S_ps = float(spec.dt_fs) * int(spec.nstxout_compressed) / 1000.0
TRESTART_PS = S_ps if S_ps > 0 else 10.0

msd_li = ANALYSIS_DIR / "msd_li.xvg"
msd_an = ANALYSIS_DIR / f"msd_{anion_res.lower()}_N.xvg"

run([GMX,"msd","-f",str(XTC),"-s",str(TPR),"-n",str(INDEX),"-o",str(msd_li),"-trestart",f"{TRESTART_PS:g}"], input_text="1\n", cwd=ANALYSIS_DIR)
run([GMX,"msd","-f",str(XTC),"-s",str(TPR),"-n",str(INDEX),"-o",str(msd_an),"-trestart",f"{TRESTART_PS:g}"], input_text="2\n", cwd=ANALYSIS_DIR)

def xvg_to_df(path: Path) -> pd.DataFrame:
    rows=[]
    for ln in path.read_text().splitlines():
        ln=ln.strip()
        if not ln or ln.startswith(("#","@")):
            continue
        parts = ln.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            pass
    arr = np.array(rows, float)
    return pd.DataFrame(arr, columns=["t","msd"])

def diffusion_from_msd(msd_path: Path) -> float:
    df = xvg_to_df(msd_path)
    t_ns = df["t"].values / 1000.0
    y    = df["msd"].values  # nm^2
    t0 = float(spec.analysis_begin_ns)
    t1 = float(spec.analysis_end_ns)
    m  = (t_ns >= t0) & (t_ns <= t1)
    if m.sum() < 5:
        t_fit, y_fit = t_ns, y
    else:
        t_fit, y_fit = t_ns[m], y[m]
    slope = np.polyfit(t_fit, y_fit, 1)[0]   # nm^2/ns
    return float((slope/6.0) * 1e-5)        # cm^2/s

D_li_cm2s_msd = diffusion_from_msd(msd_li)
D_an_cm2s_msd = diffusion_from_msd(msd_an)

# Use time-averaged MSD diffusivities for conductivity
D_li_cm2s, D_an_cm2s = D_li_cm2s_msd, D_an_cm2s_msd


# (C) box volume from GRO (nm^3)
box_vals = [float(x) for x in GRO.read_text().splitlines()[-1].split()]
V_nm3 = box_vals[0] * box_vals[1] * box_vals[2]
V_cm3 = V_nm3 * 1e-21

N_LI = spec.li_tfsi_pairs
N_AN = spec.li_tfsi_pairs
z = 1.0
TEMP_K = float(spec.temperature_prod)


sigma_NE_S_cm = (E_CHG**2)/(KB*TEMP_K*V_cm3) * ((z**2)*N_LI*D_li_cm2s + (z**2)*N_AN*D_an_cm2s)

# (D) cNE via clustering (htpmd-style: clusters over Li + anion O/S/N; anion counted by N)
n_li = len(sel_li)
n_core = len(sel_core)
n_tot = n_li + n_core
core_is_n = np.isin(sel_core.indices, sel_anionN.indices)

cutoff_A = float(spec.cluster_cutoff_nm) * 10.0
stride_ps = float(spec.cluster_stride_ps)
max_cluster = int(spec.cluster_max_cluster)

dt_ps_traj = float(u.trajectory.dt)
step = max(1, int(round(stride_ps / dt_ps_traj)))

t0_ps = float(spec.analysis_begin_ns) * 1000.0
t1_ps = float(spec.analysis_end_ns) * 1000.0

pop_mat = np.zeros((max_cluster, max_cluster), float)
n_frames_used = 0

def uf_init(n):
    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    return parent, rank

def uf_find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def uf_union(parent, rank, a, b):
    ra = uf_find(parent, a)
    rb = uf_find(parent, b)
    if ra == rb: return
    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1

for ts in u.trajectory[::step]:
    if ts.time < t0_ps or ts.time > t1_ps:
        continue

    coords = np.vstack([sel_li.positions, sel_core.positions]).astype(np.float32)  # Å
    box = ts.dimensions.astype(np.float32)  # Å,deg

    ns = FastNS(cutoff_A, coords, box, pbc=True)
    pairs = ns.self_search().get_pairs()

    parent, rank = uf_init(n_tot)
    for a, b in pairs:
        uf_union(parent, rank, int(a), int(b))

    comp_li = defaultdict(int)
    comp_an = defaultdict(int)

    for i in range(n_li):
        r = uf_find(parent, i)
        comp_li[r] += 1

    for core_pos, is_n in enumerate(core_is_n):
        if not is_n:
            continue
        node = n_li + core_pos
        r = uf_find(parent, node)
        comp_an[r] += 1

    roots = set(comp_li.keys()) | set(comp_an.keys())
    for r in roots:
        i = comp_li.get(r, 0)
        j = comp_an.get(r, 0)
        if i < max_cluster and j < max_cluster:
            pop_mat[i, j] += 1.0

    n_frames_used += 1

if n_frames_used == 0:
    raise RuntimeError("No frames used for clustering. analysis window/stride 확인.")
pop_mat /= float(n_frames_used)

sigma_cNE = 0.0
tn_num = 0.0
tn_den = 0.0
for i in range(max_cluster):
    for j in range(max_cluster):
        if i == j:
            continue
        q = (i*z - j*z)
        if q == 0.0:
            continue
        if i > j:
            sigma_cNE += (E_CHG**2)/(KB*TEMP_K*V_cm3) * (q*q) * pop_mat[i,j] * D_li_cm2s
            tn_num += (i*z) * q * pop_mat[i,j] * D_li_cm2s
            tn_den += (q*q) * pop_mat[i,j] * D_li_cm2s
        else:
            sigma_cNE += (E_CHG**2)/(KB*TEMP_K*V_cm3) * (q*q) * pop_mat[i,j] * D_an_cm2s
            tn_num += (i*z) * q * pop_mat[i,j] * D_an_cm2s
            tn_den += (q*q) * pop_mat[i,j] * D_an_cm2s

t_plus_cNE = float(tn_num/tn_den) if tn_den != 0 else float("nan")

log(f"[MSD] D_Li(msd) = {D_li_cm2s:.3e} cm^2/s, D_an(msd) = {D_an_cm2s:.3e} cm^2/s")
log(f"[MSD] D_Li(fl)  = {D_li_cm2s_fl:.3e} cm^2/s, D_an(fl)  = {D_an_cm2s_fl:.3e} cm^2/s")
log(f"[NE ] sigma_NE  = {sigma_NE_S_cm:.3e} S/cm")
log(f"[cNE] sigma_cNE = {sigma_cNE:.3e} S/cm  (frames={n_frames_used}, cutoff={cutoff_A:.2f} Å)")
log(f"[cNE] t+ (cNE)  = {t_plus_cNE:.3f}")

# save
pd.DataFrame({"metric":[
    "N_LI","N_AN","D_Li(cm^2/s)","D_an(cm^2/s)","V(nm^3)","z","sigma_NE(S/cm)","sigma_cNE(S/cm)","t+(cNE)",
    "D_Li_fl(cm^2/s)","D_an_fl(cm^2/s)",
    "cluster_cutoff(nm)","cluster_stride(ps)","cluster_frames"
], "value":[
    N_LI, N_AN, D_li_cm2s, D_an_cm2s, V_nm3, z, sigma_NE_S_cm, sigma_cNE, t_plus_cNE,
    D_li_cm2s_fl, D_an_cm2s_fl,
    spec.cluster_cutoff_nm, spec.cluster_stride_ps, n_frames_used
]}).to_csv(ANALYSIS_DIR/"conductivity_summary.csv", index=False)

np.save(ANALYSIS_DIR/"pop_mat.npy", pop_mat)
pd.DataFrame(pop_mat).to_csv(ANALYSIS_DIR/"pop_mat.csv", index=False)

(ANALYSIS_DIR/"conductivity_summary.csv", ANALYSIS_DIR/"pop_mat.csv")

# ===== Notebook Cell 6 [conductivity-analysis] =====
print("__STAGEV3__:conductivity-analysis:cell6", flush=True)
# =========================
# Cell 6) Production density + summary table
# =========================

def density_from_gro(gro_path: Path, mass_kg: float) -> float:
    box = [float(x) for x in gro_path.read_text().splitlines()[-1].split()]
    if len(box) < 3:
        raise ValueError("GRO box parse failed")
    V_nm3 = box[0] * box[1] * box[2]
    V_m3 = V_nm3 * 1e-27
    rho_kg_m3 = mass_kg / V_m3
    return rho_kg_m3 / 1000.0  # g/cm^3

prod_gro = PROD_DIR / "production.gro"

mass_poly_g = polymer_mw_g_mol * spec.n_chains
mass_salt_g = (li_mw_g_mol + tfsi_mw_g_mol) * spec.li_tfsi_pairs
total_mass_g_mol = mass_poly_g + mass_salt_g
mass_box_kg = total_mass_g_mol / NA / 1000.0

density_g_cm3 = density_from_gro(prod_gro, mass_box_kg)

monomer_smi = spec.psmiles.replace("[*]", spec.placeholder)
monomer = Chem.MolFromSmiles(monomer_smi)
monomer_mw_g_mol = float(Descriptors.MolWt(monomer)) if monomer else float("nan")

molality = molality_from_counts(
    spec.li_tfsi_pairs,
    spec.n_chains,
    polymer_mw_g_mol,
    li_mw_g_mol + tfsi_mw_g_mol,
)


def msd_at_time(msd_path: Path, t_ns: float) -> float:
    df = xvg_to_df(msd_path)
    if df.empty:
        return float("nan")
    t_ps = t_ns * 1000.0
    df = df[df["t"] <= t_ps]
    if df.empty:
        return float("nan")
    return float(df["msd"].iloc[-1])

li_msd_nm2 = msd_at_time(msd_li, spec.analysis_end_ns)
tfsi_msd_nm2 = msd_at_time(msd_an, spec.analysis_end_ns)

summary_df = pd.DataFrame([{
    "Trajectory ID": spec.name[5:],
    "PSMILES": spec.psmiles,
    "Molality": molality,
    "Monomer Molecular Weight": monomer_mw_g_mol,
    "Degree of Polymerization": spec.n_repeat,
    "Density": density_g_cm3,
    "Li-ion Conductivity": sigma_cNE,
    "TFSI MSD": tfsi_msd_nm2,
    "Li MSD": li_msd_nm2,
    "Transfer number": t_plus_cNE,
}])

summary_df.to_csv(ANALYSIS_DIR / "final_summary.csv", index=False)
summary_df

# ===== Notebook Cell 7 [conductivity-analysis] =====
print("__STAGEV3__:conductivity-analysis:cell7", flush=True)
# =========================
# Cell 7) Aggregate comparison (Trajectory ID)
# =========================
import re

AGG_PATH = ROOT.parent / "simulation-trajectory-aggregate.csv"
if not AGG_PATH.exists():
    raise FileNotFoundError(f"aggregate CSV not found: {AGG_PATH}")

agg_df = pd.read_csv(AGG_PATH)

m = re.search(r"\d+", spec.name)
if not m:
    raise ValueError(f"No Trajectory ID found in spec.name: {spec.name}")
traj_id = int(m.group())

current_df = pd.DataFrame([{
    "Trajectory ID": traj_id,
    "CONDUCTIVITY": float(summary_df["Li-ion Conductivity"].iloc[0]),
    "TFSI Diffusivity": float(D_an_cm2s),
    "Li Diffusivity": float(D_li_cm2s),
    "Transference Number": float(summary_df["Transfer number"].iloc[0]),
}])

merged = current_df.merge(agg_df, on="Trajectory ID", how="left", suffixes=("_cur", "_ref"))

metrics = ["CONDUCTIVITY", "TFSI Diffusivity", "Li Diffusivity", "Transference Number"]
rows = []
for col in metrics:
    cur = merged[f"{col}_cur"].iloc[0] if f"{col}_cur" in merged.columns else float("nan")
    ref = merged[f"{col}_ref"].iloc[0] if f"{col}_ref" in merged.columns else float("nan")
    diff = float(cur - ref) if pd.notna(cur) and pd.notna(ref) else float("nan")
    pct = float(diff / ref * 100.0) if pd.notna(diff) and ref not in (0.0, 0) else float("nan")
    rows.append({
        "metric": col,
        "current": cur,
        "reference": ref,
        "abs_diff": diff,
        "pct_diff": pct,
    })

compare_df = pd.DataFrame(rows)
compare_df.to_csv(ANALYSIS_DIR / "aggregate_compare.csv", index=False)
compare_df

# ===== Notebook Cell 8 [conductivity-analysis] =====
print("__STAGEV3__:conductivity-analysis:cell8", flush=True)
# =========================
# Cell X) Post-check: spec vs actual / NE vs cNE / block stats
# =========================
from pathlib import Path
import numpy as np, pandas as pd, re

traj_ids = None  # e.g., ['27622','14768']; None -> current spec
analysis_begin_ns = float(spec.analysis_begin_ns)
analysis_end_ns = float(spec.analysis_end_ns)
n_blocks = 4

def _parse_tid(name: str) -> str:
    m = re.search(r'(\d+)', name)
    return m.group(1) if m else name

def _read_ndx_counts(path: Path):
    counts = {}
    if not path.exists():
        return counts
    grp = None
    cnt = 0
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith(';'):
            continue
        if ln.startswith('['):
            if grp is not None:
                counts[grp] = cnt
            grp = ln.strip('[]').strip()
            cnt = 0
            continue
        if grp is None:
            continue
        cnt += len(ln.split())
    if grp is not None:
        counts[grp] = cnt
    return counts

def _xvg_to_array(path: Path):
    rows = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith(('#','@')):
            continue
        try:
            rows.append([float(x) for x in ln.split()])
        except ValueError:
            pass
    return np.array(rows, float)

def _diffusion_from_msd(arr, t0_ns, t1_ns):
    if arr.size == 0:
        return None
    t_ns = arr[:,0] / 1000.0
    y = arr[:,1]
    m = (t_ns >= t0_ns) & (t_ns <= t1_ns)
    if m.sum() < 5:
        t_fit, y_fit = t_ns, y
    else:
        t_fit, y_fit = t_ns[m], y[m]
    slope = np.polyfit(t_fit, y_fit, 1)[0]
    return float((slope/6.0) * 1e-5)  # cm^2/s

def _block_stats(arr, t0_ns, t1_ns, n_blocks):
    if arr.size == 0:
        return None, None, 0
    t_ns = arr[:,0] / 1000.0
    y = arr[:,1]
    edges = np.linspace(t0_ns, t1_ns, n_blocks + 1)
    vals = []
    for i in range(n_blocks):
        a, b = edges[i], edges[i+1]
        m = (t_ns >= a) & (t_ns <= b)
        if m.sum() < 5:
            continue
        slope = np.polyfit(t_ns[m], y[m], 1)[0]
        vals.append((slope/6.0) * 1e-5)
    if not vals:
        return None, None, 0
    vals = np.array(vals, float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, len(vals)

def _sigma_NE(D_li, D_an, z, N_LI, N_AN, V_nm3, T_K):
    V_cm3 = V_nm3 * 1e-21
    return (E_CHG**2)/(KB*T_K*V_cm3) * ((z**2)*N_LI*D_li + (z**2)*N_AN*D_an)

def _sigma_cNE_from_pop(pop_mat, D_li, D_an, z, V_nm3, T_K):
    V_cm3 = V_nm3 * 1e-21
    sigma = 0.0
    max_cluster = pop_mat.shape[0]
    for i in range(max_cluster):
        for j in range(max_cluster):
            if i == j:
                continue
            q = (i*z - j*z)
            if q == 0.0:
                continue
            if i > j:
                sigma += (E_CHG**2)/(KB*T_K*V_cm3) * (q*q) * pop_mat[i,j] * D_li
            else:
                sigma += (E_CHG**2)/(KB*T_K*V_cm3) * (q*q) * pop_mat[i,j] * D_an
    return sigma

current_tid = _parse_tid(spec.name)
if traj_ids is None:
    traj_ids = [current_tid]

base = spec.workspace.parent
rows = []
block_rows = []

for tid in traj_ids:
    traj = base / f'Traj_{tid}'
    md_dir = traj / 'md'
    analysis_dir = traj / 'analysis'
    row = {'Trajectory ID': tid}

    ndx_counts = _read_ndx_counts(md_dir / 'index.ndx')
    row['LI_in_ndx'] = ndx_counts.get('LI')
    an_group = None
    for g in [f"{spec.anion_resname_coords}_N", 'TFSI_N', spec.anion_resname_coords, 'TFSI']:
        if g in ndx_counts:
            an_group = g
            row['An_in_ndx'] = ndx_counts[g]
            break
    row['An_group'] = an_group
    if tid == current_tid:
        row['spec_li_tfsi_pairs'] = spec.li_tfsi_pairs
        if row.get('LI_in_ndx') is not None:
            row['LI_spec_diff'] = row['LI_in_ndx'] - spec.li_tfsi_pairs

    sum_path = analysis_dir / 'conductivity_summary.csv'
    metrics = {}
    if sum_path.exists():
        df_sum = pd.read_csv(sum_path)
        metrics = dict(zip(df_sum['metric'], df_sum['value']))
        row['sigma_NE_summary'] = metrics.get('sigma_NE(S/cm)')
        row['sigma_cNE_summary'] = metrics.get('sigma_cNE(S/cm)')
        row['D_Li_summary'] = metrics.get('D_Li(cm^2/s)')
        row['D_an_summary'] = metrics.get('D_an(cm^2/s)')

    # recalc NE/cNE using summary + pop_mat (sanity check)
    if metrics:
        try:
            z = float(metrics.get('z'))
            V_nm3 = float(metrics.get('V(nm^3)'))
            N_LI = float(metrics.get('N_LI'))
            N_AN = float(metrics.get('N_AN'))
            D_li = float(metrics.get('D_Li(cm^2/s)'))
            D_an = float(metrics.get('D_an(cm^2/s)'))
            sigma_ne_calc = _sigma_NE(D_li, D_an, z, N_LI, N_AN, V_nm3, float(spec.temperature_prod))
            row['sigma_NE_calc'] = sigma_ne_calc
            if metrics.get('sigma_NE(S/cm)'):
                row['sigma_NE_diff_pct'] = (sigma_ne_calc/float(metrics['sigma_NE(S/cm)'])-1.0)*100.0

            pop_path = analysis_dir / 'pop_mat.npy'
            if pop_path.exists():
                pop = np.load(pop_path)
                sigma_cne_calc = _sigma_cNE_from_pop(pop, D_li, D_an, z, V_nm3, float(spec.temperature_prod))
                row['sigma_cNE_calc'] = sigma_cne_calc
                if metrics.get('sigma_cNE(S/cm)'):
                    row['sigma_cNE_diff_pct'] = (sigma_cne_calc/float(metrics['sigma_cNE(S/cm)'])-1.0)*100.0
        except Exception:
            pass

    # block averages from msd (NE only)
    msd_li = analysis_dir / 'msd_li.xvg'
    anion_candidates = [
        analysis_dir / f"msd_{spec.anion_resname_coords.lower()}_N.xvg",
    ]
    anion_file = next((p for p in anion_candidates if p.exists()), None)
    if anion_file is None:
        for p in analysis_dir.glob('msd_*_N.xvg'):
            if p.name != 'msd_li.xvg':
                anion_file = p
                break

    if msd_li.exists() and anion_file is not None:
        arr_li = _xvg_to_array(msd_li)
        arr_an = _xvg_to_array(anion_file)
        D_li_blk, D_li_std, n1 = _block_stats(arr_li, analysis_begin_ns, analysis_end_ns, n_blocks)
        D_an_blk, D_an_std, n2 = _block_stats(arr_an, analysis_begin_ns, analysis_end_ns, n_blocks)
        n_used = min(n1, n2)
        blk = {'Trajectory ID': tid, 'blocks_used': n_used, 'D_Li_mean': D_li_blk, 'D_Li_std': D_li_std,
               'D_an_mean': D_an_blk, 'D_an_std': D_an_std}
        if metrics and D_li_blk is not None and D_an_blk is not None:
            try:
                z = float(metrics.get('z'))
                V_nm3 = float(metrics.get('V(nm^3)'))
                N_LI = float(metrics.get('N_LI'))
                N_AN = float(metrics.get('N_AN'))
                sigma_blk = _sigma_NE(D_li_blk, D_an_blk, z, N_LI, N_AN, V_nm3, float(spec.temperature_prod))
                blk['sigma_NE_mean'] = sigma_blk
            except Exception:
                pass
        block_rows.append(blk)

    rows.append(row)

diag_df = pd.DataFrame(rows)
block_df = pd.DataFrame(block_rows)
diag_df

block_df

# ===== Notebook Cell 9 [conductivity-analysis] =====
print("__STAGEV3__:conductivity-analysis:cell9", flush=True)
# =========================
# Cell Z) htpmd-reference conductivity/transference (z=1, max_cluster=spec.cluster_max_cluster)
# =========================
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import pandas as pd

PROD_DIR = MD_DIR / "production"
GRO = PROD_DIR / "production.gro"
XTC = PROD_DIR / "production.xtc"
if not GRO.exists():
    raise FileNotFoundError(f"Missing GRO: {GRO}")

analysis_dir = ANALYSIS_DIR
pop_npy = analysis_dir / "pop_mat.npy"
pop_csv = analysis_dir / "pop_mat.csv"

if pop_npy.exists():
    pop_mat = np.load(pop_npy)
elif pop_csv.exists():
    pop_mat = pd.read_csv(pop_csv).values
else:
    raise FileNotFoundError("Missing pop matrix. Run cNE analysis cell first.")

if pop_mat.ndim != 2:
    raise RuntimeError(f"Invalid pop_mat shape: {pop_mat.shape}")

anion_res = spec.anion_resname_coords
msd_li = analysis_dir / "msd_li.xvg"
anion_candidates = [analysis_dir / f"msd_{anion_res.lower()}_N.xvg"]
anion_candidates.extend(sorted(analysis_dir.glob("msd_*_N.xvg")))
msd_an = next((x for x in anion_candidates if x.exists() and x.name != "msd_li.xvg"), None)

if not msd_li.exists() or msd_an is None:
    raise FileNotFoundError("Missing MSD xvg files (msd_li.xvg / msd_*_N.xvg). Run MSD cell first.")


def _xvg_to_df(path: Path) -> pd.DataFrame:
    rows = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith(("#", "@")):
            continue
        try:
            rows.append([float(x) for x in ln.split()])
        except ValueError:
            pass
    arr = np.array(rows, dtype=float)
    if arr.size == 0:
        raise RuntimeError(f"No numeric rows in {path}")
    return pd.DataFrame(arr, columns=["t", "msd"])


def _diffusion_from_msd(path: Path, t0_ns: float, t1_ns: float) -> float:
    df = _xvg_to_df(path)
    t_ns = df["t"].values / 1000.0
    y = df["msd"].values
    m = (t_ns >= t0_ns) & (t_ns <= t1_ns)
    if int(m.sum()) >= 5:
        xfit, yfit = t_ns[m], y[m]
    else:
        xfit, yfit = t_ns, y
    slope = np.polyfit(xfit, yfit, 1)[0]  # nm^2/ns
    return float((slope / 6.0) * 1e-5)    # cm^2/s


window_begin_ns = float(spec.analysis_begin_ns)
window_end_ns = float(spec.analysis_end_ns)

D_li_cm2s = _diffusion_from_msd(msd_li, window_begin_ns, window_end_ns)
D_an_cm2s = _diffusion_from_msd(msd_an, window_begin_ns, window_end_ns)

box_vals = [float(x) for x in GRO.read_text().splitlines()[-1].split()]
if len(box_vals) < 3:
    raise RuntimeError(f"Unexpected GRO box line: {box_vals}")
V_nm3 = box_vals[0] * box_vals[1] * box_vals[2]
V_cm3 = V_nm3 * 1e-21
TEMP_K = float(spec.temperature_prod)

z_htpmd = 1.0
htpmd_max_cluster = int(getattr(spec, "htpmd_max_cluster", getattr(spec, "cluster_max_cluster", 101)))

N_LI = int(spec.li_tfsi_pairs)
N_AN = int(spec.li_tfsi_pairs)
pref = (E_CHG**2) / (KB * TEMP_K * V_cm3)

sigma_NE_htpmd_S_cm = pref * ((z_htpmd**2) * N_LI * D_li_cm2s + (z_htpmd**2) * N_AN * D_an_cm2s)

persistence_threshold_ps = float(getattr(spec, "cluster_persistence_threshold_ps", 0.0) or 0.0)
sigma_eval_mode = "pure_cNE"
cluster_frames_from_persistence = float("nan")
lifetime_stats = {
    "contact_lifetime_n_runs": 0,
    "contact_lifetime_median_ps": float("nan"),
    "contact_lifetime_mean_ps": float("nan"),
    "contact_lifetime_p90_ps": float("nan"),
    "contact_lifetime_p95_ps": float("nan"),
    "contact_lifetime_p99_ps": float("nan"),
    "contact_lifetime_max_ps": float("nan"),
    "contact_lifetime_run_frac_ge_threshold": float("nan"),
    "contact_lifetime_time_frac_ge_threshold": float("nan"),
}
pop_mat_for_sigma = pop_mat


def _add_pop_from_li_an_codes(pop: np.ndarray, codes: np.ndarray, n_li: int, n_an: int) -> None:
    if codes.size == 0:
        if pop.shape[0] > 1:
            pop[1, 0] += n_li
        if pop.shape[1] > 1:
            pop[0, 1] += n_an
        return

    n_tot = n_li + n_an
    parent = np.arange(n_tot, dtype=np.int32)
    rank = np.zeros(n_tot, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    li = (codes // n_an).astype(np.int32, copy=False)
    an = (codes % n_an).astype(np.int32, copy=False)
    for li_i, an_i in zip(li, an):
        union(int(li_i), n_li + int(an_i))

    active_li = np.unique(li)
    active_an = np.unique(an)
    comp_li = defaultdict(int)
    comp_an = defaultdict(int)
    for li_i in active_li:
        comp_li[find(int(li_i))] += 1
    for an_i in active_an:
        comp_an[find(n_li + int(an_i))] += 1

    for root in set(comp_li) | set(comp_an):
        i = comp_li.get(root, 0)
        j = comp_an.get(root, 0)
        if i < pop.shape[0] and j < pop.shape[1]:
            pop[i, j] += 1.0

    free_li = n_li - len(active_li)
    free_an = n_an - len(active_an)
    if free_li and pop.shape[0] > 1:
        pop[1, 0] += free_li
    if free_an and pop.shape[1] > 1:
        pop[0, 1] += free_an


def _selection_missing(name: str) -> bool:
    obj = globals().get(name)
    if obj is None:
        return True
    try:
        return len(obj) == 0
    except Exception:
        return True


def _ensure_persistence_inputs():
    global u, sel_li, sel_o, sel_anionN, FastNS
    if "FastNS" not in globals():
        from MDAnalysis.lib.nsgrid import FastNS as _FastNS
        FastNS = _FastNS
    if "u" not in globals() or not hasattr(u, "trajectory"):
        if not XTC.exists():
            raise FileNotFoundError(f"Missing XTC for persistence cNE: {XTC}")
        import MDAnalysis as mda
        u = mda.Universe(str(GRO), str(XTC))
    cation_res = getattr(spec, "cation_resname_coords", "LI")
    anion_res = getattr(spec, "anion_resname_coords", "TFSI")
    if _selection_missing("sel_li"):
        sel_li = u.select_atoms(f"resname {cation_res}")
    if _selection_missing("sel_o"):
        sel_o = u.select_atoms(f"resname {anion_res} and name O*")
    if _selection_missing("sel_anionN"):
        sel_anionN = u.select_atoms(f"resname {anion_res} and name N*")
    if len(sel_li) == 0 or len(sel_o) == 0 or len(sel_anionN) == 0:
        raise RuntimeError(
            f"Persistence cNE selection empty: LI={len(sel_li)}, O={len(sel_o)}, N={len(sel_anionN)}"
        )
    return u, sel_li, sel_o, sel_anionN


def _persistence_filtered_pop_mat(cutoff_nm: float, threshold_ps: float, max_cluster: int):
    u_local, li_atoms, o_atoms, anion_n_atoms = _ensure_persistence_inputs()
    n_li = len(li_atoms)
    n_an = len(anion_n_atoms)
    if n_li <= 0 or n_an <= 0:
        raise RuntimeError("Persistence cNE requires non-empty Li and anion selections.")

    pop_shape = (max(2, int(max_cluster)), max(2, int(max_cluster)))
    pop = np.zeros(pop_shape, dtype=np.float64)
    anion_resindex_to_i = {int(residx): i for i, residx in enumerate(list(anion_n_atoms.resindices))}
    try:
        o_to_an = np.array([anion_resindex_to_i[int(atom.resindex)] for atom in o_atoms], dtype=np.int32)
    except KeyError as exc:
        raise RuntimeError("Could not map every TFSI oxygen to its anion N residue for persistence cNE.") from exc

    cutoff_a = float(cutoff_nm) * 10.0
    dt_ps = float(u_local.trajectory.dt)
    if not np.isfinite(dt_ps) or dt_ps <= 0:
        dt_ps = 1.0
    stride_ps = float(getattr(spec, "cluster_stride_ps", 0.0) or 0.0)
    step = max(1, int(round(stride_ps / dt_ps))) if stride_ps > 0 else 1
    sample_dt_ps = dt_ps * step
    t0_ps = float(spec.analysis_begin_ns) * 1000.0
    t1_ps = float(spec.analysis_end_ns) * 1000.0

    frame_codes = []
    active = {}
    run_records = []

    for iframe, ts in enumerate(u_local.trajectory[::step]):
        if ts.time < t0_ps or ts.time > t1_ps:
            continue
        coords = np.vstack([li_atoms.positions, o_atoms.positions]).astype(np.float32)
        pairs = FastNS(cutoff_a, coords, np.asarray(ts.dimensions, dtype=np.float32), pbc=True).self_search().get_pairs()
        if pairs.size:
            a = pairs[:, 0]
            b = pairs[:, 1]
            mask = ((a < n_li) & (b >= n_li)) | ((b < n_li) & (a >= n_li))
            pair_contacts = pairs[mask]
            if pair_contacts.size:
                li_idx = np.where(pair_contacts[:, 0] < n_li, pair_contacts[:, 0], pair_contacts[:, 1])
                o_idx = np.where(
                    pair_contacts[:, 0] >= n_li,
                    pair_contacts[:, 0] - n_li,
                    pair_contacts[:, 1] - n_li,
                )
                an_idx = o_to_an[o_idx.astype(np.int32)]
                codes = np.unique(li_idx.astype(np.int32) * n_an + an_idx.astype(np.int32))
            else:
                codes = np.empty(0, dtype=np.int32)
        else:
            codes = np.empty(0, dtype=np.int32)

        local_frame = len(frame_codes)
        frame_codes.append(codes.astype(np.int32, copy=False))
        current = set(int(code) for code in codes)
        for code in list(active.keys()):
            if code not in current:
                first, last = active.pop(code)
                run_records.append((code, first, last, last - first + 1, (last - first) * sample_dt_ps))
        for code in current:
            if code in active:
                first, _last = active[code]
                active[code] = (first, local_frame)
            else:
                active[code] = (local_frame, local_frame)

    n_frames = len(frame_codes)
    if n_frames == 0:
        raise RuntimeError("No frames used for persistence cNE. analysis window/stride 확인.")
    for code, (first, last) in active.items():
        run_records.append((code, first, last, last - first + 1, (last - first) * sample_dt_ps))

    starts = [[] for _ in range(n_frames)]
    ends = [[] for _ in range(n_frames + 1)]
    for code, first, last, _length, tau_span in run_records:
        if tau_span >= threshold_ps:
            starts[first].append(code)
            if last + 1 <= n_frames:
                ends[last + 1].append(code)

    active_bool = np.zeros(n_li * n_an, dtype=bool)
    for iframe, codes in enumerate(frame_codes):
        if ends[iframe]:
            active_bool[np.array(ends[iframe], dtype=np.int32)] = False
        if starts[iframe]:
            active_bool[np.array(starts[iframe], dtype=np.int32)] = True
        filtered = codes[active_bool[codes]] if codes.size else codes
        _add_pop_from_li_an_codes(pop, filtered, n_li, n_an)
    pop /= float(n_frames)

    taus = np.array([rec[4] for rec in run_records], dtype=float)
    if taus.size:
        tau_sum = float(np.sum(taus))
        ge = taus >= threshold_ps
        stats = {
            "contact_lifetime_n_runs": int(taus.size),
            "contact_lifetime_median_ps": float(np.median(taus)),
            "contact_lifetime_mean_ps": float(np.mean(taus)),
            "contact_lifetime_p90_ps": float(np.percentile(taus, 90)),
            "contact_lifetime_p95_ps": float(np.percentile(taus, 95)),
            "contact_lifetime_p99_ps": float(np.percentile(taus, 99)),
            "contact_lifetime_max_ps": float(np.max(taus)),
            "contact_lifetime_run_frac_ge_threshold": float(np.mean(ge)),
            "contact_lifetime_time_frac_ge_threshold": float(np.sum(taus[ge]) / tau_sum) if tau_sum > 0 else float("nan"),
        }
    else:
        stats = {
            "contact_lifetime_n_runs": 0,
            "contact_lifetime_median_ps": float("nan"),
            "contact_lifetime_mean_ps": float("nan"),
            "contact_lifetime_p90_ps": float("nan"),
            "contact_lifetime_p95_ps": float("nan"),
            "contact_lifetime_p99_ps": float("nan"),
            "contact_lifetime_max_ps": float("nan"),
            "contact_lifetime_run_frac_ge_threshold": float("nan"),
            "contact_lifetime_time_frac_ge_threshold": float("nan"),
        }
    stats["cluster_persistence_sample_dt_ps"] = float(sample_dt_ps)
    return pop, float(n_frames), stats


if persistence_threshold_ps > 0.0:
    pop_mat_for_sigma, cluster_frames_from_persistence, lifetime_stats = _persistence_filtered_pop_mat(
        float(spec.cluster_cutoff_nm), persistence_threshold_ps, htpmd_max_cluster
    )
    sigma_eval_mode = f"cNE_persist_ge_{persistence_threshold_ps:g}ps"
    cutoff_label = f"{float(spec.cluster_cutoff_nm):g}".replace(".", "p")
    threshold_label = f"{persistence_threshold_ps:g}".replace(".", "p")
    pop_label = f"pop_mat_persist_cutoff{cutoff_label}_ge{threshold_label}ps"
    np.save(analysis_dir / f"{pop_label}.npy", pop_mat_for_sigma)
    pd.DataFrame(pop_mat_for_sigma).to_csv(analysis_dir / f"{pop_label}.csv", index=False)
    pd.DataFrame([{**lifetime_stats, "cutoff_nm": float(spec.cluster_cutoff_nm), "persistence_threshold_ps": persistence_threshold_ps}]).to_csv(
        analysis_dir / f"contact_lifetime_summary_cutoff{cutoff_label}_ge{threshold_label}ps.csv", index=False
    )

max_i = min(htpmd_max_cluster, int(pop_mat_for_sigma.shape[0]))
max_j = min(htpmd_max_cluster, int(pop_mat_for_sigma.shape[1]))

sigma_cNE_htpmd_S_cm = 0.0
ctn_num = 0.0
ctn_den = 0.0

for i in range(max_i):
    for j in range(max_j):
        nij = float(pop_mat_for_sigma[i, j])
        if nij == 0.0 or i == j:
            continue

        q = i * z_htpmd - j * z_htpmd
        Dij_eff = D_li_cm2s if i > j else D_an_cm2s

        sigma_cNE_htpmd_S_cm += pref * (q * q) * nij * Dij_eff
        ctn_num += (i * z_htpmd) * q * nij * Dij_eff
        ctn_den += (q * q) * nij * Dij_eff

c_tn_htpmd = float(ctn_num / ctn_den) if ctn_den != 0.0 else float("nan")
ii = np.arange(pop_mat_for_sigma.shape[0], dtype=float)[:, None]
jj = np.arange(pop_mat_for_sigma.shape[1], dtype=float)[None, :]
alpha_Li_sum = float((ii * pop_mat_for_sigma).sum())
alpha_AN_sum = float((jj * pop_mat_for_sigma).sum())

m = re.search(r"(\d+)", str(spec.name))
traj_id = m.group(1) if m else str(spec.name)

summary_path = analysis_dir / "conductivity_summary.csv"
cluster_frames_static_summary = float("nan")
if summary_path.exists():
    try:
        s = pd.read_csv(summary_path)
        if set(["metric", "value"]).issubset(s.columns):
            mm = dict(zip(s["metric"], s["value"]))
            cluster_frames_static_summary = float(mm.get("cluster_frames", float("nan")))
    except Exception:
        pass
cluster_frames = cluster_frames_from_persistence if np.isfinite(cluster_frames_from_persistence) else cluster_frames_static_summary

out = pd.DataFrame([{
    "Trajectory ID": traj_id,
    "analysis_begin_ns": window_begin_ns,
    "analysis_end_ns": window_end_ns,
    "analysis_charge_z": z_htpmd,
    "sigma_eval_mode": sigma_eval_mode,
    "cluster_persistence_threshold_ps": persistence_threshold_ps,
    "htpmd_max_cluster": int(htpmd_max_cluster),
    "N_LI": N_LI,
    "N_AN": N_AN,
    "alpha_Li_sum": alpha_Li_sum,
    "alpha_AN_sum": alpha_AN_sum,
    "D_Li_cm2s": D_li_cm2s,
    "D_an_cm2s": D_an_cm2s,
    "V_nm3": V_nm3,
    "sigma_NE_htpmd_S_cm": sigma_NE_htpmd_S_cm,
    "sigma_cNE_htpmd_S_cm": sigma_cNE_htpmd_S_cm,
    "c_tn_htpmd": c_tn_htpmd,
    "cluster_cutoff_nm": float(spec.cluster_cutoff_nm),
    "cluster_stride_ps": float(spec.cluster_stride_ps),
    "cluster_frames_from_summary": cluster_frames,
    "cluster_frames_static_summary": cluster_frames_static_summary,
    "cluster_frames_persistence": cluster_frames_from_persistence,
    **lifetime_stats,
}])

out_csv = analysis_dir / "conductivity_summary_htpmd_ref.csv"
out.to_csv(out_csv, index=False)

print(f"Saved: {out_csv}")
out
