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
print("__STAGEV3__:atomtyping:entry", flush=True)


# ---- phase prelude: atomtyping ----
polymer_pdb = STRUCT_DIR / f"{spec.name}_chain_fix.pdb"
tfsi_pdb = STRUCT_DIR / "tfsi.pdb"
li_pdb = STRUCT_DIR / "li.pdb"
for _p in [polymer_pdb, tfsi_pdb, li_pdb, MD_DIR / "conf_initial.gro"]:
    if not _p.exists():
        raise FileNotFoundError(f"[atomtyping phase] missing prerequisite: {_p}")

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


print("__PHASE_DONE__:atomtyping", flush=True)
