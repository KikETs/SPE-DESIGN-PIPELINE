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
print("__STAGEV3__:pysoftk:entry", flush=True)


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


print("__PHASE_DONE__:pysoftk", flush=True)
