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
print("__STAGEV3__:packmol:entry", flush=True)


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


print("__PHASE_DONE__:packmol", flush=True)
