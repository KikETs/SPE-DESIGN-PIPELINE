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
print("__STAGEV3__:analysis:entry", flush=True)


# ---- phase prelude: analysis ----
_ANALYSIS_REPLICA_STAGE = os.environ.get("GROMACS_ANALYSIS_REPLICA_STAGE", "").strip()
try:
    _ANALYSIS_REPLICA_INDEX = int(os.environ.get("GROMACS_ANALYSIS_REPLICA_INDEX", "1") or "1")
except Exception:
    _ANALYSIS_REPLICA_INDEX = 1

def _analysis_replica_count() -> int:
    raw = os.environ.get("GROMACS_PRODUCTION_REPLICAS", "1")
    try:
        return max(1, int(raw))
    except Exception:
        return 1

def _analysis_replica_entries():
    count = _analysis_replica_count()
    return [(1, "production", MD_DIR / "production")] + [
        (idx, f"production_rep{idx}", MD_DIR / f"production_rep{idx}")
        for idx in range(2, count + 1)
    ]

def _analysis_required_prod_files(stage: str, stage_dir: Path) -> list[Path]:
    return [
        stage_dir / f"{stage}.tpr",
        stage_dir / f"{stage}.xtc",
        stage_dir / f"{stage}.gro",
    ]

def _assert_analysis_replica_inputs() -> None:
    missing = []
    for _idx, stage, stage_dir in _analysis_replica_entries():
        for path in _analysis_required_prod_files(stage, stage_dir):
            if not path.exists():
                missing.append(path)
    if missing:
        shown = "; ".join(str(path) for path in missing[:12])
        if len(missing) > 12:
            shown += f"; ... ({len(missing)} missing total)"
        raise FileNotFoundError(f"[analysis phase] missing production replica outputs: {shown}")

def _numeric_stats(series) -> Optional[dict]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    n = int(vals.shape[0])
    if n == 0:
        return None
    return {
        "n": n,
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if n > 1 else 0.0,
        "var": float(vals.var(ddof=1)) if n > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
    }

def _numeric_stats_frame(df: pd.DataFrame, skip_cols: set[str]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col in skip_cols:
            continue
        stats = _numeric_stats(df[col])
        if stats is None:
            continue
        rows.append({"metric": col, **stats})
    return pd.DataFrame(rows)

def _mean_aggregate_row(df: pd.DataFrame, id_cols: set[str]) -> dict:
    first = df.iloc[0]
    out = {}
    skip_cols = {"replica", "production_stage", "analysis_csv"}
    for col in df.columns:
        if col in skip_cols:
            continue
        if col in id_cols:
            vals = [str(v) for v in df[col].dropna().unique()]
            out[col] = vals[0] if len(vals) == 1 else "|".join(vals)
            continue
        stats = _numeric_stats(df[col])
        if stats is None:
            vals = [str(v) for v in df[col].dropna().unique()]
            out[col] = vals[0] if vals else first.get(col, "")
            continue
        out[col] = stats["mean"]
        for key in ("mean", "std", "var", "min", "max"):
            out[f"{col}_{key}"] = stats[key]
    out["replica_count"] = int(_analysis_replica_count())
    out["replica_success_count"] = int(df["replica"].nunique()) if "replica" in df.columns else int(len(df))
    if "production_stage" in df.columns:
        out["production_stages"] = ",".join(str(v) for v in df["production_stage"].dropna().unique())
    if "analysis_csv" in df.columns:
        out["replica_source_csvs"] = "|".join(str(v) for v in df["analysis_csv"].dropna().unique())
    return out

def _read_replica_csvs(csv_name: str) -> pd.DataFrame:
    frames = []
    for idx, stage, _stage_dir in _analysis_replica_entries():
        src = ROOT / "analysis" / f"replica_{idx}" / csv_name
        if not src.exists():
            raise FileNotFoundError(f"[analysis phase] missing replica analysis output: {src}")
        df = pd.read_csv(src)
        df["replica"] = idx
        df["production_stage"] = stage
        df["analysis_csv"] = str(src)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def _aggregate_wide_replica_csv(csv_name: str, id_cols: set[str]) -> None:
    combined = _read_replica_csvs(csv_name)
    stem = Path(csv_name).stem
    root_analysis = ROOT / "analysis"
    combined.to_csv(root_analysis / f"{stem}_replicas.csv", index=False)
    stats = _numeric_stats_frame(
        combined,
        set(id_cols) | {"replica", "production_stage", "analysis_csv"},
    )
    stats.to_csv(root_analysis / f"{stem}_replica_stats.csv", index=False)
    out = _mean_aggregate_row(combined, id_cols)
    pd.DataFrame([out]).to_csv(root_analysis / csv_name, index=False)

def _aggregate_metric_value_csv(csv_name: str) -> None:
    combined = _read_replica_csvs(csv_name)
    stem = Path(csv_name).stem
    root_analysis = ROOT / "analysis"
    combined.to_csv(root_analysis / f"{stem}_replicas.csv", index=False)
    rows = []
    for metric, grp in combined.groupby("metric", dropna=False):
        stats = _numeric_stats(grp["value"])
        if stats is None:
            continue
        rows.append({"metric": metric, "value": stats["mean"], **stats})
    pd.DataFrame(rows).to_csv(root_analysis / f"{stem}_replica_stats.csv", index=False)
    pd.DataFrame(rows).to_csv(root_analysis / csv_name, index=False)

def _aggregate_compare_csv() -> None:
    combined = _read_replica_csvs("aggregate_compare.csv")
    root_analysis = ROOT / "analysis"
    combined.to_csv(root_analysis / "aggregate_compare_replicas.csv", index=False)
    rows = []
    for metric, grp in combined.groupby("metric", dropna=False):
        current_stats = _numeric_stats(grp["current"])
        if current_stats is None:
            continue
        ref_vals = pd.to_numeric(grp["reference"], errors="coerce").dropna()
        reference = float(ref_vals.iloc[0]) if not ref_vals.empty else float("nan")
        abs_diff = current_stats["mean"] - reference if np.isfinite(reference) else float("nan")
        pct_diff = abs_diff / reference * 100.0 if np.isfinite(abs_diff) and reference not in (0.0, 0) else float("nan")
        rows.append({
            "metric": metric,
            "current": current_stats["mean"],
            "reference": reference,
            "abs_diff": abs_diff,
            "pct_diff": pct_diff,
            "current_mean": current_stats["mean"],
            "current_std": current_stats["std"],
            "current_var": current_stats["var"],
            "current_min": current_stats["min"],
            "current_max": current_stats["max"],
            "n": current_stats["n"],
        })
    pd.DataFrame(rows).to_csv(root_analysis / "aggregate_compare.csv", index=False)

def _run_replica_analyses_and_aggregate() -> None:
    _assert_analysis_replica_inputs()
    root_analysis = ROOT / "analysis"
    root_analysis.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    entries = _analysis_replica_entries()
    for idx, stage, _stage_dir in entries:
        print(f"[production-replicas] analysis replica {idx}/{len(entries)} stage={stage}", flush=True)
        child_env = os.environ.copy()
        child_env["GROMACS_ANALYSIS_REPLICA_STAGE"] = stage
        child_env["GROMACS_ANALYSIS_REPLICA_INDEX"] = str(idx)
        child_env["GROMACS_ANALYSIS_REPLICA_COUNT"] = str(len(entries))
        child_env["GROMACS_MODULE_NAME"] = f"{os.environ.get('GROMACS_MODULE_NAME', 'gromacs_phase')}_rep{idx}"
        res = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            env=child_env,
            text=True,
            capture_output=True,
            check=False,
        )
        log_path = root_analysis / f"replica_{idx}_analysis.log"
        log_path.write_text((res.stdout or "") + "\n--- STDERR ---\n" + (res.stderr or ""))
        if res.returncode != 0:
            tail = "\n".join(((res.stdout or "") + "\n" + (res.stderr or "")).splitlines()[-40:])
            raise RuntimeError(f"[analysis phase] replica {idx} ({stage}) failed; see {log_path}\n{tail}")
    _aggregate_metric_value_csv("conductivity_summary.csv")
    _aggregate_wide_replica_csv("final_summary.csv", {"Trajectory ID", "PSMILES"})
    _aggregate_compare_csv()
    _aggregate_wide_replica_csv(
        "conductivity_summary_htpmd_ref.csv",
        {"Trajectory ID", "PSMILES", "sigma_eval_mode"},
    )
    print(f"[production-replicas] aggregated analysis for {len(entries)} replicas", flush=True)

if not _ANALYSIS_REPLICA_STAGE and _analysis_replica_count() > 1:
    _run_replica_analyses_and_aggregate()
    print("__PHASE_DONE__:analysis", flush=True)
    sys.exit(0)

PROD_STAGE = _ANALYSIS_REPLICA_STAGE or "production"
if _ANALYSIS_REPLICA_STAGE:
    ANALYSIS_DIR = ROOT / "analysis" / f"replica_{_ANALYSIS_REPLICA_INDEX}"

PROD_DIR = MD_DIR / PROD_STAGE
TPR = PROD_DIR / f"{PROD_STAGE}.tpr"
XTC = PROD_DIR / f"{PROD_STAGE}.xtc"
GRO = PROD_DIR / f"{PROD_STAGE}.gro"
for _p in [TPR, XTC, GRO]:
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
_assert_analysis_replica_inputs()
print(
    f"[production-replicas] analysis stage={PROD_STAGE} "
    f"replica_index={_ANALYSIS_REPLICA_INDEX} inputs_present={len(_analysis_replica_entries())}",
    flush=True,
)
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

prod_gro = GRO

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
    analysis_dir = ANALYSIS_DIR if tid == current_tid else traj / 'analysis'
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


print("__PHASE_DONE__:analysis", flush=True)
