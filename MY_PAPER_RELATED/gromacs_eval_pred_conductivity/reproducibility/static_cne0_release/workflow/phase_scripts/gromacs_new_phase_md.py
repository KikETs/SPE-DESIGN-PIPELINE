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
print("__STAGEV3__:md:entry", flush=True)


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


print("__PHASE_DONE__:md", flush=True)
