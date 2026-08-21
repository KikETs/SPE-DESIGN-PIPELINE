from __future__ import annotations

import json
from pathlib import Path

STRICT_MOL_TOL = 1e-3
STRICT_SYSTEM_TOL = 5e-2
STRICT_POLYMER_CHAIN_TOL = 1e-3

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


def extract_moleculetype_name_from_itp(itp_path: Path) -> str:
    in_mt = False
    for raw in itp_path.read_text(errors='ignore').splitlines():
        line = raw.strip()
        if not line or line.startswith(';'):
            continue
        if line.startswith('['):
            in_mt = line.lower().startswith('[ moleculetype')
            continue
        if in_mt:
            return line.split()[0]
    raise ValueError(f'moleculetype name not found in {itp_path}')


def parse_itp_atoms_with_indices(itp_path: Path):
    rows = []
    lines = itp_path.read_text(errors='ignore').splitlines()
    in_atoms = False
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith(';'):
            continue
        if line.startswith('['):
            in_atoms = line.lower().startswith('[ atoms')
            continue
        if not in_atoms:
            continue
        core = raw.split(';', 1)[0].split()
        if len(core) < 7:
            continue
        rows.append(
            {
                'line_idx': idx,
                'nr': int(core[0]),
                'atomtype': core[1],
                'resnr': core[2],
                'residue': core[3],
                'atomname': core[4],
                'charge': float(core[6]),
            }
        )
    return lines, rows


def write_itp_charges(itp_path: Path, lines, atoms, new_charges):
    for atom, q in zip(atoms, new_charges):
        raw = lines[atom['line_idx']]
        left, *comm = raw.split(';', 1)
        parts = left.split()
        parts[6] = f'{float(q):.6f}'
        new_line = ' '.join(parts)
        if comm:
            new_line += ' ;' + comm[0]
        lines[atom['line_idx']] = new_line
    itp_path.write_text('\n'.join(lines).rstrip() + '\n')


def rewrite_itp_resname(itp_path: Path, resname: str) -> Path:
    lines = itp_path.read_text().splitlines()
    out = []
    in_atoms = False
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith('[ atoms ]'):
            in_atoms = True
            out.append(ln)
            continue
        if in_atoms and s.startswith('['):
            in_atoms = False
            out.append(ln)
            continue
        if in_atoms:
            if not s or s.startswith(';'):
                out.append(ln)
                continue
            core, *rest = ln.split(';', 1)
            parts = core.split()
            if len(parts) >= 4:
                parts[3] = resname
                core_new = ' '.join(parts)
                ln = core_new + (' ;' + rest[0] if rest else '')
            out.append(ln)
        else:
            out.append(ln)
    itp_path.write_text('\n'.join(out).rstrip() + '\n')
    return itp_path


def canonicalize_tfsi_clean_itp(
    itp_path: Path,
    *,
    anion_target: float,
    resname: str = 'TFSI',
):
    lines, atoms = parse_itp_atoms_with_indices(itp_path)
    if not atoms:
        raise RuntimeError(f'tfsi atoms not found in {itp_path}')

    missing = [atom['atomname'] for atom in atoms if atom['atomname'] not in LAMMPS_TFSI_FQ07_CHARGES]
    if missing:
        raise KeyError(f'missing canonical TFSI charges for {sorted(set(missing))}')

    scale = float(anion_target) / 0.7
    new = [float(LAMMPS_TFSI_FQ07_CHARGES[atom['atomname']]) * scale for atom in atoms]
    total = sum(new)
    delta = (-float(anion_target)) - total
    if abs(delta) > 1e-12:
        j = max(range(len(new)), key=lambda k: abs(new[k]))
        new[j] += delta

    for atom, q in zip(atoms, new):
        raw = lines[atom['line_idx']]
        left, *comm = raw.split(';', 1)
        parts = left.split()
        parts[3] = resname
        parts[6] = f'{float(q):.6f}'
        new_line = ' '.join(parts)
        if comm:
            new_line += ' ;' + comm[0]
        lines[atom['line_idx']] = new_line
    itp_path.write_text('\n'.join(lines).rstrip() + '\n')
    return {
        'tfsi_q_after': float(sum(new)),
        'scale': scale,
        'resname': resname,
    }


def diagnose_charge_imbalance(state: dict, *, li_target: float, anion_target: float):
    polymer_total = float(state['polymer_q_chain']) * int(state['polymer_n_mol'])
    tfsi_mismatch_mol = float(state['tfsi_q_mol']) - (-float(anion_target))
    li_mismatch_mol = float(state['li_q_mol']) - float(li_target)
    tfsi_total_drift = tfsi_mismatch_mol * int(state['tfsi_n_mol'])
    li_total_drift = li_mismatch_mol * int(state['li_n_mol'])
    nonpolymer_residual = float(state['system_q']) - polymer_total

    dominant = 'mixed'
    candidates = {
        'polymer': abs(polymer_total),
        'tfsi': abs(tfsi_total_drift),
        'li': abs(li_total_drift),
        'nonpolymer_residual': abs(nonpolymer_residual),
    }
    dominant = max(candidates, key=candidates.get)

    return {
        'polymer_total_charge': polymer_total,
        'tfsi_mismatch_per_mol': tfsi_mismatch_mol,
        'tfsi_total_drift': tfsi_total_drift,
        'li_mismatch_per_mol': li_mismatch_mol,
        'li_total_drift': li_total_drift,
        'nonpolymer_residual': nonpolymer_residual,
        'dominant_source': dominant,
    }


def read_topology_charge_state(run_dir: Path):
    topo_dir = run_dir / 'topology'
    topol_top = topo_dir / 'topol.top'
    pol_itp = topo_dir / 'polymer_clean.itp'
    tfsi_itp = topo_dir / 'tfsi_clean.itp'
    li_itp = topo_dir / 'li_clean.itp'
    required = [topol_top, pol_itp, tfsi_itp, li_itp]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f'missing topology files for charge sanity: {missing}')

    mol_counts = {}
    in_molecules = False
    for raw in topol_top.read_text(errors='ignore').splitlines():
        line = raw.strip()
        if not line or line.startswith(';'):
            continue
        if line.startswith('['):
            in_molecules = line.lower().startswith('[ molecules')
            continue
        if not in_molecules:
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                mol_counts[parts[0]] = int(parts[1])
            except Exception:
                continue

    def _mol_charge(itp_path: Path):
        _, atoms = parse_itp_atoms_with_indices(itp_path)
        return (
            float(sum(a['charge'] for a in atoms)),
            len(atoms),
            sum(1 for a in atoms if not a['atomname'].upper().startswith('H')),
        )

    pol_mt = extract_moleculetype_name_from_itp(pol_itp)
    tfsi_mt = extract_moleculetype_name_from_itp(tfsi_itp)
    li_mt = extract_moleculetype_name_from_itp(li_itp)

    q_pol, n_pol_atoms, n_pol_heavy = _mol_charge(pol_itp)
    q_tfsi, n_tfsi_atoms, _ = _mol_charge(tfsi_itp)
    q_li, n_li_atoms, _ = _mol_charge(li_itp)

    n_pol = int(mol_counts.get(pol_mt, 0))
    n_tfsi = int(mol_counts.get(tfsi_mt, 0))
    n_li = int(mol_counts.get(li_mt, 0))
    system_q = q_pol * n_pol + q_tfsi * n_tfsi + q_li * n_li

    return {
        'topol_top': topol_top,
        'polymer_itp': pol_itp,
        'tfsi_itp': tfsi_itp,
        'li_itp': li_itp,
        'polymer_mt': pol_mt,
        'tfsi_mt': tfsi_mt,
        'li_mt': li_mt,
        'polymer_q_chain': q_pol,
        'tfsi_q_mol': q_tfsi,
        'li_q_mol': q_li,
        'polymer_n_mol': n_pol,
        'tfsi_n_mol': n_tfsi,
        'li_n_mol': n_li,
        'polymer_n_atoms': n_pol_atoms,
        'polymer_n_heavy': n_pol_heavy,
        'tfsi_n_atoms': n_tfsi_atoms,
        'li_n_atoms': n_li_atoms,
        'system_q': system_q,
    }


def neutralize_polymer_itp(
    polymer_itp: Path,
    *,
    max_per_atom_shift: float = 0.01,
    max_abs_chain_correction: float = 5.0,
):
    lines, atoms = parse_itp_atoms_with_indices(polymer_itp)
    if not atoms:
        raise RuntimeError(f'polymer atoms not found in {polymer_itp}')
    heavy_idx = [i for i, atom in enumerate(atoms) if not atom['atomname'].upper().startswith('H')]
    if not heavy_idx:
        raise RuntimeError('polymer has no heavy atoms available for charge neutralization')

    charges = [atom['charge'] for atom in atoms]
    q_chain = float(sum(charges))
    if abs(q_chain) <= 1e-12:
        return {
            'polymer_q_before': q_chain,
            'polymer_q_after': q_chain,
            'max_per_atom_shift': 0.0,
            'heavy_atoms_used': len(heavy_idx),
        }
    if abs(q_chain) > float(max_abs_chain_correction):
        raise RuntimeError(
            f'polymer chain net charge too large for safe auto-fix: q_chain={q_chain:.6f}, '
            f'limit={max_abs_chain_correction:.6f}'
        )

    per_atom = (-q_chain) / float(len(heavy_idx))
    if abs(per_atom) > float(max_per_atom_shift):
        raise RuntimeError(
            'polymer auto-fix would shift heavy-atom charges too much: '
            f'per_atom_shift={per_atom:.6f}, limit={max_per_atom_shift:.6f}'
        )

    new_charges = list(charges)
    for idx in heavy_idx[:-1]:
        new_charges[idx] += per_atom
    residual = -sum(new_charges)
    new_charges[heavy_idx[-1]] += residual
    write_itp_charges(polymer_itp, lines, atoms, new_charges)

    return {
        'polymer_q_before': q_chain,
        'polymer_q_after': float(sum(new_charges)),
        'max_per_atom_shift': abs(per_atom),
        'heavy_atoms_used': len(heavy_idx),
    }


def charge_sanity_report_ok(report_path: Path) -> bool:
    if (not report_path.exists()) or report_path.stat().st_size <= 0:
        return False
    try:
        data = json.loads(report_path.read_text())
    except Exception:
        return False
    if data.get('status') not in {'ok', 'repaired'}:
        return False

    state = data.get('after') if isinstance(data.get('after'), dict) else data
    try:
        polymer_q_chain = float(state.get('polymer_q_chain'))
        system_q = float(state.get('system_q'))
        li_q_mol = float(state.get('li_q_mol'))
        tfsi_q_mol = float(state.get('tfsi_q_mol'))
        li_target = float(data.get('li_target', 0.7))
        anion_target = float(data.get('anion_target', 0.7))
    except Exception:
        return False

    li_dev = abs(li_q_mol - li_target)
    tfsi_dev = abs(tfsi_q_mol + anion_target)
    return (
        abs(polymer_q_chain) <= STRICT_POLYMER_CHAIN_TOL
        and abs(system_q) <= STRICT_SYSTEM_TOL
        and li_dev <= STRICT_MOL_TOL
        and tfsi_dev <= STRICT_MOL_TOL
    )


def ensure_interphase_charge_sanity(
    run_dir: Path,
    *,
    li_target: float,
    anion_target: float,
    attempt_fix: bool = True,
    mol_tol: float = STRICT_MOL_TOL,
    system_tol: float = STRICT_SYSTEM_TOL,
    report_name: str = 'charge_sanity_interphase.json',
):
    topo_dir = run_dir / 'topology'
    report_path = topo_dir / report_name

    state = read_topology_charge_state(run_dir)
    report = {
        'status': 'unknown',
        'attempt_fix': bool(attempt_fix),
        'li_target': float(li_target),
        'anion_target': float(anion_target),
        **state,
    }

    def _write_report(payload):
        try:
            report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        except Exception:
            pass

    li_dev = abs(state['li_q_mol'] - li_target)
    tfsi_dev = abs(state['tfsi_q_mol'] + anion_target)
    polymer_explains_system = abs(state['system_q'] - state['polymer_q_chain'] * state['polymer_n_mol']) <= system_tol

    report.update(
        {
            'li_dev': li_dev,
            'tfsi_dev': tfsi_dev,
            'polymer_explains_system': polymer_explains_system,
            'diagnosis': diagnose_charge_imbalance(state, li_target=li_target, anion_target=anion_target),
        }
    )

    if tfsi_dev > mol_tol and attempt_fix:
        try:
            tfsi_repair = canonicalize_tfsi_clean_itp(
                state['tfsi_itp'],
                anion_target=float(anion_target),
                resname='TFSI',
            )
            state = read_topology_charge_state(run_dir)
            li_dev = abs(state['li_q_mol'] - li_target)
            tfsi_dev = abs(state['tfsi_q_mol'] + anion_target)
            polymer_explains_system = abs(state['system_q'] - state['polymer_q_chain'] * state['polymer_n_mol']) <= system_tol
            report.update(
                {
                    'tfsi_repair': tfsi_repair,
                    'after_tfsi_repair': state,
                    'li_dev': li_dev,
                    'tfsi_dev': tfsi_dev,
                    'polymer_explains_system': polymer_explains_system,
                    'diagnosis': diagnose_charge_imbalance(state, li_target=li_target, anion_target=anion_target),
                }
            )
        except Exception as exc:
            report.update({'tfsi_repair_error': f'{type(exc).__name__}: {exc}'})

    if li_dev > mol_tol:
        report.update({'status': 'failed', 'reason': 'li_charge_mismatch'})
        _write_report(report)
        raise RuntimeError(
            'charge_sanity_failed: Li moleculetype charge mismatch '
            f'(got {state["li_q_mol"]:.6f}, target {li_target:.6f})'
        )

    if tfsi_dev > mol_tol:
        report.update({'status': 'failed', 'reason': 'tfsi_charge_mismatch'})
        _write_report(report)
        raise RuntimeError(
            'charge_sanity_failed: TFSI moleculetype charge mismatch '
            f'(got {state["tfsi_q_mol"]:.6f}, target {-anion_target:.6f})'
        )

    if abs(state['polymer_q_chain']) <= STRICT_POLYMER_CHAIN_TOL and abs(state['system_q']) <= system_tol:
        report.update({'status': 'ok', 'reason': 'already_neutral'})
        _write_report(report)
        return report

    if not attempt_fix:
        report.update({'status': 'failed', 'reason': 'repair_disabled'})
        _write_report(report)
        raise RuntimeError(
            'charge_sanity_failed: polymer/system not neutral '
            f'(polymer_q_chain={state["polymer_q_chain"]:.6f}, system_q={state["system_q"]:.6f})'
        )

    if not polymer_explains_system:
        report.update({'status': 'failed', 'reason': 'system_charge_not_polymer_dominated'})
        _write_report(report)
        raise RuntimeError(
            'charge_sanity_failed: system net charge is not dominated by polymer; '
            'refusing to patch TFSI/Li charges'
        )

    repair = neutralize_polymer_itp(state['polymer_itp'])
    state2 = read_topology_charge_state(run_dir)
    li_dev2 = abs(state2['li_q_mol'] - li_target)
    tfsi_dev2 = abs(state2['tfsi_q_mol'] + anion_target)

    report.update(
        {
            'repair': repair,
            'after': state2,
            'after_li_dev': li_dev2,
            'after_tfsi_dev': tfsi_dev2,
        }
    )

    if li_dev2 > mol_tol or tfsi_dev2 > mol_tol:
        report.update({'status': 'failed', 'reason': 'non_polymer_charge_drift_after_fix'})
        _write_report(report)
        raise RuntimeError('charge_sanity_failed: non-polymer charge drift remained after polymer fix')

    if abs(state2['polymer_q_chain']) > STRICT_POLYMER_CHAIN_TOL or abs(state2['system_q']) > system_tol:
        report.update({'status': 'failed', 'reason': 'repair_incomplete'})
        _write_report(report)
        raise RuntimeError(
            'charge_sanity_failed: polymer/system still not neutral after repair '
            f'(polymer_q_chain={state2["polymer_q_chain"]:.6f}, system_q={state2["system_q"]:.6f})'
        )

    report.update({'status': 'repaired', 'reason': 'polymer_neutralized'})
    _write_report(report)
    return report
