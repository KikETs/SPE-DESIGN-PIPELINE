from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

EVAL_ROOT = Path(os.environ.get('GROMACS_EVAL_ROOT', '.')).expanduser().resolve()
GROMACS_ROOT = EVAL_ROOT
RUNS_DIR = EVAL_ROOT / 'runs'
RESULTS_DIR = EVAL_ROOT / 'results'
REF_CSV = GROMACS_ROOT / 'simulation-trajectory-aggregate.csv'
ACPYPE = Path(os.environ.get('ACPYPE', shutil.which('acpype') or 'acpype'))
VENDORED_PYSOFTK_ROOT = GROMACS_ROOT

if str(VENDORED_PYSOFTK_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_PYSOFTK_ROOT))

from pysoftk.linear_polymer.linear_polymer import Lp
from pysoftk.format_printers.format_mol import Fmt


@dataclass
class PrototypeRecord:
    traj_id: int
    sample_group: str
    psmiles: str
    dp: int
    status: str
    error: str
    full_atoms: int
    trimer_atoms: int
    tetramer_atoms: int
    insert_positions: str
    insert_lengths: str
    atomtype_match_rate: float
    charge_mae: float
    charge_rmse: float
    calibrated_charge_mae_sample: float
    calibrated_charge_rmse_sample: float
    sample_fit_a: float
    sample_fit_b: float


def _placeholder_smiles(psmiles: str, placeholder: str) -> str:
    return re.sub(r"\[\*\]|\*", placeholder, psmiles)


def ensure_3d_conformer(mol, seed: int = 0):
    mol = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    params.maxAttempts = 50
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise RuntimeError('RDKit EmbedMolecule failed')
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=600)
    except Exception:
        pass
    return Chem.RemoveHs(mol)


def write_pdb_strict_from_rdkit(mol, path, resname='POL', chain_id='A', resseq=1):
    conf = mol.GetConformer()
    resname3 = (resname or 'MOL').upper()[:3]
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
    lines += ['TER', 'END', '']
    path.write_text('\n'.join(lines))
    return path


def build_chain(psmiles: str, n_repeat: int, out_dir: Path, seed: int = 123, placeholder: str = 'Br'):
    out_dir.mkdir(parents=True, exist_ok=True)
    monomer_smi = _placeholder_smiles(psmiles, placeholder)
    monomer = Chem.MolFromSmiles(monomer_smi)
    if monomer is None:
        raise ValueError(f'invalid monomer smiles: {monomer_smi}')
    monomer3d = ensure_3d_conformer(monomer, seed=seed)
    chain = Lp(mol=monomer3d, atom=placeholder, n_copies=n_repeat, shift=1.0).linear_polymer(force_field='UFF')
    try:
        chain.localopt(forcefield='uff', steps=150)
    except Exception:
        pass

    mol_path = out_dir / f'chain_n{n_repeat}.mol'
    pdb_path = out_dir / f'chain_n{n_repeat}.pdb'
    Fmt(chain).mol_print(str(mol_path))
    mol = Chem.MolFromMolBlock(mol_path.read_text(), sanitize=False, removeHs=False)
    if mol is None:
        raise RuntimeError(f'failed to parse chain mol: {mol_path}')
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    write_pdb_strict_from_rdkit(mol, pdb_path)
    sig = [
        (
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            tuple(sorted(n.GetAtomicNum() for n in atom.GetNeighbors())),
        )
        for atom in mol.GetAtoms()
    ]
    return mol_path, pdb_path, sig


def run_acpype(input_mol: Path, out_dir: Path, basename: str = 'polymer') -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ac_dir = out_dir / f'{basename}.acpype'
    if ac_dir.exists() and any(ac_dir.glob('*_GMX.itp')):
        return ac_dir
    env = os.environ.copy()
    env['PATH'] = f"{ACPYPE.parent}:{env.get('PATH', '')}"
    cmd = [str(ACPYPE), '-i', str(input_mol), '-b', basename, '-c', 'bcc', '-o', 'gmx', '-n', '0']
    proc = subprocess.run(cmd, cwd=out_dir, text=True, capture_output=True, env=env)
    log_path = out_dir / f'{basename}_acpype.log'
    log_path.write_text((proc.stdout or '') + '\n--- STDERR ---\n' + (proc.stderr or ''))
    if proc.returncode != 0:
        raise RuntimeError(f'acpype failed: rc={proc.returncode} log={log_path}')
    if not ac_dir.exists():
        raise FileNotFoundError(f'missing acpype output: {ac_dir}')
    return ac_dir


def parse_itp_atoms(itp_path: Path):
    atoms = []
    in_atoms = False
    for ln in itp_path.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith(';'):
            continue
        if s.startswith('['):
            in_atoms = s.lower().startswith('[ atoms')
            continue
        if not in_atoms:
            continue
        core = ln.split(';', 1)[0].split()
        if len(core) < 8:
            continue
        atoms.append({
            'nr': int(core[0]),
            'type': core[1],
            'resnr': int(core[2]),
            'residue': core[3],
            'atom': core[4],
            'cgnr': int(core[5]),
            'charge': float(core[6]),
            'mass': float(core[7]),
        })
    if not atoms:
        raise ValueError(f'no [ atoms ] parsed from {itp_path}')
    return atoms


def infer_insert_scheme(sig3: Sequence[Tuple], sig4: Sequence[Tuple]):
    sm = SequenceMatcher(a=list(sig3), b=list(sig4), autojunk=False)
    inserts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        if tag != 'insert':
            raise ValueError(f'unsupported diff opcode: {(tag, i1, i2, j1, j2)}')
        inserts.append({'pos': i1, 'at_end': i1 == len(sig3), 'j1': j1, 'j2': j2})
    if not inserts:
        raise ValueError('no insert scheme inferred from trimer/tetramer')
    return inserts


def apply_insert_scheme(base_atoms, insert_specs, extra_repeats: int):
    atoms = list(base_atoms)
    for _ in range(extra_repeats):
        out = []
        cursor = 0
        for spec in insert_specs:
            pos = len(atoms) if spec['at_end'] else int(spec['pos'])
            pos = max(cursor, min(pos, len(atoms)))
            out.extend(atoms[cursor:pos])
            out.extend(spec['block'])
            cursor = pos
        out.extend(atoms[cursor:])
        atoms = out
    renum = []
    for idx, atom in enumerate(atoms, start=1):
        rec = dict(atom)
        rec['nr'] = idx
        rec['cgnr'] = idx
        renum.append(rec)
    return renum


def compare_atom_blocks(full_atoms, expanded_atoms):
    if len(full_atoms) != len(expanded_atoms):
        raise ValueError(f'length mismatch: full={len(full_atoms)} expanded={len(expanded_atoms)}')
    full_q = np.array([a['charge'] for a in full_atoms], dtype=float)
    exp_q = np.array([a['charge'] for a in expanded_atoms], dtype=float)
    type_match = np.mean([fa['type'] == ea['type'] for fa, ea in zip(full_atoms, expanded_atoms)])
    raw_mae = float(np.mean(np.abs(full_q - exp_q)))
    raw_rmse = float(np.sqrt(np.mean((full_q - exp_q) ** 2)))
    A = np.column_stack([exp_q, np.ones_like(exp_q)])
    fit_a, fit_b = np.linalg.lstsq(A, full_q, rcond=None)[0]
    pred = fit_a * exp_q + fit_b
    fit_mae = float(np.mean(np.abs(full_q - pred)))
    fit_rmse = float(np.sqrt(np.mean((full_q - pred) ** 2)))
    return {
        'atomtype_match_rate': float(type_match),
        'charge_mae': raw_mae,
        'charge_rmse': raw_rmse,
        'calibrated_charge_mae_sample': fit_mae,
        'calibrated_charge_rmse_sample': fit_rmse,
        'sample_fit_a': float(fit_a),
        'sample_fit_b': float(fit_b),
        'full_q': full_q,
        'exp_q': exp_q,
    }


def load_successful_candidates(manifest: pd.DataFrame, per_group: int) -> pd.DataFrame:
    rows = []
    for group, gdf in manifest.groupby('sample_group', sort=False):
        picked = 0
        for _, row in gdf.iterrows():
            tid = int(row['Trajectory ID'])
            run_dir = RUNS_DIR / f'Traj_{tid}'
            ok = (run_dir / 'topology' / 'polymer_clean.itp').exists() and (run_dir / 'md' / 'conf_initial_fixed.gro').exists()
            if ok:
                rows.append(row)
                picked += 1
            if picked >= per_group:
                break
    if not rows:
        raise RuntimeError('no successful atomtyping candidates found')
    return pd.DataFrame(rows).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-group', type=int, default=2)
    ap.add_argument('--workspace', type=str, default=str(RESULTS_DIR / 'trimer_typing_workspace'))
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(RESULTS_DIR / 'sample_manifest.csv')
    ref = pd.read_csv(REF_CSV)
    ref['Trajectory ID'] = ref['Trajectory ID'].astype(int)
    candidates = load_successful_candidates(manifest, per_group=max(1, int(args.per_group)))

    records = []
    full_q_all = []
    exp_q_all = []

    for _, row in candidates.iterrows():
        tid = int(row['Trajectory ID'])
        grp = str(row['sample_group'])
        spec_row = ref.loc[ref['Trajectory ID'] == tid]
        if spec_row.empty:
            continue
        psmiles = str(spec_row.iloc[0]['SMILES'])
        dp = int(float(spec_row.iloc[0]['Degree of Polymerization']))
        full_itp = RUNS_DIR / f'Traj_{tid}' / 'topology' / 'polymer_clean.itp'
        full_atoms = parse_itp_atoms(full_itp)
        rec = PrototypeRecord(
            traj_id=tid,
            sample_group=grp,
            psmiles=psmiles,
            dp=dp,
            status='ok',
            error='',
            full_atoms=len(full_atoms),
            trimer_atoms=0,
            tetramer_atoms=0,
            insert_positions='',
            insert_lengths='',
            atomtype_match_rate=float('nan'),
            charge_mae=float('nan'),
            charge_rmse=float('nan'),
            calibrated_charge_mae_sample=float('nan'),
            calibrated_charge_rmse_sample=float('nan'),
            sample_fit_a=float('nan'),
            sample_fit_b=float('nan'),
        )
        try:
            sample_dir = workspace / f'Traj_{tid}'
            tri_mol, tri_pdb, sig3 = build_chain(psmiles, 3, sample_dir / 'trimer')
            tet_mol, tet_pdb, sig4 = build_chain(psmiles, 4, sample_dir / 'tetramer')
            scheme = infer_insert_scheme(sig3, sig4)
            tri_ac_dir = run_acpype(tri_mol, sample_dir / 'trimer_topology', basename='polymer')
            tet_ac_dir = run_acpype(tet_mol, sample_dir / 'tetramer_topology', basename='polymer')
            tri_atoms = parse_itp_atoms(next(tri_ac_dir.glob('*_GMX.itp')))
            tet_atoms = parse_itp_atoms(next(tet_ac_dir.glob('*_GMX.itp')))
            rec.trimer_atoms = len(tri_atoms)
            rec.tetramer_atoms = len(tet_atoms)
            insert_specs = []
            for spec in scheme:
                block = tet_atoms[spec['j1']:spec['j2']]
                insert_specs.append({
                    'pos': int(spec['pos']),
                    'at_end': bool(spec['at_end']),
                    'block': block,
                })
            rec.insert_positions = '|'.join(['end' if s['at_end'] else str(s['pos']) for s in insert_specs])
            rec.insert_lengths = '|'.join([str(len(s['block'])) for s in insert_specs])
            expanded_atoms = apply_insert_scheme(tri_atoms, insert_specs, max(0, dp - 3))
            metrics = compare_atom_blocks(full_atoms, expanded_atoms)
            rec.atomtype_match_rate = metrics['atomtype_match_rate']
            rec.charge_mae = metrics['charge_mae']
            rec.charge_rmse = metrics['charge_rmse']
            rec.calibrated_charge_mae_sample = metrics['calibrated_charge_mae_sample']
            rec.calibrated_charge_rmse_sample = metrics['calibrated_charge_rmse_sample']
            rec.sample_fit_a = metrics['sample_fit_a']
            rec.sample_fit_b = metrics['sample_fit_b']
            full_q_all.append(metrics['full_q'])
            exp_q_all.append(metrics['exp_q'])
        except Exception as e:
            rec.status = 'failed'
            rec.error = f'{type(e).__name__}: {e}'
        records.append(rec.__dict__)

    df = pd.DataFrame(records)
    per_sample_csv = RESULTS_DIR / 'trimer_typing_compare_per_sample.csv'
    df.to_csv(per_sample_csv, index=False)

    calib_rows = []
    if full_q_all and exp_q_all:
        full_all = np.concatenate(full_q_all)
        exp_all = np.concatenate(exp_q_all)
        A = np.column_stack([exp_all, np.ones_like(exp_all)])
        a, b = np.linalg.lstsq(A, full_all, rcond=None)[0]
        pred = a * exp_all + b
        calib_rows.append({
            'segment': 'all',
            'n_atoms': int(full_all.size),
            'fit_a': float(a),
            'fit_b': float(b),
            'raw_mae': float(np.mean(np.abs(full_all - exp_all))),
            'calibrated_mae': float(np.mean(np.abs(full_all - pred))),
            'raw_rmse': float(np.sqrt(np.mean((full_all - exp_all) ** 2))),
            'calibrated_rmse': float(np.sqrt(np.mean((full_all - pred) ** 2))),
        })
    calib_df = pd.DataFrame(calib_rows)
    calib_csv = RESULTS_DIR / 'trimer_typing_charge_calibration.csv'
    calib_df.to_csv(calib_csv, index=False)

    summary = {
        'per_group': int(args.per_group),
        'n_candidates': int(len(candidates)),
        'n_success': int((df['status'] == 'ok').sum()) if not df.empty else 0,
        'n_failed': int((df['status'] != 'ok').sum()) if not df.empty else 0,
        'per_sample_csv': str(per_sample_csv),
        'calibration_csv': str(calib_csv),
    }
    (RESULTS_DIR / 'trimer_typing_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
