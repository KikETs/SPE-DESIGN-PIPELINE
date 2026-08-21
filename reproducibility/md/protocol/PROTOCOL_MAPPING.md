# GROMACS protocol mapping

The source prefix below is
`MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/examples/`.
Generated files are under `generated_Traj_912118_rep3/mdp/`; reference files are
under `reference_Traj_13430/mdp/`. Files are preserved unchanged at those paths.

The workflow runs each stage with the archived `grompp()`/`mdrun()` wrappers in
`.../static_cne0_release/workflow/phase_scripts/gromacs_new_phase_md.py`:
`gmx grompp -f <mdp> -c <previous.gro> -p topol.top -o <stage>.tpr` with the
recorded restraint/checkpoint options, then `gmx mdrun -deffnm <stage>` with
the archived execution options.

| Manuscript/SI stage | Actual file(s), for generated and reference examples | Key command | Status |
|---|---|---|---|
| Energy minimization | `mdp/em.mdp` | workflow `grompp`; `mdrun -deffnm em` | FOUND |
| Initial heating | `mdp/nvt0_pre.mdp`, `mdp/nvt0.mdp` | sequential `run_stage` | FOUND |
| Initial relaxation | `mdp/nvt1.mdp` | sequential `run_stage` | FOUND |
| Compression | `mdp/npt01_p50.mdp` through `mdp/npt07_p4k_hold.mdp` | sequential `run_stage` | FOUND |
| Decompression | `mdp/npt08_p2k_down.mdp` through `mdp/npt11_p1_down.mdp` | sequential `run_stage` | FOUND |
| Thermal cycle | `mdp/npt12_heat.mdp`, `mdp/npt13_cool.mdp` | sequential `run_stage` | FOUND |
| Recompression | `mdp/npt14_p200_up.mdp` through `mdp/npt17_p4k_up.mdp` | sequential `run_stage` | FOUND |
| Final decompression/density relaxation | `mdp/npt18_p1_down.mdp`, `mdp/npt19_p1.mdp` | sequential `run_stage` | FOUND |
| Final NVT relaxation | `mdp/nvt2_pre.mdp`, `mdp/nvt2.mdp` | sequential `run_stage` | FOUND |
| Generated production NVT | `mdp/production.mdp`, `production_rep2.mdp`, `production_rep3.mdp` | `run_stage`; replicas 2/3 generate velocities with archived seeds | FOUND, 1 fs archived input |
| Reference production NVT | `mdp/production.mdp` | `run_stage` | FOUND, 2 fs archived input |

Every parsed key/value pair is in `gromacs_parameters.csv`. The table is built
from the real MDP files by `scripts/build_jcim_reproducibility_package.py`.

