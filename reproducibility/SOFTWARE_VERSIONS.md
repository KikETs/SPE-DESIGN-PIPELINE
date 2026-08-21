# Historical software versions

These are versions used for the archived generated-candidate GROMACS runs, not
versions installed on the current machine. Evidence was checked on 2026-08-21.

| software | reported_version | repository_evidence | verification_status | notes |
|---|---:|---|---|---|
| GROMACS | 2025.3 | `MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/environment/software_versions.csv` | VERIFIED_FROM_60_LOGS | All 60 generated production logs referenced by `run_results.csv`; executable `/usr/local/gromacs/bin/gmx` |
| AmberTools | 23.3 | same version table and archived environment metadata | PARTIAL | Package metadata says 23.3; Antechamber banner says 22.0, so both are retained |
| ACPYPE | 2023.10.27 | topology headers in representative `tfsi_GMX.itp` and polymer ITPs | VERIFIED_FROM_OUTPUT_HEADERS | Historical commands are in `ACPYPE_INVOCATIONS.md` |
| Packmol | 21.1.2 | generated Packmol seed logs summarized in the static release version table | VERIFIED_FROM_LOGS | Executable recorded as `packmol` |
| PySoftK | 1.0.0 reported environment; upstream `a3458567...` plus local patch | `MY_PAPER_RELATED/gromacs_eval_pred_conductivity/pysoftk/` and `md/inputs/PYSOFTK_PROVENANCE.md` | VERIFIED_SOURCE_WITH_LOCAL_PATCH | 43/44 source files match upstream commit; only `linear_polymer.py` differs; BSD-3-Clause license retained |
| RDKit | 2023.09.6 | archived MD-environment metadata | VERIFIED_FROM_ENVIRONMENT_METADATA | Historical environment, not current shell |
| Python | 3.9.25 | archived notebook interpreter/environment evidence | VERIFIED_FROM_ENVIRONMENT_METADATA | Root `environment.yml` is a separate current reproduction environment |
| PyTorch | 2.8.0+cu128 | archived MD-environment metadata | VERIFIED_FROM_ENVIRONMENT_METADATA | Used environment version, not a portability requirement for GROMACS analysis |

Canonical evidence is preserved in
`MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/environment/software_versions.md`
and `.csv`.
