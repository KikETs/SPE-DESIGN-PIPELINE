# Software versions used for the generated-candidate GROMACS reassessment

| Program | Version used |
|---|---:|
| GROMACS | 2025.3 |
| AmberTools | 23.3 |
| ACPYPE | 2023.10.27 |
| Packmol | 21.1.2 |
| PySoftK | 1.0.0-derived local vendored copy |
| RDKit | 2023.09.6 |
| Python | 3.9.25 |
| PyTorch | 2.8.0+cu128 |

## Provenance

- Dataset scope: `github_results/latest_notebook_manifest_60/run_results.csv`, 60 generated candidates.
- Python executable used by the notebook: `python`.
- GROMACS executable used by production logs: `/usr/local/gromacs/bin/gmx`.
- Packmol executable used by phase logs: `packmol`.
- ACPYPE executable used by phase logs: `acpype`.
- GROMACS version was confirmed as `2025.3` in all 60 generated-candidate production logs referenced by `run_results.csv`.
- Packmol version was confirmed as `21.1.2` in the generated-candidate Packmol seed logs.
- ACPYPE-generated topology headers report `acpype (v: 2023.10.27)`.
- AmberTools was taken from the `MD` conda environment package metadata (`ambertools 23.3`); the `antechamber` executable banner reports `antechamber 22.0`.
- PySoftK was executed through the repository-local vendored copy at `pysoftk/`. The vendored copy has no embedded version metadata; the installed package metadata in the same `MD` environment is `pysoftk 1.0.0`, and the local copy differs by a local `linear_polymer.py` patch.
