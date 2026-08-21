# Supplementary Table S12a. Generated-system construction summary

| Item | Summary |
|---|---:|
| Generated candidates | 60 |
| Production replicas | 3 per candidate |
| Polymer chains | 16-20 per system |
| Ion count | 100 Li and 100 TFSI per system |
| Target density | 1.300 g cm^-3 |
| Production box length | 4.4600-4.5035 nm |
| Topology route | 47 direct ACPYPE; 13 bonded fallback |
| Charge neutrality | 54 neutral as built; 6 neutralized after polymer charge repair |

## Updated Evidence

- Topology route counted from `atomtyping_attempt*.log` fallback messages and topology fallback files across the 60 generated candidates: `47 direct ACPYPE; 13 bonded fallback`.
- Charge neutrality counted from `charge_sanity_interphase.json` `status`/`reason` across the 60 generated candidates: `54 neutral as built; 6 neutralized after polymer charge repair`.
- Per-candidate audit CSV: `s12a_topology_charge_per_candidate.csv`
- Summary CSV: `s12a_generated_system_construction_summary.csv`

## Counts By Group

| sample_group           | topology_route   | charge_neutrality_class                 |   n |
|:-----------------------|:-----------------|:----------------------------------------|----:|
| HIGH_bottom            | bonded fallback  | neutral as built                        |   3 |
| HIGH_bottom            | direct ACPYPE    | neutral as built                        |   7 |
| HIGH_middle_stratified | bonded fallback  | neutral as built                        |   2 |
| HIGH_middle_stratified | direct ACPYPE    | neutral as built                        |   6 |
| HIGH_middle_stratified | direct ACPYPE    | neutralized after polymer charge repair |   2 |
| HIGH_top               | direct ACPYPE    | neutral as built                        |   9 |
| HIGH_top               | direct ACPYPE    | neutralized after polymer charge repair |   1 |
| LOW_bottom             | bonded fallback  | neutral as built                        |   1 |
| LOW_bottom             | direct ACPYPE    | neutral as built                        |   8 |
| LOW_bottom             | direct ACPYPE    | neutralized after polymer charge repair |   1 |
| LOW_middle_stratified  | bonded fallback  | neutral as built                        |   4 |
| LOW_middle_stratified  | direct ACPYPE    | neutral as built                        |   6 |
| LOW_top                | bonded fallback  | neutral as built                        |   3 |
| LOW_top                | direct ACPYPE    | neutral as built                        |   5 |
| LOW_top                | direct ACPYPE    | neutralized after polymer charge repair |   2 |

## Software Versions

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

Notes:

- GROMACS version was confirmed as `2025.3` in all 60 generated-candidate production logs referenced by `run_results.csv`.
- Packmol version was confirmed as `21.1.2` in the generated-candidate Packmol seed logs.
- ACPYPE-generated topology headers report `acpype (v: 2023.10.27)`.
- AmberTools was taken from the `MD` conda environment package metadata (`ambertools 23.3`); the `antechamber` executable banner reports `antechamber 22.0`.
- PySoftK was executed through the repository-local vendored copy at `pysoftk/`. That copy has no embedded version metadata; the installed package metadata in the same `MD` environment is `pysoftk 1.0.0`, and the local copy differs by a local `linear_polymer.py` patch.
