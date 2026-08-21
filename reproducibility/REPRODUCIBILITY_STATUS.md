# Reproducibility status

Status date: 2026-08-21

| Area | Status | Evidence or limitation |
|---|---|---|
| 6,270 labels and structures | Available | `MY_PAPER_RELATED/polybert_con/fold_assignment.csv` |
| Final four-fold assignment | Available and validated | `ml/splits/canonical_grouped_split_6270.csv` |
| Baseline and weighted OOF predictions | Available | `MY_PAPER_RELATED/polybert_con/oof_predictions.csv`; `MY_PAPER_RELATED/polybert_weighted_evidence/tables/grouped_oof_predictions_selected.csv` |
| Generated pool predictions | Available | `MY_PAPER_RELATED/machine_readable/generated_surrogate_predictions_32611.csv` |
| Generated MD denominator | Available | 60 candidates, six archived groups, 180 replica rows |
| Reference MD denominator | Available | 120 selected, 108 completed, 12 failed retained |
| Staged MDP files | Available for representative generated/reference systems | Canonical compact MD release |
| Representative structures/topologies | Available | Generated, reference, fallback, and charge-repair examples |
| Production replica seeds | Partially available | Replica 1 is a continuation; replicas 2/3 use 730002/730003; initial equilibration used unrecoverable `gen-seed=-1` |
| Force-field route | Supported with provenance limitation | ACPYPE 2023.10.27 command omitted `-a`; versioned CLI default is GAFF2 |
| Historical software versions | Available with evidence grades | `SOFTWARE_VERSIONS.md`; exact PySoftK upstream commit and local patch recovered |
| Raw trajectories | Private upload in progress | Two representative trajectories are designated for Zenodo draft 22043456 |
| Immutable publication DOI | Reserved, not active | `10.5281/zenodo.22043456`; Zenodo publication action required |
| Full MD environment lockfile | Not recovered | Version evidence exists; historical Python 3.9.25 environment export does not |

Detailed gaps and source paths are in `../REPRODUCIBILITY_AUDIT.md` and
`../JCIM_REPRODUCIBILITY_REPORT.md`.
