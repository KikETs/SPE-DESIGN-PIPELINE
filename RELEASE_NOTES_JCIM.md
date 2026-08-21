# Proposed v1.0.0-jcim-submission release notes

Status: prepared locally; no tag or GitHub release has been created. Zenodo
draft `22043456` has reserved DOI `10.5281/zenodo.22043456`, which remains
inactive until the unsubmitted draft is published.

## Frozen scope

- Endpoint-aware SELFIES-PSMILES preprocessing and TransCVAE generation code.
- The 6,270-row labeled HTP-MD-derived table and final canonical-grouped
  four-fold split.
- Baseline and selected weighted OOF predictions, deployment provenance, and
  generated-candidate predictions.
- The 60-candidate selection and complete 180-replica generated static-cNE0
  numerical release.
- The full 120-system reference selection/status denominator and 108 completed
  static-cNE0 results.
- Representative generated/reference GROMACS inputs, MDPs, structures,
  topologies, Packmol inputs, fallback/charge-repair examples, and analysis code.
- Vendored PySoftK source provenance: upstream commit `a3458567...`,
  BSD-3-Clause, and the single local multicore construction patch.
- JCIM audit, protocol/force-field/software/seed provenance, manuscript data
  map, deterministic consolidated tables, checksums, and integrity tests.

## Scientific boundary

This release does not recalculate or alter surrogate metrics, rankings,
generation metrics, conductivity values, replica results, MALogE, bootstrap
statistics, or conclusions. Consolidated files are deterministic joins or
projections of tracked sources. The generated 60-row aggregation is checked
against the archived candidate-level result.

## External archive boundary

Large XTC, checkpoint, restart, energy, and complete raw run directories are
not placed in Git. The proposed DOI deposit contains the frozen repository and
the two representative trajectory bundles listed in
`reproducibility/zenodo/ZENODO_UPLOAD_MANIFEST.md`, subject to reference-data
redistribution approval. Reserved DOI: `10.5281/zenodo.22043456`; publication is
pending.

## Known gaps at release-preparation time

- Concrete initial-equilibration random seeds selected by GROMACS from
  `gen-seed = -1` were not recovered.
- Historical ACPYPE commands omit `-a`; GAFF2 is identified through the exact
  ACPYPE 2023.10.27 default, not an explicit archived `-a gaff2` argument.
- The manuscript summary of a 2 fs generated production step conflicts with the
  archived generated MDPs (1 fs). Saved coordinate spacing is also mixed at
  0.5/1 ps although analysis uses a uniform 1 ps grid.
- SI table-specific frozen sources remain partial for S2, S3a-S6b, S10a-S10b,
  and S11b.

## Validation

```bash
python scripts/build_jcim_reproducibility_package.py
python -m pytest tests/reproducibility -q
python scripts/repro_smoke.py
git diff --check
```
