# JCIM reproducibility compliance report

Assessment date: 2026-08-21

Targets: JCIM method/data/software reproducibility policy
(`https://doi.org/10.1021/acs.jcim.0c01389`) and MD reporting guideline
(`https://doi.org/10.1021/acs.jcim.3c00599`). This is a repository-preparation
report. The source snapshot and two representative trajectory bundles were
published at Zenodo DOI `10.5281/zenodo.22043456` on 2026-08-21.

## A. Fully addressed

- A machine-readable 6,270-row label/split table is available, with 6,026
  canonical groups, four folds, six stratification bins, seed 42 provenance,
  and zero cross-fold canonical-group overlap.
- Baseline and selected weighted grouped-OOF predictions, threshold/top-k/tail
  evaluation, generated deployment predictions, and model provenance are
  available.
- Generated MD data preserve all 60 candidates, six archived groups, and three
  production replicas per candidate (180 replica rows). A deterministic 60-row
  candidate aggregation is provided and checked against the archived source.
- Reference data preserve all 120 selected systems, including 108 completed and
  12 failed systems with structured status fields.
- Real generated/reference MDP stage files, Packmol inputs, representative
  PDB/GRO/TPR/index files, TOP/ITP files, Li/TFSI topologies, charge scaling,
  direct/fallback topology logic, and charge-repair diagnostics are tracked.
- Actual MDP parameters are extracted to a 1,622-row machine-readable table.
- The manuscript static-cNE0 route is separated from RDF-peak/zero-persistence
  sensitivity code. RDF-first-minimum, at-least-two-oxygen filtering, static
  graph, formal charge, cNE0/NE, MSD, beta, aggregation, bootstrap, and
  reference statistics entry points are documented.
- Production replica randomness is disclosed for all 180 generated records:
  replica 1 continuation semantics and seeds 730002/730003 for replicas 2/3.
- Historical software versions have repository-evidence and verification grades.
- The vendored PySoftK tree is traced to upstream commit `a3458567...`; 43/44
  files are byte-identical, the single local patch is documented, and the
  BSD-3-Clause license is retained.
- A manuscript/SI source map, checksum manifest, release metadata, proposed
  release notes, Zenodo manifest, and ten read-only integrity tests are present.

## B. Partially addressed

- GAFF2 is supported by the archived omitted-argument ACPYPE command, exact
  ACPYPE 2023.10.27 output version, and that version's documented CLI default.
  The historical command does not explicitly contain `-a gaff2`.
- AmberTools package metadata reports 23.3, while the Antechamber executable
  banner reports 22.0; both are disclosed.
- Representative full generated/reference bundles, a fallback topology, and a
  charge-repair example are included. Full bundles for all six generated groups
  are not included.
- Major figures, Tables 1-4, and most SI MD tables have direct numerical source
  mappings. Final table-specific exports remain partial for S2, S3a-S6b,
  S10a-S10b, and S11b.
- Historical software versions are evidenced, but the exact Python 3.9.25 MD
  environment is not available as a complete lockfile.

## C. Not addressed

- The concrete RNG value selected during initial equilibration from
  `gen-seed = -1` was not recovered.
- The complete production trajectory and restart/checkpoint set is not publicly
  archived. The Zenodo record contains two representative 50 ns trajectories.
- A specific upstream parameter-file citation for the local Li fallback values
  was not archived.
- New experimental validation and additional reference replicas were not
  created. They are new scientific work, not repository cleanup.

## D. Requires external action

1. Confirm complete author metadata; only the existing repository author name
   `KikET` was used in preliminary release metadata.
2. Confirm redistribution rights for reference `Traj_13430` and topology.
3. Review and authorize tag `v1.0.0-jcim-submission`.
4. Create the GitHub tag/release if required. The Zenodo source snapshot and
   representative trajectory bundle are published at
   `10.5281/zenodo.22043456`.
5. Insert the immutable Zenodo citation/DOI into the manuscript Data and
   Software Availability section.
6. Decide whether to freeze additional table-specific exports for the partial
   SI mappings and whether to archive more raw trajectories externally.

## E. Recommended manuscript changes

- Replace moving-branch references with the immutable Zenodo record and DOI
  `10.5281/zenodo.22043456`.
- State the topology route as GAFF2 via ACPYPE 2023.10.27's default, with
  AmberTools 23.3 package/Antechamber 22.0 banner provenance, unless a historical
  explicit command provides stronger evidence.
- Resolve the generated production timestep conflict: archived generated MDPs
  use 1 fs/50,000,000 steps; the reference representative uses
  2 fs/25,000,000 steps.
- Distinguish saved trajectory spacing (120 generated replicas at 0.5 ps, 60 at
  1 ps) from the uniform 1 ps analysis grid.
- Explicitly map archived `HIGH_*`/`LOW_*` group labels to TT/LT terminology.
- Disclose that replicas 2/3 use seeds 730002/730003, replica 1 continues from
  equilibration, and the initial automatic equilibration seed was not recovered.
- Cite the HTP-MD portal/data URL and platform DOI, while noting that the exact
  downloaded object accession/date was not retained locally.

## Validation result

The following commands completed successfully on 2026-08-21:

```bash
python scripts/build_jcim_reproducibility_package.py
python -m pytest tests/reproducibility -q
python scripts/repro_smoke.py
git diff --check
```

Result: `10 passed`; repository structure, tracked raw-MD exclusion,
tracked-file size limit, and notebook JSON checks passed; `git diff --check`
passed. Both Zenodo metadata files also validate against Zenodo's official
legacy deposit JSON schema.
