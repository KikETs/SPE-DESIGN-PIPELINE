# HTP-MD dataset curation and split audit

## Source and local table

The supervised conductivity dataset was downloaded from the public HTP-MD
data portal (`https://www.htpmd.matr.io/`). The repository-level public table
is `../polybert_con/fold_assignment.csv`, which preserves the 6,270 trajectory
rows and all 11 source columns before appending normalized PSMILES, canonical
PSMILES, the log-transformed target, and fold ID. The source download date and
a more specific portal object identifier are not encoded in the local CSV, so
they cannot be recovered from the repository alone. The associated HTP-MD
platform paper is available at `https://doi.org/10.1063/5.0160937`; the public
dataset landing page is `https://data.matr.io/htp/`.

## Validation and duplicate handling

No source row was removed or collapsed locally. All 6,270 trajectory IDs are
unique, and the required trajectory ID, structure, conductivity, diffusivity,
transference-number, density, molality, and degree-of-polymerization fields are
complete. All conductivity values are finite and strictly positive. RDKit
parsing, exact two-endpoint counting, and PSMILES canonicalization succeed for
all 6,270 rows.

The raw structure column contains 6,042 unique strings. Canonical endpoint-aware
PSMILES produces 6,026 unique structure groups: 133 groups contain duplicate
structures, comprising 377 rows, with a maximum multiplicity of five. These
rows are retained because they are distinct HTP-MD trajectory records. They are
grouped during cross-validation to prevent equivalent structures from appearing
in both training and held-out folds.

## Canonicalization and fold assignment

Endpoint markers are normalized to `[*]`, then canonicalized with the vendored
PSMILES implementation. The supervised target is
`log10(CONDUCTIVITY / (S cm-1))`. Fold assignment uses
`StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)`, with six
quantile bins of the log-conductivity target and canonical PSMILES as the group
key. The resulting fold sizes are 1,561, 1,559, 1,583, and 1,567 rows; the
numbers of canonical groups are 1,506, 1,507, 1,507, and 1,506. No canonical
group crosses folds.

The previous row-level split placed 119 canonical duplicate groups (343 rows)
across multiple folds. It has therefore been replaced by the canonical-group
split in the public fold assignment, OOF predictions, weighted-model
sensitivity outputs, and Fig. 7 inputs. The baseline OOF R2 changes from
0.466742 to 0.456832; this small decrease is consistent with removal of direct
canonical-structure overlap. The selected weighted model remains
`smooth_sigmoid_tail_a6_t0p05__ridge_alpha_100`.

## Public audit files

- `../polybert_con/fold_assignment.csv`: source fields, normalized and
  canonical PSMILES, target, and fold for all 6,270 trajectories.
- `../polybert_con/oof_predictions.csv`: one held-out prediction per trajectory.
- `../polybert_con/split_diagnostics.csv`: fold-level row, group, top-tail, and
  target-distribution diagnostics.
- `../polybert_con/canonical_group_split_metric_comparison.csv`: previous versus
  canonical-group OOF metrics.
- `htpmd_dataset_curation_summary.csv`: compact machine-readable curation audit.

The split prevents leakage of identical canonical endpoint-aware structures.
It does not assert that more general scaffold, substructure, or near-neighbor
similarity is absent across folds.
