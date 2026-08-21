# Machine-readable manuscript data

This directory indexes the tabular data used for the manuscript and adds the
selection/status tables needed to interpret the MD reassessments. All paths in
the tables are repository-relative. Raw trajectories, local filesystem paths,
and long execution logs are not included.

## Coverage

| Requirement | Public file | Coverage |
|---|---|---:|
| Training labels, structures, and canonical-group four-fold assignments | `../polybert_con/fold_assignment.csv` | 6,270 trajectories / 6,026 canonical groups |
| Out-of-fold surrogate predictions | `../polybert_con/oof_predictions.csv` | 6,270 trajectories |
| Generated structures and surrogate predictions | `generated_surrogate_predictions_32611.csv` | 32,611 structures |
| Generated MD selections | `../gromacs_eval_pred_conductivity/github_results/latest_notebook_manifest_60/sample_manifest.csv` | 60 candidates |
| Generated candidate-level MD results | `../gromacs_eval_pred_conductivity/github_results/latest_notebook_manifest_60/run_results.csv` | 60 candidates |
| Generated replica-level static cNE0 results | `../gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/data/generated/generated_static_cNE0_replica_180.csv` | 180 replicas |
| Reference reassessment selections | `reference_selection_manifest_120.csv` | 120 trajectories |
| Reference reassessment execution status | `reference_run_status_120.csv` | 120 trajectories |
| Completed reference static cNE0 results | `../gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/data/reference/reference_static_cNE0_reassessment_108.csv` | 108 trajectories |
| Manuscript figure source map | `figure_source_manifest.csv` | 12 figure-source relations |

`generated_surrogate_predictions_32611.csv` preserves the complete generated
pool order. Surrogate predictions are available for 32,610 structures. The one
remaining structure, `[*][*]`, is explicitly marked `excluded` because the
existing screening code classifies it as an endpoint artifact; no value was
imputed.

The reference selection and status files distinguish the 120 preselected
trajectories from the 108 completed reassessments. The 12 failed trajectory IDs
remain in `reference_run_status_120.csv` with compact stage/type fields.
Blank analysis fields mean that the corresponding stage was not reached or did
not export a structured status value; they are not imputed values.

The PolyBERT split uses `StratifiedGroupKFold` with four folds, six
log-conductivity quantile bins, shuffle enabled, and random seed 42. The group
key is the canonicalized endpoint-aware PSMILES string. Fold sizes are 1,561,
1,559, 1,583, and 1,567 trajectories, and no canonical structure group occurs
in more than one fold. See `HTPMD_DATASET_CURATION.md` and
`htpmd_dataset_curation_summary.csv` for the complete curation and leakage
audit.

## Rebuild

The generated tables can be rebuilt from a checkout plus the reference
reassessment output directory:

```bash
python scripts/build_machine_readable_release.py \
  --reference-root /path/to/eval_top10_bottom10_stratified100
```

The builder checks row counts, primary-key uniqueness, required values,
canonical-group non-overlap, the 60-candidate/180-replica relationship, and the
120-selected/108-completed reference relationship. Results are recorded in
`data_quality_report.csv`.

`data_inventory.csv` is the compact data-availability index.
`figure_source_manifest.csv` maps each numerical manuscript figure to its
source table and plotting entry point. Fig. 1B is a schematic and therefore has
no numerical source table.
