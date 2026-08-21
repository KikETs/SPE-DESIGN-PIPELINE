# Manuscript and SI numerical source map

Checked against `manuscript.docx` and `supplementary_information.docx` supplied
for audit, 2026-08-21. Paths are repository-relative. `DIRECT` means a tracked
machine-readable source and plotting/analysis entry point were identified;
`PARTIAL` means the underlying code/data exist but a final table-specific
export was not identified.

## Main figures and tables

| Item | Numerical source | Generation/plot code | Status |
|---|---|---|---|
| Figures 1A and 2 | `MY_PAPER_RELATED/polybert_con/fold_assignment.csv` | `plot.ipynb` / exported plotting workflow | DIRECT |
| Figures 1B, 1C, 3-5 | schematic/architecture/workflow panels | manuscript figure assets and notebooks | NOT_NUMERICAL |
| Figure 6, Table 1 | `MY_PAPER_RELATED/MODELS/FCD_runs/_conductivity_eval/all_models_scorecard.csv`; `conductivity_eval_summary.csv`; `figures/generated_pool_enrichment/cache/proposed_transcvae_polybert_predictions.csv` | `plot.ipynb` | DIRECT |
| Figure 7, Table 2 | `MY_PAPER_RELATED/polybert_con/oof_predictions.csv`; `weighted_model_selection_canonical_group.csv`; `MY_PAPER_RELATED/polybert_weighted_evidence/tables/grouped_oof_predictions_selected.csv` | `scripts/figures/make_fig7_surrogate_reliability.py` | DIRECT |
| Figure 8, Table 3 | `MY_PAPER_RELATED/machine_readable/figure_data/fig8_reference_stratified_static_cne0.csv`; `reference_run_status_120.csv`; static release reference group/overall CSVs | `scripts/figures/make_fig8_reference_stratified_static_cne0.py`; `static_cne0_release/analysis/manuscript_static_cne0.py` | DIRECT |
| Figure 9, Table 4 | static release `data/generated/generated_md_results_60.csv`; `generated_static_cNE0_replica_180.csv`; `generated_group_summary_candidate_level.csv` | `scripts/figures/make_fig9_generated_static_cne0.py`; `static_cne0_release/analysis/manuscript_static_cne0.py` | DIRECT |

The pre-existing machine-readable panel-level map is
`MY_PAPER_RELATED/machine_readable/figure_source_manifest.csv`.

## Supplementary tables

| SI item | Source/provenance | Status |
|---|---|---|
| S1 dataset/split | `MY_PAPER_RELATED/polybert_con/fold_assignment.csv`, `split_diagnostics.csv`, and `machine_readable/HTPMD_DATASET_CURATION.md` | DIRECT |
| S2 target definitions | model/notebook configuration under `MY_PAPER_RELATED/MODELS/` | PARTIAL - no final table-specific CSV identified |
| S3a-S4 endpoint filtering/reconstruction/examples | `MY_PAPER_RELATED/selfies-psmiles/` and model preprocessing outputs | PARTIAL - no final table-specific provenance export identified |
| S5a-S6b model/settings/objectives | TransCVAE/baseline model source and training notebooks under `MY_PAPER_RELATED/MODELS/` | PARTIAL - settings exist in code/notebooks, no frozen table-specific configuration file |
| S7a validity/novelty | `MY_PAPER_RELATED/MODELS/FCD_runs/_conductivity_eval/all_models_scorecard.csv` and `paper_table_all_conditions_raw.csv` | DIRECT |
| S7b enrichment | `MY_PAPER_RELATED/MODELS/FCD_runs/_conductivity_eval/conductivity_eval_summary.csv` | DIRECT |
| S8a regression | `MY_PAPER_RELATED/polybert_con/oof_predictions.csv` and grouped metric tables | DIRECT |
| S8b threshold/top-rank | `MY_PAPER_RELATED/polybert_weighted_evidence/tables/` model-selection and threshold/top-k tables | DIRECT |
| S9 baseline/weighted generated scores | `MY_PAPER_RELATED/polybert_weighted_evidence/tables/supplementary_table_s9_generated_weighted_predictions.md`; `weighted_generated_condition_summary.csv`; `weighted_generated_md_selection_60.csv` | DIRECT |
| S10a-S10b SMARTS motifs | motif analysis data/code within model analysis outputs | PARTIAL - final motif-source CSV not identified by repository audit |
| S11a selected identities/rank/DP | generated selection `sample_manifest.csv` and weighted `weighted_generated_md_selection_60.csv` | DIRECT |
| S11b selected motifs | selected candidate table plus motif analysis output | PARTIAL - final table-specific motif export not identified |
| S12 settings | static release README, charge model, force-field provenance, and representative MDP/topology inputs | DIRECT |
| S12a construction summary | `github_results/latest_notebook_manifest_60/s12a_construction_summary/s12a_generated_system_construction_summary.csv` and `s12a_topology_charge_per_candidate.csv` | DIRECT |
| S12b charge scaling | static release `inputs/charge_scaling_parameters.csv` and generated replica/candidate CSVs | DIRECT |
| S13 equilibration/production | representative MDPs and `reproducibility/md/protocol/gromacs_parameters.csv` | DIRECT |
| S14/S14a static cNE0 settings/logic | static release `analysis/manuscript_static_cne0.py`, README, and generated replica table protocol fields | DIRECT |
| S14b MSD fit R2 | `github_results/latest_notebook_manifest_60/s14b_msd_fit_r2/` CSVs; `analysis/msd_fit_r2_expanded.py` | DIRECT |
| S14c beta windows | `github_results/latest_notebook_manifest_60/s14c_msd_beta_windows/` CSVs; `analysis/msd_beta_windows.py` | DIRECT |
| S15 group summary | static release `data/generated/generated_group_summary_candidate_level.csv` | DIRECT |
| S15a bootstrap intervals | same group summary; bootstrap seed/resamples recorded by `analysis/manuscript_static_cne0.py` | DIRECT |
| S15b pairwise comparisons | static release `data/generated/generated_pairwise_group_tests.csv` | DIRECT |
| S16 top five generated | static release `data/generated/generated_static_cNE0_candidate_mean_60.csv`, deterministically sorted by candidate-mean static cNE0 | DIRECT |
| S17 reference group summary | static release `data/reference/reference_group_summary_trajectory_level.csv` | DIRECT |
| S18 reference cross-protocol | static release `data/reference/reference_overall_statistics.csv` | DIRECT |
| S19 reference labels vs reassessment | static release `data/reference/reference_static_cNE0_reassessment_108.csv` plus the 120-row selection/status files | DIRECT |
| S20 aggregation/statistics definitions | `analysis/manuscript_static_cne0.py` and `MANUSCRIPT_STATIC_CNE0_UPDATE.md` | DIRECT |

## Required follow-up

Before publication, freeze table-specific machine-readable exports for S2,
S3a-S6b, S10a-S10b, and S11b or document the exact notebook cells that create
them. This package does not infer those final values from prose.

