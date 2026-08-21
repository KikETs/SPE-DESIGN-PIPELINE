# Supplementary Table S14b. Generated-candidate MSD fit diagnostics

Data source: `github_results/latest_notebook_manifest_60/run_results.csv` and the 180 replica-level `analysis_replicas_csv` paths listed there.

| Diagnostic | Value |
|---|---:|
| Fit window | 0-50 ns |
| MSD restart interval | 1 ps |
| Generated candidates | 60 |
| Production replicas | 180 |
| Li mean-fit R2 range | 0.958456-0.998991 |
| Li minimum replica-fit R2 | 0.942648 |
| TFSI mean-fit R2 range | 0.948681-0.998952 |
| TFSI minimum replica-fit R2 | 0.896464 |

## Fit Counts

- Li replica fits used: 180
- TFSI/TFSI_N replica fits used: 180
- Missing or failed fits: 0
- Missing trajectory/replica IDs: none

## Source Files

- Replica-level R2 table with repository-relative `analysis_summary_csv` and `msd_xvg` paths: `msd_fit_r2_per_replica.csv`
- Candidate-level mean R2 table: `msd_fit_r2_candidate_mean.csv`
- Summary CSV: `msd_fit_r2_s14b_summary.csv`
- Missing/failed file audit: `msd_fit_r2_missing_or_failed.csv`

## Method Match Check

- The R2 calculation uses the same local fit window used by the static cNE0 analysis code: `analysis_begin_ns <= t_ns <= analysis_end_ns`, followed by a first-order polynomial fit of MSD versus time.
- The 180 replica summary CSVs all report `analysis_begin_ns = 0.0` and `analysis_end_ns = 50.0`; the run notebook/config also records `analysis_window_ns=0~50`.
- I did not find a separate manuscript Methods source file containing S14b text in the repo. Therefore this matches the repository/run-config Methods value; if the external manuscript Methods says a different window, that manuscript text needs to be updated or checked separately.

## QA

- All 360 MSD files exist: 180 `msd_li.xvg` and 180 `msd_tfsi_N.xvg`.
- All parsed `gmx msd` command lines used `-trestart 1`.
- Max absolute diffusivity reproduction delta from the same fits: Li/TFSI_N combined `2.912e-21` cm^2/s; max relative delta `2.391e-14`.
