# Predicted-Conductivity GROMACS Evaluation Results

This folder contains path-portable summary outputs for the 20 PolyBERT-selected candidates evaluated with GROMACS.
Raw `runs/`, `results/`, and trajectory files are intentionally excluded from Git.

## Run Scope

- Total candidates: 20
- Successful analyses: 20 / 20
- Sampling: HIGH top 5, HIGH bottom 5, LOW top 5, LOW bottom 5 by PolyBERT-predicted conductivity
- Production replicas per trajectory: 3
- Sigma mode: cNE_persist_ge_20ps
- Reported variance columns are sample variances across the 3 production replicas.

## Overall Metrics

- Overall MA log10 error for conductivity: 0.485983
- Overall median absolute log10 error for conductivity: 0.409209
- Overall MAE for NE transference number: 0.467465

## Files

- `per_trajectory_replica_summary.csv`: main per-trajectory PolyBERT-vs-MD table with mean/std/variance across replicas.
- `group_summary.csv`: group-level descriptive statistics.
- `metrics_summary.csv`: notebook-produced metric summary by sample group.
- `sample_manifest_summary.csv`: selected candidate manifest with SMILES, DP, molality, density, and PolyBERT ranking.
- `overall_summary.csv`: one-row global summary.
- `top5_by_md_sigma.csv`: highest MD cNE conductivity candidates.

## Group Summary

| sample_group   |   n | all_ok   |   replica_success_min |   polybert_pred_cond_mean_S_cm |   polybert_pred_cond_min_S_cm |   polybert_pred_cond_max_S_cm |   md_sigma_cNE_mean_S_cm |   md_sigma_cNE_median_S_cm |   md_sigma_cNE_min_S_cm |   md_sigma_cNE_max_S_cm |   md_sigma_cNE_replica_std_mean_S_cm |   mean_abs_log10_error_sigma |   median_abs_log10_error_sigma |   md_tplus_NE_mean |   md_c_tn_mean |
|:---------------|----:|:---------|----------------------:|-------------------------------:|------------------------------:|------------------------------:|-------------------------:|---------------------------:|------------------------:|------------------------:|-------------------------------------:|-----------------------------:|-------------------------------:|-------------------:|---------------:|
| HIGH_bottom    |   5 | True     |                     3 |                    0.000197642 |                   0.000166187 |                   0.000218711 |              0.000327507 |                0.000235868 |             0.000196917 |             0.000598332 |                          3.87635e-05 |                     0.182288 |                       0.131307 |           0.437286 |       0.393368 |
| HIGH_top       |   5 | True     |                     3 |                    0.00178004  |                   0.00173705  |                   0.00184985  |              0.00133293  |                0.00143864  |             0.000759557 |             0.00201666  |                          0.000109151 |                     0.173702 |                       0.084928 |           0.258175 |       0.251799 |
| LOW_bottom     |   5 | True     |                     3 |                    1.00075e-05 |                   9.41923e-06 |                   1.04332e-05 |              1.55046e-06 |                9.87304e-07 |             5.04948e-07 |             3.82727e-06 |                          3.32043e-07 |                     0.91624  |                       0.998455 |           0.692792 |      -1.02304  |
| LOW_top        |   5 | True     |                     3 |                    0.000367382 |                   0.000268305 |                   0.000538632 |              0.00105183  |                0.000647918 |             8.00723e-05 |             0.00316314  |                          8.31784e-05 |                     0.6717   |                       0.606697 |           0.481607 |       0.284177 |

## Top 5 By MD cNE Conductivity

|   Trajectory ID | sample_group   |   polybert_pred_cond_S_cm |   md_sigma_cNE_mean_S_cm |   md_sigma_cNE_std_S_cm |   md_tplus_NE_mean |
|----------------:|:---------------|--------------------------:|-------------------------:|------------------------:|-------------------:|
|          920003 | LOW_top        |               0.00027978  |               0.00316314 |             0.000181813 |           0.641609 |
|          910000 | HIGH_top       |               0.00184985  |               0.00201666 |             0.000173086 |           0.265075 |
|          910001 | HIGH_top       |               0.00182476  |               0.00167    |             6.04339e-05 |           0.215318 |
|          910002 | HIGH_top       |               0.00174936  |               0.00143864 |             0.000227028 |           0.261586 |
|          920002 | LOW_top        |               0.000302348 |               0.00122237 |             6.60438e-05 |           0.33572  |

