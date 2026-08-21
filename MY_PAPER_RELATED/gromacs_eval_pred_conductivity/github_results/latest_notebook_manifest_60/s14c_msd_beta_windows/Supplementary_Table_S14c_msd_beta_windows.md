# Supplementary Table S14c. Generated-candidate MSD beta-window diagnostics

This table was recomputed from the expanded 60-candidate / 180-production-replica dataset using the raw replica-level MSD `.xvg` files referenced by `github_results/latest_notebook_manifest_60/run_results.csv`.

| Dataset | Species | Series | With beta window | Median beta | Median duration / ns |
|---|---|---:|---:|---:|---:|
| Generated | Li | 180 | 117 | 0.982128 | 4.080718 |
| Generated | TFSI_N | 180 | 140 | 0.972469 | 7.926556 |

## Beta-Window Settings

- Local beta method: rolling linear slope of `log(MSD)` vs `log(t)`, using `scripts/analysis/msd_beta_windows.py`.
- Log-spaced grid points: `500`
- Rolling fit window points: `31`
- Minimum lag time: `10 ps`
- Maximum lag time: `none`
- Accepted beta threshold: `0.9 <= beta <= 1.1`
- Minimum accepted beta-window duration: `500 ps`
- Per-series value: longest accepted beta window, sorted by duration then mean local-fit R2, matching the existing script.
- Table median duration follows the existing script summary convention: no-window series contribute `0 ns` duration; median beta ignores no-window `NaN` values.

## Dataset Checks

- Generated candidates: 60
- Production replicas: 180
- Species labels in output: `Li`, `TFSI_N`
- Li series expected/observed: 180/180
- TFSI_N series expected/observed: 180/180
- Missing or failed series: 0
- Existing github_results/msd_beta/msd_beta_summary.csv pred_replica n_series=[60]; not used for S14c expanded table.

## Additional Accepted-Only Medians

| Dataset | Species | Median beta among accepted windows | Median duration among accepted windows / ns |
|---|---|---:|---:|
| Generated | Li | 0.982128 | 8.493316 |
| Generated | TFSI_N | 0.972469 | 10.100737 |

## Source Files

- Longest-window per series, including repository-relative source paths: `msd_beta_windows.csv`
- All accepted beta windows: `msd_beta_all_windows.csv`
- Summary CSV: `msd_beta_summary_s14c.csv`
- Missing/failed audit: `msd_beta_missing_or_failed.csv`

The 58 MB pointwise local-beta curve dump is not included because the committed window-level tables contain the diagnostics used for S14c.

## Missing Or Failed Series

None.
