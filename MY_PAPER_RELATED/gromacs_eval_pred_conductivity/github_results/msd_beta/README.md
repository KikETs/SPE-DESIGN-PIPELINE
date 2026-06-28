# MSD log-log beta analysis

Local anomalous-diffusion exponent beta was estimated by rolling linear fits of log(MSD) vs log(t).

Settings:
- log-spaced grid points: 500
- rolling fit window points: 31
- minimum lag time: 10 ps
- beta≈1 threshold: 0.9 <= beta <= 1.1
- minimum accepted beta≈1 window length: 500 ps

Outputs:
- `msd_beta_windows.csv`: longest beta≈1 interval for each trajectory/replica/species.
- `msd_beta_all_windows.csv`: every accepted beta≈1 interval for each trajectory/replica/species.
- `msd_beta_curves.csv`: local beta curve for every analyzed series.
- `msd_beta_summary.csv`: dataset/species-level summary.
- `figures/`: beta curves and longest-window bar plots.

Summary:

| dataset      | species   |   n_series |   n_with_beta_window |   median_window_duration_ns |   mean_window_duration_ns |   median_beta_mean |   mean_beta_mean |   median_window_start_ns |   median_window_end_ns |
|:-------------|:----------|-----------:|---------------------:|----------------------------:|--------------------------:|-------------------:|-----------------:|-------------------------:|-----------------------:|
| pred_replica | Li        |         60 |                   44 |                     4.17008 |                   9.29163 |           0.988528 |         0.977632 |                  32.0803 |                42.88   |
| pred_replica | TFSI_N    |         60 |                   49 |                    12.2555  |                  14.7459  |           0.970168 |         0.969074 |                  17.3532 |                38.051  |
| ref_traj     | Li        |        108 |                   72 |                     3.73583 |                   5.90122 |           0.9824   |         0.974882 |                  32.9134 |                42.1543 |
| ref_traj     | TFSI_N    |        108 |                   74 |                     5.20423 |                   7.85648 |           0.978468 |         0.974174 |                  34.352  |                46.7002 |

Input counts:
- reference trajectories: 108 per species
- predicted production replicas: 60 per species
- predicted single-run `Traj_*/analysis/msd_*.xvg` files are intentionally excluded.
