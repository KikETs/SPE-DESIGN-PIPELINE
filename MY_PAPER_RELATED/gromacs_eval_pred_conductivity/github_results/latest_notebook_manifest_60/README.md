# Latest Notebook Manifest GROMACS Results

This folder contains summary outputs for the 60 trajectories selected and run by
`gromacs_new_batch_eval_top_bottom_stratified.ipynb`.

Raw trajectory and simulation artifacts are intentionally excluded. Do not add
`runs/`, `*.xtc`, `*.trr`, `*.edr`, `*.tpr`, `*.cpt`, `*.gro`, or `*.pdb` files
to this folder.

## Files

- `sample_manifest.csv`: the 60 trajectory IDs selected by the notebook.
- `run_results.csv`: notebook batch-run status and merged analysis outputs.
- `per_traj_eval.csv`: per-trajectory evaluation table from the notebook summary.
- `metrics_summary.csv`: notebook metric summary.
- `completed_production_logs.tsv`: production log index for the same 60
  trajectory IDs, including log modification time and GROMACS performance line.

## Verification

- `sample_manifest.csv`, `run_results.csv`, and `completed_production_logs.tsv`
  contain the same 60 unique trajectory IDs.
- All 60 `run_results.csv` rows have `status=ok`, `md_status=ok`, and
  `analysis_status=ok`.
