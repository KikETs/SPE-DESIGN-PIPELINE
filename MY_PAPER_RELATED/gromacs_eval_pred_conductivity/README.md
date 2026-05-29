# GROMACS Predicted-Conductivity Eval

This folder uses the same batch-eval notebook structure as the existing-data workflow, but the input candidate CSV is built from:

```text
source/all_novel_smiles_condz_high_with_pred_conductivity.csv
source/all_novel_smiles_condz_low_with_pred_conductivity.csv
```

Run notebook:

```text
gromacs_new_batch_eval_top_bottom_stratified.ipynb
```

Prediction-eval settings:

```python
selection_group_col = "design_condition"
selection_groups = ["HIGH", "LOW"]
top_k_per_group = 5
bottom_k_per_group = 5
stratified_n = 0
production_replicas = 3
start_phase = "pysoftk"
```

For `production_replicas = 3`, the analysis phase evaluates all three production trajectories:

```text
md/production
md/production_rep2
md/production_rep3
```

Per-replica outputs are saved under `analysis/replica_1`, `analysis/replica_2`, and `analysis/replica_3`. The root `analysis/conductivity_summary_htpmd_ref.csv` contains the replica mean for the main conductivity fields plus `_std`, `_var`, `_min`, and `_max` columns. The full per-replica table and long-form statistics are also written to:

```text
analysis/conductivity_summary_htpmd_ref_replicas.csv
analysis/conductivity_summary_htpmd_ref_replica_stats.csv
```

The GROMACS input table is:

```text
simulation-trajectory-aggregate.csv
```

Its `CONDUCTIVITY` column is the PolyBERT-predicted conductivity (`pred_cond`), not measured MD conductivity. The notebook samples 20 trajectories total: HIGH top 5, HIGH bottom 5, LOW top 5, and LOW bottom 5 by predicted conductivity. `Degree of Polymerization` is computed per molecule as `ceil(150 / repeat_heavy_atoms)`, where `repeat_heavy_atoms` excludes polymer dummy atoms (`[*]`). Molality and density are set from typical values in the existing dataset: molality 1.42 and density 1.30.
