# Static cNE0 reassessment reproducibility release

This directory contains the compact numerical results, simulation inputs, and analysis code for the 60-generated-candidate / 180-production-replica GROMACS static cNE0 reassessment. It also contains the completed 108-system reference reassessment table. Production trajectories are deliberately excluded from GitHub.

## Scope and conditions

- Generated systems: 60 candidates, three independent 50 ns production replicas per candidate, 180 replicas total.
- Reference reassessment: 108 completed systems, one 50 ns production trajectory per system.
- Static cNE0 temperature: 353 K.
- Static cNE0 contact definition: replica/trajectory-specific Li-TFSI oxygen RDF first-minimum cutoff, followed by retention of only Li-TFSI pairs with at least two oxygen contacts in the same frame.
- Analysis grid: 0-50 ns at 1.0 ps spacing (50,001 frames). Generated trajectories saved at 0.5 ps are analyzed with stride 2; trajectories saved at 1.0 ps use stride 1.
- Cluster definition: static association graph with no persistence filter, maximum cluster size 101, and formal cluster charge `z_ij = i - j`.
- Charge model: Li and TFSI formal ionic charges scaled by 0.7; polymer ACPYPE gas charges with neutrality repair when required.
- Raw trajectory files (`.xtc`, `.trr`, `.edr`, and `.cpt`) are not included in this directory.

## Requested public artifacts

| Requested artifact | Public location |
|---|---|
| Production and staged equilibration MDP files | `examples/*/mdp/` |
| Representative GRO and TPR files | `examples/*/representative_structure/` |
| Representative TOP/ITP files | `examples/*/topology/` |
| Li and TFSI topology | `examples/*/topology/li_*.itp` and `tfsi_*.itp` |
| Charge scaling parameters | `inputs/charge_scaling_parameters.csv` and `inputs/CHARGE_MODEL.md` |
| Packmol inputs and structures | `examples/*/packmol/` and `examples/*/structures/` |
| Polymer construction workflow | `workflow/phase_scripts/gromacs_new_phase_pysoftk.py` and `workflow/gromacs_new_pipeline_importable.py` |
| ACPYPE invocation | `ACPYPE_INVOCATIONS.md` and `workflow/phase_scripts/gromacs_new_phase_atomtyping.py` |
| Bonded fallback procedure | `FALLBACK_TOPOLOGY_PROCEDURE.md` and `examples/fallback_Traj_912122/` |
| Manuscript static cNE0 analysis | `analysis/manuscript_static_cne0.py` and `analysis/persistence_sweep_and_hybrid.py` |
| RDF-peak/zero-persistence sensitivity code | `analysis/rdf_peak_tau0_replica_eval.py` (not the manuscript result source) |
| MSD fit and beta-window analysis | `analysis/msd_fit_r2_expanded.py` and `analysis/msd_beta_windows.py` |
| Candidate DP/composition table | `data/generated/generated_candidate_composition_60.csv` |
| Generated-candidate MD table | `data/generated/generated_md_results_60.csv` |
| Generated replica static cNE0 values | `data/generated/generated_static_cNE0_replica_180.csv` |
| Generated candidate means | `data/generated/generated_static_cNE0_candidate_mean_60.csv` |
| Generated group and pairwise summaries | `data/generated/generated_group_summary_candidate_level.csv` and `generated_pairwise_group_tests.csv` |
| Completed reference reassessment | `data/reference/reference_static_cNE0_reassessment_108.csv` |
| Reference group and overall summaries | `data/reference/reference_group_summary_trajectory_level.csv` and `reference_overall_statistics.csv` |
| MSD R2 and beta-window numerical data | `data/generated/msd_fit_r2_*.csv` and `msd_beta_*.csv` |
| Software versions | `environment/software_versions.md` and `.csv` |
| External trajectory checksums | `external_data/representative_trajectory_files.csv` |

## Representative systems

The generated example is `Traj_912118`, replica 3. It is the lowest robust median-distance candidate among 41 systems with three complete replicas, direct ACPYPE topology, and neutral-as-built charge status. The distance used DP, log10 candidate-mean static cNE0, and log10 candidate-mean NE conductivity, each centered by its median and scaled by its MAD. Replica 3 was selected because its static cNE0 is closest to the candidate mean.

- Generated `Traj_912118`, replica 3: 1 fs step, 50,000,000 steps, compressed coordinates every 0.5 ps, 50 ns, 353 K.
- Candidate-mean static cNE0: `2.960759709188e-4 S cm^-1`.
- Replica-3 static cNE0: `3.099696942041e-4 S cm^-1`.
- Replica-3 RDF first-minimum cutoff: `0.342 nm`.

The reference example is `Traj_13430`, the lowest robust median-distance system among the 108 completed reference trajectories. The distance used DP, density, molality, log10 reference conductivity, log10 static cNE0, and log10 NE conductivity.

- Reference `Traj_13430`: 2 fs step, 25,000,000 steps, compressed coordinates every 1 ps, 50 ns, 353 K.
- Static cNE0: `2.776388001129e-5 S cm^-1`.
- RDF first-minimum cutoff: `0.326 nm`.

## Rebuilding the compact release

Run from the `gromacs_eval_pred_conductivity` project directory after the generated and reference run trees are available:

```bash
python scripts/analysis/msd_fit_r2_expanded.py
python scripts/analysis/msd_beta_windows.py \
  --pred-runs-dir runs \
  --ref-runs-dir /path/to/eval_top10_bottom10_stratified100/runs \
  --expected-pred-replicas 180 \
  --expected-ref-trajs 108 \
  --out-dir github_results/latest_notebook_manifest_60/s14c_msd_beta_windows
python scripts/analysis/manuscript_static_cne0.py \
  --dataset all \
  --reference-base /path/to/eval_top10_bottom10_stratified100 \
  --temperature-k 353 \
  --frame-spacing-ps 1.0 \
  --max-cluster-size 101 \
  --workers 8
python scripts/release/build_static_cne0_release.py \
  --reference-base /path/to/eval_top10_bottom10_stratified100 \
  --hash-trajectories
```

`data_quality_report.csv` must contain only `PASS` rows before publication.

## External trajectory deposit

One generated trajectory and one reference-reassessment trajectory are designated for Zenodo or Figshare. Their archive filenames, sizes, and SHA256 checksums are in `external_data/representative_trajectory_files.csv`; the upload procedure and unresolved licensing check are in `external_data/DEPOSITION_PLAN.md`.
