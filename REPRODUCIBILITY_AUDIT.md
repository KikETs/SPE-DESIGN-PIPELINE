# JCIM reproducibility audit

Audit date: 2026-08-21

Post-audit recovery: subsequent official-upstream comparison identified the
vendored PySoftK source as commit
`a3458567d9b61e5caf6d891861e5fa016dfe97b5` plus one local
`linear_polymer.py` patch. Of 44 source files, 43 are byte-identical; the
upstream BSD-3-Clause license is now retained in the vendored directory. This
recovery supersedes the initial local-only provenance assessment.

Post-audit archive update: Zenodo record `22043456` was published on 2026-08-21
at DOI `10.5281/zenodo.22043456`. It contains the frozen Git-tracked source
snapshot and two representative 50 ns GROMACS trajectory bundles.

Scope: repository state on branch `codex/gromacs-conductivity-results`. This
audit treats archived inputs and numerical outputs as evidence. Manuscript/SI
statements are claims to compare against that evidence, not targets to force.
No scientific result was recomputed or changed during this audit.

The compliance targets are the JCIM method/data/software reproducibility policy
(`https://doi.org/10.1021/acs.jcim.0c01389`) and MD reporting guideline
(`https://doi.org/10.1021/acs.jcim.3c00599`), both checked on 2026-08-21.

## FOUND

### Generation and surrogate workflow

- The 6,270-row HTP-MD-derived label table, endpoint-normalized PSMILES,
  canonical PSMILES, log target, and final fold assignment are in
  `MY_PAPER_RELATED/polybert_con/fold_assignment.csv`.
- Dataset curation, duplicate retention, invalid/missing-value checks,
  canonicalization, and leakage audit are documented in
  `MY_PAPER_RELATED/machine_readable/HTPMD_DATASET_CURATION.md` and
  `MY_PAPER_RELATED/machine_readable/htpmd_dataset_curation_summary.csv`.
- Endpoint-aware SELFIES-PSMILES conversion and validation code is under
  `MY_PAPER_RELATED/selfies-psmiles/`.
- TransCVAE model code is in
  `MY_PAPER_RELATED/MODELS/models/Trans_TransCVAE_PSMILES.py`; generation code
  is in `MY_PAPER_RELATED/MODELS/utils/generate_TransCVAE_PSMILES.py`, with
  training/evaluation notebooks under `MY_PAPER_RELATED/MODELS/notebooks/`.
- Canonical-structure-grouped four-fold PolyBERT/Ridge training and split
  generation are implemented by
  `MY_PAPER_RELATED/polybert_con/train_polybert_conductivity_4fold.py`.
- Baseline OOF predictions are in
  `MY_PAPER_RELATED/polybert_con/oof_predictions.csv`. Paired canonical-grouped
  baseline/selected-weighted OOF predictions are in
  `MY_PAPER_RELATED/polybert_weighted_evidence/tables/grouped_oof_predictions_selected.csv`.
- Weighted model fitting and deployment scripts are
  `MY_PAPER_RELATED/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py`
  and
  `MY_PAPER_RELATED/polybert_weighted_evidence/scripts/predict_weighted_generated_candidates.py`.
- Generated-pool and selected-candidate machine-readable predictions are
  indexed by `MY_PAPER_RELATED/machine_readable/README.md`. The 60-candidate
  selection is in
  `MY_PAPER_RELATED/gromacs_eval_pred_conductivity/github_results/latest_notebook_manifest_60/sample_manifest.csv`.

### MD preparation and GROMACS inputs

- The compact, tracked MD release is
  `MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/`.
  It contains staged equilibration and production MDPs, representative GRO/TPR,
  TOP/ITP files, Packmol inputs, construction structures, preparation drivers,
  analysis code, and compact numerical outputs.
- Polymer construction, Packmol, atom typing, charge checks, MD staging, and
  analysis drivers are preserved under
  `MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/workflow/`.
- The generated representative system is
  `.../static_cne0_release/examples/generated_Traj_912118_rep3/`; the reference
  representative is `.../static_cne0_release/examples/reference_Traj_13430/`.
- A bonded-fallback example is
  `.../static_cne0_release/examples/fallback_Traj_912122/`. A polymer-charge
  repair example is `.../static_cne0_release/examples/charge_repair_Traj_911758/`.
- Direct ACPYPE and TFSI command forms are recorded in
  `.../static_cne0_release/ACPYPE_INVOCATIONS.md`; the bonded fallback is
  documented in `.../static_cne0_release/FALLBACK_TOPOLOGY_PROCEDURE.md`.
- Li/TFSI charge scaling and neutrality handling are recorded in
  `.../static_cne0_release/inputs/CHARGE_MODEL.md` and
  `.../static_cne0_release/inputs/charge_scaling_parameters.csv`.
- The actual generated representative production input uses `dt = 0.001 ps`,
  `nsteps = 50000000`, `ref-t = 353 K`, and `nstxout-compressed = 500`; see
  `.../generated_Traj_912118_rep3/mdp/production*.mdp`.
- The actual reference representative production input uses `dt = 0.002 ps`
  and 25,000,000 steps; see
  `.../reference_Traj_13430/mdp/production.mdp`.

### MD analysis and numerical outputs

- Manuscript-protocol RDF-first-minimum, at-least-two-oxygen association,
  static graph, formal cluster charge `z_ij = i - j`, cNE0, and NE processing
  are preserved in
  `.../static_cne0_release/analysis/manuscript_static_cne0.py` and
  `.../static_cne0_release/analysis/persistence_sweep_and_hybrid.py`.
- RDF-peak/zero-persistence sensitivity code is separately labeled as a
  sensitivity analysis in
  `.../static_cne0_release/analysis/rdf_peak_tau0_replica_eval.py`; it is not
  the manuscript static-cNE0 source.
- MSD fit-R2 and beta-window scripts are
  `.../static_cne0_release/analysis/msd_fit_r2_expanded.py` and
  `.../static_cne0_release/analysis/msd_beta_windows.py`.
- Generated results include 60 candidate rows and 180 replica rows in
  `.../static_cne0_release/data/generated/generated_md_results_60.csv` and
  `.../static_cne0_release/data/generated/generated_static_cNE0_replica_180.csv`.
- The full reference denominator is retained in
  `MY_PAPER_RELATED/machine_readable/reference_selection_manifest_120.csv` and
  `MY_PAPER_RELATED/machine_readable/reference_run_status_120.csv`; 108 completed
  outputs are in
  `.../static_cne0_release/data/reference/reference_static_cNE0_reassessment_108.csv`.
- Existing figure-source relations are in
  `MY_PAPER_RELATED/machine_readable/figure_source_manifest.csv`.

### Software and release support

- Historical versions and evidence are preserved in
  `.../static_cne0_release/environment/software_versions.md` and `.csv`.
- `LICENSE`, `NOTICE.md`, and `CITATION.cff` exist at repository root.
- Representative external-archive filenames, sizes, and SHA256 values are in
  `.../static_cne0_release/external_data/representative_trajectory_files.csv`;
  the proposed two-trajectory deposit is described in
  `.../static_cne0_release/external_data/DEPOSITION_PLAN.md`.

## PARTIALLY FOUND

- **Force-field identity:** archived ACPYPE commands omit `-a`, while topology
  headers identify ACPYPE `2023.10.27`. The versioned ACPYPE CLI source gives
  `gaff2` as that release's default atom type, so GAFF2 is strongly supported
  by the command plus tool-version evidence. The historical log does not print
  an explicit `-a gaff2` argument, so this is not a direct command-line record.
- **AmberTools detail:** conda metadata reports AmberTools 23.3 while the
  `antechamber` executable banner reports 22.0. Both facts must be retained.
- **PySoftK version labeling:** historical environment metadata reported
  PySoftK 1.0.0, while the recovered source commit is described in upstream Git
  as `v.1.0.1-151-ga345856`. Both are retained rather than collapsing package
  metadata and source-tree provenance into one version string.
- **Replica randomness:** replicas 2 and 3 use recoverable velocity seeds
  `730002` and `730003` in the production MDPs/workflow. Replica 1 continues
  from the equilibrated state and does not generate new production velocities.
  Earlier equilibration uses `gen-seed = -1`, so the automatically chosen
  equilibration seed is not recoverable from the archived MDP alone.
- **Representative group coverage:** the compact release provides generated,
  reference, direct-ACPYPE, bonded-fallback, and charge-repair examples, but it
  does not provide six separate full input bundles, one for every generated
  selection group.
- **Large MD output availability:** representative trajectory checksums and an
  upload plan exist, local run storage contains large trajectory files, and two
  representative trajectories are published in Zenodo record `22043456`. The
  complete production trajectory and restart/checkpoint set is not archived.
- **Source dataset accession:** the public HTP-MD URL and platform-paper DOI are
  documented, but the precise downloaded object/accession and download date
  cannot be recovered from the local CSV metadata.
- **Figure/table traceability:** major numerical figures have source mappings,
  but Tables 1-4 and every SI table S7-S20 do not yet have a complete explicit
  row-by-row source map.
- **Environment reconstruction:** historical version evidence exists, but the
  root `environment.yml` is a modern paper-reproduction environment and is not
  a frozen export of the Python 3.9.25 MD environment.

## MISSING

- `MISSING — could not be recovered from repository or archived outputs`:
  exact automatically generated random seed used by the initial equilibration
  stage (`gen-seed = -1`).
- `MISSING — could not be recovered from repository or archived outputs`:
  the complete publicly archived production trajectory and restart/checkpoint
  set. Git deliberately excludes these files; the Zenodo record contains two
  representative trajectories instead.
- `MISSING — could not be recovered from repository or archived outputs`:
  an archived generated-system command log that explicitly prints
  `ACPYPE ... -a gaff2`. The force-field identity can only be resolved from the
  archived omitted-argument command plus the versioned ACPYPE default.
- `MISSING — could not be recovered from repository or archived outputs`:
  experimental benchmarking created specifically for this repository cleanup.
  This is new scientific work, not a packaging task.

## AMBIGUOUS

- **Generated production timestep:** the task/manuscript summary claims a 2 fs
  generated production timestep, but the archived generated representative
  MDPs and release README record 1 fs for 50,000,000 steps. The reference
  representative uses 2 fs. The repository evidence does not support silently
  rewriting generated production as 2 fs.
- **Generated output interval:** the task/manuscript summary claims 1 ps. The
  180-row result table records 120 replicas at 0.5 ps and 60 at 1.0 ps, while
  all three representative generated production MDP variants use 0.5 ps.
  Analysis is consistently downsampled to a 1 ps grid, but saved trajectory
  spacing and analysis spacing must not be conflated.
- **Generated group labels:** the archived selection labels are
  `HIGH_top`, `HIGH_middle_stratified`, `HIGH_bottom`, `LOW_top`,
  `LOW_middle_stratified`, and `LOW_bottom`. The requested TT/LT terminology
  requires an explicit manuscript mapping; it must not be silently inferred.
- **Li parameter source:** the implementation labels the monatomic Li values as
  an `amber99sb-ildn` fallback and records mass, sigma, and epsilon in
  `.../workflow/phase_scripts/gromacs_new_phase_atomtyping.py`. A specific
  upstream parameter-file citation was not archived.
- **Reference redistribution:** the existing deposit plan requires confirmation
  that the upstream license permits redistribution of reference `Traj_13430`
  and its topology before external upload.

## Audit conclusion

The repository already contains a substantial compact MD release and complete
machine-readable generated/reference result denominators. The remaining work is
an additive index and documentation layer, deterministic consolidated tables,
validation tests, and archive metadata. It must preserve the timestep/output
discrepancies above and distinguish the public representative subset from the
unarchived complete trajectory set.
