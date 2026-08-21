# Planned immutable archive manifest

Zenodo has reserved DOI `10.5281/zenodo.22043456` for the private draft. The DOI
is not an active public record until the draft is published.

## Current private draft

- Zenodo draft: `https://zenodo.org/deposit/22043456`
- Reserved DOI: `10.5281/zenodo.22043456`
- State: `unsubmitted`
- Publish action called: no
- Upload source: `scripts/zenodo_upload_representative.py`

The reserved DOI must not be described as an accessible public archive until
publication and DOI resolution are verified.

## GitHub release

Include the repository snapshot at proposed version
`v1.0.0-jcim-submission`, including:

- source code, notebooks, configuration, MDP, topology, Packmol, and small
  representative structure files;
- machine-readable labels, split assignments, OOF/generated predictions, and
  generated/reference MD numerical data;
- this reproducibility package, audit, compliance report, license, citation,
  and release notes.

Do not include ignored model caches or all production trajectories.

## Zenodo publication record

Upload the frozen GitHub release plus the designated representative trajectory
bundle in
`MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/external_data/representative_trajectory_files.csv`:

1. Generated `Traj_912118`, replica 3: 50 ns XTC, matching TPR, final GRO,
   production MDP, and index.
2. Reference reassessment `Traj_13430`: 50 ns XTC, matching TPR, final GRO,
   production MDP, and index, only after upstream redistribution rights are
   confirmed.

The recorded XTC total is 4,675,157,648 bytes. Verify every upload against the
listed SHA256. Do not upload all 180 generated trajectories merely to satisfy
the GitHub release; the compact replica-level numerical data are already
publicable. If broader raw preservation is desired, create a separate
restricted/large archive inventory rather than placing it in Git history.

## External public data

- HTP-MD portal: `https://www.htpmd.matr.io/`
- HTP-MD data landing page: `https://data.matr.io/htp/`
- Platform paper: `https://doi.org/10.1063/5.0160937`

Reference third-party data by its original URL/DOI when redistribution terms do
not support bundling.

## Publication checklist

1. Confirm author metadata and reference trajectory redistribution rights.
2. Create signed/annotated tag `v1.0.0-jcim-submission` only after authorization.
3. Publish the GitHub release and archive files.
4. Verify SHA256 values, publish the draft, and confirm that reserved DOI
   `10.5281/zenodo.22043456` resolves.
5. Update manuscript availability text from reserved to published status; do not
   rewrite historical scientific outputs.
