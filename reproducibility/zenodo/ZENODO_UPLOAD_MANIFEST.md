# Published immutable archive manifest

Zenodo record `22043456` was published on 2026-08-21 at DOI
[`10.5281/zenodo.22043456`](https://doi.org/10.5281/zenodo.22043456).

## Published record

- Public record: `https://zenodo.org/records/22043456`
- DOI: `10.5281/zenodo.22043456`
- State: `done`
- Publish action called: yes
- Published files: 11
- Published bytes: 4,776,170,247
- Upload source: `scripts/zenodo_upload_representative.py`

The DOI resolver, public record API, remote sizes, and remote checksums were
verified after publication.

## Source snapshot

The record includes `SPE-DESIGN-PIPELINE-830fa3c.tar.gz`, a 99,173,551-byte
snapshot of GitHub `main` merge commit
`830fa3ce5770a9b43bfc3439a381288c2a1a5254`. Its SHA256 is
`5953e885baa6571e82bdc1eeaec415d55bc561659a6ec1e38e1a23c4f3802050`.
It includes:

- source code, notebooks, configuration, MDP, topology, Packmol, and small
  representative structure files;
- machine-readable labels, split assignments, OOF/generated predictions, and
  generated/reference MD numerical data;
- this reproducibility package, audit, compliance report, license, citation,
  and release notes.

Ignored model caches and the complete production trajectory set are excluded.

## Zenodo publication record

The record also contains the designated representative trajectory bundle in
`MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/external_data/representative_trajectory_files.csv`:

1. Generated `Traj_912118`, replica 3: 50 ns XTC, matching TPR, final GRO,
   production MDP, and index.
2. Reference reassessment `Traj_13430`: 50 ns XTC, matching TPR, final GRO,
   production MDP, and index. The redistribution basis must be documented in
   the final submission record.

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

## Remaining release checklist

1. Confirm author metadata and reference trajectory redistribution rights.
2. Create the GitHub tag/release if required by the final submission workflow.
3. Completed: verify files, publish the Zenodo record, and confirm DOI
   `10.5281/zenodo.22043456` resolves.
4. Update manuscript availability text with the published DOI; do not rewrite
   historical scientific outputs.
