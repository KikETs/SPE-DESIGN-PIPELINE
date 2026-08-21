# JCIM reproducibility package

This directory is an additive index over the archived scientific sources. It
does not replace or rewrite the source MD/ML outputs. All paths are relative to
the repository root.

## Scope

- `md/`: protocol mapping, extracted MDP parameters, topology provenance,
  representative-system index, production-replica seeds, and analysis order.
- `ml/`: final canonical-grouped split and model-workflow entry points.
- `data/`: deterministic consolidated generated/reference tables and their
  provenance.
- `zenodo/`: immutable archive plan and private-draft status. Zenodo reserved
  DOI `10.5281/zenodo.22043456`; it is inactive until publication.
- `MANIFEST.csv`: SHA256 and size index for package files and key source files.

The canonical compact MD source is
`MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release/`.
It already contains representative inputs/topologies/structures and the
published numerical release, so those files are referenced instead of copied.

## Build and validate

```bash
python scripts/build_jcim_reproducibility_package.py
python -m pytest tests/reproducibility -q
```

The builder only reads archived source files and writes indexes under this
directory. It verifies the generated candidate aggregation against the existing
60-row archived candidate table before writing.

## Availability boundary

Small code, inputs, structures, and numerical tables belong in GitHub. The two
designated representative trajectories and an immutable publication snapshot
belong in a DOI-providing repository. Zenodo has reserved DOI
`10.5281/zenodo.22043456`, but the trajectory deposit is not public until the
draft is published. See `zenodo/ZENODO_UPLOAD_MANIFEST.md`.
