# Paper Reproducibility Release

This repository contains the curated `MY_PAPER_RELATED` release for reproducing
the polymer generation and surrogate-screening results.

## What Is Included

- `MY_PAPER_RELATED/MODELS`: integrated generation models, training/evaluation
  notebooks, data, FCD outputs, and summary tables.
- `MY_PAPER_RELATED/polybert_con`: PolyBERT conductivity screening code and
  generated-candidate summary outputs.
- `MY_PAPER_RELATED/polybert_weighted_evidence`: interval-weighted
  PolyBERT analysis scripts, figure data, and summary tables.
- `MY_PAPER_RELATED/selfies-psmiles`: local package source used by the model
  notebooks.
- `vendor/`: local `psmiles` packaging and its small canonicalization
  dependency.

Unfinished atomistic-simulation batch pipelines and large model/cache artifacts
are intentionally omitted from this release.

## Setup

Python 3.10 or newer is recommended. A CUDA-enabled PyTorch installation is
optional; CPU is enough for structure checks and small smoke tests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conda users can start from:

```bash
conda env create -f environment.yml
conda activate paper-repro
```

## External Reference Data

Reference CSV files matching `simulation*.csv` are intentionally excluded from
Git. Download the reference data separately and place the files at the expected
paths before running training, PolyBERT screening, MD reassessment, or full
reproduction checks:

```text
MY_PAPER_RELATED/MODELS/data/simulation-trajectory-aggregate_aligned.csv
MY_PAPER_RELATED/polybert_con/simulation-trajectory-aggregate.csv
MY_PAPER_RELATED/gromacs_eval_pred_conductivity/simulation-trajectory-aggregate.csv
MY_PAPER_RELATED/polybert_weighted_evidence/source_data/polybert_con/simulation-trajectory-aggregate.csv
```

These local files remain ignored by `.gitignore` after placement.

## Quick Validation

Run the repository-level smoke check:

```bash
python scripts/repro_smoke.py
```

After installing dependencies, include import checks:

```bash
python scripts/repro_smoke.py --check-imports
```

## Optional Pretrained Checkpoints

Pretrained model weights are distributed through the `v0.1.0` GitHub Release,
not through Git history:

```bash
python scripts/download_checkpoints.py
```

See `CHECKPOINTS.md` for the asset URL, checksum, included paths, and excluded
cache artifacts.

## Reproduction Entry Points

1. Inspect the release structure and expected files:

   ```bash
   python scripts/repro_smoke.py
   ```

2. Rebuild combined FCD repeat summaries:

   ```bash
   python MY_PAPER_RELATED/MODELS/scripts/build_repeat51200_summary.py
   ```

3. Recompute conductivity-screening summary tables when PolyBERT embeddings are
   available:

   ```bash
   python MY_PAPER_RELATED/MODELS/scripts/update_conductivity_eval.py --device cpu
   ```

4. Open notebooks from:

   ```text
   MY_PAPER_RELATED/MODELS/notebooks/
   ```

Training notebooks write checkpoints under `MY_PAPER_RELATED/MODELS/checkpoints/`.
That directory is intentionally ignored except for `.gitkeep`.

## Machine-readable Data Index

The [machine-readable data index](MY_PAPER_RELATED/machine_readable/README.md)
maps the manuscript requirements and numerical figures to repository CSVs. It
includes all 6,270 labeled structures with leakage-controlled,
canonical-structure-grouped four-fold assignments, all 32,611 generated
structures with prediction/exclusion status, the 60-candidate and
180-replica generated-MD results, and the full 120-selected/108-completed
reference reassessment status.

## Data And Artifacts

Tracked outputs include compact CSV/Markdown summary tables, figure data, and
the 6,270-row OOF prediction table. Large model checkpoints and cache tensors
are not tracked. The canonical-group fold and OOF tables can be regenerated
from cached PolyBERT embeddings with:

```bash
python MY_PAPER_RELATED/polybert_con/train_polybert_conductivity_4fold.py \
  --csv MY_PAPER_RELATED/polybert_con/fold_assignment.csv \
  --outdir MY_PAPER_RELATED/polybert_con \
  --embeddings_path MY_PAPER_RELATED/polybert_weighted_evidence/source_data/polybert_run/embeddings.npy
```

The weighted sensitivity analysis can then be rerun with
`MY_PAPER_RELATED/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py`.
After OOF model selection, regenerate the 32,610 generated-candidate weighted
predictions with
`MY_PAPER_RELATED/polybert_weighted_evidence/scripts/predict_weighted_generated_candidates.py`.

The current tracked tree is designed to stay below normal GitHub file-size
limits without requiring Git LFS.

## Citation And License

Use `CITATION.cff` for software citation metadata. Original source code and
documentation are released under the top-level `LICENSE`; third-party and
vendored components retain their own license terms as listed in `NOTICE.md`.
