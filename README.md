# Paper Reproducibility Release

This repository contains the curated `MY_PAPER_RELATED` release for reproducing
the polymer generation and surrogate-screening results.

## What Is Included

- `MY_PAPER_RELATED/MODELS`: integrated generation models, training/evaluation
  notebooks, data, FCD outputs, and summary tables.
- `MY_PAPER_RELATED/polybert_con`: PolyBERT conductivity screening code and
  generated-candidate summary outputs.
- `MY_PAPER_RELATED/revised/polybert_weighted_evidence`: interval-weighted
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

## Quick Validation

Run the repository-level smoke check:

```bash
python scripts/repro_smoke.py
```

After installing dependencies, include import checks:

```bash
python scripts/repro_smoke.py --check-imports
```

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

## Data And Artifacts

Tracked outputs include compact CSV/Markdown summary tables and figure data.
Large model checkpoints, cache tensors, and one oversized full OOF prediction
CSV are not tracked. The excluded full OOF table can be regenerated from:

```bash
python MY_PAPER_RELATED/revised/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py
```

The current tracked tree is designed to stay below normal GitHub file-size
limits without requiring Git LFS.
