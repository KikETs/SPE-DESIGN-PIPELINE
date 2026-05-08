# MODELS

Integrated model workspace for the paper reproducibility release.

## Structure

- `notebooks/`: training, evaluation, pretraining, and FCD notebooks.
- `models/`: `Encoder_Only`, `TransCVAE`, and `LSTM_CVAE` model definitions.
- `utils/`: dataloader, loss, evaluation, generation, dispatch, and PI1M
  pretraining utilities.
- `data/`: input data, including
  `simulation-trajectory-aggregate_aligned.csv`.
- `checkpoints/`: local output location for trained weights and cache tensors.
  Only `.gitkeep` is tracked.
- `FCD_runs/`: tracked compact result outputs and summary tables.

## Variant Selection

Set `MODELS_VARIANT` before importing `models` or `utils`:

- `Encoder_Only`
- `Encoder_Only_PSMILES`
- `TransCVAE`
- `TransCVAE_PSMILES`
- `LSTM_CVAE`
- `LSTM_CVAE_PSMILES`

The dispatch modules `models/_dispatch.py` and `utils/_dispatch.py` re-export
variant-specific implementations based on this environment variable.

## Quick Start

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/repro_smoke.py --check-imports
```

To run a notebook, open files under `MY_PAPER_RELATED/MODELS/notebooks/`.
Training notebooks write new artifacts to `MY_PAPER_RELATED/MODELS/checkpoints/`.

## Rebuild Summary Tables

```bash
python MY_PAPER_RELATED/MODELS/scripts/build_repeat51200_summary.py
```

To recompute PolyBERT conductivity evaluation tables:

```bash
python MY_PAPER_RELATED/MODELS/scripts/update_conductivity_eval.py --device cpu
```

## Release Hygiene

```bash
python scripts/prepare_release.py
python ../../scripts/strip_notebook_outputs.py ../../
```

Do not commit local checkpoints, cache tensors, or generated training artifacts.
