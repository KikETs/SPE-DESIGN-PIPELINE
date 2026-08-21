# Weighted Predictor File Inventory

## Detected Source Files

- `source_data/polybert_con/counts_pred_cond_ge_1e4_by_model.csv` (514 bytes)
- `source_data/polybert_con/counts_pred_cond_ge_1e4_by_model_condition.csv` (389 bytes)
- `source_data/polybert_con/fold_assignment.csv` (1105639 bytes)
- `source_data/polybert_con/fold_summary.csv` (79 bytes)
- `source_data/polybert_con/simulation-trajectory-aggregate.csv` (717818 bytes)
- `source_data/polybert_con/train_polybert_conductivity_4fold.py` (7647 bytes)
- `source_data/polybert_run/all_novel_smiles_with_pred_conductivity.csv` (3861002 bytes)
- `source_data/polybert_run/cv_metrics.csv` (433 bytes)
- `source_data/polybert_run/cv_metrics_by_conductivity_band.csv` (742 bytes)
- `source_data/polybert_run/cv_metrics_cond_ge_threshold.csv` (238 bytes)
- `source_data/polybert_run/embeddings.npy` (15048128 bytes)
- `source_data/polybert_run/fold_assignment.csv` (1105153 bytes)
- `source_data/polybert_run/oof_predictions.csv` (886167 bytes)

## Fold Assignments

- Authoritative split: `source_data/polybert_con/fold_assignment.csv`.
- Canonical structure groups: 6026
- Canonical groups crossing folds: 0
- Number of folds: [0, 1, 2, 3]
- Samples per fold: {0: 1561, 1: 1559, 2: 1583, 3: 1567}

## Embeddings

- Cached PolyBERT training embeddings available: yes
- Embedding shape: (6270, 600)
- Generated embedding caches are reproducible intermediates and are excluded from Git.
- Candidate-level weighted predictions available: yes.

## Existing OOF Predictions

- This script regenerates baseline and interval-weighted Ridge OOF predictions using canonical-structure-grouped folds.
- Selected baseline/weighted OOF predictions are exported to `tables/grouped_oof_predictions_selected.csv`.

## Baseline Reproduction Command

```bash
python polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py
```
