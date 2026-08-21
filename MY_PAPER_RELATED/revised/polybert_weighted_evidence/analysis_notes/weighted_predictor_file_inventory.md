# Weighted Predictor File Inventory

## Detected Source Files

- `source_data/polybert_con/counts_pred_cond_ge_1e4_by_model.csv` (514 bytes)
- `source_data/polybert_con/counts_pred_cond_ge_1e4_by_model_condition.csv` (389 bytes)
- `source_data/polybert_con/fold_assignment.csv` (1105639 bytes)
- `source_data/polybert_con/fold_summary.csv` (196 bytes)
- `source_data/polybert_con/simulation-trajectory-aggregate.csv` (717818 bytes)
- `source_data/polybert_con/train_polybert_conductivity_4fold.py` (10061 bytes)
- `source_data/polybert_run/all_novel_smiles_with_pred_conductivity.csv` (3861002 bytes)
- `source_data/polybert_run/cv_metrics.csv` (433 bytes)
- `source_data/polybert_run/cv_metrics_by_conductivity_band.csv` (741 bytes)
- `source_data/polybert_run/cv_metrics_cond_ge_threshold.csv` (237 bytes)
- `source_data/polybert_run/embeddings.npy` (15048128 bytes)
- `source_data/polybert_run/fold_assignment.csv` (1105639 bytes)
- `source_data/polybert_run/oof_predictions.csv` (886558 bytes)

## Fold Assignments

- Existing fold column detected in `polybert_run/oof_predictions.csv`: yes
- Number of folds: [0, 1, 2, 3]
- Samples per fold: {0: 1561, 1: 1559, 2: 1583, 3: 1567}

## Embeddings

- Cached PolyBERT training embeddings available: yes
- Embedding shape: (6270, 600)
- Generated-candidate PolyBERT embeddings available: no detected file.
- Because generated-candidate embeddings are absent and SentenceTransformers is not required for this script, weighted generated-candidate prediction is marked as not feasible in this evidence pass.

## Existing OOF Predictions

- Existing unweighted OOF predictions available: yes (`polybert_run/oof_predictions.csv`).
- This script regenerates Ridge OOF predictions from cached embeddings and the same fold assignments for baseline and interval-weighted schemes.

## Baseline Reproduction Command

```powershell
python polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py
```
