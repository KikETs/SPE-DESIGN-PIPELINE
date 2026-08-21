# Weighted Predictor File Inventory

## Detected Source Files

- `source_data\polybert_con\counts_pred_cond_ge_1e4_by_model.csv` (514 bytes)
- `source_data\polybert_con\counts_pred_cond_ge_1e4_by_model_condition.csv` (389 bytes)
- `source_data\polybert_con\fold_assignment.csv` (578016 bytes)
- `source_data\polybert_con\fold_summary.csv` (157 bytes)
- `source_data\polybert_con\simulation-trajectory-aggregate.csv` (717818 bytes)
- `source_data\polybert_con\train_polybert_conductivity_4fold.py` (7647 bytes)
- `source_data\polybert_run\all_novel_smiles_with_pred_conductivity.csv` (3861002 bytes)
- `source_data\polybert_run\cv_metrics.csv` (433 bytes)
- `source_data\polybert_run\cv_metrics_by_conductivity_band.csv` (742 bytes)
- `source_data\polybert_run\cv_metrics_cond_ge_threshold.csv` (238 bytes)
- `source_data\polybert_run\embeddings.npy` (15048128 bytes)
- `source_data\polybert_run\fold_assignment.csv` (916513 bytes)
- `source_data\polybert_run\oof_predictions.csv` (697510 bytes)

## Fold Assignments

- Existing fold column detected in `polybert_run/oof_predictions.csv`: yes
- Number of folds: [0, 1, 2, 3]
- Samples per fold: {0: 1568, 1: 1568, 2: 1567, 3: 1567}

## Embeddings

- Cached PolyBERT training embeddings available: yes
- Embedding shape: (6270, 600)
- Generated-candidate PolyBERT embedding cache available locally: yes.
- The 32,610 x 600 embedding cache is excluded from Git because it is a large reproducible intermediate.
- Candidate-level weighted predictions available: yes (`tables/weighted_generated_candidate_predictions.csv`).
- Reproduction script: `scripts/predict_weighted_generated_candidates.py`.

## Existing OOF Predictions

- Existing unweighted OOF predictions available: yes (`polybert_run/oof_predictions.csv`).
- This script regenerates Ridge OOF predictions from cached embeddings and the same fold assignments for baseline and interval-weighted schemes.

## Existing Candidate Predictions

- Baseline generated-candidate predictions available: yes (`polybert_run/all_novel_smiles_with_pred_conductivity.csv`).
- Candidate-level weighted predictions available: no.
- Required to score generated candidates with weighted Ridge: generated-candidate PolyBERT embeddings aligned row-by-row to the generated-candidate CSV.
- Existing baseline count tables are present, but they are not overwritten or relabeled as weighted-model results.

## Model Scope

- Ridge with `sample_weight` was evaluated because it is the cleanest extension of the existing PolyBERT-Ridge prescreener.
- MLP was not run in this pass. The original script includes an MLP option, but adding weighted MSE and early-stopping control would introduce a separate model experiment beyond the requested interval-weighted Ridge comparison.

## Baseline Reproduction Command

```powershell
python polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py
```
