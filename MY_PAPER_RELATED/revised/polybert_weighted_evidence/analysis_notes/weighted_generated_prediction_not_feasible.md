# Weighted Generated-Candidate Prediction Status

Weighted generated-candidate prediction was not run in this pass.

Reason:
- The available generated-candidate table contains baseline PolyBERT-Ridge predictions, but no generated-candidate PolyBERT embedding cache was detected.
- The weighted Ridge models require the same 600-dimensional PolyBERT embeddings used for training.
- Re-extracting embeddings would require `sentence_transformers` and access to `kuelumbus/polyBERT`; this evidence pass intentionally avoids modifying the original pipeline or inventing weighted predictions.

Required input to complete this step:
- A candidate-level PolyBERT embedding matrix aligned to `all_novel_smiles_with_pred_conductivity.csv`, or a reproducible embedding extraction run using the original PolyBERT model.

Claim boundary:
- The weighted OOF results can support a predictor sensitivity analysis.
- They cannot yet replace baseline generated-candidate counts or candidate rankings.
