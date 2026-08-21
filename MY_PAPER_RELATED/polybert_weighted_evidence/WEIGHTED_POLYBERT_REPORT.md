# Weighted PolyBERT Report

# Final Status
WEIGHTED MODEL IMPROVES SCREENING

# Best Weighted Scheme
`smooth_sigmoid_tail_a6_t0p05__ridge_alpha_100` (Scheme E: smooth_sigmoid_tail; threshold=-4; alpha=6; temperature=0.05; Ridge alpha=100.0)

# Baseline vs Weighted Summary

| metric | baseline | best weighted | delta |
| --- | ---: | ---: | ---: |
| MAE log10 | 0.136053 | 0.138399 | 0.002346 |
| RMSE log10 | 0.176553 | 0.181957 | 0.005405 |
| R2 | 0.466742 | 0.433594 | -0.033148 |
| Spearman | 0.642409 | 0.652364 | 0.009955 |
| precision at 1e-4 | 0.667364 | 0.586721 | -0.080643 |
| recall at 1e-4 | 0.455064 | 0.617689 | 0.162625 |
| F1 at 1e-4 | 0.541137 | 0.601807 | 0.060670 |
| threshold enrichment at 1e-4 | 5.969147 | 5.247846 | -0.721302 |
| top-100 enrichment | 8.407703 | 8.586591 | 0.178887 |
| high-tail MAE true >=1e-4 | 0.170468 | 0.126244 | -0.044224 |
| high-tail RMSE true >=1e-4 | 0.208820 | 0.161771 | -0.047049 |

# CEJ-Safe Interpretation

- The weighted experiment tests whether conductivity-interval sample weights improve surrogate screening behavior in the high-conductivity tail.
- The selected weighted Ridge model was applied to all 32,610 unique generated candidates as a sensitivity analysis; baseline ranking remains the primary MD-selection route.
- The current weighted results are OOF diagnostics on labeled MD-derived training data; they do not validate generated candidates.
- Generated-candidate weighted predictions were computed after verifying the encoder against cached training embeddings and refitting the selected model on all 6,270 labeled rows.
- Candidate selection should remain multi-criteria and should not rely on weighted predicted conductivity alone.

# Risks

- False positives can increase when recall-focused weighting shifts predictions upward near the high-conductivity threshold.
- Calibration may degrade even when recall improves.
- Weighted rankings are available, but the original 60-candidate MD selection remains attributed to the baseline ranking.
- The model remains a surrogate prescreener and not a physical conductivity validator.

# Recommended Manuscript Changes

- Methods: describe interval-weighted Ridge as a sensitivity experiment using training-fold-only target-derived weights.
- Results: report OOF threshold, top-k, and high-tail diagnostics against the unweighted baseline.
- Supplementary: report generated-pool weighted predictions separately from OOF model-selection diagnostics.
- Supplementary: place the full weighting grid, fold-wise metrics, calibration deciles, and threshold sensitivity tables.

# Release Checklist

- `polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py`
- `polybert_weighted_evidence/scripts/predict_weighted_generated_candidates.py`
- `polybert_weighted_evidence/tables/weighted_oof_metrics_all.csv`
- `polybert_weighted_evidence/tables/weighted_threshold_metrics_all.csv`
- `polybert_weighted_evidence/tables/weighted_topk_metrics_all.csv`
- `polybert_weighted_evidence/tables/weighted_model_selection.csv`
- `polybert_weighted_evidence/tables/weighted_generated_candidate_predictions.csv`
- `polybert_weighted_evidence/figures_data/*.csv`
