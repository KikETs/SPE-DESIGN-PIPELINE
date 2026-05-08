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
- The selected weighted Ridge model may be used as a sensitivity analysis or auxiliary recall-focused filter only if generated-candidate embeddings are later produced.
- The current weighted results are OOF diagnostics on labeled MD-derived training data; they do not validate generated candidates.
- Generated-candidate weighted predictions were not computed because candidate PolyBERT embeddings were unavailable.
- Candidate selection should remain multi-criteria and should not rely on weighted predicted conductivity alone.

# Risks

- False positives can increase when recall-focused weighting shifts predictions upward near the high-conductivity threshold.
- Calibration may degrade even when recall improves.
- Existing applicability-domain analysis shows many generated candidates are outside the training distribution.
- Weighted rankings for generated candidates remain unavailable until embeddings are generated.
- The model remains a surrogate prescreener and not a physical conductivity validator.

# Recommended Manuscript Changes

- Methods: describe interval-weighted Ridge as a sensitivity experiment using training-fold-only target-derived weights.
- Results: report OOF threshold, top-k, and high-tail diagnostics against the unweighted baseline.
- Limitations: state that weighted generated-candidate prediction requires candidate embeddings and that OOD sensitivity remains unresolved.
- Supplementary: place the full weighting grid, fold-wise metrics, calibration deciles, and threshold sensitivity tables.

# Release Checklist

- `revised/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py`
- `revised/polybert_weighted_evidence/tables/weighted_oof_metrics_all.csv`
- `revised/polybert_weighted_evidence/tables/weighted_threshold_metrics_all.csv`
- `revised/polybert_weighted_evidence/tables/weighted_topk_metrics_all.csv`
- `revised/polybert_weighted_evidence/tables/weighted_model_selection.csv`
- `revised/polybert_weighted_evidence/figures_data/*.csv`
