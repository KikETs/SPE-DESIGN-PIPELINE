# Weighted PolyBERT Report

# Final Status
WEIGHTED MODEL IMPROVES SCREENING

# Best Weighted Scheme
`smooth_sigmoid_tail_a6_t0p05__ridge_alpha_100` (Scheme E: smooth_sigmoid_tail; threshold=-4; alpha=6; temperature=0.05; Ridge alpha=100.0)

# Baseline vs Weighted Summary

| metric | baseline | best weighted | delta |
| --- | ---: | ---: | ---: |
| MAE log10 | 0.137052 | 0.139319 | 0.002267 |
| RMSE log10 | 0.178186 | 0.183801 | 0.005615 |
| R2 | 0.456832 | 0.422057 | -0.034775 |
| Spearman | 0.639149 | 0.649013 | 0.009864 |
| precision at 1e-4 | 0.665263 | 0.588710 | -0.076553 |
| recall at 1e-4 | 0.450785 | 0.624822 | 0.174037 |
| F1 at 1e-4 | 0.537415 | 0.606228 | 0.068813 |
| threshold enrichment at 1e-4 | 5.950357 | 5.265634 | -0.684722 |
| top-100 enrichment | 8.407703 | 8.586591 | 0.178887 |
| high-tail MAE true >=1e-4 | 0.170562 | 0.128247 | -0.042315 |
| high-tail RMSE true >=1e-4 | 0.211366 | 0.168942 | -0.042423 |

# CEJ-Safe Interpretation

- The weighted experiment tests whether conductivity-interval sample weights improve surrogate screening behavior in the high-conductivity tail.
- The selected weighted Ridge model may be used as a sensitivity analysis or auxiliary recall-focused filter only if generated-candidate embeddings are later produced.
- The current weighted results are OOF diagnostics on labeled MD-derived training data; they do not validate generated candidates.
- Generated-candidate weighted predictions were not computed because candidate PolyBERT embeddings were unavailable.
- Candidate selection should remain multi-criteria and should not rely on weighted predicted conductivity alone.

# Risks

- False positives can increase when recall-focused weighting shifts predictions upward near the high-conductivity threshold.
- Calibration may degrade even when recall improves.
- Weighted rankings for generated candidates remain unavailable until embeddings are generated.
- The model remains a surrogate prescreener and not a physical conductivity validator.

# Recommended Manuscript Changes

- Methods: describe interval-weighted Ridge as a sensitivity experiment using training-fold-only target-derived weights.
- Results: report OOF threshold, top-k, and high-tail diagnostics against the unweighted baseline.
- Limitations: state that weighted generated-candidate prediction requires candidate embeddings.
- Supplementary: place the full weighting grid, fold-wise metrics, calibration deciles, and threshold sensitivity tables.

# Release Checklist

- `polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py`
- `polybert_weighted_evidence/tables/weighted_oof_metrics_all.csv`
- `polybert_weighted_evidence/tables/weighted_threshold_metrics_all.csv`
- `polybert_weighted_evidence/tables/weighted_topk_metrics_all.csv`
- `polybert_weighted_evidence/tables/weighted_model_selection.csv`
- `polybert_weighted_evidence/figures_data/*.csv`
