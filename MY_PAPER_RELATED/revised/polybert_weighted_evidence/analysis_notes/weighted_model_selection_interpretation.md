# Weighted Model Selection Interpretation

- Baseline reference: `baseline_unweighted__ridge_alpha_1`.
- Best CEJ screening candidate: `smooth_sigmoid_tail_a6_t0p05__ridge_alpha_100`.
- Baseline 1e-4 precision/recall/F1: 0.665 / 0.451 / 0.537.
- Best 1e-4 precision/recall/F1: 0.589 / 0.625 / 0.606.
- Baseline MAE/RMSE/R2/Spearman: 0.1371 / 0.1782 / 0.4568 / 0.6391.
- Best MAE/RMSE/R2/Spearman: 0.1393 / 0.1838 / 0.4221 / 0.6490.
- Baseline top-100 enrichment: 8.408.
- Best top-100 enrichment: 8.587.

Claim-safe conclusion: interval weighting should be discussed as a surrogate-predictor sensitivity experiment. It does not validate generated-candidate conductivity.
