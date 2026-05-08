# Weighted Model Selection Interpretation

- Baseline reference: `baseline_unweighted__ridge_alpha_1`.
- Best CEJ screening candidate: `smooth_sigmoid_tail_a6_t0p05__ridge_alpha_100`.
- Baseline 1e-4 precision/recall/F1: 0.667 / 0.455 / 0.541.
- Best 1e-4 precision/recall/F1: 0.587 / 0.618 / 0.602.
- Baseline MAE/RMSE/R2/Spearman: 0.1361 / 0.1766 / 0.4667 / 0.6424.
- Best MAE/RMSE/R2/Spearman: 0.1384 / 0.1820 / 0.4336 / 0.6524.
- Baseline top-100 enrichment: 8.408.
- Best top-100 enrichment: 8.587.

Claim-safe conclusion: interval weighting should be discussed as a surrogate-predictor sensitivity experiment. It does not validate generated-candidate conductivity.
