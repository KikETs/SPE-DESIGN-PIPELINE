# Weighted Generated-Candidate Prediction Status

Weighted generated-candidate prediction is complete for all 32,610 unique candidates.

Method:
- Encoder: `xushijie/polyBERT` revision `e7dce434fb3eff37905dc114008660e5479ca9a8` with 600-dimensional, unnormalized embeddings.
- Encoder identity was checked against cached training embeddings before candidate encoding.
- Baseline deployment model: unweighted Ridge alpha = 1, refit on all 6,270 labeled rows.
- Weighted deployment model: smooth sigmoid weights centered at log10 conductivity = -4, alpha = 6, temperature = 0.05; Ridge alpha = 100; refit on all 6,270 labeled rows.

Generated-pool result:
- Baseline hit fraction at 1e-4 S cm-1: 0.523520 (17,072/32,610).
- Weighted hit fraction at 1e-4 S cm-1: 0.645569 (21,052/32,610).
- Baseline-weighted rank Spearman: 0.942686.

Claim boundary:
- Weighted generated-candidate predictions can now be reported as a sensitivity analysis.
- The original 60-candidate MD selection remains based on the baseline ranking and is not retrospectively relabeled as weighted selection.


Final target-conditioned generated pool:
- LOW: 44,999 candidates; baseline/weighted hit fractions 0.009734 / 0.014578.
- HIGH: 2,126 candidates; baseline/weighted hit fractions 1.000000 / 1.000000.
- Combined: 47,125 candidates; baseline/weighted hit fractions 0.054408 / 0.059034.
- Baseline-weighted rank Spearman across the combined pool: 0.915828.
- All 60 MD-selected candidates were matched to exact baseline-refit and weighted predictions.
