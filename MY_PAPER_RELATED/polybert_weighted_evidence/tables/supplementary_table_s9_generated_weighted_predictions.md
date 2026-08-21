# Supplementary Table S9

Baseline and interval-weighted generated-candidate score comparison.

| Metric | Baseline | Weighted |
|---|---:|---:|
| Generated candidates | 32,610 | 32,610 |
| Mean log10 predicted conductivity | -3.955 | -3.864 |
| Median log10 predicted conductivity | -3.981 | -3.887 |
| Count at or above 1.00 x 10^-4 S cm^-1 | 17,072 | 21,052 |
| Hit fraction | 0.524 | 0.646 |
| Rank Spearman vs baseline | 1.000 | 0.943 |

The selected weighted model used smooth sigmoid tail weights centered at log10 conductivity = -4 (alpha = 6; temperature = 0.05) and Ridge alpha = 100. Both deployment models were refit on all 6,270 labeled rows after OOF model selection. The table reports the current deduplicated 32,610-candidate pool.
