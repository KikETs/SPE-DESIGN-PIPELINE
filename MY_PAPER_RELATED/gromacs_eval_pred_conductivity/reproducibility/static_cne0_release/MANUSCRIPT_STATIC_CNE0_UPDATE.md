# Manuscript and SI static cNE0 replacement values

These values supersede the RDF-peak/zero-persistence release and the generated 363 K method-sweep values. They use the manuscript/SI protocol: Li-TFSI oxygen RDF first minimum, at least two oxygen contacts per retained Li-TFSI pair, 353 K, 0-50 ns at 1 ps spacing, 50,001 frames, static clusters, maximum cluster size 101, and formal cluster charge z_ij = i - j.

Generated candidate values are arithmetic means of three replica-level linear conductivities. Group statistics use the 10 candidate-level values in each group.

## Main-text reference reassessment table

| Metric | Value |
|---|---:|
| Completed static cNE0 analyses | 108 / 120 |
| Group coverage | 8 bottom, 91 middle, 9 top |
| Simulation length | 50 ns |
| Primary reassessment protocol | RDF first minimum + Li-TFSI O contacts >= 2 + static cNE0 |
| Overall MALogE (log10 scale) | 0.329 |
| Overall median static cNE0/reference-label ratio | 1.06 |
| Bottom MALogE / median ratio | 0.473 / 2.8 |
| Middle-stratified MALogE / median ratio | 0.299 / 0.786 |
| Top MALogE / median ratio | 0.508 / 4.07 |
| Median RDF first-minimum cutoff | 0.330 nm overall; 0.328 nm bottom; 0.330 nm middle; 0.328 nm top |
| Top-bottom median shift (log10 scale) | 2.2406 |
| Spearman rho vs reference labels | 0.4883 |
| Kendall tau vs reference labels | 0.3519 |
| Mann-Whitney U and exact p-value | 72.0; p = 8.227e-05 |

## Main-text generated-candidate table

| Group | n candidates / replicas | Mean abs delta log10 | Median PolyBERT/static cNE0 | RDF cutoff / nm |
|---|---:|---:|---:|---:|
| TT-top | 10 / 30 | 0.283 | 2.149 | 0.330 |
| TT-middle | 10 / 30 | 0.433 | 2.664 | 0.332 |
| TT-bottom | 10 / 30 | 0.333 | 0.832 | 0.334 |
| LT-top | 10 / 30 | 0.538 | 0.371 | 0.335 |
| LT-middle | 10 / 30 | 0.318 | 1.294 | 0.330 |
| LT-bottom | 10 / 30 | 0.217 | 1.776 | 0.322 |
| Overall | 60 / 180 | 0.354 | 1.495 | 0.330 |

## SI generated-candidate diagnostics

| Group | n | Mean abs delta log10 | Median abs delta log10 | Median PolyBERT/static cNE0 | Median cNE0/NE |
|---|---:|---:|---:|---:|---:|
| TT-top | 10 | 0.283 | 0.332 | 2.149 | 0.946 |
| TT-middle | 10 | 0.433 | 0.425 | 2.664 | 0.920 |
| TT-bottom | 10 | 0.333 | 0.363 | 0.832 | 0.853 |
| LT-top | 10 | 0.538 | 0.533 | 0.371 | 0.527 |
| LT-middle | 10 | 0.318 | 0.342 | 1.294 | 0.293 |
| LT-bottom | 10 | 0.217 | 0.250 | 1.776 | 0.294 |
| Overall | 60 | 0.354 | 0.338 | 1.495 | 0.640 |

## SI generated-candidate log10 static cNE0

Percentile bootstrap intervals use 5000 candidate-level resamples. The seed is recorded in the CSV.

| Group | n | Mean log10 static cNE0 | Mean 95% CI | Median log10 static cNE0 | Median 95% CI |
|---|---:|---:|---:|---:|---:|
| TT-top | 10 | -3.033 | -3.144 to -2.907 | -3.100 | -3.172 to -2.833 |
| TT-middle | 10 | -3.463 | -3.638 to -3.285 | -3.500 | -3.726 to -3.180 |
| TT-bottom | 10 | -3.621 | -3.849 to -3.411 | -3.556 | -3.932 to -3.310 |
| LT-top | 10 | -3.213 | -3.481 to -2.948 | -3.158 | -3.572 to -2.900 |
| LT-middle | 10 | -4.624 | -4.833 to -4.389 | -4.638 | -4.879 to -4.387 |
| LT-bottom | 10 | -5.200 | -5.261 to -5.133 | -5.234 | -5.280 to -5.134 |

## SI pairwise generated-group tests

| Comparison | Median delta log10 | Fold difference | Cliff's delta | Exact p |
|---|---:|---:|---:|---:|
| TT-top vs TT-middle | 0.400 | 2.51 | 0.720 | 0.0052 |
| TT-top vs TT-bottom | 0.456 | 2.86 | 0.900 | 0.000206 |
| TT-top vs LT-top | 0.058 | 1.14 | 0.260 | 0.353 |
| TT-top vs LT-middle | 1.538 | 34.5 | 1.000 | 1.08e-05 |
| TT-top vs LT-bottom | 2.135 | 136 | 1.000 | 1.08e-05 |
| TT-middle vs TT-bottom | 0.056 | 1.14 | 0.200 | 0.481 |
| TT-middle vs LT-top | -0.342 | 0.455 | -0.340 | 0.218 |
| TT-middle vs LT-middle | 1.138 | 13.7 | 1.000 | 1.08e-05 |
| TT-middle vs LT-bottom | 1.734 | 54.3 | 1.000 | 1.08e-05 |
| TT-bottom vs LT-top | -0.398 | 0.4 | -0.520 | 0.0524 |
| TT-bottom vs LT-middle | 1.082 | 12.1 | 0.920 | 0.00013 |
| TT-bottom vs LT-bottom | 1.678 | 47.7 | 1.000 | 1.08e-05 |
| LT-top vs LT-middle | 1.480 | 30.2 | 0.980 | 2.17e-05 |
| LT-top vs LT-bottom | 2.076 | 119 | 1.000 | 1.08e-05 |
| LT-middle vs LT-bottom | 0.596 | 3.95 | 0.920 | 0.00013 |

## SI top five generated candidates by corrected static cNE0

| ID | Group | Repeat-unit string | Rank | PolyBERT / S cm-1 | Static cNE0 / S cm-1 |
|---:|---|---|---:|---:|---:|
| 920003 | LT-top | [*]CNCC([*])=O | 4 | 2.798e-04 | 3.602e-03 |
| 910000 | TT-top | [*]CNOCCOCCOCOCCOCCONCOC[*] | 1 | 1.850e-03 | 2.102e-03 |
| 910001 | TT-top | [*]COCCOCNOCCOCCOCCSCCOC[*] | 2 | 1.825e-03 | 1.730e-03 |
| 920007 | LT-top | [*]COCCNC([*])=O | 8 | 2.049e-04 | 1.690e-03 |
| 910002 | TT-top | [*]COCCOCNOCCOCCOCCOCOCCOC[*] | 3 | 1.749e-03 | 1.470e-03 |

## Source files

- `data/generated/generated_static_cNE0_replica_180.csv`
- `data/generated/generated_static_cNE0_candidate_mean_60.csv`
- `data/generated/generated_group_summary_candidate_level.csv`
- `data/generated/generated_pairwise_group_tests.csv`
- `data/reference/reference_static_cNE0_reassessment_108.csv`
- `data/reference/reference_group_summary_trajectory_level.csv`
- `data/reference/reference_overall_statistics.csv`
- `analysis/manuscript_static_cne0.py`
