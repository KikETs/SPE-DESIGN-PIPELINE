# Final surrogate split

`canonical_grouped_split_6270.csv` is a projection of the archived final split,
not a newly assigned split. It adds deterministic SHA256 group identifiers and
the six quantile-bin labels used by the original assignment.

The archived algorithm uses endpoint-normalized/canonical PSMILES as groups and
`StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)` over six
`log10_cond` quantile bins. Validation requires 6,270 rows, 6,026 groups, four
folds, and zero group overlap.

Source and reproducer:
`MY_PAPER_RELATED/polybert_con/train_polybert_conductivity_4fold.py`.

