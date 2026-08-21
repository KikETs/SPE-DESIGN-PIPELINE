# Weighted Candidate Counts by Model

| model | condition | weighted_predictions_available | generated_candidates | count_ge_3e-5 | count_ge_5e-5 | count_ge_1e-4 | count_ge_2e-4 | count_ge_3e-4 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline PolyBERT-Ridge | all | True | 32610 | 31547 | 27653 | 17072 | 7236 | 3567 | full 6,270-row deployment refit on the deduplicated generated pool |
| Interval-weighted Ridge | all | True | 32610 | 32388 | 30455 | 21052 | 9361 | 4568 | full 6,270-row deployment refit on the deduplicated generated pool |
