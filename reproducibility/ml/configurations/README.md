# Model configuration sources

Configurations remain next to their actual implementations to avoid stale
copies:

- TransCVAE model and notebook settings: `MY_PAPER_RELATED/MODELS/`.
- Baseline PolyBERT/Ridge arguments and defaults:
  `MY_PAPER_RELATED/polybert_con/train_polybert_conductivity_4fold.py`.
- Weighted candidate grid and selected configuration:
  `MY_PAPER_RELATED/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py`
  and `analysis_notes/canonical_grouped_cv_provenance.json`.
- Deployment model and full-6,270-row refit provenance:
  `MY_PAPER_RELATED/polybert_weighted_evidence/analysis_notes/weighted_generated_prediction_provenance.json`.

The selected weighted model identifier is
`smooth_sigmoid_tail_a6_t0p05__ridge_alpha_100`.

