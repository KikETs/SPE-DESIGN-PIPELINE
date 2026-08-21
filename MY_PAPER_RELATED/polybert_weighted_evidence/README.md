# Weighted PolyBERT Evidence

This folder contains the interval-weighted PolyBERT analysis used for
surrogate-screening evidence.

Tracked content:

- `scripts/train_polybert_weighted_interval.py`
- `scripts/predict_weighted_generated_candidates.py`
- `tables/`: compact model-selection, threshold, tail, top-k, and candidate
  summary tables.
- `figures_data/`: CSV data used to build diagnostic figures.
- `source_data/`: compact source CSV files and cached embeddings used by the
  weighted-analysis script.

Model selection uses four-fold OOF predictions grouped by
`canonical_psmiles` (6,026 groups across 6,270 rows). No canonical structure
group is allowed to cross folds. The compact baseline/selected-weighted OOF
release is `tables/grouped_oof_predictions_selected.csv`; split and model IDs
are recorded in `analysis_notes/canonical_grouped_cv_provenance.json`.

Not tracked:

- `tables/weighted_oof_predictions_all.csv`, because it exceeds normal GitHub
  file-size limits. Regenerate it with:

```bash
python MY_PAPER_RELATED/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py
```

Generated-candidate weighted predictions can be regenerated on a CUDA-capable
machine after the OOF analysis with:

```bash
python MY_PAPER_RELATED/polybert_weighted_evidence/scripts/predict_weighted_generated_candidates.py \
  --device cuda \
  --batch-size 256
```

The generated embedding cache is intentionally excluded from Git. The
candidate-level weighted predictions, summary statistics, provenance hashes,
and manuscript-ready Supplementary Table S9 are tracked.
The selected weighted model is refit on all 6,270 labeled rows before
generated-candidate inference; generated predictions are not OOF values.

The same command also scores the final target-conditioned generated pool
(44,999 LOW and 2,126 HIGH candidates) and joins all 60 MD-selected candidates.
The tracked outputs are:

- `tables/weighted_generated_condition_predictions_47125.csv`
- `tables/weighted_generated_condition_summary.csv`
- `tables/weighted_generated_md_selection_60.csv`
