# Weighted PolyBERT Evidence

This folder contains the interval-weighted PolyBERT analysis used for
surrogate-screening evidence.

Tracked content:

- `scripts/train_polybert_weighted_interval.py`
- `tables/`: compact model-selection, threshold, tail, top-k, and candidate
  summary tables.
- `figures_data/`: CSV data used to build diagnostic figures.
- `source_data/`: compact source CSV files and cached embeddings used by the
  weighted-analysis script.

Not tracked:

- `tables/weighted_oof_predictions_all.csv`, because it exceeds normal GitHub
  file-size limits. Regenerate it with:

```bash
python MY_PAPER_RELATED/revised/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py
```
