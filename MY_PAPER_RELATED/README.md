# MY_PAPER_RELATED

Curated reproducibility package for the paper-related polymer generation and
surrogate-screening workflow.

## Directory Map

- `MODELS/`: integrated generation models, training/evaluation notebooks, input
  data, and FCD summary outputs.
- `polybert_con/`: PolyBERT conductivity predictor script and generated-candidate
  screening outputs.
- `revised/polybert_weighted_evidence/`: weighted PolyBERT analysis scripts,
  figure data, and compact result tables.
- `selfies-psmiles/`: local dependency used for endpoint-aware SELFIES/PSMILES
  handling.

## Reproducibility Notes

- Notebook outputs are stripped so results can be regenerated from code.
- Large learned weights and cache tensors are excluded from Git tracking.
- `MODELS/FCD_runs/` keeps compact result CSV/Markdown/Tex outputs.
- Training notebooks write new checkpoints into `MODELS/checkpoints/`.
- The full weighted OOF prediction table is not tracked because it is larger
  than normal GitHub file-size limits; regenerate it from the weighted evidence
  script when needed.
