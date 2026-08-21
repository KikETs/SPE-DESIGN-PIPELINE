# Machine-learning reproducibility index

## Workflow entry points

1. Dataset/canonicalization: `MY_PAPER_RELATED/polybert_con/train_polybert_conductivity_4fold.py`
   and `MY_PAPER_RELATED/selfies-psmiles/`.
2. TransCVAE: `MY_PAPER_RELATED/MODELS/models/Trans_TransCVAE_PSMILES.py`,
   training notebooks, and `MY_PAPER_RELATED/MODELS/utils/generate_TransCVAE_PSMILES.py`.
3. Baseline PolyBERT/Ridge grouped OOF: `train_polybert_conductivity_4fold.py`.
4. Weighted OOF selection:
   `MY_PAPER_RELATED/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py`.
5. Full-data weighted deployment:
   `MY_PAPER_RELATED/polybert_weighted_evidence/scripts/predict_weighted_generated_candidates.py`.
6. Final MD selection:
   `MY_PAPER_RELATED/gromacs_eval_pred_conductivity/github_results/latest_notebook_manifest_60/sample_manifest.csv`.

The generated embedding cache is intentionally not tracked. Precomputed
machine-readable predictions and model-selection provenance are tracked.

