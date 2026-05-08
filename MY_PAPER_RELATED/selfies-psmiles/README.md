# SELFIES-PSMILES

**An unofficial, upstream-compatible extension of SELFIES with first-class PSMILES `*` endpoint support.**
This package is designed to **co-exist** with the original [`selfies`](https://github.com/aspuru-guzik-group/selfies) package—install both side by side and import whichever you need.

<p align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>
  <a href="https://github.com/KikETs/selfies-psmiles/graphs/commit-activity">
    <img alt="Maintained?" src="https://img.shields.io/badge/Maintained%3F-yes-blue.svg">
  </a>
  <a href="https://github.com/KikETs/selfies-psmiles/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/KikETs/selfies-psmiles.svg">
  </a>
</p>

---

## ✨ Key Features

- **Drop-in API** mirroring upstream: `encoder`, `decoder`, constraints and encoding utilities are re-exported.
- **PSMILES helpers**:
  - `encoder_psmiles(smiles, psmiles=True, ...)` — normalizes `* → [*]` before encoding.
  - `decoder_psmiles(selfies, psmiles=True, ...)` — restores `[*] → *` after decoding.
  - `normalize_psmiles_stars`, `denormalize_psmiles_stars` are available if you just want the transforms.
- **Co-exists with upstream**: import names won’t clash (`selfies_psmiles` vs `selfies`).
- **Opt-in behavior**: PSMILES handling is explicit; vanilla behavior remains unchanged unless you use the helpers.

> **Why this exists?** In PSMILES, polymer repeat-unit endpoints may be written as `[*]` **or** `*`. This library adds minimal pre/post-processing so you can keep the `*` style in your workflow without sacrificing the robustness of SELFIES.

---

## 📦 Installation

### From PyPI (when available)
```bash
pip install selfies-psmiles
```

### From GitHub
```bash
pip install "git+https://github.com/KikETs/selfies-psmiles.git"
```

## 🚀 Quick Start
```bash
# Upstream can still be used normally
import selfies as sf

# This package provides PSMILES-friendly helpers
import selfies_psmiles as sp

# Keep '*' endpoints in your I/O while enjoying SELFIES robustness
s = "*CCO*"
sf_str = sp.encoder_psmiles(s)          # SMILES/PSMILES -> SELFIES  (normalizes * → [*])
roundtrip = sp.decoder_psmiles(sf_str)  # SELFIES -> PSMILES/SMILES  (restores [*] → *)
assert roundtrip == "*CCO*"

# Vanilla upstream continues to work unchanged
benzene = "c1ccccc1"
benzene_sf = sf.encoder(benzene)
assert sf.decoder(benzene_sf) == "C1=CC=CC=C1"
```

## 🧩 API Surface
This package re-exports common upstream symbols so existing code feels familiar:

| Symbol                                                                                                           			| Purpose                         |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| ``encoder``, ``decoder``                                                                                             		| SMILES ↔ SELFIES translation    |
| ``get_preset_constraints``, ``get_semantic_constraints``, ``set_semantic_constraints``, ``get_semantic_robust_alphabet``  | Constraint & alphabet utilities |
| ``len_selfies``, ``split_selfies``, ``get_alphabet_from_selfies``                                                      	| SELFIES utilities               |
| ``selfies_to_encoding``, ``encoding_to_selfies``, ``batch_selfies_to_flat_hot``, ``batch_flat_hot_to_selfies``            | Encoding helpers                |
| ``EncoderError``, ``DecoderError``                                                                                   		| Exceptions                      |

### Extras (this package):

* normalize_psmiles_stars(smiles) -> smiles
* denormalize_psmiles_stars(smiles) -> smiles
* encoder_psmiles(smiles, *, psmiles=True, **kwargs) -> selfies
* decoder_psmiles(selfies, *, psmiles=True, **kwargs) -> smiles
All extra functions are opt-in and won’t affect upstream behavior unless you call them.

## 🧪 Testing
We use pytest. From the repository root:
```bash
pip install -e ".[test]"   # or just `pip install pytest`
pytest -q
```
Minimal round-trip checks you can try:
```bash
import selfies_psmiles as sp

# PSMILES endpoints
for p in ("*CC*", "*CCO*", "[*]CC[*]"):
    sf = sp.encoder_psmiles(p)
    back = sp.decoder_psmiles(sf)
    assert back in ("*CC*", "*CCO*") or back == p
```

## ⚙️ Compatibility Notes
* Python 3.8+ recommended.
* If your environment enforces strict SELFIES constraints, pass strict=False to the encoder/decoder (same as upstream) when working with endpoint placeholders.
* This project is unaffiliated with the upstream maintainers. It aims to track the upstream API but may evolve independently.

## 📚 Citation
If this package helps your work, please consider citing the original SELFIES papers:
* SELFIES (MLST, 2020) — Krenn et al., Machine Learning: Science and Technology 1, 045024 (2020).
* SELFIES Code Paper (2023) — A hands-on tutorial and API details.
And, for polymers, see the PSMILES specification regarding endpoint notation ([*] and *).

## 🙏 Acknowledgments

* Upstream: [aspuru-guzik-group/selfies](https://github.com/aspuru-guzik-group/selfies). We’re grateful to the authors and contributors of SELFIES for making a robust, ML-friendly molecular string representation widely available.
* This project is maintained independently and is not endorsed by the upstream team.

## 🪪 License

This repository includes and modifies code originally distributed under the Apache License 2.0.
See LICENSE in the repository root for details. Please retain copyright and
license notices in source files where modifications were made.

---

## Endpoint Framework (PI1M)

This repository now includes a reproducible deep learning framework for:
- canonical endpoint prediction (exactly two endpoints),
- constrained reconstruction with exactly two `[*]`,
- and evaluation with reproducible splits/metrics.

### Problem Definition
- Scope v1:
  - linear homopolymer repeat units only,
  - exactly two polymerization endpoints,
  - out-of-scope topologies are excluded and logged.
- Input:
  - PI1M p-SMILES records (`PI1M.csv` or `PI1M_v2.csv`).
- Output:
  - canonicalized base sequence + endpoint pair labels,
  - reconstructed canonical SELFIES-PSMILES with exactly two `[*]`.

### Repository Additions
- `data/`
  - `build_pi1m_dataset.py`
  - `canonical_labeler.py`
  - `filters.py`
  - `split_dataset.py`
- `models/`
  - `endpoint_pointer.py`
- `decode/`
  - `constrained_reconstruction.py`
  - `validity_checks.py`
- `train/`
  - `datasets.py`
  - `metrics.py`
  - `train_endpoint_model.py`
  - `eval_endpoint_model.py`
  - `error_analysis.py`
- `configs/`
  - `pi1m_endpoint_baseline.yaml`
  - `pi1m_endpoint_ablation.yaml`
- `scripts/`
  - `run_preprocess.sh`
  - `run_train.sh`
  - `run_eval.sh`
- `notebooks/`
  - `pi1m_endpoint_pipeline_logs.ipynb`

### Environment
- Required:
  - Python 3.8+
  - `torch`, `pandas`, `pyyaml`, `selfies-psmiles`
- Optional:
  - `psmiles` for canonicalization backend
  - `rdkit` for graph-equivalence metric

### Data Preparation
```bash
cd MY_PAPER_RELATED/selfies-psmiles
bash scripts/run_preprocess.sh ../../PI1M/PI1M_v2.csv
```

Debug/smoke mode:
```bash
DEBUG=1 bash scripts/run_preprocess.sh ../../PI1M/PI1M_v2.csv
```

Generated files:
- `outputs/pi1m_endpoint_dataset/train.jsonl`
- `outputs/pi1m_endpoint_dataset/valid.jsonl`
- `outputs/pi1m_endpoint_dataset/test.jsonl`
- `outputs/pi1m_endpoint_dataset/dataset_summary.json`
- `outputs/pi1m_endpoint_dataset/excluded_samples.jsonl` (if enabled)

### Training
```bash
cd MY_PAPER_RELATED/selfies-psmiles
bash scripts/run_train.sh configs/pi1m_endpoint_baseline.yaml
```

Debug/smoke mode:
```bash
DEBUG=1 bash scripts/run_train.sh configs/pi1m_endpoint_baseline.yaml
```

Training artifacts:
- `outputs/experiments/<run_name>/checkpoints/best.pt`
- `outputs/experiments/<run_name>/train_history.json`
- `outputs/experiments/<run_name>/train_summary.json`

### Evaluation
Constrained decoding:
```bash
bash scripts/run_eval.sh configs/pi1m_endpoint_baseline.yaml <PATH_TO_BEST_PT> constrained
```

Ablation (unconstrained decoding):
```bash
bash scripts/run_eval.sh configs/pi1m_endpoint_ablation.yaml <PATH_TO_BEST_PT> unconstrained
```

Evaluation artifacts:
- `outputs/eval/<eval_run>/metrics.json`
- `outputs/eval/<eval_run>/predictions.jsonl`
- `outputs/eval/<eval_run>/qualitative_samples.csv`
- `outputs/eval/<eval_run>/error_analysis.json`

### Metrics
Primary metrics:
- endpoint exact match accuracy
- endpoint pair accuracy
- exact-two-`[*]` reconstruction rate
- syntax validity rate
- canonical reconstruction exact match rate

Secondary metrics:
- round-trip success rate
- graph-equivalence rate (if RDKit available)
- augmentation consistency on equivalent variants
- per-length breakdown

### Reproducibility
- Deterministic split by hash + seed
- Seeded training/evaluation
- Config-driven runs
- Exclusion reasons are logged and auditable

### Notebook Runner
- `notebooks/pi1m_endpoint_pipeline_logs.ipynb`
  - runs preprocess/train/eval commands,
  - stores command logs in `outputs/notebook_logs/`,
  - and loads metrics/qualitative outputs in-notebook.
