
# -*- coding: utf-8 -*-
"""
polyBERT (HF) + Regression head for ionic conductivity (log10 scale)
- Canonical-structure-grouped 4-fold CV with log-conductivity stratification
- Input: PSMILES-like strings (we convert "*" -> "[*]" by default)

Data expected (CSV):
- SMILES (polymer repeat-unit with "*" end markers)
- CONDUCTIVITY (float > 0)

References:
- polyBERT HF model card: maps PSMILES to 600-d fingerprints. https://huggingface.co/kuelumbus/polyBERT
- polyBERT paper: https://www.nature.com/articles/s41467-023-39868-6

Usage:
  pip install -U numpy pandas scikit-learn torch sentence-transformers tqdm

  python train_polybert_conductivity_4fold.py \
    --csv simulation-trajectory-aggregate.csv \
    --outdir runs/polybert_cv \
    --batch_size 64 \
    --regressor ridge

Notes:
- Requires internet (or cached HF model) to download "kuelumbus/polyBERT".
- We do NOT fine-tune polyBERT by default (feature extraction). This is usually sufficient for a screening model.
  If you want end-to-end fine-tuning, see the TODO section at the bottom.
"""
from __future__ import annotations
import argparse, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    from psmiles import PolymerSmiles as PS
except ModuleNotFoundError:
    vendor_root = Path(__file__).resolve().parents[2] / "vendor"
    sys.path.insert(
        0, str(vendor_root / "canonicalize_psmiles-0.1.2-py3-none-any.whl")
    )
    sys.path.insert(0, str(vendor_root / "psmiles_local"))
    from psmiles import PolymerSmiles as PS

def to_psmiles(s: str) -> str:
    if not isinstance(s, str):
        return ""
    tmp = s.replace("[*]", "__STAR__")
    tmp = tmp.replace("*", "[*]")
    tmp = tmp.replace("__STAR__", "[*]")
    return tmp

def canonicalize_psmiles(s: str) -> str:
    psmiles = to_psmiles(s)
    canonical = PS(psmiles).canonicalize.psmiles
    if canonical.count("[*]") != 2:
        raise ValueError(f"Expected exactly two [*] endpoints: {s}")
    return canonical


def make_folds(
    df: pd.DataFrame,
    k: int = 4,
    strat_bins: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """Assign canonical structures to stratified folds without group overlap."""
    y_bins = pd.qcut(df["log10_cond"], q=strat_bins, labels=False, duplicates="drop")
    y_bins = np.asarray(y_bins, dtype=int)
    groups = df["canonical_psmiles"].astype(str).to_numpy()
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for fold_id, (_, holdout_idx) in enumerate(cv.split(df, y_bins, groups=groups)):
        fold[holdout_idx] = fold_id

    if np.any(fold < 0):
        raise RuntimeError("StratifiedGroupKFold did not assign every row")
    group_fold_counts = (
        pd.DataFrame({"group": groups, "fold": fold})
        .groupby("group")["fold"]
        .nunique()
    )
    if group_fold_counts.gt(1).any():
        raise RuntimeError("Canonical-structure leakage detected across folds")
    return fold

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def compute_polybert_embeddings(texts: list[str], batch_size: int = 64, device: str | None = None) -> np.ndarray:
    """
    Returns: [N, 600] embeddings
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("kuelumbus/polyBERT", device=device)  # downloads if not cached
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return emb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to simulation-trajectory-aggregate.csv")
    ap.add_argument("--outdir", default="polybert_cv_out", help="Output directory")
    ap.add_argument("--kfold", type=int, default=4)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--strat_bins", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default=None, help="e.g., 'cuda' or 'cpu' (SentenceTransformers device)")
    ap.add_argument("--regressor", choices=["ridge","mlp"], default="ridge")
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--mlp_hidden", type=int, default=256)
    ap.add_argument("--mlp_max_iter", type=int, default=500)
    ap.add_argument("--cache_embeddings", action="store_true", help="Cache embeddings to outdir/embeddings.npy")
    ap.add_argument(
        "--embeddings_path",
        default=None,
        help="Optional precomputed embedding .npy aligned row-for-row with the input CSV",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "SMILES" not in df.columns or "CONDUCTIVITY" not in df.columns:
        raise ValueError("CSV must contain columns: SMILES, CONDUCTIVITY")

    df = df.copy()
    df["PSMILES"] = df["SMILES"].map(to_psmiles)
    df["canonical_psmiles"] = df["SMILES"].map(canonicalize_psmiles)
    df["log10_cond"] = np.log10(df["CONDUCTIVITY"].astype(float))

    # Fold assignment
    df["fold"] = make_folds(
        df,
        k=args.kfold,
        strat_bins=args.strat_bins,
        seed=args.seed,
    )
    df.to_csv(outdir / "fold_assignment.csv", index=False)

    # Report fold balance and top-N distribution.
    top_idx = df["CONDUCTIVITY"].nlargest(args.top_n).index.to_numpy()
    top_counts = df.loc[top_idx, "fold"].value_counts().sort_index()
    with open(outdir / "topN_distribution.txt", "w", encoding="utf-8") as f:
        f.write("Top-N fold distribution (by CONDUCTIVITY):\n")
        for k in range(args.kfold):
            f.write(f"  fold {k}: {int(top_counts.get(k,0))}\n")

    split_rows = []
    for fold_id in range(args.kfold):
        subset = df[df["fold"].eq(fold_id)]
        split_rows.append(
            {
                "fold": fold_id,
                "n_rows": int(len(subset)),
                "n_canonical_groups": int(subset["canonical_psmiles"].nunique()),
                "n_top10": int(top_counts.get(fold_id, 0)),
                "log10_cond_min": float(subset["log10_cond"].min()),
                "log10_cond_median": float(subset["log10_cond"].median()),
                "log10_cond_max": float(subset["log10_cond"].max()),
            }
        )
    pd.DataFrame(split_rows).to_csv(outdir / "split_diagnostics.csv", index=False)

    # Compute embeddings (cache optional)
    emb_path = outdir / "embeddings.npy"
    if args.embeddings_path:
        X = np.load(args.embeddings_path)
    elif args.cache_embeddings and emb_path.exists():
        X = np.load(emb_path)
    else:
        X = compute_polybert_embeddings(df["PSMILES"].tolist(), batch_size=args.batch_size, device=args.device)
        if args.cache_embeddings:
            np.save(emb_path, X)
    if len(X) != len(df):
        raise ValueError(f"Embedding rows ({len(X)}) do not match input rows ({len(df)})")

    y = df["log10_cond"].to_numpy().astype(np.float32)

    # Choose regressor
    if args.regressor == "ridge":
        reg = Ridge(alpha=args.ridge_alpha, random_state=args.seed)
        model = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
    else:
        reg = MLPRegressor(
            hidden_layer_sizes=(args.mlp_hidden, args.mlp_hidden//2),
            activation="relu",
            random_state=args.seed,
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            n_iter_no_change=20,
        )
        model = Pipeline([("scaler", StandardScaler()), ("reg", reg)])

    rows = []
    preds_all = np.full_like(y, np.nan, dtype=float)

    for k in range(args.kfold):
        tr = np.where(df["fold"].to_numpy() != k)[0]
        va = np.where(df["fold"].to_numpy() == k)[0]

        model.fit(X[tr], y[tr])
        pred = model.predict(X[va]).astype(float)
        preds_all[va] = pred

        mae = float(mean_absolute_error(y[va], pred))
        r = rmse(y[va], pred)
        rows.append({
            "fold": k,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "mae_log10": mae,
            "rmse_log10": r,
            "mae_factor_approx": float(10**mae),   # multiplicative error scale
            "rmse_factor_approx": float(10**r),
        })

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "cv_metrics.csv", index=False)

    # Overall metrics (OOF)
    oof_mae = float(mean_absolute_error(y, preds_all))
    oof_rmse = rmse(y, preds_all)

    with open(outdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("polyBERT + regressor (4-fold OOF)\n")
        f.write(res.to_string(index=False))
        f.write("\n\n")
        f.write(f"OOF MAE (log10):  {oof_mae:.6f}  (~x{10**oof_mae:.2f})\n")
        f.write(f"OOF RMSE (log10): {oof_rmse:.6f} (~x{10**oof_rmse:.2f})\n")

    # Save per-sample predictions
    out = df[
        [
            "Trajectory ID",
            "SMILES",
            "PSMILES",
            "canonical_psmiles",
            "CONDUCTIVITY",
            "log10_cond",
            "fold",
        ]
    ].copy()
    out["pred_log10_cond"] = preds_all
    out.to_csv(outdir / "oof_predictions.csv", index=False)

    print(res)
    print(f"\nOOF MAE(log10)={oof_mae:.4f} (~x{10**oof_mae:.2f})")
    print(f"OOF RMSE(log10)={oof_rmse:.4f} (~x{10**oof_rmse:.2f})")

if __name__ == "__main__":
    main()

# TODO (optional): end-to-end fine-tuning
# - Wrap polyBERT as a Transformer encoder with a regression head, train with a small lr (e.g., 1e-5 to 5e-5),
#   and keep the same fold assignment file for fair comparison.
