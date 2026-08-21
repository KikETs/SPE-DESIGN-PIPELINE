
# -*- coding: utf-8 -*-
"""
polyBERT (HF) + Regression head for ionic conductivity (log10 scale)
- 4-fold CV with enforced distribution of top-10 highest-conductivity samples across folds
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
import argparse, math, os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def to_psmiles(s: str) -> str:
    if not isinstance(s, str):
        return ""
    tmp = s.replace("[*]", "__STAR__")
    tmp = tmp.replace("*", "[*]")
    tmp = tmp.replace("__STAR__", "[*]")
    return tmp

def make_folds(df: pd.DataFrame, k: int = 4, top_n: int = 10, seed: int = 42) -> np.ndarray:
    """Assign each row to a fold id in [0..k-1], ensuring top_n (by CONDUCTIVITY) are evenly distributed."""
    n = len(df)
    fold = np.full(n, -1, dtype=int)
    top_idx = df["CONDUCTIVITY"].nlargest(top_n).index.to_numpy()
    for i, idx in enumerate(top_idx):
        fold[idx] = i % k

    remaining = np.where(fold < 0)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    chunks = np.array_split(remaining, k)
    for i in range(k):
        fold[chunks[i]] = i
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default=None, help="e.g., 'cuda' or 'cpu' (SentenceTransformers device)")
    ap.add_argument("--regressor", choices=["ridge","mlp"], default="ridge")
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--mlp_hidden", type=int, default=256)
    ap.add_argument("--mlp_max_iter", type=int, default=500)
    ap.add_argument("--cache_embeddings", action="store_true", help="Cache embeddings to outdir/embeddings.npy")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "SMILES" not in df.columns or "CONDUCTIVITY" not in df.columns:
        raise ValueError("CSV must contain columns: SMILES, CONDUCTIVITY")

    df = df.copy()
    df["PSMILES"] = df["SMILES"].map(to_psmiles)
    df["log10_cond"] = np.log10(df["CONDUCTIVITY"].astype(float))

    # Fold assignment
    df["fold"] = make_folds(df, k=args.kfold, top_n=args.top_n, seed=args.seed)
    df.to_csv(outdir / "fold_assignment.csv", index=False)

    # Report top-N distribution
    top_idx = df["CONDUCTIVITY"].nlargest(args.top_n).index.to_numpy()
    top_counts = df.loc[top_idx, "fold"].value_counts().sort_index()
    with open(outdir / "topN_distribution.txt", "w", encoding="utf-8") as f:
        f.write("Top-N fold distribution (by CONDUCTIVITY):\n")
        for k in range(args.kfold):
            f.write(f"  fold {k}: {int(top_counts.get(k,0))}\n")

    # Compute embeddings (cache optional)
    emb_path = outdir / "embeddings.npy"
    if args.cache_embeddings and emb_path.exists():
        X = np.load(emb_path)
    else:
        X = compute_polybert_embeddings(df["PSMILES"].tolist(), batch_size=args.batch_size, device=args.device)
        if args.cache_embeddings:
            np.save(emb_path, X)

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
    out = df[["Trajectory ID","SMILES","PSMILES","CONDUCTIVITY","log10_cond","fold"]].copy()
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
