#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def to_psmiles(s: str) -> str:
    if not isinstance(s, str):
        return ""
    tmp = s.replace("[*]", "__STAR__")
    tmp = tmp.replace("*", "[*]")
    tmp = tmp.replace("__STAR__", "[*]")
    return tmp


def resolve_repo_root(start: Path) -> Path:
    start = start.resolve()
    for base in [start] + list(start.parents):
        cand = base / "MY_PAPER_RELATED" / "MODELS" / "FCD_runs"
        if cand.exists():
            return base
    raise FileNotFoundError("Could not locate repo root containing MY_PAPER_RELATED/MODELS/FCD_runs")


def compute_polybert_embeddings(
    texts: list[str],
    batch_size: int,
    device: str | None,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("kuelumbus/polyBERT", device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return emb


def pick_smiles_col(frame: pd.DataFrame) -> str | None:
    for c in ("SMILES", "smiles", "mol_smiles"):
        if c in frame.columns:
            return c
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rebuild MODELS/FCD_runs/_conductivity_eval from generated CSVs.")
    ap.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root.")
    ap.add_argument(
        "--ref-csv",
        type=Path,
        default=Path("MY_PAPER_RELATED/polybert_con/simulation-trajectory-aggregate.csv"),
        help="Reference training CSV for conductivity regressor.",
    )
    ap.add_argument(
        "--embeddings",
        type=Path,
        default=Path("MY_PAPER_RELATED/revised/polybert_weighted_evidence/source_data/polybert_run/embeddings.npy"),
        help="Cached reference embeddings path.",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)

    fcd_runs_root = repo_root / "MY_PAPER_RELATED" / "MODELS" / "FCD_runs"
    out_dir = fcd_runs_root / "_conductivity_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_csv = args.ref_csv if args.ref_csv.is_absolute() else (repo_root / args.ref_csv)
    emb_path = args.embeddings if args.embeddings.is_absolute() else (repo_root / args.embeddings)

    df = pd.read_csv(ref_csv).copy()
    required_cols = {"SMILES", "CONDUCTIVITY"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Reference CSV missing columns: {sorted(missing)}")

    df["PSMILES"] = df["SMILES"].map(to_psmiles)
    y = np.log10(df["CONDUCTIVITY"].astype(float).to_numpy(dtype=np.float32))

    if emb_path.exists():
        X = np.load(emb_path)
        if X.shape[0] != len(df):
            raise ValueError(
                f"Cached embedding row mismatch: embeddings={X.shape[0]}, ref_rows={len(df)} @ {emb_path}"
            )
    else:
        X = compute_polybert_embeddings(df["PSMILES"].tolist(), batch_size=args.batch_size, device=args.device)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, X)

    reg = Ridge(alpha=args.ridge_alpha, random_state=42)
    model = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
    model.fit(X, y)

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("kuelumbus/polyBERT", device=args.device)

    def predict_cond(smiles_series: pd.Series) -> np.ndarray:
        psmiles = smiles_series.astype(str).map(to_psmiles)
        emb = encoder.encode(
            psmiles.tolist(),
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        pred_log10 = model.predict(emb).astype(float)
        pred_cond = np.power(10.0, pred_log10)
        pred_cond = np.asarray(pred_cond, dtype=float)
        pred_cond = pred_cond[np.isfinite(pred_cond) & (pred_cond > 0)]
        return pred_cond

    exclude_patterns = ["[*][*]"]
    rows_repeat: list[dict] = []

    eval_models = [
        "TransCVAE",
        "TransCVAE_PSMILES",
        "LSTM_CVAE",
        "LSTM_CVAE_PSMILES",
        "Encoder_Only",
        "Encoder_Only_PSMILES",
    ]
    eval_cond_dirs = {"LOW": "condz_low", "HIGH": "condz_high"}

    for model_name in eval_models:
        model_dir = fcd_runs_root / model_name
        if not model_dir.exists():
            print(f"[WARN] missing model dir: {model_dir}")
            continue

        for cond_label, cond_dir_name in eval_cond_dirs.items():
            cond_dir = model_dir / cond_dir_name
            if not cond_dir.exists():
                print(f"[WARN] missing cond dir: {cond_dir}")
                continue

            repeat_dirs = sorted(p for p in cond_dir.glob("repeat_*") if p.is_dir())
            if not repeat_dirs:
                print(f"[WARN] no repeat_* dirs under: {cond_dir}")
                continue

            for repeat_dir in repeat_dirs:
                csv_path = repeat_dir / f"all_novel_smiles_{cond_dir_name}.csv"
                if not csv_path.exists():
                    print(f"[SKIP] missing csv: {csv_path}")
                    continue

                frame = pd.read_csv(csv_path).copy()
                smiles_col = pick_smiles_col(frame)
                if smiles_col is None:
                    print(f"[SKIP] no smiles column: {csv_path}")
                    continue

                mask_bad = pd.Series(False, index=frame.index)
                for pat in exclude_patterns:
                    mask_bad = mask_bad | frame[smiles_col].astype(str).str.contains(pat, regex=False, na=False)
                frame = frame.loc[~mask_bad].copy()
                frame = frame.drop_duplicates(subset=[smiles_col]).copy()

                if frame.empty:
                    print(f"[SKIP] empty after filtering: {csv_path}")
                    continue

                pred_cond = predict_cond(frame[smiles_col])
                if pred_cond.size == 0:
                    print(f"[SKIP] no positive predictions: {csv_path}")
                    continue

                logv = np.log10(pred_cond)
                rows_repeat.append(
                    {
                        "model": model_name,
                        "condition": cond_label,
                        "repeat": repeat_dir.name,
                        "source_csv": str(csv_path),
                        "n_samples": int(pred_cond.size),
                        "mean_log10_cond": float(np.mean(logv)),
                        "median_log10_cond": float(np.median(logv)),
                        "q90_log10_cond": float(np.quantile(logv, 0.9)),
                        "hit_rate_ge_1e-4": float(np.mean(pred_cond >= 1e-4)),
                        "hit_rate_ge_1e-3": float(np.mean(pred_cond >= 1e-3)),
                    }
                )

    # minGPT (stored in a different layout)
    min_dir = fcd_runs_root / "minGPT_cond_repeat5_of_50_results" / "generated_per_repeat"
    if min_dir.exists():
        pat = re.compile(r"generated_cond(?P<cid>\d+)_repeat(?P<rid>\d+)_all\.csv$")
        for csv_path in sorted(min_dir.glob("generated_cond*_repeat*_all.csv")):
            m = pat.match(csv_path.name)
            if not m:
                continue
            cid = int(m.group("cid"))
            rid = int(m.group("rid"))
            cond_label = "LOW" if cid == 0 else "HIGH"
            repeat_name = f"repeat_{rid:02d}"

            frame = pd.read_csv(csv_path).copy()
            smiles_col = pick_smiles_col(frame)
            if smiles_col is None:
                print(f"[SKIP] no smiles column: {csv_path}")
                continue

            if "validity" in frame.columns:
                frame = frame.loc[frame["validity"].astype(str) == "ok"].copy()
            if "diversity" in frame.columns:
                frame = frame.loc[frame["diversity"].astype(str) == "novel"].copy()

            mask_bad = pd.Series(False, index=frame.index)
            for ep in exclude_patterns:
                mask_bad = mask_bad | frame[smiles_col].astype(str).str.contains(ep, regex=False, na=False)
            frame = frame.loc[~mask_bad].copy()
            frame = frame.drop_duplicates(subset=[smiles_col]).copy()

            if frame.empty:
                print(f"[SKIP] empty after filtering: {csv_path}")
                continue

            pred_cond = predict_cond(frame[smiles_col])
            if pred_cond.size == 0:
                print(f"[SKIP] no positive predictions: {csv_path}")
                continue

            logv = np.log10(pred_cond)
            rows_repeat.append(
                {
                    "model": "minGPT",
                    "condition": cond_label,
                    "repeat": repeat_name,
                    "source_csv": str(csv_path),
                    "n_samples": int(pred_cond.size),
                    "mean_log10_cond": float(np.mean(logv)),
                    "median_log10_cond": float(np.median(logv)),
                    "q90_log10_cond": float(np.quantile(logv, 0.9)),
                    "hit_rate_ge_1e-4": float(np.mean(pred_cond >= 1e-4)),
                    "hit_rate_ge_1e-3": float(np.mean(pred_cond >= 1e-3)),
                }
            )
    else:
        print(f"[WARN] minGPT repeat dir not found: {min_dir}")

    by_repeat = pd.DataFrame(rows_repeat)
    if by_repeat.empty:
        raise RuntimeError("No rows were generated for conductivity_eval_by_repeat.")

    by_repeat = by_repeat.sort_values(["condition", "model", "repeat"]).reset_index(drop=True)

    summary = (
        by_repeat.groupby(["model", "condition"], as_index=False)
        .agg(
            repeats=("repeat", "nunique"),
            n_samples_mean=("n_samples", "mean"),
            mean_log10_cond_mean=("mean_log10_cond", "mean"),
            mean_log10_cond_std=("mean_log10_cond", "std"),
            median_log10_cond_mean=("median_log10_cond", "mean"),
            median_log10_cond_std=("median_log10_cond", "std"),
            q90_log10_cond_mean=("q90_log10_cond", "mean"),
            q90_log10_cond_std=("q90_log10_cond", "std"),
            hit_1e4_mean=("hit_rate_ge_1e-4", "mean"),
            hit_1e4_std=("hit_rate_ge_1e-4", "std"),
            hit_1e3_mean=("hit_rate_ge_1e-3", "mean"),
            hit_1e3_std=("hit_rate_ge_1e-3", "std"),
        )
        .sort_values(["condition", "model"])
        .reset_index(drop=True)
    )

    by_repeat_path = out_dir / "conductivity_eval_by_repeat.csv"
    summary_path = out_dir / "conductivity_eval_summary.csv"
    by_repeat.to_csv(by_repeat_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"saved: {by_repeat_path}")
    print(f"saved: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
