from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures_data"
NOTES = ROOT / "analysis_notes"
SRC_RUN = ROOT / "source_data" / "polybert_run"
SRC_CON = ROOT / "source_data" / "polybert_con"

for directory in [TABLES, FIGURES, NOTES]:
    directory.mkdir(parents=True, exist_ok=True)

THRESHOLDS = {
    "3e-5": 3e-5,
    "5e-5": 5e-5,
    "1e-4": 1e-4,
    "2e-4": 2e-4,
    "3e-4": 3e-4,
}
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
TOP_KS = [10, 25, 50, 100, 200]


@dataclass(frozen=True)
class Scheme:
    scheme_id: str
    scheme_name: str
    weight_fn: Callable[[np.ndarray], np.ndarray]
    params: str = ""


def log_thr(value: float) -> float:
    return float(np.log10(value))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normalize_mean_one(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0:
        return np.ones_like(weights, dtype=float)
    return weights / mean


def weights_unweighted(y_train: np.ndarray) -> np.ndarray:
    return np.ones_like(y_train, dtype=float)


def weights_inverse_frequency_bins(y_train: np.ndarray, q: int = 5) -> np.ndarray:
    labels = pd.qcut(pd.Series(y_train), q=q, duplicates="drop")
    counts = labels.value_counts()
    weights = labels.map(lambda x: 1.0 / float(counts.loc[x])).astype(float).to_numpy()
    return normalize_mean_one(weights)


def weights_high_tail_step(y_train: np.ndarray) -> np.ndarray:
    t3 = log_thr(3e-5)
    t1 = log_thr(1e-4)
    t30 = log_thr(3e-4)
    weights = np.ones_like(y_train, dtype=float)
    weights[(y_train >= t3) & (y_train < t1)] = 1.5
    weights[(y_train >= t1) & (y_train < t30)] = 3.0
    weights[y_train >= t30] = 5.0
    return normalize_mean_one(weights)


def weights_aggressive_high_tail_step(y_train: np.ndarray) -> np.ndarray:
    t3 = log_thr(3e-5)
    t1 = log_thr(1e-4)
    t30 = log_thr(3e-4)
    weights = np.ones_like(y_train, dtype=float)
    weights[(y_train >= t3) & (y_train < t1)] = 2.0
    weights[(y_train >= t1) & (y_train < t30)] = 5.0
    weights[y_train >= t30] = 8.0
    return normalize_mean_one(weights)


def weights_recall_focused(y_train: np.ndarray) -> np.ndarray:
    t5 = log_thr(5e-5)
    t1 = log_thr(1e-4)
    weights = np.ones_like(y_train, dtype=float)
    weights[(y_train >= t5) & (y_train < t1)] = 3.0
    weights[y_train >= t1] = 5.0
    return normalize_mean_one(weights)


def make_sigmoid_weight(alpha: float, temperature: float) -> Callable[[np.ndarray], np.ndarray]:
    def _fn(y_train: np.ndarray) -> np.ndarray:
        weights = 1.0 + alpha * sigmoid((y_train - (-4.0)) / temperature)
        return normalize_mean_one(weights)

    return _fn


def build_schemes() -> list[Scheme]:
    schemes = [
        Scheme("baseline_unweighted", "Scheme A: baseline_unweighted", weights_unweighted),
        Scheme("inverse_frequency_bins", "Scheme B: inverse_frequency_bins", weights_inverse_frequency_bins, "q=5"),
        Scheme("high_tail_step", "Scheme C: high_tail_step", weights_high_tail_step),
        Scheme("aggressive_high_tail_step", "Scheme D: aggressive_high_tail_step", weights_aggressive_high_tail_step),
        Scheme("recall_focused_near_threshold", "Scheme F: recall_focused_near_threshold", weights_recall_focused),
    ]
    for alpha in [1, 2, 4, 6]:
        for temperature in [0.05, 0.10, 0.20]:
            scheme_id = f"smooth_sigmoid_tail_a{alpha}_t{temperature:g}".replace(".", "p")
            schemes.append(
                Scheme(
                    scheme_id,
                    "Scheme E: smooth_sigmoid_tail",
                    make_sigmoid_weight(float(alpha), float(temperature)),
                    f"threshold=-4; alpha={alpha}; temperature={temperature}",
                )
            )
    return schemes


def corr_values(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan"), float("nan")
    pearson = float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"))
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    return pearson, spearman


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = np.asarray(y_pred) - np.asarray(y_true)
    abs_err = np.abs(err)
    pearson, spearman = corr_values(y_true, y_pred)
    return {
        "mae_log10": float(mean_absolute_error(y_true, y_pred)),
        "rmse_log10": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "pearson": pearson,
        "spearman": spearman,
        "median_abs_error_log10": float(np.median(abs_err)),
        "p90_abs_error_log10": float(np.percentile(abs_err, 90)),
        "max_abs_error_log10": float(np.max(abs_err)),
    }


def threshold_rows(model_meta: dict[str, object], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, threshold in THRESHOLDS.items():
        lt = log_thr(threshold)
        true_pos = y_true >= lt
        pred_pos = y_pred >= lt
        tp = int(np.sum(true_pos & pred_pos))
        fp = int(np.sum(~true_pos & pred_pos))
        tn = int(np.sum(~true_pos & ~pred_pos))
        fn = int(np.sum(true_pos & ~pred_pos))
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else float("nan")
        pred_rate = (tp + fp) / len(y_true)
        true_rate = (tp + fn) / len(y_true)
        enrichment = precision / true_rate if true_rate and np.isfinite(precision) else float("nan")
        rows.append(
            {
                **model_meta,
                "threshold_label": label,
                "threshold_s_cm": threshold,
                "threshold_log10": lt,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_positive_rate": pred_rate,
                "true_positive_rate": true_rate,
                "enrichment": enrichment,
            }
        )
    return rows


def topk_rows(model_meta: dict[str, object], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    high_true = y_true >= -4.0
    prevalence = float(np.mean(high_true))
    order = np.argsort(-y_pred)
    for k in TOP_KS:
        idx = order[: min(k, len(order))]
        hits = int(np.sum(high_true[idx]))
        hit_rate = hits / len(idx) if len(idx) else float("nan")
        rows.append(
            {
                **model_meta,
                "k": k,
                "n_selected": int(len(idx)),
                "true_ge_1e4_hits": hits,
                "true_ge_1e4_hit_rate": hit_rate,
                "base_prevalence_true_ge_1e4": prevalence,
                "enrichment": hit_rate / prevalence if prevalence else float("nan"),
                "mean_true_log10_cond": float(np.mean(y_true[idx])) if len(idx) else float("nan"),
                "median_true_log10_cond": float(np.median(y_true[idx])) if len(idx) else float("nan"),
            }
        )
    return rows


def tail_rows(model_meta: dict[str, object], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    definitions = [
        ("true_top_5pct", y_true >= np.quantile(y_true, 0.95)),
        ("true_top_10pct", y_true >= np.quantile(y_true, 0.90)),
        ("true_ge_1e-4", y_true >= -4.0),
    ]
    for tail_name, mask in definitions:
        yt = y_true[mask]
        yp = y_pred[mask]
        metrics = regression_metrics(yt, yp) if len(yt) >= 2 else {}
        rows.append(
            {
                **model_meta,
                "tail_definition": tail_name,
                "n_tail": int(len(yt)),
                "tail_mae_log10": metrics.get("mae_log10", float("nan")),
                "tail_rmse_log10": metrics.get("rmse_log10", float("nan")),
                "tail_r2": metrics.get("r2", float("nan")),
                "tail_median_abs_error_log10": metrics.get("median_abs_error_log10", float("nan")),
                "tail_p90_abs_error_log10": metrics.get("p90_abs_error_log10", float("nan")),
            }
        )
    return rows


def decile_rows(model_meta: dict[str, object], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, object]]:
    tmp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    tmp["prediction_decile"] = pd.qcut(tmp["y_pred"], q=10, labels=False, duplicates="drop")
    rows: list[dict[str, object]] = []
    for decile, group in tmp.groupby("prediction_decile", dropna=False):
        rows.append(
            {
                **model_meta,
                "prediction_decile": int(decile) if pd.notna(decile) else -1,
                "count": int(len(group)),
                "mean_pred_log10": float(group["y_pred"].mean()),
                "mean_true_log10": float(group["y_true"].mean()),
                "std_true_log10": float(group["y_true"].std(ddof=1)),
                "mae_log10": float(mean_absolute_error(group["y_true"], group["y_pred"])),
                "min_pred_log10": float(group["y_pred"].min()),
                "max_pred_log10": float(group["y_pred"].max()),
            }
        )
    return rows


def markdown_table(df: pd.DataFrame, path: Path, title: str, max_rows: int | None = None) -> None:
    data = df.head(max_rows).copy() if max_rows else df.copy()
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if data.empty:
            f.write("No rows available.\n")
            return
        cols = list(data.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in data.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append("" if pd.isna(val) else f"{val:.6g}")
                else:
                    vals.append(str(val))
            f.write("| " + " | ".join(vals) + " |\n")


def write_inventory(oof: pd.DataFrame, embeddings: np.ndarray) -> None:
    source_files = []
    for p in sorted((ROOT / "source_data").rglob("*")):
        if p.is_file():
            source_files.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size})
    lines = [
        "# Weighted Predictor File Inventory",
        "",
        "## Detected Source Files",
        "",
    ]
    for item in source_files:
        lines.append(f"- `{item['path']}` ({item['bytes']} bytes)")
    lines.extend(
        [
            "",
            "## Fold Assignments",
            "",
            f"- Existing fold column detected in `polybert_run/oof_predictions.csv`: yes",
            f"- Number of folds: {sorted(oof['fold'].dropna().unique().tolist())}",
            f"- Samples per fold: {oof['fold'].value_counts().sort_index().to_dict()}",
            "",
            "## Embeddings",
            "",
            f"- Cached PolyBERT training embeddings available: yes",
            f"- Embedding shape: {tuple(embeddings.shape)}",
            "- Generated-candidate PolyBERT embeddings available: no detected file.",
            "- Because generated-candidate embeddings are absent and SentenceTransformers is not required for this script, weighted generated-candidate prediction is marked as not feasible in this evidence pass.",
            "",
            "## Existing OOF Predictions",
            "",
            "- Existing unweighted OOF predictions available: yes (`polybert_run/oof_predictions.csv`).",
            "- This script regenerates Ridge OOF predictions from cached embeddings and the same fold assignments for baseline and interval-weighted schemes.",
            "",
            "## Baseline Reproduction Command",
            "",
            "```powershell",
            "python revised/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py",
            "```",
        ]
    )
    (NOTES / "weighted_predictor_file_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_inputs() -> tuple[pd.DataFrame, np.ndarray]:
    oof_path = SRC_RUN / "oof_predictions.csv"
    emb_path = SRC_RUN / "embeddings.npy"
    if not oof_path.exists():
        raise FileNotFoundError(oof_path)
    if not emb_path.exists():
        raise FileNotFoundError(emb_path)
    oof = pd.read_csv(oof_path)
    embeddings = np.load(emb_path)
    if len(oof) != embeddings.shape[0]:
        raise ValueError(f"OOF rows ({len(oof)}) do not match embeddings ({embeddings.shape[0]}).")
    if "log10_cond" not in oof.columns or "fold" not in oof.columns:
        raise ValueError("OOF file must contain `log10_cond` and `fold` columns.")
    return oof, embeddings.astype(np.float64, copy=False)


def run_weighted_oof(oof: pd.DataFrame, X: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = oof["log10_cond"].to_numpy(dtype=float)
    folds = oof["fold"].to_numpy(dtype=int)
    fold_ids = sorted(np.unique(folds).tolist())
    sample_ids = oof["Trajectory ID"].to_numpy() if "Trajectory ID" in oof.columns else np.arange(len(oof))

    scaled_by_fold: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for fold in fold_ids:
        tr = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        scaled_by_fold[fold] = (tr, va, Xtr, Xva, y[tr])

    schemes = build_schemes()
    global_rows = []
    fold_rows = []
    threshold_all = []
    topk_all = []
    tail_all = []
    decile_all = []
    oof_rows = []

    total_configs = len(schemes) * len(ALPHAS)
    config_counter = 0
    for scheme in schemes:
        for ridge_alpha in ALPHAS:
            config_counter += 1
            model_id = f"{scheme.scheme_id}__ridge_alpha_{ridge_alpha:g}".replace(".", "p")
            print(f"[{config_counter}/{total_configs}] {model_id}")
            preds = np.full(len(y), np.nan, dtype=float)
            for fold in fold_ids:
                tr, va, Xtr, Xva, ytr = scaled_by_fold[fold]
                weights = scheme.weight_fn(ytr)
                model = Ridge(alpha=ridge_alpha)
                model.fit(Xtr, ytr, sample_weight=weights)
                pred = model.predict(Xva).astype(float)
                preds[va] = pred

                fm = regression_metrics(y[va], pred)
                fold_rows.append(
                    {
                        "model_id": model_id,
                        "scheme_id": scheme.scheme_id,
                        "scheme_name": scheme.scheme_name,
                        "scheme_params": scheme.params,
                        "regressor": "Ridge",
                        "ridge_alpha": ridge_alpha,
                        "fold": fold,
                        "n_val": int(len(va)),
                        "n_high_true_ge_1e4": int(np.sum(y[va] >= -4.0)),
                        **fm,
                    }
                )

            meta = {
                "model_id": model_id,
                "scheme_id": scheme.scheme_id,
                "scheme_name": scheme.scheme_name,
                "scheme_params": scheme.params,
                "regressor": "Ridge",
                "ridge_alpha": ridge_alpha,
            }
            gm = regression_metrics(y, preds)
            global_rows.append(
                {
                    **meta,
                    "n_samples": int(len(y)),
                    "n_high_true_ge_1e4": int(np.sum(y >= -4.0)),
                    **gm,
                }
            )
            threshold_all.extend(threshold_rows(meta, y, preds))
            topk_all.extend(topk_rows(meta, y, preds))
            tail_all.extend(tail_rows(meta, y, preds))
            decile_all.extend(decile_rows(meta, y, preds))

            oof_rows.append(
                pd.DataFrame(
                    {
                        "model_id": model_id,
                        "scheme_id": scheme.scheme_id,
                        "scheme_name": scheme.scheme_name,
                        "scheme_params": scheme.params,
                        "regressor": "Ridge",
                        "ridge_alpha": ridge_alpha,
                        "sample_id": sample_ids,
                        "fold": folds,
                        "y_true_log10_conductivity": y,
                        "y_pred_log10_conductivity": preds,
                        "residual_log10": preds - y,
                        "abs_error_log10": np.abs(preds - y),
                        "is_high_true_1e4": y >= -4.0,
                        "is_high_pred_1e4": preds >= -4.0,
                    }
                )
            )

    global_df = pd.DataFrame(global_rows)
    fold_df = pd.DataFrame(fold_rows)
    threshold_df = pd.DataFrame(threshold_all)
    topk_df = pd.DataFrame(topk_all)
    tail_df = pd.DataFrame(tail_all)
    decile_df = pd.DataFrame(decile_all)
    oof_all_df = pd.concat(oof_rows, ignore_index=True)
    return global_df, threshold_df, topk_df, fold_df, tail_df, decile_df, oof_all_df


def add_selection_scores(global_df: pd.DataFrame, threshold_df: pd.DataFrame, topk_df: pd.DataFrame, tail_df: pd.DataFrame) -> pd.DataFrame:
    thr1 = threshold_df[threshold_df["threshold_label"] == "1e-4"][
        ["model_id", "precision", "recall", "f1", "enrichment"]
    ].rename(
        columns={
            "precision": "precision_at_1e4",
            "recall": "recall_at_1e4",
            "f1": "f1_at_1e4",
            "enrichment": "threshold_enrichment_at_1e4",
        }
    )
    top100 = topk_df[topk_df["k"] == 100][
        ["model_id", "true_ge_1e4_hit_rate", "enrichment", "mean_true_log10_cond", "median_true_log10_cond"]
    ].rename(
        columns={
            "true_ge_1e4_hit_rate": "top100_true_ge_1e4_hit_rate",
            "enrichment": "top100_enrichment",
            "mean_true_log10_cond": "top100_mean_true_log10_cond",
            "median_true_log10_cond": "top100_median_true_log10_cond",
        }
    )
    tail = tail_df[tail_df["tail_definition"] == "true_ge_1e-4"][
        ["model_id", "tail_mae_log10", "tail_rmse_log10"]
    ].rename(columns={"tail_mae_log10": "tail_mae_true_ge_1e4", "tail_rmse_log10": "tail_rmse_true_ge_1e4"})
    selection = global_df.merge(thr1, on="model_id", how="left").merge(top100, on="model_id", how="left").merge(tail, on="model_id", how="left")

    baseline_row = selection[selection["model_id"] == "baseline_unweighted__ridge_alpha_1"]
    if baseline_row.empty:
        baseline_row = selection[selection["scheme_id"] == "baseline_unweighted"].sort_values("mae_log10").head(1)
    baseline = baseline_row.iloc[0]
    selection["baseline_model_id"] = str(baseline["model_id"])
    selection["delta_mae_vs_baseline"] = selection["mae_log10"] - float(baseline["mae_log10"])
    selection["delta_rmse_vs_baseline"] = selection["rmse_log10"] - float(baseline["rmse_log10"])
    selection["delta_spearman_vs_baseline"] = selection["spearman"] - float(baseline["spearman"])
    selection["delta_recall_at_1e4_vs_baseline"] = selection["recall_at_1e4"] - float(baseline["recall_at_1e4"])
    selection["delta_top100_enrichment_vs_baseline"] = selection["top100_enrichment"] - float(baseline["top100_enrichment"])
    selection["passes_screening_constraints"] = (
        (selection["mae_log10"] <= float(baseline["mae_log10"]) * 1.10)
        & (selection["rmse_log10"] <= float(baseline["rmse_log10"]) * 1.10)
        & (selection["spearman"] >= float(baseline["spearman"]) - 0.03)
        & (selection["precision_at_1e4"] >= 0.50)
    )

    def norm_high(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - lo) / (hi - lo)

    def norm_low(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (hi - series) / (hi - lo)

    selection["score_recall_at_1e4"] = norm_high(selection["recall_at_1e4"])
    selection["score_top100_enrichment"] = norm_high(selection["top100_enrichment"])
    selection["score_spearman"] = norm_high(selection["spearman"])
    selection["score_tail_mae_inverse"] = norm_low(selection["tail_mae_true_ge_1e4"])
    selection["score_global_mae_inverse"] = norm_low(selection["mae_log10"])
    selection["cej_screening_score"] = (
        0.25 * selection["score_recall_at_1e4"]
        + 0.25 * selection["score_top100_enrichment"]
        + 0.20 * selection["score_spearman"]
        + 0.15 * selection["score_tail_mae_inverse"]
        + 0.15 * selection["score_global_mae_inverse"]
    )

    selection["pareto_candidate"] = False
    values = selection[["recall_at_1e4", "top100_enrichment", "spearman", "tail_mae_true_ge_1e4", "mae_log10"]].to_numpy(dtype=float)
    for i in range(len(selection)):
        dominated = False
        for j in range(len(selection)):
            if i == j:
                continue
            better_or_equal = (
                values[j, 0] >= values[i, 0]
                and values[j, 1] >= values[i, 1]
                and values[j, 2] >= values[i, 2]
                and values[j, 3] <= values[i, 3]
                and values[j, 4] <= values[i, 4]
            )
            strictly_better = (
                values[j, 0] > values[i, 0]
                or values[j, 1] > values[i, 1]
                or values[j, 2] > values[i, 2]
                or values[j, 3] < values[i, 3]
                or values[j, 4] < values[i, 4]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        selection.loc[selection.index[i], "pareto_candidate"] = not dominated

    selection["recommended_role"] = "not recommended"
    eligible = selection[selection["passes_screening_constraints"]].copy()
    if not eligible.empty:
        best_id = eligible.sort_values("cej_screening_score", ascending=False).iloc[0]["model_id"]
        selection.loc[selection["model_id"] == best_id, "recommended_role"] = "best weighted screening candidate"
        baseline_id = baseline["model_id"]
        selection.loc[selection["model_id"] == baseline_id, "recommended_role"] = "unweighted baseline reference"
    return selection.sort_values(["passes_screening_constraints", "cej_screening_score"], ascending=[False, False])


def write_figures(oof_all: pd.DataFrame, selection: pd.DataFrame, threshold_df: pd.DataFrame, topk_df: pd.DataFrame, tail_df: pd.DataFrame, decile_df: pd.DataFrame) -> None:
    baseline_id = selection.iloc[0]["baseline_model_id"]
    best_row = selection[selection["recommended_role"] == "best weighted screening candidate"]
    best_id = best_row.iloc[0]["model_id"] if not best_row.empty else selection.iloc[0]["model_id"]
    selected_ids = sorted(set([baseline_id, best_id] + selection[selection["pareto_candidate"]]["model_id"].head(5).tolist()))

    oof_plot = oof_all[oof_all["model_id"].isin(selected_ids)].copy()
    oof_plot.to_csv(FIGURES / "figure_weighted_pred_vs_true.csv", index=False)

    bins = np.linspace(-1.0, 1.0, 81)
    hist_rows = []
    for model_id, group in oof_plot.groupby("model_id"):
        counts, edges = np.histogram(group["residual_log10"], bins=bins)
        for count, left, right in zip(counts, edges[:-1], edges[1:]):
            hist_rows.append({"model_id": model_id, "bin_left": left, "bin_right": right, "count": int(count)})
    pd.DataFrame(hist_rows).to_csv(FIGURES / "figure_weighted_residual_histogram.csv", index=False)

    threshold_df[threshold_df["model_id"].isin(selected_ids)].to_csv(FIGURES / "figure_weighted_threshold_recall_precision.csv", index=False)
    topk_df[topk_df["model_id"].isin(selected_ids)].to_csv(FIGURES / "figure_weighted_topk_enrichment.csv", index=False)

    trade = selection.merge(tail_df[tail_df["tail_definition"] == "true_ge_1e-4"][["model_id", "tail_mae_log10", "tail_rmse_log10"]], on="model_id", how="left", suffixes=("", "_dup"))
    trade.to_csv(FIGURES / "figure_weighted_tradeoff_global_vs_tail.csv", index=False)
    decile_df[decile_df["model_id"].isin(selected_ids)].to_csv(FIGURES / "figure_weighted_calibration_deciles.csv", index=False)


def write_generated_candidate_notes(selection: pd.DataFrame) -> None:
    note = """# Weighted Generated-Candidate Prediction Status

Weighted generated-candidate prediction was not run in this pass.

Reason:
- The available generated-candidate table contains baseline PolyBERT-Ridge predictions, but no generated-candidate PolyBERT embedding cache was detected.
- The weighted Ridge models require the same 600-dimensional PolyBERT embeddings used for training.
- Re-extracting embeddings would require `sentence_transformers` and access to `kuelumbus/polyBERT`; this evidence pass intentionally avoids modifying the original pipeline or inventing weighted predictions.

Required input to complete this step:
- A candidate-level PolyBERT embedding matrix aligned to `all_novel_smiles_with_pred_conductivity.csv`, or a reproducible embedding extraction run using the original PolyBERT model.

Claim boundary:
- The weighted OOF results can support a predictor sensitivity analysis.
- They cannot yet replace baseline generated-candidate counts or candidate rankings.
"""
    (NOTES / "weighted_generated_prediction_not_feasible.md").write_text(note, encoding="utf-8")

    pd.DataFrame(
        [
            {
                "status": "not_computed",
                "reason": "generated_candidate_polybert_embeddings_missing",
                "required_input": "candidate-level 600-d PolyBERT embedding matrix aligned to generated candidates",
                "manuscript_use": "do not report weighted generated-candidate counts as results",
            }
        ]
    ).to_csv(TABLES / "weighted_generated_candidate_predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "all",
                "condition": "all",
                "weighted_predictions_available": False,
                "count_ge_3e-5": np.nan,
                "count_ge_5e-5": np.nan,
                "count_ge_1e-4": np.nan,
                "count_ge_2e-4": np.nan,
                "count_ge_3e-4": np.nan,
                "note": "weighted generated-candidate predictions require candidate embeddings",
            }
        ]
    ).to_csv(TABLES / "weighted_candidate_counts_by_model.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "all",
                "condition": "unknown",
                "weighted_predictions_available": False,
                "count_ge_3e-5": np.nan,
                "count_ge_5e-5": np.nan,
                "count_ge_1e-4": np.nan,
                "count_ge_2e-4": np.nan,
                "count_ge_3e-4": np.nan,
                "note": "condition-level weighted generated-candidate counts not computed",
            }
        ]
    ).to_csv(TABLES / "weighted_candidate_counts_by_model_condition.csv", index=False)
    markdown_table(
        pd.DataFrame(
            [
                {
                    "item": "weighted generated-candidate counts",
                    "status": "not computed",
                    "reason": "candidate PolyBERT embeddings unavailable",
                    "required_fix": "extract/cache embeddings for generated candidates and rerun selected weighted Ridge models",
                }
            ]
        ),
        TABLES / "table_weighted_candidate_counts_by_model.md",
        "Weighted Candidate Counts by Model",
    )

    best = selection[selection["recommended_role"] == "best weighted screening candidate"].head(1)
    best_model = best.iloc[0]["model_id"] if not best.empty else ""
    pd.DataFrame(
        [
            {
                "candidate_id": "",
                "bucket": "not_available",
                "baseline_pred_log10_conductivity": np.nan,
                "weighted_pred_log10_conductivity": np.nan,
                "agreement_status": "not_computed",
                "ad_flag": "not_evaluated",
                "recommendation": "defer weighted candidate selection until generated-candidate embeddings are available",
                "selected_weighted_model": best_model,
                "final_manuscript_use": "exclude_pending_weighted_generated_predictions",
            }
        ]
    ).to_csv(TABLES / "weighted_candidate_selection_recommendation.csv", index=False)
    markdown_table(
        pd.DataFrame(
            [
                {
                    "selection_bucket": "high-confidence predicted hits",
                    "status": "not computed",
                    "reason": "weighted predictions for generated candidates unavailable",
                },
                {
                    "selection_bucket": "weighted-only rescued candidates",
                    "status": "not computed",
                    "reason": "requires weighted generated-candidate predictions",
                },
                {
                    "selection_bucket": "AD-risk candidates",
                    "status": "use existing P1 AD analysis only",
                    "reason": "weighted predictions not linked to generated candidates",
                },
            ]
        ),
        TABLES / "table_weighted_candidate_selection_summary.md",
        "Weighted Candidate Selection Summary",
    )


def write_interpretation_reports(selection: pd.DataFrame, threshold_df: pd.DataFrame, topk_df: pd.DataFrame, tail_df: pd.DataFrame) -> None:
    baseline_id = str(selection.iloc[0]["baseline_model_id"])
    baseline = selection[selection["model_id"] == baseline_id].iloc[0]
    best = selection[selection["recommended_role"] == "best weighted screening candidate"]
    best = best.iloc[0] if not best.empty else selection.iloc[0]
    weighted_help = (
        str(best["model_id"]) != baseline_id
        and float(best["recall_at_1e4"]) > float(baseline["recall_at_1e4"])
        and bool(best["passes_screening_constraints"])
    )
    if weighted_help:
        status = "WEIGHTED MODEL IMPROVES SCREENING"
    elif str(best["model_id"]) != baseline_id and bool(best["passes_screening_constraints"]):
        status = "WEIGHTED MODEL PARTIALLY IMPROVES SCREENING"
    else:
        status = "WEIGHTED MODEL DOES NOT HELP"

    lines = [
        "# Weighted Model Selection Interpretation",
        "",
        f"- Baseline reference: `{baseline_id}`.",
        f"- Best CEJ screening candidate: `{best['model_id']}`.",
        f"- Baseline 1e-4 precision/recall/F1: {baseline['precision_at_1e4']:.3f} / {baseline['recall_at_1e4']:.3f} / {baseline['f1_at_1e4']:.3f}.",
        f"- Best 1e-4 precision/recall/F1: {best['precision_at_1e4']:.3f} / {best['recall_at_1e4']:.3f} / {best['f1_at_1e4']:.3f}.",
        f"- Baseline MAE/RMSE/R2/Spearman: {baseline['mae_log10']:.4f} / {baseline['rmse_log10']:.4f} / {baseline['r2']:.4f} / {baseline['spearman']:.4f}.",
        f"- Best MAE/RMSE/R2/Spearman: {best['mae_log10']:.4f} / {best['rmse_log10']:.4f} / {best['r2']:.4f} / {best['spearman']:.4f}.",
        f"- Baseline top-100 enrichment: {baseline['top100_enrichment']:.3f}.",
        f"- Best top-100 enrichment: {best['top100_enrichment']:.3f}.",
        "",
        "Claim-safe conclusion: interval weighting should be discussed as a surrogate-predictor sensitivity experiment. It does not validate generated-candidate conductivity.",
    ]
    (NOTES / "weighted_model_selection_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = [
        "# Weighted PolyBERT Report",
        "",
        "# Final Status",
        status,
        "",
        "# Best Weighted Scheme",
        f"`{best['model_id']}` ({best['scheme_name']}; {best['scheme_params']}; Ridge alpha={best['ridge_alpha']})",
        "",
        "# Baseline vs Weighted Summary",
        "",
        "| metric | baseline | best weighted | delta |",
        "| --- | ---: | ---: | ---: |",
        f"| MAE log10 | {baseline['mae_log10']:.6f} | {best['mae_log10']:.6f} | {best['delta_mae_vs_baseline']:.6f} |",
        f"| RMSE log10 | {baseline['rmse_log10']:.6f} | {best['rmse_log10']:.6f} | {best['delta_rmse_vs_baseline']:.6f} |",
        f"| R2 | {baseline['r2']:.6f} | {best['r2']:.6f} | {float(best['r2']) - float(baseline['r2']):.6f} |",
        f"| Spearman | {baseline['spearman']:.6f} | {best['spearman']:.6f} | {best['delta_spearman_vs_baseline']:.6f} |",
        f"| precision at 1e-4 | {baseline['precision_at_1e4']:.6f} | {best['precision_at_1e4']:.6f} | {float(best['precision_at_1e4']) - float(baseline['precision_at_1e4']):.6f} |",
        f"| recall at 1e-4 | {baseline['recall_at_1e4']:.6f} | {best['recall_at_1e4']:.6f} | {best['delta_recall_at_1e4_vs_baseline']:.6f} |",
        f"| F1 at 1e-4 | {baseline['f1_at_1e4']:.6f} | {best['f1_at_1e4']:.6f} | {float(best['f1_at_1e4']) - float(baseline['f1_at_1e4']):.6f} |",
        f"| threshold enrichment at 1e-4 | {baseline['threshold_enrichment_at_1e4']:.6f} | {best['threshold_enrichment_at_1e4']:.6f} | {float(best['threshold_enrichment_at_1e4']) - float(baseline['threshold_enrichment_at_1e4']):.6f} |",
        f"| top-100 enrichment | {baseline['top100_enrichment']:.6f} | {best['top100_enrichment']:.6f} | {best['delta_top100_enrichment_vs_baseline']:.6f} |",
        f"| high-tail MAE true >=1e-4 | {baseline['tail_mae_true_ge_1e4']:.6f} | {best['tail_mae_true_ge_1e4']:.6f} | {float(best['tail_mae_true_ge_1e4']) - float(baseline['tail_mae_true_ge_1e4']):.6f} |",
        f"| high-tail RMSE true >=1e-4 | {baseline['tail_rmse_true_ge_1e4']:.6f} | {best['tail_rmse_true_ge_1e4']:.6f} | {float(best['tail_rmse_true_ge_1e4']) - float(baseline['tail_rmse_true_ge_1e4']):.6f} |",
        "",
        "# CEJ-Safe Interpretation",
        "",
        "- The weighted experiment tests whether conductivity-interval sample weights improve surrogate screening behavior in the high-conductivity tail.",
        "- The selected weighted Ridge model may be used as a sensitivity analysis or auxiliary recall-focused filter only if generated-candidate embeddings are later produced.",
        "- The current weighted results are OOF diagnostics on labeled MD-derived training data; they do not validate generated candidates.",
        "- Generated-candidate weighted predictions were not computed because candidate PolyBERT embeddings were unavailable.",
        "- Candidate selection should remain multi-criteria and should not rely on weighted predicted conductivity alone.",
        "",
        "# Risks",
        "",
        "- False positives can increase when recall-focused weighting shifts predictions upward near the high-conductivity threshold.",
        "- Calibration may degrade even when recall improves.",
        "- Existing applicability-domain analysis shows many generated candidates are outside the training distribution.",
        "- Weighted rankings for generated candidates remain unavailable until embeddings are generated.",
        "- The model remains a surrogate prescreener and not a physical conductivity validator.",
        "",
        "# Recommended Manuscript Changes",
        "",
        "- Methods: describe interval-weighted Ridge as a sensitivity experiment using training-fold-only target-derived weights.",
        "- Results: report OOF threshold, top-k, and high-tail diagnostics against the unweighted baseline.",
        "- Limitations: state that weighted generated-candidate prediction requires candidate embeddings and that OOD sensitivity remains unresolved.",
        "- Supplementary: place the full weighting grid, fold-wise metrics, calibration deciles, and threshold sensitivity tables.",
        "",
        "# Release Checklist",
        "",
        "- `revised/polybert_weighted_evidence/scripts/train_polybert_weighted_interval.py`",
        "- `revised/polybert_weighted_evidence/tables/weighted_oof_metrics_all.csv`",
        "- `revised/polybert_weighted_evidence/tables/weighted_threshold_metrics_all.csv`",
        "- `revised/polybert_weighted_evidence/tables/weighted_topk_metrics_all.csv`",
        "- `revised/polybert_weighted_evidence/tables/weighted_model_selection.csv`",
        "- `revised/polybert_weighted_evidence/figures_data/*.csv`",
    ]
    (ROOT / "WEIGHTED_POLYBERT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    oof, X = load_inputs()
    write_inventory(oof, X)
    global_df, threshold_df, topk_df, fold_df, tail_df, decile_df, oof_all_df = run_weighted_oof(oof, X)

    global_df.to_csv(TABLES / "weighted_oof_metrics_all.csv", index=False)
    threshold_df.to_csv(TABLES / "weighted_threshold_metrics_all.csv", index=False)
    topk_df.to_csv(TABLES / "weighted_topk_metrics_all.csv", index=False)
    fold_df.to_csv(TABLES / "weighted_foldwise_metrics_all.csv", index=False)
    tail_df.to_csv(TABLES / "weighted_tail_metrics_all.csv", index=False)
    decile_df.to_csv(TABLES / "weighted_decile_calibration_all.csv", index=False)
    oof_all_df.to_csv(TABLES / "weighted_oof_predictions_all.csv", index=False)

    selection = add_selection_scores(global_df, threshold_df, topk_df, tail_df)
    selection.to_csv(TABLES / "weighted_model_selection.csv", index=False)
    display_cols = [
        "model_id",
        "scheme_name",
        "ridge_alpha",
        "mae_log10",
        "rmse_log10",
        "r2",
        "spearman",
        "precision_at_1e4",
        "recall_at_1e4",
        "f1_at_1e4",
        "top100_enrichment",
        "tail_mae_true_ge_1e4",
        "passes_screening_constraints",
        "pareto_candidate",
        "cej_screening_score",
        "recommended_role",
    ]
    markdown_table(selection[display_cols], TABLES / "table_weighted_model_selection.md", "Weighted Model Selection", max_rows=25)

    write_figures(oof_all_df, selection, threshold_df, topk_df, tail_df, decile_df)
    write_generated_candidate_notes(selection)
    write_interpretation_reports(selection, threshold_df, topk_df, tail_df)

    required = [
        ROOT / "scripts" / "train_polybert_weighted_interval.py",
        TABLES / "weighted_oof_metrics_all.csv",
        TABLES / "weighted_threshold_metrics_all.csv",
        TABLES / "weighted_topk_metrics_all.csv",
        TABLES / "table_weighted_model_selection.md",
        FIGURES / "figure_weighted_threshold_recall_precision.csv",
        FIGURES / "figure_weighted_topk_enrichment.csv",
        ROOT / "WEIGHTED_POLYBERT_REPORT.md",
    ]
    status_lines = ["# Weighted PolyBERT Final Self-Check", ""]
    for path in required:
        status_lines.append(f"- `{path.relative_to(ROOT)}`: {'exists' if path.exists() else 'missing'}")
    (NOTES / "weighted_polybert_final_self_check.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    print("\n".join(status_lines))


if __name__ == "__main__":
    main()
