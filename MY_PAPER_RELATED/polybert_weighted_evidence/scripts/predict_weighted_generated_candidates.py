from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SRC_RUN = ROOT / "source_data" / "polybert_run"
TABLES = ROOT / "tables"
NOTES = ROOT / "analysis_notes"

OOF_PATH = SRC_RUN / "oof_predictions.csv"
TRAIN_EMBEDDINGS_PATH = SRC_RUN / "embeddings.npy"
GENERATED_PATH = SRC_RUN / "all_novel_smiles_with_pred_conductivity.csv"
GENERATED_EMBEDDINGS_PATH = SRC_RUN / "generated_candidate_embeddings_32610.npy"
CONDITION_EMBEDDINGS_PATH = SRC_RUN / "generated_condition_embeddings_47125.npy"
MODEL_SELECTION_PATH = TABLES / "weighted_model_selection.csv"

PREDICTIONS_PATH = TABLES / "weighted_generated_candidate_predictions.csv"
SUMMARY_PATH = TABLES / "weighted_generated_candidate_summary.csv"
CONDITION_PREDICTIONS_PATH = TABLES / "weighted_generated_condition_predictions_47125.csv"
CONDITION_SUMMARY_PATH = TABLES / "weighted_generated_condition_summary.csv"
MD_SELECTION_PATH = TABLES / "weighted_generated_md_selection_60.csv"
S9_PATH = TABLES / "supplementary_table_s9_generated_weighted_predictions.md"
PROVENANCE_PATH = NOTES / "weighted_generated_prediction_provenance.json"
STATUS_PATH = NOTES / "weighted_generated_prediction_status.md"
LEGACY_STATUS_PATH = NOTES / "weighted_generated_prediction_not_feasible.md"
COUNTS_PATH = TABLES / "weighted_candidate_counts_by_model.csv"
CONDITION_COUNTS_PATH = TABLES / "weighted_candidate_counts_by_model_condition.csv"
SELECTION_SUMMARY_PATH = TABLES / "weighted_candidate_selection_recommendation.csv"
COUNTS_MD_PATH = TABLES / "table_weighted_candidate_counts_by_model.md"
SELECTION_MD_PATH = TABLES / "table_weighted_candidate_selection_summary.md"

EXPECTED_TRAIN_ROWS = 6270
EXPECTED_CANDIDATE_ROWS = 32610
EXPECTED_CONDITION_ROWS = 47125
EXPECTED_DIMENSION = 600
BASELINE_ALPHA = 1.0
SELECTED_SCHEME = "smooth_sigmoid_tail_a6_t0p05"
SELECTED_WEIGHT_ALPHA = 6.0
SELECTED_WEIGHT_TEMPERATURE = 0.05
EMBEDDING_MODEL = "xushijie/polyBERT"
EMBEDDING_MODEL_REVISION = "e7dce434fb3eff37905dc114008660e5479ca9a8"
ANCHOR_INDICES = [0, 1, 2, 17, 101, 999, 3000, 6269]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_sigmoid_weights(y: np.ndarray) -> np.ndarray:
    raw = 1.0 + SELECTED_WEIGHT_ALPHA / (
        1.0 + np.exp(-(y - (-4.0)) / SELECTED_WEIGHT_TEMPERATURE)
    )
    return raw / raw.mean()


def load_inputs() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, float]:
    train = pd.read_csv(OOF_PATH)
    train_embeddings = np.load(TRAIN_EMBEDDINGS_PATH)
    generated = pd.read_csv(GENERATED_PATH)
    selection = pd.read_csv(MODEL_SELECTION_PATH)

    if len(train) != EXPECTED_TRAIN_ROWS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_ROWS} training rows, found {len(train)}")
    if train_embeddings.shape != (EXPECTED_TRAIN_ROWS, EXPECTED_DIMENSION):
        raise ValueError(f"Unexpected training embedding shape: {train_embeddings.shape}")
    if len(generated) != EXPECTED_CANDIDATE_ROWS:
        raise ValueError(f"Expected {EXPECTED_CANDIDATE_ROWS} candidates, found {len(generated)}")
    if not generated["smiles"].is_unique or not generated["PSMILES"].is_unique:
        raise ValueError("Generated candidate strings must be unique")
    if generated[["smiles", "PSMILES", "pred_log10_cond"]].isna().any().any():
        raise ValueError("Generated candidate source contains missing required values")

    selected = selection.loc[selection["recommended_role"].eq("best weighted screening candidate")]
    if len(selected) != 1:
        raise ValueError(f"Expected one selected weighted model, found {len(selected)}")
    row = selected.iloc[0]
    if row["scheme_id"] != SELECTED_SCHEME:
        raise ValueError(f"Unexpected selected weighting scheme: {row['scheme_id']}")
    ridge_alpha = float(row["ridge_alpha"])
    return train, train_embeddings, generated, ridge_alpha


def load_encoder(device: str):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        EMBEDDING_MODEL,
        revision=EMBEDDING_MODEL_REVISION,
        device=device,
    )
    if encoder.get_sentence_embedding_dimension() != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSION}-d embeddings, found "
            f"{encoder.get_sentence_embedding_dimension()}"
        )
    return encoder


def verify_encoder(
    encoder,
    train: pd.DataFrame,
    train_embeddings: np.ndarray,
    batch_size: int,
) -> dict[str, float | bool]:
    current = encoder.encode(
        train.loc[ANCHOR_INDICES, "PSMILES"].astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    reference = np.asarray(train_embeddings[ANCHOR_INDICES])
    max_abs = float(np.max(np.abs(current - reference)))
    mean_abs = float(np.mean(np.abs(current - reference)))
    cosines = np.sum(current * reference, axis=1) / (
        np.linalg.norm(current, axis=1) * np.linalg.norm(reference, axis=1)
    )
    min_cosine = float(np.min(cosines))
    passed = max_abs <= 3e-6 and min_cosine >= 0.999999
    if not passed:
        raise ValueError(
            "Loaded encoder does not reproduce cached training embeddings: "
            f"max_abs={max_abs:.6g}, min_cosine={min_cosine:.9g}"
        )
    return {
        "anchor_max_abs_difference": max_abs,
        "anchor_mean_abs_difference": mean_abs,
        "anchor_min_cosine_similarity": min_cosine,
        "anchor_check_passed": passed,
    }


def load_or_compute_generated_embeddings(
    encoder,
    generated: pd.DataFrame,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if GENERATED_EMBEDDINGS_PATH.exists() and not force:
        embeddings = np.load(GENERATED_EMBEDDINGS_PATH)
    else:
        embeddings = encoder.encode(
            generated["PSMILES"].astype(str).tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        np.save(GENERATED_EMBEDDINGS_PATH, embeddings)
    if embeddings.shape != (EXPECTED_CANDIDATE_ROWS, EXPECTED_DIMENSION):
        raise ValueError(f"Unexpected generated embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Generated embeddings contain non-finite values")
    return embeddings


def load_condition_pool() -> pd.DataFrame:
    sources = {
        "LOW": REPO_ROOT
        / "MY_PAPER_RELATED/polybert_con/TransCVAE/all_novel_smiles_condz_low_with_pred_conductivity.csv",
        "HIGH": REPO_ROOT
        / "MY_PAPER_RELATED/polybert_con/TransCVAE/all_novel_smiles_condz_high_with_pred_conductivity.csv",
    }
    frames = []
    for condition, path in sources.items():
        frame = pd.read_csv(path)
        required = {"smiles", "PSMILES", "pred_log10_cond", "pred_cond"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame.insert(0, "condition", condition)
        frame.insert(1, "condition_row_index", np.arange(len(frame), dtype=int))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != EXPECTED_CONDITION_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_CONDITION_ROWS} condition-pool rows, found {len(combined)}"
        )
    if not combined["smiles"].is_unique or not combined["PSMILES"].is_unique:
        raise ValueError("LOW/HIGH condition pools must be mutually unique")
    return combined


def load_or_compute_condition_embeddings(
    encoder,
    generated: pd.DataFrame,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    if CONDITION_EMBEDDINGS_PATH.exists() and not force:
        embeddings = np.load(CONDITION_EMBEDDINGS_PATH)
    else:
        embeddings = encoder.encode(
            generated["PSMILES"].astype(str).tolist(),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        np.save(CONDITION_EMBEDDINGS_PATH, embeddings)
    if embeddings.shape != (EXPECTED_CONDITION_ROWS, EXPECTED_DIMENSION):
        raise ValueError(f"Unexpected condition-pool embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Condition-pool embeddings contain non-finite values")
    return embeddings


def fit_and_predict(
    train: pd.DataFrame,
    train_embeddings: np.ndarray,
    generated_embeddings: np.ndarray,
    weighted_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = train["log10_cond"].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_embeddings.astype(np.float64, copy=False))
    x_generated = scaler.transform(generated_embeddings.astype(np.float64, copy=False))

    baseline = Ridge(alpha=BASELINE_ALPHA)
    baseline.fit(x_train, y)
    baseline_predictions = baseline.predict(x_generated).astype(float)

    weighted = Ridge(alpha=weighted_alpha)
    weighted.fit(x_train, y, sample_weight=normalized_sigmoid_weights(y))
    weighted_predictions = weighted.predict(x_generated).astype(float)
    return baseline_predictions, weighted_predictions


def rank_descending(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="min", ascending=False).astype(int).to_numpy()


def make_outputs(
    generated: pd.DataFrame,
    baseline_refit: np.ndarray,
    weighted: np.ndarray,
    weighted_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    existing = generated["pred_log10_cond"].to_numpy(dtype=float)
    baseline_delta = baseline_refit - existing
    validation = {
        "baseline_refit_max_abs_difference": float(np.max(np.abs(baseline_delta))),
        "baseline_refit_mean_abs_difference": float(np.mean(np.abs(baseline_delta))),
        "baseline_refit_pearson": float(pd.Series(existing).corr(pd.Series(baseline_refit))),
        "baseline_refit_spearman": float(
            pd.Series(existing).corr(pd.Series(baseline_refit), method="spearman")
        ),
    }

    output = pd.DataFrame(
        {
            "generated_pool_index": np.arange(len(generated), dtype=int),
            "smiles": generated["smiles"].astype(str),
            "PSMILES": generated["PSMILES"].astype(str),
            "baseline_pred_log10_conductivity_existing": existing,
            "baseline_pred_log10_conductivity_refit": baseline_refit,
            "weighted_pred_log10_conductivity": weighted,
        }
    )
    output["baseline_pred_conductivity_s_cm"] = 10.0 ** output[
        "baseline_pred_log10_conductivity_refit"
    ]
    output["weighted_pred_conductivity_s_cm"] = 10.0 ** output[
        "weighted_pred_log10_conductivity"
    ]
    output["baseline_rank"] = rank_descending(baseline_refit)
    output["weighted_rank"] = rank_descending(weighted)
    output["rank_change_toward_top"] = output["baseline_rank"] - output["weighted_rank"]
    output["baseline_hit_ge_1e4"] = baseline_refit >= -4.0
    output["weighted_hit_ge_1e4"] = weighted >= -4.0
    output["selected_weighted_model"] = (
        f"{SELECTED_SCHEME}__ridge_alpha_{weighted_alpha:g}"
    )
    output["deployment_fit"] = "full_6270_row_refit"
    output["embedding_model"] = EMBEDDING_MODEL

    def summary_row(model: str, values: np.ndarray) -> dict[str, object]:
        return {
            "model": model,
            "generated_candidates": int(len(values)),
            "mean_log10_predicted_conductivity": float(np.mean(values)),
            "median_log10_predicted_conductivity": float(np.median(values)),
            "count_ge_1e4_s_cm": int(np.sum(values >= -4.0)),
            "hit_fraction_ge_1e4_s_cm": float(np.mean(values >= -4.0)),
        }

    summary = pd.DataFrame(
        [
            summary_row("Baseline PolyBERT-Ridge", baseline_refit),
            summary_row("Interval-weighted Ridge", weighted),
        ]
    )
    rank_spearman = float(pd.Series(baseline_refit).corr(pd.Series(weighted), method="spearman"))
    summary["baseline_weighted_rank_spearman"] = rank_spearman
    summary["weighted_minus_baseline_mean_log10"] = float(np.mean(weighted - baseline_refit))
    summary["weighted_model"] = f"{SELECTED_SCHEME}__ridge_alpha_{weighted_alpha:g}"
    summary["deployment_fit"] = "full_6270_row_refit"
    return output, summary, validation


def write_s9(summary: pd.DataFrame) -> None:
    baseline = summary.iloc[0]
    weighted = summary.iloc[1]
    lines = [
        "# Supplementary Table S9",
        "",
        "Baseline and interval-weighted generated-candidate score comparison.",
        "",
        "| Metric | Baseline | Weighted |",
        "|---|---:|---:|",
        f"| Generated candidates | {int(baseline['generated_candidates']):,} | {int(weighted['generated_candidates']):,} |",
        f"| Mean log10 predicted conductivity | {baseline['mean_log10_predicted_conductivity']:.3f} | {weighted['mean_log10_predicted_conductivity']:.3f} |",
        f"| Median log10 predicted conductivity | {baseline['median_log10_predicted_conductivity']:.3f} | {weighted['median_log10_predicted_conductivity']:.3f} |",
        f"| Count at or above 1.00 x 10^-4 S cm^-1 | {int(baseline['count_ge_1e4_s_cm']):,} | {int(weighted['count_ge_1e4_s_cm']):,} |",
        f"| Hit fraction | {baseline['hit_fraction_ge_1e4_s_cm']:.3f} | {weighted['hit_fraction_ge_1e4_s_cm']:.3f} |",
        f"| Rank Spearman vs baseline | 1.000 | {weighted['baseline_weighted_rank_spearman']:.3f} |",
        "",
        "The selected weighted model used smooth sigmoid tail weights centered at log10 conductivity = -4 (alpha = 6; temperature = 0.05) and Ridge alpha = 100. Both deployment models were refit on all 6,270 labeled rows after OOF model selection. The table reports the current deduplicated 32,610-candidate pool.",
    ]
    S9_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_supporting_outputs(predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    model_columns = {
        "Baseline PolyBERT-Ridge": "baseline_pred_log10_conductivity_refit",
        "Interval-weighted Ridge": "weighted_pred_log10_conductivity",
    }
    thresholds = {
        "count_ge_3e-5": float(np.log10(3e-5)),
        "count_ge_5e-5": float(np.log10(5e-5)),
        "count_ge_1e-4": -4.0,
        "count_ge_2e-4": float(np.log10(2e-4)),
        "count_ge_3e-4": float(np.log10(3e-4)),
    }
    count_rows = []
    for model, column in model_columns.items():
        values = predictions[column].to_numpy(dtype=float)
        row = {
            "model": model,
            "condition": "all",
            "weighted_predictions_available": True,
            "generated_candidates": len(values),
        }
        row.update({name: int(np.sum(values >= threshold)) for name, threshold in thresholds.items()})
        row["note"] = "full 6,270-row deployment refit on the deduplicated generated pool"
        count_rows.append(row)
    counts = pd.DataFrame(count_rows)
    counts.to_csv(COUNTS_PATH, index=False)

    condition_counts = pd.DataFrame(
        [
            {
                "model": "all",
                "condition": "not_retained_in_deduplicated_pool",
                "weighted_predictions_available": True,
                "count_ge_3e-5": np.nan,
                "count_ge_5e-5": np.nan,
                "count_ge_1e-4": np.nan,
                "count_ge_2e-4": np.nan,
                "count_ge_3e-4": np.nan,
                "note": "overall predictions complete; source pool has no condition column",
            }
        ]
    )
    condition_counts.to_csv(CONDITION_COUNTS_PATH, index=False)

    baseline_hit = predictions["baseline_hit_ge_1e4"].to_numpy(dtype=bool)
    weighted_hit = predictions["weighted_hit_ge_1e4"].to_numpy(dtype=bool)
    selection = pd.DataFrame(
        [
            {"selection_bucket": "consensus hit", "count": int(np.sum(baseline_hit & weighted_hit))},
            {"selection_bucket": "weighted-only hit", "count": int(np.sum(~baseline_hit & weighted_hit))},
            {"selection_bucket": "baseline-only hit", "count": int(np.sum(baseline_hit & ~weighted_hit))},
            {"selection_bucket": "neither hit", "count": int(np.sum(~baseline_hit & ~weighted_hit))},
        ]
    )
    selection["threshold_s_cm"] = 1e-4
    selection["manuscript_use"] = "sensitivity analysis; baseline ranking retained for MD selection"
    selection.to_csv(SELECTION_SUMMARY_PATH, index=False)

    def markdown_table(frame: pd.DataFrame, path: Path, title: str) -> None:
        lines = [f"# {title}", "", "| " + " | ".join(frame.columns) + " |"]
        lines.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    markdown_table(counts, COUNTS_MD_PATH, "Weighted Candidate Counts by Model")
    markdown_table(selection, SELECTION_MD_PATH, "Weighted Candidate Selection Summary")

    baseline = summary.iloc[0]
    weighted = summary.iloc[1]
    status = f"""# Weighted Generated-Candidate Prediction Status

Weighted generated-candidate prediction is complete for all 32,610 unique candidates.

Method:
- Encoder: `{EMBEDDING_MODEL}` revision `{EMBEDDING_MODEL_REVISION}` with 600-dimensional, unnormalized embeddings.
- Encoder identity was checked against cached training embeddings before candidate encoding.
- Baseline deployment model: unweighted Ridge alpha = 1, refit on all 6,270 labeled rows.
- Weighted deployment model: smooth sigmoid weights centered at log10 conductivity = -4, alpha = 6, temperature = 0.05; Ridge alpha = 100; refit on all 6,270 labeled rows.

Generated-pool result:
- Baseline hit fraction at 1e-4 S cm-1: {baseline['hit_fraction_ge_1e4_s_cm']:.6f} ({int(baseline['count_ge_1e4_s_cm']):,}/{int(baseline['generated_candidates']):,}).
- Weighted hit fraction at 1e-4 S cm-1: {weighted['hit_fraction_ge_1e4_s_cm']:.6f} ({int(weighted['count_ge_1e4_s_cm']):,}/{int(weighted['generated_candidates']):,}).
- Baseline-weighted rank Spearman: {weighted['baseline_weighted_rank_spearman']:.6f}.

Claim boundary:
- Weighted generated-candidate predictions can now be reported as a sensitivity analysis.
- The original 60-candidate MD selection remains based on the baseline ranking and is not retrospectively relabeled as weighted selection.
"""
    STATUS_PATH.write_text(status, encoding="utf-8")
    if LEGACY_STATUS_PATH.exists():
        LEGACY_STATUS_PATH.unlink()


def write_condition_outputs(
    condition_pool: pd.DataFrame,
    baseline_refit: np.ndarray,
    weighted: np.ndarray,
    weighted_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    predictions, _, validation = make_outputs(
        condition_pool, baseline_refit, weighted, weighted_alpha
    )
    predictions = predictions.rename(columns={"generated_pool_index": "condition_pool_index"})
    predictions.insert(1, "condition", condition_pool["condition"].to_numpy())
    predictions.insert(2, "condition_row_index", condition_pool["condition_row_index"].to_numpy())
    predictions.to_csv(CONDITION_PREDICTIONS_PATH, index=False)

    summary_rows = []
    for condition, indices in [
        ("LOW", np.flatnonzero(condition_pool["condition"].eq("LOW").to_numpy())),
        ("HIGH", np.flatnonzero(condition_pool["condition"].eq("HIGH").to_numpy())),
        ("ALL", np.arange(len(condition_pool))),
    ]:
        for model, values in [
            ("Baseline PolyBERT-Ridge", baseline_refit[indices]),
            ("Interval-weighted Ridge", weighted[indices]),
        ]:
            summary_rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "generated_candidates": len(indices),
                    "mean_log10_predicted_conductivity": float(np.mean(values)),
                    "median_log10_predicted_conductivity": float(np.median(values)),
                    "count_ge_1e4_s_cm": int(np.sum(values >= -4.0)),
                    "hit_fraction_ge_1e4_s_cm": float(np.mean(values >= -4.0)),
                }
            )
    condition_summary = pd.DataFrame(summary_rows)
    condition_summary["baseline_weighted_rank_spearman_all"] = float(
        pd.Series(baseline_refit).corr(pd.Series(weighted), method="spearman")
    )
    condition_summary["weighted_model"] = (
        f"{SELECTED_SCHEME}__ridge_alpha_{weighted_alpha:g}"
    )
    condition_summary["deployment_fit"] = "full_6270_row_refit"
    condition_summary.to_csv(CONDITION_SUMMARY_PATH, index=False)

    selection_source = pd.read_csv(
        REPO_ROOT
        / "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/github_results/"
        "latest_notebook_manifest_60/sample_manifest.csv"
    )
    selected = selection_source.merge(
        predictions,
        on="PSMILES",
        how="left",
        validate="one_to_one",
        suffixes=("_selection", "_weighted_pool"),
        indicator=True,
    )
    if not selected["_merge"].eq("both").all():
        missing = selected.loc[~selected["_merge"].eq("both"), "Trajectory ID"].tolist()
        raise ValueError(f"Selected MD candidates missing weighted predictions: {missing}")
    selected = selected.drop(columns="_merge")
    selected.to_csv(MD_SELECTION_PATH, index=False)

    low = condition_summary.loc[
        condition_summary["condition"].eq("LOW")
    ].set_index("model")
    high = condition_summary.loc[
        condition_summary["condition"].eq("HIGH")
    ].set_index("model")
    all_pool = condition_summary.loc[
        condition_summary["condition"].eq("ALL")
    ].set_index("model")
    status_append = f"""

Final target-conditioned generated pool:
- LOW: 44,999 candidates; baseline/weighted hit fractions {low.loc['Baseline PolyBERT-Ridge', 'hit_fraction_ge_1e4_s_cm']:.6f} / {low.loc['Interval-weighted Ridge', 'hit_fraction_ge_1e4_s_cm']:.6f}.
- HIGH: 2,126 candidates; baseline/weighted hit fractions {high.loc['Baseline PolyBERT-Ridge', 'hit_fraction_ge_1e4_s_cm']:.6f} / {high.loc['Interval-weighted Ridge', 'hit_fraction_ge_1e4_s_cm']:.6f}.
- Combined: 47,125 candidates; baseline/weighted hit fractions {all_pool.loc['Baseline PolyBERT-Ridge', 'hit_fraction_ge_1e4_s_cm']:.6f} / {all_pool.loc['Interval-weighted Ridge', 'hit_fraction_ge_1e4_s_cm']:.6f}.
- Baseline-weighted rank Spearman across the combined pool: {all_pool.iloc[0]['baseline_weighted_rank_spearman_all']:.6f}.
- All 60 MD-selected candidates were matched to exact baseline-refit and weighted predictions.
"""
    STATUS_PATH.write_text(STATUS_PATH.read_text(encoding="utf-8") + status_append, encoding="utf-8")
    return predictions, condition_summary, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force-embeddings", action="store_true")
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)
    train, train_embeddings, generated, weighted_alpha = load_inputs()
    encoder = load_encoder(args.device)
    encoder_check = verify_encoder(encoder, train, train_embeddings, args.batch_size)
    generated_embeddings = load_or_compute_generated_embeddings(
        encoder, generated, args.batch_size, args.force_embeddings
    )
    baseline_refit, weighted = fit_and_predict(
        train, train_embeddings, generated_embeddings, weighted_alpha
    )
    predictions, summary, baseline_check = make_outputs(
        generated, baseline_refit, weighted, weighted_alpha
    )
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_s9(summary)
    write_supporting_outputs(predictions, summary)

    condition_pool = load_condition_pool()
    condition_embeddings = load_or_compute_condition_embeddings(
        encoder, condition_pool, args.batch_size, args.force_embeddings
    )
    condition_baseline, condition_weighted = fit_and_predict(
        train, train_embeddings, condition_embeddings, weighted_alpha
    )
    condition_predictions, condition_summary, condition_baseline_check = write_condition_outputs(
        condition_pool, condition_baseline, condition_weighted, weighted_alpha
    )

    provenance = {
        "training_rows": len(train),
        "candidate_rows": len(generated),
        "embedding_dimension": int(generated_embeddings.shape[1]),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "candidate_embedding_cache": str(GENERATED_EMBEDDINGS_PATH.relative_to(ROOT)),
        "candidate_embedding_sha256": sha256(GENERATED_EMBEDDINGS_PATH),
        "training_embedding_sha256": sha256(TRAIN_EMBEDDINGS_PATH),
        "candidate_source_sha256": sha256(GENERATED_PATH),
        "condition_pool_rows": len(condition_pool),
        "condition_embedding_cache": str(CONDITION_EMBEDDINGS_PATH.relative_to(ROOT)),
        "condition_embedding_sha256": sha256(CONDITION_EMBEDDINGS_PATH),
        "condition_predictions_sha256": sha256(CONDITION_PREDICTIONS_PATH),
        "md_selection_rows_with_weighted_predictions": 60,
        "selected_weighted_scheme": SELECTED_SCHEME,
        "selected_weighted_ridge_alpha": weighted_alpha,
        "deployment_fit": "full_6270_row_refit",
        **encoder_check,
        **baseline_check,
        **{f"condition_{key}": value for key, value in condition_baseline_check.items()},
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(condition_summary.to_string(index=False))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
