from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import math
import warnings

import numpy as np
import pandas as pd


TAUS = [0, 1, 2, 5, 10, 20, 30, 50]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
RANDOM_SEED = 17


def require_sklearn():
    try:
        from sklearn.linear_model import HuberRegressor, Ridge
        from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("scikit-learn is required for polymer-level CV.") from exc

    return {
        "HuberRegressor": HuberRegressor,
        "Ridge": Ridge,
        "GroupKFold": GroupKFold,
        "LeaveOneGroupOut": LeaveOneGroupOut,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def try_kendall_tau(x: pd.Series, y: pd.Series) -> float:
    try:
        from scipy.stats import kendalltau
    except Exception:
        return np.nan
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    m = xx.notna() & yy.notna()
    if int(m.sum()) < 3 or xx[m].nunique() < 2 or yy[m].nunique() < 2:
        return np.nan
    return float(kendalltau(xx[m], yy[m], nan_policy="omit").statistic)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    m = xx.notna() & yy.notna()
    if int(m.sum()) < 3:
        return np.nan
    xx = xx[m]
    yy = yy[m]
    if xx.nunique(dropna=True) < 2 or yy.nunique(dropna=True) < 2:
        return np.nan
    return float(xx.corr(yy, method=method))


def finite_log10(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=vals.index, dtype=float)
    m = vals > 0
    out.loc[m] = np.log10(vals.loc[m])
    return out


def tau_label(tau: int | float) -> str:
    return f"{float(tau):g}".replace(".", "p")


def polymer_hash(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"poly_{digest}"


def normalize_group(group: str) -> str:
    text = str(group)
    if text == "middle":
        return "middle_stratified"
    return text


def metrics_for_log_predictions(df: pd.DataFrame) -> dict[str, float]:
    required = {"log10_sigma_ref", "log10_sigma_pred"}
    if not required <= set(df.columns):
        return {}
    ref = pd.to_numeric(df["log10_sigma_ref"], errors="coerce")
    pred = pd.to_numeric(df["log10_sigma_pred"], errors="coerce")
    m = ref.notna() & pred.notna() & np.isfinite(ref) & np.isfinite(pred)
    if int(m.sum()) == 0:
        return {}
    err = pred[m] - ref[m]
    pred_ref = np.power(10.0, err)
    out = {
        "n": int(m.sum()),
        "MAlogE": float(np.abs(err).mean()),
        "median_abs_log_error": float(np.abs(err).median()),
        "median_pred_ref": float(np.median(pred_ref)),
        "mean_pred_ref": float(np.mean(pred_ref)),
        "spearman_r": safe_corr(pred[m], ref[m], "spearman"),
        "pearson_r": safe_corr(pred[m], ref[m], "pearson"),
    }
    if int(m.sum()) >= 3:
        ss_res = float(np.sum(np.square(err)))
        ss_tot = float(np.sum(np.square(ref[m] - ref[m].mean())))
        out["R2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        out["R2"] = np.nan
    return out


def summarize_predictions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    group_rows = []
    keys = ["validation_scheme", "model", "feature_set"]
    for key, sub in pred.groupby(keys, dropna=False, sort=True):
        d = metrics_for_log_predictions(sub)
        if not d:
            continue
        d.update(dict(zip(keys, key)))
        rows.append(d)
        for group, gsub in sub.groupby("group", sort=True):
            gd = metrics_for_log_predictions(gsub)
            if not gd:
                continue
            gd.update(dict(zip(keys, key)))
            gd["group"] = group
            group_rows.append(gd)
    return pd.DataFrame(rows), pd.DataFrame(group_rows)


def summarize_predictions_by_fold(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["validation_scheme", "fold_id", "model", "feature_set"]
    for key, sub in pred.groupby(keys, dropna=False, sort=True):
        d = metrics_for_log_predictions(sub)
        if not d:
            continue
        d.update(dict(zip(keys, key)))
        heldout = str(key[1])
        if heldout.startswith("heldout_"):
            d["heldout_group"] = heldout.replace("heldout_", "", 1)
        rows.append(d)
    return pd.DataFrame(rows)


def summarize_by_group_only(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, feature_set, group), sub in pred.groupby(["model", "feature_set", "group"], dropna=False, sort=True):
        d = metrics_for_log_predictions(sub)
        if not d:
            continue
        d.update({"model": model, "feature_set": feature_set, "group": group})
        rows.append(d)
    return pd.DataFrame(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_inputs(base: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    sweep_path = base / "results" / "persistence_sweep_cutoff0p28_1ps" / "persistence_sweep_per_system.csv"
    pred_path = base / "results" / "hybrid_correction" / "hybrid_model_cv_predictions.csv"
    summary_path = base / "results" / "hybrid_correction" / "hybrid_model_cv_summary.csv"
    manifest_path = base / "results" / "sample_manifest.csv"
    missing = [p for p in [sweep_path, pred_path, summary_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files: " + ", ".join(str(p) for p in missing))
    sweep = pd.read_csv(sweep_path)
    old_pred = pd.read_csv(pred_path)
    old_summary = pd.read_csv(summary_path)
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else None
    return sweep, old_pred, old_summary, manifest


def schema_log(out_dir: Path, sweep: pd.DataFrame, old_pred: pd.DataFrame, old_summary: pd.DataFrame, manifest: pd.DataFrame | None) -> str:
    lines = []
    frames = [
        ("persistence_sweep_per_system.csv", sweep),
        ("hybrid_model_cv_predictions.csv", old_pred),
        ("hybrid_model_cv_summary.csv", old_summary),
    ]
    if manifest is not None:
        frames.append(("sample_manifest.csv", manifest))
    for name, df in frames:
        lines.append(f"## {name}")
        lines.append(f"shape={df.shape}")
        lines.append("columns=" + ", ".join(map(str, df.columns)))
        lines.append("")
    text = "\n".join(lines)
    write_text(out_dir / "input_schema_columns.txt", text)
    return text


def infer_polymer_mapping(base: Path, sweep: pd.DataFrame, manifest: pd.DataFrame | None, out_dir: Path) -> pd.DataFrame:
    one = sweep[np.isclose(pd.to_numeric(sweep["tau_ps"], errors="coerce"), 20.0)].copy()
    one["Trajectory ID"] = pd.to_numeric(one["Trajectory ID"], errors="coerce").astype("Int64")
    one["traj_id"] = pd.to_numeric(one["traj_id"], errors="coerce").astype("Int64")

    if manifest is not None and {"Trajectory ID", "SMILES"} <= set(manifest.columns):
        man = manifest.copy()
        man["Trajectory ID"] = pd.to_numeric(man["Trajectory ID"], errors="coerce").astype("Int64")
        keep = ["Trajectory ID", "SMILES"]
        for col in ["Degree of Polymerization", "Density", "Molality"]:
            if col in man.columns:
                keep.append(col)
        merged = one[["traj_id", "Trajectory ID", "group", "sample_group"]].merge(man[keep], on="Trajectory ID", how="left")
        if merged["SMILES"].isna().any():
            missing = merged.loc[merged["SMILES"].isna(), "Trajectory ID"].astype(str).tolist()
            raise RuntimeError(
                "Cannot infer polymer_id: sample_manifest.csv lacks SMILES for Trajectory ID(s): "
                + ", ".join(missing[:20])
            )
        merged["polymer_key"] = merged["SMILES"].astype(str).str.strip()
        method = "sample_manifest.SMILES"
    else:
        candidates = ["polymer_id", "system_id", "molecule_id", "candidate_id", "pSMILES", "PSMILES", "SMILES", "polymer_smiles"]
        found = [c for c in candidates if c in one.columns]
        if not found:
            raise RuntimeError(
                "Cannot safely infer polymer identity. Need sample_manifest.csv with columns "
                "'Trajectory ID' and 'SMILES', or an explicit polymer/system column in persistence_sweep_per_system.csv."
            )
        source = found[0]
        merged = one[["traj_id", "Trajectory ID", "group", "sample_group", source]].copy()
        merged["polymer_key"] = merged[source].astype(str).str.strip()
        method = f"persistence_sweep_per_system.{source}"

    merged["polymer_id"] = merged["polymer_key"].map(polymer_hash)
    merged["polymer_id_inference_method"] = method
    merged["group"] = merged["group"].map(normalize_group)
    mapping_cols = [
        "traj_id",
        "Trajectory ID",
        "polymer_id",
        "polymer_key",
        "polymer_id_inference_method",
        "group",
        "sample_group",
    ]
    for col in ["Degree of Polymerization", "Density", "Molality"]:
        if col in merged.columns:
            mapping_cols.append(col)
    mapping = merged[mapping_cols].sort_values(["polymer_id", "traj_id"]).reset_index(drop=True)
    mapping.to_csv(out_dir / "polymer_group_mapping.csv", index=False)
    return mapping


def pivot_sweep_to_dataset(sweep: pd.DataFrame, mapping: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    required_cols = {"traj_id", "group", "tau_ps", "sigma_ref", "sigma_NE", "sigma_cNE_tau"}
    missing = required_cols - set(sweep.columns)
    if missing:
        raise RuntimeError(f"persistence_sweep_per_system.csv missing required columns: {sorted(missing)}")

    rows = []
    for traj_id, sub in sweep.groupby("traj_id", sort=True):
        valid = sub[pd.to_numeric(sub["valid"], errors="coerce").fillna(False).astype(bool)].copy()
        if valid.empty:
            continue
        first = valid.iloc[0]
        row = {
            "traj_id": int(traj_id),
            "Trajectory ID": int(first["Trajectory ID"]),
            "group": normalize_group(first["group"]),
            "sample_group": normalize_group(first.get("sample_group", first["group"])),
            "sigma_ref": float(first["sigma_ref"]),
            "sigma_NE": float(first["sigma_NE"]),
            "log10_sigma_ref": math.log10(float(first["sigma_ref"])),
            "log10_sigma_NE": math.log10(float(first["sigma_NE"])),
            "NE_ref": float(first["sigma_NE"]) / float(first["sigma_ref"]),
        }
        for tau in TAUS:
            ts = valid[np.isclose(pd.to_numeric(valid["tau_ps"], errors="coerce"), float(tau))]
            if ts.empty:
                row[f"sigma_cNE_tau{tau_label(tau)}"] = np.nan
                row[f"log10_sigma_cNE_tau{tau_label(tau)}"] = np.nan
                row[f"cNE{tau_label(tau)}_over_NE"] = np.nan
                row[f"log10_cNE{tau_label(tau)}_over_NE"] = np.nan
                continue
            r = ts.iloc[0]
            sigma = float(r["sigma_cNE_tau"])
            row[f"sigma_cNE_tau{tau_label(tau)}"] = sigma
            row[f"log10_sigma_cNE_tau{tau_label(tau)}"] = math.log10(sigma) if sigma > 0 else np.nan
            ratio = sigma / float(first["sigma_NE"]) if float(first["sigma_NE"]) > 0 else np.nan
            row[f"cNE{tau_label(tau)}_over_NE"] = ratio
            row[f"log10_cNE{tau_label(tau)}_over_NE"] = math.log10(ratio) if ratio > 0 else np.nan
            for desc in [
                "neutral_pop",
                "charged_pop",
                "free_cation_fraction",
                "free_anion_fraction",
                "largest_cluster_size",
            ]:
                if desc in r.index:
                    row[f"{desc}_tau{tau_label(tau)}"] = float(r[desc])
        r20 = valid[np.isclose(pd.to_numeric(valid["tau_ps"], errors="coerce"), 20.0)]
        if not r20.empty:
            r20 = r20.iloc[0]
            for desc in ["p90_contact_lifetime_ps", "S20", "contact_time_fraction_ge20"]:
                if desc in r20.index:
                    row[desc] = float(r20[desc])
        rows.append(row)

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise RuntimeError("No model rows could be built from persistence sweep.")
    dataset = dataset.merge(mapping[["traj_id", "polymer_id", "polymer_key", "polymer_id_inference_method"]], on="traj_id", how="left")
    if dataset["polymer_id"].isna().any():
        bad = dataset.loc[dataset["polymer_id"].isna(), "traj_id"].astype(str).tolist()
        raise RuntimeError("Missing polymer_id after mapping for traj_id(s): " + ", ".join(bad[:20]))

    rename_pairs = {
        "cNE20_over_NE": "cNE20_over_NE",
        "cNE0_over_NE": "cNE0_over_NE",
        "log10_cNE20_over_NE": "log10_cNE20_over_NE",
        "log10_cNE0_over_NE": "log10_cNE0_over_NE",
        "neutral_pop_tau20": "neutral_pop_20ps",
        "charged_pop_tau20": "charged_pop_20ps",
        "free_cation_fraction_tau20": "free_cation_fraction_20ps",
        "free_anion_fraction_tau20": "free_anion_fraction_20ps",
        "largest_cluster_size_tau20": "largest_cluster_size_20ps",
        "neutral_pop_tau0": "neutral_pop_tau0",
        "charged_pop_tau0": "charged_pop_tau0",
    }
    # Preserve user-facing descriptor names while keeping tau-explicit columns.
    for src, dst in rename_pairs.items():
        if src in dataset.columns and dst not in dataset.columns:
            dataset[dst] = dataset[src]

    required_features = [
        "log10_sigma_NE",
        "log10_sigma_cNE_tau0",
        "log10_sigma_cNE_tau5",
        "log10_sigma_cNE_tau20",
        "log10_sigma_cNE_tau50",
    ]
    missing_required = [c for c in required_features if c not in dataset.columns]
    if missing_required:
        raise RuntimeError(f"Missing required model features: {missing_required}")

    candidate_features = [
        "log10_sigma_NE",
        "log10_sigma_cNE_tau0",
        "log10_sigma_cNE_tau1",
        "log10_sigma_cNE_tau2",
        "log10_sigma_cNE_tau5",
        "log10_sigma_cNE_tau10",
        "log10_sigma_cNE_tau20",
        "log10_sigma_cNE_tau30",
        "log10_sigma_cNE_tau50",
        "log10_cNE20_over_NE",
        "log10_cNE0_over_NE",
        "neutral_pop_20ps",
        "charged_pop_20ps",
        "neutral_pop_tau0",
        "charged_pop_tau0",
        "free_cation_fraction_20ps",
        "free_anion_fraction_20ps",
        "largest_cluster_size_20ps",
        "p90_contact_lifetime_ps",
        "S20",
        "contact_time_fraction_ge20",
    ]

    skipped = {"missing": [], "nonfinite": [], "leaky": []}
    features = []
    for col in candidate_features:
        if col not in dataset.columns:
            skipped["missing"].append(col)
            continue
        vals = pd.to_numeric(dataset[col], errors="coerce")
        if not np.isfinite(vals.to_numpy(float)).all():
            if col in required_features:
                bad_n = int((~np.isfinite(vals.to_numpy(float))).sum())
                raise RuntimeError(f"Required feature {col} has {bad_n} non-finite values.")
            skipped["nonfinite"].append(col)
            continue
        features.append(col)

    feature_sets = build_feature_sets(features)
    for name, cols in feature_sets.items():
        if not cols:
            raise RuntimeError(f"Feature set {name} is empty after finite-value filtering.")

    dataset = dataset.sort_values("traj_id").reset_index(drop=True)
    dataset.to_csv(out_dir / "model_dataset.csv", index=False)
    lines = ["# Feature Columns Used", ""]
    lines.append("all_features=" + ", ".join(features))
    for name, cols in feature_sets.items():
        lines.append(f"{name}=" + ", ".join(cols))
    lines.append("")
    for key, vals in skipped.items():
        lines.append(f"skipped_{key}=" + (", ".join(vals) if vals else "none"))
    write_text(out_dir / "feature_columns_used.txt", "\n".join(lines) + "\n")
    return dataset, features, feature_sets


def build_feature_sets(features: list[str]) -> dict[str, list[str]]:
    def keep(cols: list[str]) -> list[str]:
        return [c for c in cols if c in features]

    set_a = keep(["log10_sigma_NE", "log10_sigma_cNE_tau0", "log10_sigma_cNE_tau20"])
    set_b = keep(
        [
            "log10_sigma_NE",
            "log10_sigma_cNE_tau0",
            "log10_sigma_cNE_tau5",
            "log10_sigma_cNE_tau20",
            "log10_sigma_cNE_tau50",
        ]
    )
    set_c = set_b + keep(["log10_cNE20_over_NE", "neutral_pop_20ps", "charged_pop_20ps"])
    set_c = list(dict.fromkeys(set_c))
    set_d = list(features)
    return {
        "A_minimal": set_a,
        "B_recommended_compact": set_b,
        "C_compact_descriptor": set_c,
        "D_full_available": set_d,
    }


def check_feature_leakage(features: list[str]) -> None:
    banned_exact = {"group", "sample_group", "polymer_id", "polymer_key", "traj_id", "Trajectory ID", "tau_star"}
    banned_parts = ["sigma_ref", "s_target", "pred_ref", "abs_log_error", "MAlogE", "maloge", "target", "tau_star"]
    bad = []
    for feature in features:
        if feature in banned_exact or any(part.lower() in feature.lower() for part in banned_parts):
            bad.append(feature)
    if bad:
        raise RuntimeError("Leaky or label-derived features detected: " + ", ".join(bad))


def make_pipeline(sklearn: dict, model_name: str, alpha: float | None = None):
    if model_name == "hybrid_ridge":
        model = sklearn["Ridge"](alpha=1.0 if alpha is None else float(alpha))
    elif model_name == "hybrid_huber":
        model = sklearn["HuberRegressor"](alpha=1e-3, max_iter=10000)
    else:
        raise ValueError(f"Unknown hybrid model: {model_name}")
    return sklearn["Pipeline"]([("scaler", sklearn["StandardScaler"]()), ("model", model)])


def select_ridge_alpha(sklearn: dict, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3 or len(y) < 8:
        return 1.0
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        return 1.0
    scores = []
    splitter = sklearn["GroupKFold"](n_splits=n_splits)
    for alpha in RIDGE_ALPHAS:
        fold_errs = []
        for tr, va in splitter.split(x, y, groups):
            if len(tr) < 2 or len(va) == 0:
                continue
            pipe = make_pipeline(sklearn, "hybrid_ridge", alpha=alpha)
            pipe.fit(x[tr], y[tr])
            pred = pipe.predict(x[va])
            fold_errs.append(float(np.mean(np.abs(pred - y[va]))))
        if fold_errs:
            scores.append((float(np.mean(fold_errs)), alpha))
    if not scores:
        return 1.0
    return float(sorted(scores)[0][1])


def extract_coefficients(pipe, feature_set: str, model_name: str, features: list[str], context: dict) -> list[dict]:
    model = pipe.named_steps["model"]
    rows = []
    intercept = getattr(model, "intercept_", np.nan)
    if np.ndim(intercept):
        intercept = float(np.ravel(intercept)[0])
    rows.append({**context, "model": model_name, "feature_set": feature_set, "feature": "intercept", "coef": float(intercept)})
    coefs = np.ravel(getattr(model, "coef_", np.full(len(features), np.nan)))
    for feature, coef in zip(features, coefs):
        rows.append({**context, "model": model_name, "feature_set": feature_set, "feature": feature, "coef": float(coef)})
    return rows


def baseline_columns(dataset: pd.DataFrame) -> dict[str, str]:
    cols = {"NE": "log10_sigma_NE"}
    for tau in TAUS:
        col = f"log10_sigma_cNE_tau{tau_label(tau)}"
        if col in dataset.columns:
            cols[f"cNE_tau{tau_label(tau)}"] = col
    return cols


def add_prediction_rows(
    rows: list[dict],
    data: pd.DataFrame,
    indices: np.ndarray,
    log_pred: np.ndarray,
    validation_scheme: str,
    fold_id: str,
    model: str,
    feature_set: str,
    selected_alpha: float | None = None,
) -> None:
    for idx, pred in zip(indices, log_pred):
        ref = float(data.loc[idx, "log10_sigma_ref"])
        sigma_pred = float(10.0 ** float(pred))
        rows.append(
            {
                "validation_scheme": validation_scheme,
                "fold_id": fold_id,
                "model": model,
                "feature_set": feature_set,
                "selected_alpha": selected_alpha,
                "traj_id": int(data.loc[idx, "traj_id"]),
                "Trajectory ID": int(data.loc[idx, "Trajectory ID"]),
                "polymer_id": data.loc[idx, "polymer_id"],
                "group": data.loc[idx, "group"],
                "log10_sigma_ref": ref,
                "log10_sigma_pred": float(pred),
                "sigma_ref": float(data.loc[idx, "sigma_ref"]),
                "sigma_pred": sigma_pred,
                "pred_ref": float(10.0 ** (float(pred) - ref)),
                "abs_log_error": float(abs(float(pred) - ref)),
            }
        )


def best_global_tau_on_training(data: pd.DataFrame, train_idx: np.ndarray) -> tuple[str, str]:
    candidates = {k: v for k, v in baseline_columns(data).items() if k.startswith("cNE_tau")}
    scores = []
    y = data.loc[train_idx, "log10_sigma_ref"].to_numpy(float)
    for model, col in candidates.items():
        pred = data.loc[train_idx, col].to_numpy(float)
        if np.isfinite(pred).all():
            scores.append((float(np.mean(np.abs(pred - y))), model, col))
    if not scores:
        raise RuntimeError("No finite cNE tau columns available for best_global_tau selection.")
    _score, model, col = sorted(scores)[0]
    return model, col


@dataclass(frozen=True)
class CVResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    summary_by_group: pd.DataFrame
    coefficients: pd.DataFrame


def run_cv_scheme(
    data: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    splits: list[tuple[str, np.ndarray, np.ndarray]],
    validation_scheme: str,
    sklearn: dict,
) -> CVResult:
    all_features = sorted(set(c for cols in feature_sets.values() for c in cols))
    check_feature_leakage(all_features)
    pred_rows: list[dict] = []
    coef_rows: list[dict] = []
    base_cols = baseline_columns(data)

    for fold_id, train_idx, test_idx in splits:
        train_poly = set(data.loc[train_idx, "polymer_id"])
        test_poly = set(data.loc[test_idx, "polymer_id"])
        if validation_scheme in {"polymer_groupkfold", "leave_one_polymer_out"} and train_poly & test_poly:
            overlap = sorted(train_poly & test_poly)
            raise RuntimeError(f"polymer_id overlap in {validation_scheme} {fold_id}: {overlap[:5]}")

        for model, col in base_cols.items():
            add_prediction_rows(
                pred_rows,
                data,
                test_idx,
                data.loc[test_idx, col].to_numpy(float),
                validation_scheme,
                fold_id,
                model,
                "baseline",
            )

        best_tau_model, best_tau_col = best_global_tau_on_training(data, train_idx)
        add_prediction_rows(
            pred_rows,
            data,
            test_idx,
            data.loc[test_idx, best_tau_col].to_numpy(float),
            validation_scheme,
            fold_id,
            f"best_global_tau_train_selected:{best_tau_model}",
            "baseline",
        )

        y_train = data.loc[train_idx, "log10_sigma_ref"].to_numpy(float)
        groups_train = data.loc[train_idx, "polymer_id"].astype(str).to_numpy()
        for feature_set, features in feature_sets.items():
            x_all = data[features].to_numpy(float)
            if not np.isfinite(x_all).all():
                raise RuntimeError(f"Non-finite model input detected for feature set {feature_set}.")
            x_train = x_all[train_idx]
            x_test = x_all[test_idx]

            alpha = select_ridge_alpha(sklearn, x_train, y_train, groups_train)
            ridge = make_pipeline(sklearn, "hybrid_ridge", alpha=alpha)
            ridge.fit(x_train, y_train)
            add_prediction_rows(
                pred_rows,
                data,
                test_idx,
                ridge.predict(x_test),
                validation_scheme,
                fold_id,
                "hybrid_ridge",
                feature_set,
                selected_alpha=alpha,
            )
            coef_rows.extend(
                extract_coefficients(
                    ridge,
                    feature_set,
                    "hybrid_ridge",
                    features,
                    {"validation_scheme": validation_scheme, "fold_id": fold_id, "selected_alpha": alpha},
                )
            )

            if len(train_idx) >= 10:
                try:
                    huber = make_pipeline(sklearn, "hybrid_huber")
                    huber.fit(x_train, y_train)
                    add_prediction_rows(
                        pred_rows,
                        data,
                        test_idx,
                        huber.predict(x_test),
                        validation_scheme,
                        fold_id,
                        "hybrid_huber",
                        feature_set,
                        selected_alpha=1e-3,
                    )
                    coef_rows.extend(
                        extract_coefficients(
                            huber,
                            feature_set,
                            "hybrid_huber",
                            features,
                            {"validation_scheme": validation_scheme, "fold_id": fold_id, "selected_alpha": 1e-3},
                        )
                    )
                except Exception as exc:
                    warnings.warn(f"Huber failed for {validation_scheme} {fold_id} {feature_set}: {exc}")

    pred = pd.DataFrame(pred_rows)
    summary, summary_by_group = summarize_predictions(pred)
    coefs = pd.DataFrame(coef_rows)
    return CVResult(pred, summary, summary_by_group, coefs)


def make_groupkfold_splits(data: pd.DataFrame, sklearn: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups = data["polymer_id"].astype(str).to_numpy()
    unique = np.unique(groups)
    if len(unique) < 2:
        raise RuntimeError("Need at least two unique polymers for polymer-level GroupKFold.")
    n_splits = min(5, len(unique))
    splitter = sklearn["GroupKFold"](n_splits=n_splits)
    x_dummy = np.zeros((len(data), 1))
    y_dummy = data["log10_sigma_ref"].to_numpy(float)
    return [(f"fold_{i}", tr, te) for i, (tr, te) in enumerate(splitter.split(x_dummy, y_dummy, groups), start=1)]


def make_lopo_splits(data: pd.DataFrame, sklearn: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups = data["polymer_id"].astype(str).to_numpy()
    splitter = sklearn["LeaveOneGroupOut"]()
    x_dummy = np.zeros((len(data), 1))
    y_dummy = data["log10_sigma_ref"].to_numpy(float)
    splits = []
    for i, (tr, te) in enumerate(splitter.split(x_dummy, y_dummy, groups), start=1):
        held = sorted(set(groups[te]))[0]
        splits.append((f"polymer_{i}_{held}", tr, te))
    return splits


def make_leave_group_out_splits(data: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    groups = data["group"].astype(str).map(normalize_group).to_numpy()
    out = []
    for group in sorted(set(groups)):
        te = np.where(groups == group)[0]
        tr = np.where(groups != group)[0]
        if len(tr) and len(te):
            out.append((f"heldout_{group}", tr, te))
    return out


def direct_baseline_predictions(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = np.arange(len(data))
    for model, col in baseline_columns(data).items():
        add_prediction_rows(rows, data, idx, data[col].to_numpy(float), "direct_all_rows", "all", model, "baseline")
    return pd.DataFrame(rows)


def coefficient_stability(coefs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if coefs.empty:
        return pd.DataFrame()
    for key, sub in coefs.groupby(["validation_scheme", "model", "feature_set", "feature"], sort=True):
        vals = pd.to_numeric(sub["coef"], errors="coerce").dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        nonzero = vals[np.abs(vals) > 1e-12]
        if len(nonzero):
            sign_mode = 1.0 if np.sum(nonzero > 0) >= np.sum(nonzero < 0) else -1.0
            sign_consistency = float(np.mean(np.sign(nonzero) == sign_mode))
        else:
            sign_consistency = np.nan
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(
            {
                "validation_scheme": key[0],
                "model": key[1],
                "feature_set": key[2],
                "feature": key[3],
                "n_folds": int(len(vals)),
                "coef_mean": mean,
                "coef_std": std,
                "sign_consistency_fraction": sign_consistency,
                "coef_of_variation_abs": float(std / abs(mean)) if abs(mean) > 1e-12 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_coefficients(data: pd.DataFrame, feature_sets: dict[str, list[str]], sklearn: dict, n_boot: int = 200) -> pd.DataFrame:
    feature_set = "B_recommended_compact"
    features = feature_sets[feature_set]
    rng = np.random.default_rng(RANDOM_SEED)
    polymers = np.array(sorted(data["polymer_id"].unique()))
    rows = []
    for b in range(1, n_boot + 1):
        sampled = rng.choice(polymers, size=len(polymers), replace=True)
        parts = []
        for pid in sampled:
            parts.append(data[data["polymer_id"] == pid])
        boot = pd.concat(parts, ignore_index=True)
        x = boot[features].to_numpy(float)
        y = boot["log10_sigma_ref"].to_numpy(float)
        alpha = select_ridge_alpha(sklearn, x, y, boot["polymer_id"].astype(str).to_numpy())
        pipe = make_pipeline(sklearn, "hybrid_ridge", alpha=alpha)
        pipe.fit(x, y)
        rows.extend(
            extract_coefficients(
                pipe,
                feature_set,
                "hybrid_ridge",
                features,
                {"validation_scheme": "polymer_bootstrap", "fold_id": f"bootstrap_{b}", "selected_alpha": alpha},
            )
        )
    return pd.DataFrame(rows)


def choose_recommended_and_best(groupkfold_summary: pd.DataFrame) -> tuple[tuple[str, str], tuple[str, str]]:
    overall = groupkfold_summary.copy()
    if "group" in overall.columns:
        overall = overall[overall["group"].isna()] if overall["group"].isna().any() else overall
    hybrid = overall[overall["model"].astype(str).str.startswith("hybrid_")].copy()
    if hybrid.empty:
        raise RuntimeError("No hybrid model rows found in polymer GroupKFold summary.")
    rec = hybrid[hybrid["feature_set"].eq("B_recommended_compact")].sort_values("MAlogE")
    if rec.empty:
        rec = hybrid[hybrid["feature_set"].str.contains("compact", na=False)].sort_values("MAlogE")
    recommended = (str(rec.iloc[0]["model"]), str(rec.iloc[0]["feature_set"]))
    best = hybrid.sort_values("MAlogE").iloc[0]
    return recommended, (str(best["model"]), str(best["feature_set"]))


def ranking_metrics_for_predictions(pred: pd.DataFrame, methods: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for label, (model, feature_set) in methods.items():
        sub = pred[(pred["model"] == model) & (pred["feature_set"] == feature_set)].copy()
        if sub.empty:
            continue
        sub = sub.drop_duplicates("traj_id")
        n = len(sub)
        if n == 0:
            continue
        spearman = safe_corr(sub["log10_sigma_pred"], sub["log10_sigma_ref"], "spearman")
        kendall = try_kendall_tau(sub["log10_sigma_pred"], sub["log10_sigma_ref"])
        for frac in [0.05, 0.10, 0.20]:
            k = max(1, int(math.ceil(frac * n)))
            true_top = set(sub.nlargest(k, "log10_sigma_ref")["traj_id"])
            pred_top = set(sub.nlargest(k, "log10_sigma_pred")["traj_id"])
            hits = len(true_top & pred_top)
            precision = hits / k
            recall = hits / len(true_top) if true_top else np.nan
            rows.append(
                {
                    "method": label,
                    "model": model,
                    "feature_set": feature_set,
                    "n": n,
                    "top_fraction": frac,
                    "k": k,
                    "hits": hits,
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "top_k_hit_rate": recall,
                    "enrichment_factor": recall / frac if frac > 0 and np.isfinite(recall) else np.nan,
                    "spearman_r": spearman,
                    "kendall_tau": kendall,
                }
            )
    return pd.DataFrame(rows)


def make_plots(
    out_dir: Path,
    data: pd.DataFrame,
    groupkfold_pred: pd.DataFrame,
    groupkfold_summary: pd.DataFrame,
    old_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    boot: pd.DataFrame,
    recommended: tuple[str, str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_methods = [
        ("NE", "baseline", "NE"),
        ("cNE_tau0", "baseline", "cNE tau0"),
        ("cNE_tau20", "baseline", "cNE tau20"),
        (recommended[0], recommended[1], "hybrid recommended compact"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (model, feature_set, title) in zip(axes, plot_methods):
        sub = groupkfold_pred[(groupkfold_pred["model"] == model) & (groupkfold_pred["feature_set"] == feature_set)]
        ax.scatter(sub["log10_sigma_ref"], sub["log10_sigma_pred"], s=22, alpha=0.75)
        lo = min(sub["log10_sigma_ref"].min(), sub["log10_sigma_pred"].min())
        hi = max(sub["log10_sigma_ref"].max(), sub["log10_sigma_pred"].max())
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("log10 reference")
        ax.set_ylabel("log10 prediction")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "predicted_vs_reference_polymer_groupkfold.png", dpi=220)
    plt.close(fig)

    hybrid = groupkfold_pred[(groupkfold_pred["model"] == recommended[0]) & (groupkfold_pred["feature_set"] == recommended[1])].copy()
    hybrid["residual_log10"] = hybrid["log10_sigma_pred"] - hybrid["log10_sigma_ref"]
    groups = sorted(hybrid["group"].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([hybrid[hybrid["group"] == g]["residual_log10"].to_numpy(float) for g in groups], tick_labels=groups, showfliers=True)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("log10(pred/ref)")
    ax.set_title("Hybrid residual by group")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "hybrid_residual_by_group_polymer_cv.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([hybrid[hybrid["group"] == g]["pred_ref"].to_numpy(float) for g in groups], tick_labels=groups, showfliers=True)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("pred/ref")
    ax.set_title("Hybrid pred/ref by group")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "hybrid_pred_ref_distribution_by_group.png", dpi=220)
    plt.close(fig)

    overall = groupkfold_summary[groupkfold_summary["group"].isna()] if "group" in groupkfold_summary.columns else groupkfold_summary
    display = overall.copy()
    display["label"] = display["model"].astype(str) + "\n" + display["feature_set"].astype(str)
    display = display.sort_values("MAlogE").head(18)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(np.arange(len(display)), display["MAlogE"].to_numpy(float), color="tab:blue", alpha=0.85)
    ax.set_xticks(np.arange(len(display)))
    ax.set_xticklabels(display["label"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("MAlogE")
    ax.set_title("Polymer GroupKFold MAlogE")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "MAlogE_comparison_polymer_cv.png", dpi=220)
    plt.close(fig)

    old = old_summary[(old_summary["validation_scheme"] == "LOOCV") & (old_summary["group"].isna())].copy()
    rows = []
    for model in ["hybrid_huber", "hybrid_ridge", "NE", "cNE_tau0", "cNE_tau20"]:
        old_row = old[old["model"] == model]
        if not old_row.empty:
            rows.append({"source": "trajectory_LOOCV", "model": model, "MAlogE": float(old_row.iloc[0]["MAlogE"])})
        new_row = overall[(overall["model"] == model) & (overall["feature_set"].isin(["baseline", recommended[1]]))]
        if not new_row.empty:
            best = new_row.sort_values("MAlogE").iloc[0]
            rows.append({"source": "polymer_GroupKFold", "model": model, "MAlogE": float(best["MAlogE"])})
    comp = pd.DataFrame(rows)
    if not comp.empty:
        labels = sorted(comp["model"].unique())
        x = np.arange(len(labels))
        width = 0.36
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for offset, source in [(-width / 2, "trajectory_LOOCV"), (width / 2, "polymer_GroupKFold")]:
            vals = [comp[(comp["model"] == m) & (comp["source"] == source)]["MAlogE"].min() if not comp[(comp["model"] == m) & (comp["source"] == source)].empty else np.nan for m in labels]
            ax.bar(x + offset, vals, width=width, label=source)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("MAlogE")
        ax.set_title("Trajectory LOOCV vs polymer GroupKFold")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "trajectory_loocv_vs_polymer_cv.png", dpi=220)
        plt.close(fig)

    rank_methods = [
        ("NE", "baseline", "NE"),
        ("cNE_tau0", "baseline", "cNE tau0"),
        ("cNE_tau20", "baseline", "cNE tau20"),
        (recommended[0], recommended[1], "hybrid"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (model, feature_set, title) in zip(axes, rank_methods):
        sub = groupkfold_pred[(groupkfold_pred["model"] == model) & (groupkfold_pred["feature_set"] == feature_set)].copy()
        sub["ref_rank"] = sub["log10_sigma_ref"].rank(ascending=False, method="average")
        sub["pred_rank"] = sub["log10_sigma_pred"].rank(ascending=False, method="average")
        ax.scatter(sub["ref_rank"], sub["pred_rank"], s=20, alpha=0.7)
        lo, hi = 1, len(sub)
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("reference rank")
        ax.set_ylabel("predicted rank")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "rank_scatter_hybrid_vs_baselines.png", dpi=220)
    plt.close(fig)

    r10 = ranking[np.isclose(ranking["top_fraction"], 0.10)].copy()
    if not r10.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        r10 = r10.sort_values("enrichment_factor", ascending=False)
        ax.bar(np.arange(len(r10)), r10["enrichment_factor"].to_numpy(float), color="tab:green", alpha=0.85)
        ax.set_xticks(np.arange(len(r10)))
        ax.set_xticklabels(r10["method"], rotation=35, ha="right")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("enrichment factor at top 10%")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "enrichment_barplot.png", dpi=220)
        plt.close(fig)

    bsub = boot[(boot["model"] == "hybrid_ridge") & (boot["feature_set"] == "B_recommended_compact") & (boot["feature"] != "intercept")].copy()
    if not bsub.empty:
        features = list(dict.fromkeys(bsub["feature"]))
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.boxplot([bsub[bsub["feature"] == f]["coef"].to_numpy(float) for f in features], tick_labels=features, showfliers=False)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("scaled coefficient")
        ax.set_title("Bootstrap coefficient distribution, recommended compact ridge")
        ax.tick_params(axis="x", rotation=55)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "coefficient_bootstrap_boxplot.png", dpi=220)
        plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    show = df[cols].head(max_rows).copy()
    return show.to_markdown(index=False)


def write_report(
    out_dir: Path,
    data: pd.DataFrame,
    mapping: pd.DataFrame,
    old_summary: pd.DataFrame,
    g_summary: pd.DataFrame,
    lopo_summary: pd.DataFrame,
    logo_summary: pd.DataFrame,
    coef_summary: pd.DataFrame,
    ranking: pd.DataFrame,
    recommended: tuple[str, str],
    best: tuple[str, str],
) -> None:
    old_overall = old_summary[(old_summary["validation_scheme"] == "LOOCV") & (old_summary["group"].isna())].copy()
    old_keep = old_overall[old_overall["model"].isin(["hybrid_huber", "hybrid_ridge", "NE", "cNE_tau0", "cNE_tau20"])].sort_values("MAlogE")

    def overall(summary: pd.DataFrame) -> pd.DataFrame:
        if summary.empty:
            return summary
        return summary[summary["group"].isna()].copy() if "group" in summary.columns else summary.copy()

    g_overall = overall(g_summary).sort_values("MAlogE")
    lopo_overall = overall(lopo_summary).sort_values("MAlogE")
    logo_overall = logo_summary.sort_values(["fold_id" if "fold_id" in logo_summary.columns else "model", "MAlogE"]) if not logo_summary.empty else logo_summary
    rec_rows = g_overall[(g_overall["model"] == recommended[0]) & (g_overall["feature_set"] == recommended[1])]
    rec_text = "not available"
    if not rec_rows.empty:
        r = rec_rows.iloc[0]
        rec_text = f"{recommended[0]} / {recommended[1]}: MAlogE={r['MAlogE']:.3f}, median pred/ref={r['median_pred_ref']:.3g}x"

    best_rows = g_overall[(g_overall["model"] == best[0]) & (g_overall["feature_set"] == best[1])]
    best_text = "not available"
    if not best_rows.empty:
        r = best_rows.iloc[0]
        best_text = f"{best[0]} / {best[1]}: MAlogE={r['MAlogE']:.3f}, median pred/ref={r['median_pred_ref']:.3g}x"

    stable = coef_summary[
        (coef_summary["validation_scheme"] == "polymer_groupkfold")
        & (coef_summary["model"] == recommended[0])
        & (coef_summary["feature_set"] == recommended[1])
        & (coef_summary["feature"] != "intercept")
    ].sort_values("coef_std")

    rank10 = ranking[np.isclose(ranking["top_fraction"], 0.10)].sort_values("enrichment_factor", ascending=False)
    method = mapping["polymer_id_inference_method"].iloc[0] if not mapping.empty else "unknown"

    text = f"""# Polymer-Level Hybrid Validation Report

## Purpose

Trajectory-level LOOCV can be optimistic when multiple trajectories or replicas belong to the same polymer. This analysis holds out all trajectories from the same polymer together to test whether the hybrid correction generalizes at the polymer/system level.

Rows: {len(data)}

Unique trajectories: {data['traj_id'].nunique()}

Unique polymers: {data['polymer_id'].nunique()}

Polymer id inference: `{method}`. The saved mapping keeps the original polymer key/SMILES.

## Main Interpretation

The persistence-threshold sweep revealed that no single lifetime threshold is transferable across conductivity regimes. A 20 ps persistence filter accurately corrected the low-conductivity group, consistent with transport suppression by long-lived Li-anion associations, but it overcorrected the middle-conductivity group and failed to correct the high-conductivity group where long-lived contacts are rare. Therefore, persistence-filtered cNE is used as a physically interpretable diagnostic rather than a universal conductivity estimator.

To obtain a transferable screening-level validation metric, we combined NE and persistence-filtered cNE baselines using a cross-validated hybrid estimator. Polymer-level cross-validation was used to ensure that trajectories from the same polymer did not appear in both training and test sets.

The hybrid estimator should not be interpreted as a universal transport model. Instead, it is a calibrated screening correction for short-trajectory MD validation of generated polymer electrolyte candidates.

## Existing Trajectory-Level LOOCV

{md_table(old_keep, ['model', 'n', 'MAlogE', 'median_abs_log_error', 'median_pred_ref', 'spearman_r', 'pearson_r', 'R2'])}

## Polymer GroupKFold

Recommended compact model: {rec_text}

Best polymer-level CV model: {best_text}

{md_table(g_overall, ['model', 'feature_set', 'n', 'MAlogE', 'median_abs_log_error', 'median_pred_ref', 'spearman_r', 'pearson_r', 'R2'], max_rows=18)}

## Leave-One-Polymer-Out

{md_table(lopo_overall, ['model', 'feature_set', 'n', 'MAlogE', 'median_abs_log_error', 'median_pred_ref', 'spearman_r', 'pearson_r', 'R2'], max_rows=18)}

## Leave-One-Conductivity-Group-Out

Diagnostic only. Failures, especially on top, should be reported rather than hidden because they indicate that high-conductivity systems may require collective conductivity estimators.

{md_table(logo_overall, ['fold_id', 'heldout_group', 'model', 'feature_set', 'n', 'MAlogE', 'median_pred_ref', 'spearman_r', 'pearson_r', 'R2'], max_rows=60)}

## Coefficient Stability

Recommended compact model coefficients from polymer GroupKFold:

{md_table(stable, ['feature', 'n_folds', 'coef_mean', 'coef_std', 'sign_consistency_fraction', 'coef_of_variation_abs'], max_rows=20)}

## Ranking and Enrichment

Top-10% enrichment:

{md_table(rank10, ['method', 'n', 'k', 'hits', 'precision_at_k', 'recall_at_k', 'enrichment_factor', 'spearman_r', 'kendall_tau'], max_rows=20)}

## Key Plots

- [Predicted vs reference](plots/predicted_vs_reference_polymer_groupkfold.png)
- [Hybrid residual by group](plots/hybrid_residual_by_group_polymer_cv.png)
- [Hybrid pred/ref by group](plots/hybrid_pred_ref_distribution_by_group.png)
- [MAlogE comparison](plots/MAlogE_comparison_polymer_cv.png)
- [Trajectory LOOCV vs polymer CV](plots/trajectory_loocv_vs_polymer_cv.png)
- [Rank scatter](plots/rank_scatter_hybrid_vs_baselines.png)
- [Enrichment barplot](plots/enrichment_barplot.png)
- [Coefficient bootstrap](plots/coefficient_bootstrap_boxplot.png)

## Manuscript-Ready Wording

> The persistence-threshold sweep revealed that no single lifetime threshold is transferable across conductivity regimes. A 20 ps persistence filter accurately corrected the low-conductivity group, consistent with transport suppression by long-lived Li-anion associations, but it overcorrected the middle-conductivity group and failed to correct the high-conductivity group where long-lived contacts are rare. Therefore, persistence-filtered cNE is used as a physically interpretable diagnostic rather than a universal conductivity estimator.

> To obtain a transferable screening-level validation metric, we combined NE and persistence-filtered cNE baselines using a cross-validated hybrid estimator. Polymer-level cross-validation was used to ensure that trajectories from the same polymer did not appear in both training and test sets.

> The hybrid estimator should not be interpreted as a universal transport model. Instead, it is a calibrated screening correction for short-trajectory MD validation of generated polymer electrolyte candidates.

## Output Files

- `model_dataset.csv`
- `polymer_group_mapping.csv`
- `polymer_groupkfold_predictions.csv`
- `polymer_groupkfold_summary.csv`
- `leave_one_polymer_out_predictions.csv`
- `leave_one_polymer_out_summary.csv`
- `leave_one_conductivity_group_out_predictions.csv`
- `leave_one_conductivity_group_out_summary.csv`
- `coefficient_stability_summary.csv`
- `bootstrap_coefficient_distribution.csv`
- `ranking_enrichment_summary.csv`
- `validation_checks.txt`
"""
    write_text(out_dir / "README_polymer_level_validation_report.md", text)


def validation_checks_text(
    data: pd.DataFrame,
    features: list[str],
    feature_sets: dict[str, list[str]],
    all_cv_pred: pd.DataFrame,
    mapping: pd.DataFrame,
    schema: str,
) -> str:
    lines = ["# Validation Checks", ""]
    lines.append(f"rows={len(data)}")
    lines.append(f"unique_trajectories={data['traj_id'].nunique()}")
    lines.append(f"unique_polymers={data['polymer_id'].nunique()}")
    dup_poly = mapping.groupby("polymer_id")["traj_id"].nunique()
    lines.append(f"polymers_with_multiple_trajectories={int((dup_poly > 1).sum())}")
    lines.append("")
    lines.append("## Feature safety")
    check_feature_leakage(features)
    lines.append("no_ref_or_target_derived_features=true")
    lines.append("no_group_label_features=true")
    for name, cols in feature_sets.items():
        arr = data[cols].to_numpy(float)
        lines.append(f"{name}_finite_inputs={bool(np.isfinite(arr).all())}")
    lines.append("")
    lines.append("## Polymer overlap")
    lines.append("polymer_train_test_overlap_detected=false")
    lines.append("overlap_policy=enforced inside polymer-level CV; the script raises RuntimeError on any train/test polymer_id intersection")
    for fold in sorted(all_cv_pred.loc[all_cv_pred["validation_scheme"].eq("polymer_groupkfold"), "fold_id"].unique()):
        lines.append(f"{fold}:overlap_check_passed")
    lines.append("")
    lines.append("## Positive predictions")
    lines.append(f"all_sigma_pred_positive={bool((all_cv_pred['sigma_pred'] > 0).all())}")
    lines.append("")
    lines.append("## Input schema")
    lines.append(schema)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()

    base = args.base.resolve()
    out_dir = base / "results" / "hybrid_correction_polymer_cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    sklearn = require_sklearn()
    sweep, old_pred, old_summary, manifest = load_inputs(base)
    schema = schema_log(out_dir, sweep, old_pred, old_summary, manifest)
    print(schema)

    mapping = infer_polymer_mapping(base, sweep, manifest, out_dir)
    data, features, feature_sets = pivot_sweep_to_dataset(sweep, mapping, out_dir)
    check_feature_leakage(features)

    direct_pred = direct_baseline_predictions(data)
    direct_summary, direct_by_group = summarize_predictions(direct_pred)

    g_splits = make_groupkfold_splits(data, sklearn)
    g_res = run_cv_scheme(data, feature_sets, g_splits, "polymer_groupkfold", sklearn)
    g_res.predictions.to_csv(out_dir / "polymer_groupkfold_predictions.csv", index=False)
    g_res.summary.to_csv(out_dir / "polymer_groupkfold_summary.csv", index=False)
    g_res.summary_by_group.to_csv(out_dir / "polymer_groupkfold_summary_by_group.csv", index=False)

    lopo_splits = make_lopo_splits(data, sklearn)
    lopo_res = run_cv_scheme(data, feature_sets, lopo_splits, "leave_one_polymer_out", sklearn)
    lopo_res.predictions.to_csv(out_dir / "leave_one_polymer_out_predictions.csv", index=False)
    lopo_res.summary.to_csv(out_dir / "leave_one_polymer_out_summary.csv", index=False)
    lopo_res.summary_by_group.to_csv(out_dir / "leave_one_polymer_out_summary_by_group.csv", index=False)

    logo_splits = make_leave_group_out_splits(data)
    logo_res = run_cv_scheme(data, feature_sets, logo_splits, "leave_one_conductivity_group_out", sklearn)
    logo_fold_summary = summarize_predictions_by_fold(logo_res.predictions)
    logo_res.predictions.to_csv(out_dir / "leave_one_conductivity_group_out_predictions.csv", index=False)
    logo_fold_summary.to_csv(out_dir / "leave_one_conductivity_group_out_summary.csv", index=False)

    baseline_cv = g_res.predictions[g_res.predictions["feature_set"].eq("baseline")].copy()
    baseline_cv_summary, baseline_cv_by_group = summarize_predictions(baseline_cv)
    baseline_summary = pd.concat([direct_summary, baseline_cv_summary], ignore_index=True)
    baseline_summary_by_group = pd.concat([direct_by_group, baseline_cv_by_group], ignore_index=True)
    baseline_summary.to_csv(out_dir / "baseline_summary.csv", index=False)
    baseline_summary_by_group.to_csv(out_dir / "baseline_summary_by_group.csv", index=False)

    coefs = pd.concat([g_res.coefficients, lopo_res.coefficients, logo_res.coefficients], ignore_index=True)
    coefs.to_csv(out_dir / "coefficient_stability_by_fold.csv", index=False)
    coef_summary = coefficient_stability(coefs)
    coef_summary.to_csv(out_dir / "coefficient_stability_summary.csv", index=False)

    boot = bootstrap_coefficients(data, feature_sets, sklearn, n_boot=max(1, int(args.bootstrap)))
    boot.to_csv(out_dir / "bootstrap_coefficient_distribution.csv", index=False)

    recommended, best = choose_recommended_and_best(g_res.summary)
    methods = {
        "NE": ("NE", "baseline"),
        "cNE_tau0": ("cNE_tau0", "baseline"),
        "cNE_tau20": ("cNE_tau20", "baseline"),
    }
    best_global_rows = g_res.summary[g_res.summary["model"].astype(str).str.startswith("best_global_tau_train_selected")]
    if not best_global_rows.empty:
        bg = best_global_rows[best_global_rows["feature_set"].eq("baseline")].sort_values("MAlogE").iloc[0]
        methods["best_global_tau"] = (str(bg["model"]), "baseline")
    methods["hybrid_recommended_compact"] = recommended
    methods["hybrid_best_polymer_cv"] = best
    ranking = ranking_metrics_for_predictions(g_res.predictions, methods)
    ranking.to_csv(out_dir / "ranking_enrichment_summary.csv", index=False)

    make_plots(out_dir, data, g_res.predictions, g_res.summary, old_summary, ranking, boot, recommended)
    write_report(out_dir, data, mapping, old_summary, g_res.summary, lopo_res.summary, logo_fold_summary, coef_summary, ranking, recommended, best)
    all_cv_pred = pd.concat([g_res.predictions, lopo_res.predictions, logo_res.predictions], ignore_index=True)
    write_text(out_dir / "validation_checks.txt", validation_checks_text(data, features, feature_sets, all_cv_pred, mapping, schema))

    print(f"[done] {out_dir / 'model_dataset.csv'}")
    print(f"[done] {out_dir / 'polymer_groupkfold_summary.csv'}")
    print(f"[done] {out_dir / 'leave_one_polymer_out_summary.csv'}")
    print(f"[done] {out_dir / 'README_polymer_level_validation_report.md'}")


if __name__ == "__main__":
    main()
