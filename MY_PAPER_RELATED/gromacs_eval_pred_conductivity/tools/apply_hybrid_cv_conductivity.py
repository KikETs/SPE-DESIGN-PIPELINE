from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any

import numpy as np
import pandas as pd


KB = 1.380649e-23
E_CHG = 1.602176634e-19
DEFAULT_TEMPERATURE_K = 353.0
_CSV_CACHE: dict[Path, pd.DataFrame | None] = {}


def _cfg(config: Any) -> dict[str, Any]:
    return getattr(config, "cfg", {}) or {}


def _path_from_config(config: Any, key: str, default: Path) -> Path:
    value = _cfg(config).get(key, None)
    return Path(value).expanduser().resolve() if value else default.expanduser().resolve()


def load_artifact(config: Any) -> tuple[dict[str, Any], Path]:
    artifact_dir = _path_from_config(
        config,
        "hybrid_cv_artifact_dir",
        Path(getattr(config, "out_dir", Path.cwd())) / "hybrid_cv_artifacts",
    )
    artifact_file = str(_cfg(config).get("hybrid_cv_artifact_file", "") or "").strip()
    model = str(_cfg(config).get("hybrid_cv_model", "polybert_md_huber"))
    feature_set = str(_cfg(config).get("hybrid_cv_feature_set", "polybert_md_minimal"))
    if not artifact_file:
        artifact_file = f"final_{model}_{feature_set}.json"
    path = artifact_dir / artifact_file
    if not path.exists():
        raise FileNotFoundError(f"Hybrid CV artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8")), path


def _read_analysis_summary(config: Any, row: pd.Series) -> tuple[pd.Series | None, Path | None]:
    candidates: list[Path] = []
    analysis_csv = row.get("analysis_csv", None)
    if isinstance(analysis_csv, str) and analysis_csv.strip() and analysis_csv.strip().lower() != "nan":
        candidates.append(Path(analysis_csv).expanduser())
    traj_id = int(row["Trajectory ID"])
    runs_dir = Path(getattr(config, "runs_dir", Path.cwd() / "runs"))
    candidates.append(runs_dir / f"Traj_{traj_id}" / "analysis" / "conductivity_summary_htpmd_ref.csv")
    for path in candidates:
        if path.exists():
            return pd.read_csv(path).iloc[0], path
    return None, None


def _read_csv_cached(path: Path) -> pd.DataFrame | None:
    path = path.expanduser().resolve()
    if path not in _CSV_CACHE:
        if path.exists():
            try:
                _CSV_CACHE[path] = pd.read_csv(path)
            except Exception:
                _CSV_CACHE[path] = None
        else:
            _CSV_CACHE[path] = None
    return _CSV_CACHE[path]


def _polybert_lookup_tables(config: Any) -> list[Path]:
    out_dir = Path(getattr(config, "out_dir", Path.cwd()))
    artifact_dir = _path_from_config(config, "hybrid_cv_artifact_dir", out_dir / "hybrid_cv_artifacts")
    paths: list[Path] = []
    for key in ("hybrid_cv_polybert_prediction_csv", "hybrid_cv_polybert_oof_csv"):
        val = str(_cfg(config).get(key, "") or "").strip()
        if val:
            paths.append(Path(val).expanduser())
    paths.extend(
        [
            out_dir / "simulation-trajectory-aggregate.csv",
            out_dir / "results" / "sample_manifest.csv",
            artifact_dir / "polybert_oof_predictions.csv",
            artifact_dir / "oof_predictions.csv",
        ]
    )
    seen = set()
    unique = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _first_finite_log10(row: pd.Series) -> float:
    for col in ("log10_sigma_polybert", "pred_log10_cond", "pred_log10_conductivity", "polybert_pred_log10_cond"):
        if col in row.index:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if np.isfinite(val):
                return float(val)
    for col in ("sigma_polybert", "pred_cond", "polybert_pred_cond"):
        if col in row.index:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if np.isfinite(val) and val > 0:
                return float(math.log10(float(val)))
    return float("nan")


def _polybert_log10_for_row(config: Any, row: pd.Series) -> tuple[float, str]:
    # Reference-set rows must use PolyBERT OOF predictions, not the true
    # CONDUCTIVITY column. Generated-candidate rows use the PolyBERT prediction
    # used for candidate selection.
    val = _first_finite_log10(row)
    if np.isfinite(val):
        return val, "run_results"

    traj_id = pd.to_numeric(row.get("Trajectory ID", np.nan), errors="coerce")
    if np.isfinite(traj_id):
        for path in _polybert_lookup_tables(config):
            table = _read_csv_cached(path)
            if table is None or "Trajectory ID" not in table.columns:
                continue
            ids = pd.to_numeric(table["Trajectory ID"], errors="coerce")
            hit = table.loc[ids == int(traj_id)]
            if hit.empty:
                continue
            val = _first_finite_log10(hit.iloc[0])
            if np.isfinite(val):
                return val, str(path)

    if bool(_cfg(config).get("hybrid_cv_allow_conductivity_as_polybert_prior", False)):
        cond = pd.to_numeric(row.get("CONDUCTIVITY", np.nan), errors="coerce")
        if np.isfinite(cond) and cond > 0:
            return float(math.log10(float(cond))), "CONDUCTIVITY_fallback"
    return float("nan"), ""


def _read_pop_matrix(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path).to_numpy(float)
    except Exception:
        return None


def _sigma_cne_from_pop(pop: np.ndarray, d_li_cm2s: float, d_an_cm2s: float, volume_nm3: float, temperature_k: float, max_cluster: int) -> float:
    # Formal cluster net charge is z_ij = i - j. alpha_ij is a population count,
    # not a normalized fraction.
    if pop is None or not np.isfinite(volume_nm3) or volume_nm3 <= 0:
        return float("nan")
    v_cm3 = volume_nm3 * 1e-21
    pref = (E_CHG**2) / (KB * temperature_k * v_cm3)
    sigma = 0.0
    imax = min(int(max_cluster), pop.shape[0])
    jmax = min(int(max_cluster), pop.shape[1])
    for i in range(imax):
        for j in range(jmax):
            nij = float(pop[i, j])
            if nij == 0.0 or i == j:
                continue
            q = i - j
            d_eff = d_li_cm2s if q > 0 else d_an_cm2s
            sigma += pref * (q * q) * nij * d_eff
    return float(sigma)


def _positive_log10(value: Any) -> float:
    val = pd.to_numeric(value, errors="coerce")
    if np.isfinite(val) and val > 0:
        return float(math.log10(float(val)))
    return float("nan")


def _analysis_dir(summary_path: Path | None, config: Any, traj_id: int) -> Path:
    if summary_path is not None:
        return summary_path.parent
    return Path(getattr(config, "runs_dir", Path.cwd() / "runs")) / f"Traj_{traj_id}" / "analysis"


def build_features_for_row(config: Any, row: pd.Series) -> tuple[dict[str, float], list[str], dict[str, float]]:
    summary, summary_path = _read_analysis_summary(config, row)
    traj_id = int(row["Trajectory ID"])
    analysis_dir = _analysis_dir(summary_path, config, traj_id)
    temperature_k = float(_cfg(config).get("hybrid_cv_temperature_k", DEFAULT_TEMPERATURE_K))
    max_cluster = int(_cfg(config).get("hybrid_cv_max_cluster", 101))
    features: dict[str, float] = {}
    raw: dict[str, float] = {}

    if summary is not None:
        sigma_ne = pd.to_numeric(summary.get("sigma_NE_htpmd_S_cm", np.nan), errors="coerce")
        d_li = float(pd.to_numeric(summary.get("D_Li_cm2s", np.nan), errors="coerce"))
        d_an = float(pd.to_numeric(summary.get("D_an_cm2s", np.nan), errors="coerce"))
        v_nm3 = float(pd.to_numeric(summary.get("V_nm3", np.nan), errors="coerce"))
        summary_cne = pd.to_numeric(summary.get("sigma_cNE_htpmd_S_cm", np.nan), errors="coerce")
        threshold = pd.to_numeric(summary.get("cluster_persistence_threshold_ps", np.nan), errors="coerce")
    else:
        sigma_ne = pd.to_numeric(row.get("sigma_NE_htpmd_S_cm_pred", np.nan), errors="coerce")
        d_li = float(pd.to_numeric(row.get("D_Li_cm2s_pred", np.nan), errors="coerce"))
        d_an = float(pd.to_numeric(row.get("D_an_cm2s_pred", np.nan), errors="coerce"))
        v_nm3 = float("nan")
        summary_cne = pd.to_numeric(row.get("sigma_cNE_htpmd_S_cm_pred", np.nan), errors="coerce")
        threshold = float("nan")

    raw["sigma_NE"] = float(sigma_ne) if np.isfinite(sigma_ne) else float("nan")
    features["log10_sigma_NE"] = _positive_log10(sigma_ne)

    log10_polybert, polybert_source = _polybert_log10_for_row(config, row)
    raw["log10_sigma_polybert_prior"] = log10_polybert
    raw["sigma_polybert_prior"] = float(10.0 ** log10_polybert) if np.isfinite(log10_polybert) else float("nan")
    raw["polybert_prior_source"] = polybert_source
    features["log10_sigma_polybert"] = log10_polybert

    pop0 = _read_pop_matrix(analysis_dir / "pop_mat.csv")
    sigma0 = _sigma_cne_from_pop(pop0, d_li, d_an, v_nm3, temperature_k, max_cluster)
    raw["sigma_cNE_tau0"] = sigma0
    features["log10_sigma_cNE_tau0"] = _positive_log10(sigma0)

    for tau in (5, 20, 50):
        label = f"{tau:g}".replace(".", "p")
        pop_path = analysis_dir / f"pop_mat_persist_cutoff0p28_ge{label}ps.csv"
        sigma = _sigma_cne_from_pop(_read_pop_matrix(pop_path), d_li, d_an, v_nm3, temperature_k, max_cluster)
        if tau == 20 and not (np.isfinite(sigma) and sigma > 0):
            if np.isfinite(threshold) and abs(float(threshold) - 20.0) < 1e-9:
                sigma = float(summary_cne) if np.isfinite(summary_cne) else float("nan")
            elif str(row.get("sigma_eval_mode_pred", "")).strip() == "cNE_persist_ge_20ps":
                sigma = float(row.get("sigma_cNE_htpmd_S_cm_pred", np.nan))
        raw[f"sigma_cNE_tau{tau:g}"] = sigma
        features[f"log10_sigma_cNE_tau{tau:g}"] = _positive_log10(sigma)

    missing = [k for k, v in features.items() if not np.isfinite(v)]
    return features, missing, raw


def predict_from_features(
    artifact: dict[str, Any],
    features: dict[str, float],
    *,
    clip_polybert: bool = True,
) -> tuple[float, list[str], dict[str, float], list[str]]:
    feature_order = list(artifact["features"])
    missing = [f for f in feature_order if f not in features or not np.isfinite(features[f])]
    if missing:
        return float("nan"), missing, {}, []
    model_features = {f: float(features[f]) for f in feature_order}
    clipped: list[str] = []
    ranges = artifact.get("feature_ranges", {}) or {}
    if clip_polybert and "log10_sigma_polybert" in model_features and "log10_sigma_polybert" in ranges:
        bounds = ranges["log10_sigma_polybert"]
        lo = pd.to_numeric(bounds.get("min", np.nan), errors="coerce")
        hi = pd.to_numeric(bounds.get("max", np.nan), errors="coerce")
        val = model_features["log10_sigma_polybert"]
        if np.isfinite(lo) and np.isfinite(hi):
            new_val = float(np.clip(val, float(lo), float(hi)))
            if new_val != val:
                clipped.append("log10_sigma_polybert")
                model_features["log10_sigma_polybert"] = new_val
    x = np.array([model_features[f] for f in feature_order], dtype=float)
    mean = np.array(artifact["standard_scaler_mean"], dtype=float)
    scale = np.array(artifact["standard_scaler_scale"], dtype=float)
    coef = np.array(artifact["coef"], dtype=float)
    z = (x - mean) / scale
    log_pred = float(artifact["intercept"] + np.dot(coef, z))
    return log_pred, [], model_features, clipped


def apply_hybrid_cv_from_config(config: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact, artifact_path = load_artifact(config)
    results_dir = Path(getattr(config, "results_dir", Path.cwd() / "results"))
    run_csv = results_dir / "run_results.csv"
    if not run_csv.exists():
        raise FileNotFoundError(f"run_results.csv not found: {run_csv}")
    run_df = pd.read_csv(run_csv)
    rows = []
    for _, row in run_df.iterrows():
        traj_id = int(row["Trajectory ID"])
        status = str(row.get("status", ""))
        features, feature_missing, raw = build_features_for_row(config, row)
        clip_polybert = bool(_cfg(config).get("hybrid_cv_clip_polybert_to_training_range", True))
        log_pred, model_missing, model_features, clipped = predict_from_features(
            artifact,
            features,
            clip_polybert=clip_polybert,
        )
        sigma_pred = 10.0 ** log_pred if np.isfinite(log_pred) else float("nan")
        ref = pd.to_numeric(row.get(str(_cfg(config).get("sigma_ref_col", "CONDUCTIVITY")), row.get("CONDUCTIVITY", np.nan)), errors="coerce")
        pred_ref = sigma_pred / float(ref) if np.isfinite(sigma_pred) and np.isfinite(ref) and ref > 0 else float("nan")
        rows.append(
            {
                "Trajectory ID": traj_id,
                "sample_group": row.get("sample_group", np.nan),
                "status": status,
                "hybrid_cv_model": artifact["model_name"],
                "hybrid_cv_feature_set": artifact["feature_set"],
                "hybrid_cv_artifact": str(artifact_path),
                "sigma_hybrid_cv_S_cm": sigma_pred,
                "log10_sigma_hybrid_cv": log_pred,
                "CONDUCTIVITY": float(ref) if np.isfinite(ref) else float("nan"),
                "hybrid_cv_pred_ref": pred_ref,
                "hybrid_cv_abs_log_error": abs(math.log10(pred_ref)) if np.isfinite(pred_ref) and pred_ref > 0 else float("nan"),
                "hybrid_cv_missing_features": ",".join(model_missing),
                "hybrid_cv_clipped_features": ",".join(clipped),
                **raw,
                **features,
                **{f"{k}_model_input": v for k, v in model_features.items()},
            }
        )
    pred_df = pd.DataFrame(rows).sort_values(["sample_group", "Trajectory ID"])
    out_csv = Path(str(_cfg(config).get("hybrid_cv_output_csv", "") or results_dir / "hybrid_cv_predictions.csv")).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_csv, index=False)

    ok = pred_df["sigma_hybrid_cv_S_cm"].notna() & np.isfinite(pred_df["sigma_hybrid_cv_S_cm"])
    err = pred_df.loc[ok, "hybrid_cv_abs_log_error"]
    summary = pd.DataFrame(
        [
            {
                "artifact": str(artifact_path),
                "model": artifact["model_name"],
                "feature_set": artifact["feature_set"],
                "n_total": int(len(pred_df)),
                "n_pred": int(ok.sum()),
                "n_missing": int((~ok).sum()),
                "MAlogE": float(err.mean(skipna=True)) if err.notna().any() else float("nan"),
                "median_abs_log_error": float(err.median(skipna=True)) if err.notna().any() else float("nan"),
                "median_pred_ref": float(pred_df.loc[ok, "hybrid_cv_pred_ref"].median(skipna=True)) if ok.any() else float("nan"),
                "comparison_label": str(_cfg(config).get("hybrid_cv_comparison_label", "CONDUCTIVITY")),
                "output_csv": str(out_csv),
            }
        ]
    )
    summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
    summary.to_csv(summary_csv, index=False)
    return pred_df, summary
