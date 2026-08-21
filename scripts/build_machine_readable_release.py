#!/usr/bin/env python
"""Build the compact, machine-readable manuscript data release."""

from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "MY_PAPER_RELATED" / "machine_readable"

GENERATED_POOL = Path("MY_PAPER_RELATED/polybert_con/all_novel_smiles.csv")
GENERATED_PREDICTIONS = Path(
    "MY_PAPER_RELATED/polybert_weighted_evidence/source_data/"
    "polybert_run/all_novel_smiles_with_pred_conductivity.csv"
)
WEIGHTED_GENERATED_PREDICTIONS = Path(
    "MY_PAPER_RELATED/polybert_weighted_evidence/tables/"
    "weighted_generated_candidate_predictions.csv"
)
WEIGHTED_CONDITION_PREDICTIONS = Path(
    "MY_PAPER_RELATED/polybert_weighted_evidence/tables/"
    "weighted_generated_condition_predictions_47125.csv"
)
WEIGHTED_MD_SELECTION = Path(
    "MY_PAPER_RELATED/polybert_weighted_evidence/tables/"
    "weighted_generated_md_selection_60.csv"
)
TRAINING_FOLDS = Path("MY_PAPER_RELATED/polybert_con/fold_assignment.csv")
OOF_PREDICTIONS = Path("MY_PAPER_RELATED/polybert_con/oof_predictions.csv")
WEIGHTED_SELECTION = Path(
    "MY_PAPER_RELATED/polybert_con/weighted_model_selection_canonical_group.csv"
)
MD_SELECTION = Path(
    "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/github_results/"
    "latest_notebook_manifest_60/sample_manifest.csv"
)
MD_CANDIDATE_RESULTS = Path(
    "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/"
    "static_cne0_release/data/generated/generated_md_results_60.csv"
)
MD_REPLICA_RESULTS = Path(
    "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/"
    "static_cne0_release/data/generated/generated_static_cNE0_replica_180.csv"
)
REFERENCE_RESULTS = Path(
    "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/"
    "static_cne0_release/data/reference/reference_static_cNE0_reassessment_108.csv"
)
MODEL_SCORECARD = Path(
    "MY_PAPER_RELATED/MODELS/FCD_runs/_conductivity_eval/all_models_scorecard.csv"
)
CONDUCTIVITY_SUMMARY = Path(
    "MY_PAPER_RELATED/MODELS/FCD_runs/_conductivity_eval/"
    "conductivity_eval_summary.csv"
)
GENERATED_ENRICHMENT = Path(
    "figures/generated_pool_enrichment/cache/"
    "proposed_transcvae_polybert_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        required=True,
        help="Root of eval_top10_bottom10_stratified100 containing results/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT.relative_to(ROOT)}).",
    )
    return parser.parse_args()


def read_repo_csv(relative_path: Path) -> pd.DataFrame:
    path = ROOT / relative_path
    if path.exists():
        return pd.read_csv(path)

    result = subprocess.run(
        ["git", "show", f"HEAD:{relative_path.as_posix()}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return pd.read_csv(io.BytesIO(result.stdout))


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def as_integer_ids(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if not numeric.mod(1).eq(0).all():
        raise ValueError("Trajectory IDs must be integers")
    return numeric.astype("int64")


def build_generated_predictions(output_dir: Path) -> pd.DataFrame:
    pool = read_repo_csv(GENERATED_POOL)
    predictions = read_repo_csv(GENERATED_PREDICTIONS)
    weighted_predictions = read_repo_csv(WEIGHTED_GENERATED_PREDICTIONS)
    require_columns(pool, {"smiles"}, "generated pool")
    require_columns(
        predictions,
        {"smiles", "PSMILES", "pred_log10_cond", "pred_cond", "conductivity_for_summary"},
        "generated predictions",
    )
    if len(pool) != 32611 or not pool["smiles"].is_unique:
        raise ValueError("Expected 32,611 unique generated structures")
    if len(predictions) != 32610 or not predictions["smiles"].is_unique:
        raise ValueError("Expected 32,610 unique generated predictions")
    weighted_columns = {
        "smiles",
        "baseline_pred_log10_conductivity_refit",
        "weighted_pred_log10_conductivity",
        "weighted_pred_conductivity_s_cm",
        "baseline_rank",
        "weighted_rank",
        "rank_change_toward_top",
        "baseline_hit_ge_1e4",
        "weighted_hit_ge_1e4",
        "selected_weighted_model",
        "deployment_fit",
        "embedding_model",
    }
    require_columns(weighted_predictions, weighted_columns, "weighted generated predictions")
    if len(weighted_predictions) != 32610 or not weighted_predictions["smiles"].is_unique:
        raise ValueError("Expected 32,610 unique weighted generated predictions")

    predictions = predictions.merge(
        weighted_predictions[list(weighted_columns)],
        on="smiles",
        how="left",
        validate="one_to_one",
    )
    if predictions["weighted_pred_log10_conductivity"].isna().any():
        raise ValueError("Weighted generated predictions do not cover the baseline candidate pool")

    merged = pool.reset_index(names="generated_pool_index").merge(
        predictions,
        on="smiles",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    merged["generated_pool_index"] += 1
    excluded = merged["_merge"].eq("left_only")
    excluded_structures = set(merged.loc[excluded, "smiles"])
    if excluded_structures != {"[*][*]"}:
        raise ValueError(
            "Unexpected structures without predictions: "
            f"{sorted(excluded_structures)}"
        )

    merged["prediction_status"] = "ok"
    merged.loc[excluded, "prediction_status"] = "excluded"
    merged["exclusion_reason"] = ""
    merged.loc[excluded, "exclusion_reason"] = "endpoint_artifact_[*][*]"
    merged.loc[excluded, "PSMILES"] = merged.loc[excluded, "smiles"]
    merged = merged.rename(columns={"smiles": "SMILES"})
    columns = [
        "generated_pool_index",
        "SMILES",
        "PSMILES",
        "prediction_status",
        "pred_log10_cond",
        "pred_cond",
        "conductivity_for_summary",
        "baseline_pred_log10_conductivity_refit",
        "weighted_pred_log10_conductivity",
        "weighted_pred_conductivity_s_cm",
        "baseline_rank",
        "weighted_rank",
        "rank_change_toward_top",
        "baseline_hit_ge_1e4",
        "weighted_hit_ge_1e4",
        "selected_weighted_model",
        "deployment_fit",
        "embedding_model",
        "exclusion_reason",
    ]
    result = merged[columns]
    result.to_csv(output_dir / "generated_surrogate_predictions_32611.csv", index=False)
    return result


def build_reference_tables(reference_root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_path = reference_root / "results" / "sample_manifest.csv"
    status_path = reference_root / "results" / "run_results.csv"
    selection = pd.read_csv(selection_path)
    status = pd.read_csv(status_path)
    completed = read_repo_csv(REFERENCE_RESULTS)

    require_columns(
        selection,
        {
            "Trajectory ID",
            "sample_group",
            "CONDUCTIVITY",
            "Transference Number",
            "Li Diffusivity",
            "TFSI Diffusivity",
            "SMILES",
            "Degree of Polymerization",
            "Density",
            "Molality",
        },
        "reference selection manifest",
    )
    if len(selection) != 120 or not selection["Trajectory ID"].is_unique:
        raise ValueError("Expected 120 unique reference selections")

    selection = selection.rename(
        columns={
            "Trajectory ID": "trajectory_id",
            "CONDUCTIVITY": "reference_conductivity_S_cm",
            "Transference Number": "reference_transference_number",
            "Li Diffusivity": "reference_D_Li_cm2s",
            "TFSI Diffusivity": "reference_D_TFSI_cm2s",
            "Degree of Polymerization": "degree_of_polymerization",
            "Density": "density_g_cm3",
            "Molality": "molality_mol_kg",
        }
    )
    selection["trajectory_id"] = as_integer_ids(selection["trajectory_id"])
    selection.to_csv(output_dir / "reference_selection_manifest_120.csv", index=False)

    status_columns = [
        "Trajectory ID",
        "sample_group",
        "status",
        "analysis_status",
        "analysis_attempts",
        "analysis_last_stage",
        "analysis_last_return_code",
        "analysis_error_type",
        "attempts_used",
        "error_phase",
        "error_type",
    ]
    require_columns(status, set(status_columns), "reference run status")
    status_public = status[status_columns].rename(
        columns={"Trajectory ID": "trajectory_id", "status": "run_status"}
    )
    status_public["trajectory_id"] = as_integer_ids(status_public["trajectory_id"])
    status_public["has_completed_static_cNE0"] = status_public["trajectory_id"].isin(
        set(as_integer_ids(completed["trajectory_id"]))
    )
    if len(status_public) != 120 or not status_public["trajectory_id"].is_unique:
        raise ValueError("Expected 120 unique reference run-status rows")
    if int(status_public["has_completed_static_cNE0"].sum()) != 108:
        raise ValueError("Expected 108 completed reference reassessments")
    if set(selection["trajectory_id"]) != set(status_public["trajectory_id"]):
        raise ValueError("Reference selection and status trajectory IDs differ")
    status_public.to_csv(output_dir / "reference_run_status_120.csv", index=False)
    return selection, status_public


def build_fig8_source(reference_root: Path, output_dir: Path) -> pd.DataFrame:
    data = read_repo_csv(REFERENCE_RESULTS)
    require_columns(
        data,
        {
            "trajectory_id",
            "sample_group",
            "protocol",
            "cutoff_source",
            "association_filter",
            "sigma_static_cNE0_S_cm",
            "selection_reference_conductivity_S_cm",
        },
        "Fig. 8 source",
    )
    filtered = data[
        data["sample_group"].isin(["bottom", "middle_stratified", "top"])
        & data["protocol"].eq("manuscript_static_cNE0")
        & data["cutoff_source"].eq("Li_TFSI_O_RDF_first_minimum")
        & data["association_filter"].eq("Li_TFSI_O_contacts_ge_2")
    ].copy()
    filtered["trajectory_id"] = as_integer_ids(filtered["trajectory_id"])
    if len(filtered) != 108 or not filtered["trajectory_id"].is_unique:
        raise ValueError("Expected 108 unique filtered Fig. 8 rows")
    figure_dir = output_dir / "figure_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(figure_dir / "fig8_reference_stratified_static_cne0.csv", index=False)
    return filtered


def build_figure_manifest(output_dir: Path, fig8_rows: int) -> pd.DataFrame:
    rows = [
        {
            "figure": "Fig1A",
            "panel": "A",
            "source_role": "training structures and labels",
            "source_file": TRAINING_FOLDS.as_posix(),
            "plotting_code": "plot.ipynb cell 8",
            "row_count": 6270,
            "data_status": "available",
            "notes": "Full label table with canonical-structure-grouped four-fold assignments.",
        },
        {
            "figure": "Fig1B",
            "panel": "B",
            "source_role": "workflow schematic",
            "source_file": "",
            "plotting_code": "plot.ipynb cell 9",
            "row_count": "",
            "data_status": "not_applicable",
            "notes": "Schematic panel; no numerical source data.",
        },
        {
            "figure": "Fig2",
            "panel": "",
            "source_role": "training-label distribution",
            "source_file": TRAINING_FOLDS.as_posix(),
            "plotting_code": "plot.ipynb cell 7",
            "row_count": 6270,
            "data_status": "available",
            "notes": "Conductivity labels for all reference trajectories.",
        },
        {
            "figure": "Fig6",
            "panel": "",
            "source_role": "model scorecard",
            "source_file": MODEL_SCORECARD.as_posix(),
            "plotting_code": "plot.ipynb cell 4",
            "row_count": 14,
            "data_status": "available",
            "notes": "Generator-level scorecard.",
        },
        {
            "figure": "Fig6",
            "panel": "",
            "source_role": "conductivity screening summary",
            "source_file": CONDUCTIVITY_SUMMARY.as_posix(),
            "plotting_code": "plot.ipynb cell 4",
            "row_count": 14,
            "data_status": "available",
            "notes": "Generator-level surrogate summary.",
        },
        {
            "figure": "Fig6",
            "panel": "",
            "source_role": "generated-pool surrogate predictions",
            "source_file": GENERATED_ENRICHMENT.as_posix(),
            "plotting_code": "plot.ipynb cell 4",
            "row_count": 47125,
            "data_status": "available",
            "notes": "LOW/HIGH generated-pool predictions used for enrichment plots.",
        },
        {
            "figure": "Fig7",
            "panel": "A-C",
            "source_role": "out-of-fold predictions",
            "source_file": OOF_PREDICTIONS.as_posix(),
            "plotting_code": "scripts/figures/make_fig7_surrogate_reliability.py",
            "row_count": 6270,
            "data_status": "available",
            "notes": "One OOF prediction per training trajectory.",
        },
        {
            "figure": "Fig7",
            "panel": "A-C",
            "source_role": "weighted-model selection",
            "source_file": WEIGHTED_SELECTION.as_posix(),
            "plotting_code": "scripts/figures/make_fig7_surrogate_reliability.py",
            "row_count": 85,
            "data_status": "available",
            "notes": "Global and tail model-selection diagnostics.",
        },
        {
            "figure": "Fig8",
            "panel": "A-C",
            "source_role": "reference static cNE0 reassessment",
            "source_file": (
                "MY_PAPER_RELATED/machine_readable/figure_data/"
                "fig8_reference_stratified_static_cne0.csv"
            ),
            "plotting_code": "scripts/figures/make_fig8_reference_stratified_static_cne0.py",
            "row_count": fig8_rows,
            "data_status": "available",
            "notes": "Manuscript protocol: RDF first minimum and Li-TFSI O contacts >= 2.",
        },
        {
            "figure": "Fig8",
            "panel": "A-C",
            "source_role": "reference selection and completion denominator",
            "source_file": "MY_PAPER_RELATED/machine_readable/reference_run_status_120.csv",
            "plotting_code": "scripts/figures/make_fig8_reference_stratified_static_cne0.py",
            "row_count": 120,
            "data_status": "available",
            "notes": "Records all 120 selected trajectories, including 12 failures.",
        },
        {
            "figure": "Fig9",
            "panel": "",
            "source_role": "generated-candidate means",
            "source_file": MD_CANDIDATE_RESULTS.as_posix(),
            "plotting_code": "scripts/figures/make_fig9_generated_static_cne0.py",
            "row_count": 60,
            "data_status": "available",
            "notes": "One summary row per selected generated candidate.",
        },
        {
            "figure": "Fig9",
            "panel": "",
            "source_role": "generated replica-level static cNE0",
            "source_file": MD_REPLICA_RESULTS.as_posix(),
            "plotting_code": "scripts/figures/make_fig9_generated_static_cne0.py",
            "row_count": 180,
            "data_status": "available",
            "notes": "Three production replicas per generated candidate.",
        },
    ]
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "figure_source_manifest.csv", index=False)
    return manifest


def build_inventory(
    output_dir: Path,
    generated_predictions: pd.DataFrame,
    reference_selection: pd.DataFrame,
    reference_status: pd.DataFrame,
    figure_manifest: pd.DataFrame,
) -> pd.DataFrame:
    inventory = pd.DataFrame(
        [
            ["training labels and structures", "complete", 6270, "trajectory", "Trajectory ID", TRAINING_FOLDS.as_posix(), "Includes conductivity labels and structure strings."],
            ["four-fold assignments", "complete", 6270, "trajectory", "Trajectory ID", TRAINING_FOLDS.as_posix(), "Fold values 0-3 are assigned by canonical structure; no canonical group crosses folds."],
            ["out-of-fold surrogate predictions", "complete", 6270, "trajectory", "Trajectory ID", OOF_PREDICTIONS.as_posix(), "One OOF prediction per labeled trajectory."],
            ["generated structures and surrogate predictions", "complete_with_documented_exclusion", len(generated_predictions), "generated structure", "generated_pool_index", "MY_PAPER_RELATED/machine_readable/generated_surrogate_predictions_32611.csv", "32,610 baseline and weighted predictions; [*][*] excluded as an endpoint artifact."],
            ["target-conditioned generated baseline and weighted predictions", "complete", 47125, "generated structure", "condition + condition_row_index", "MY_PAPER_RELATED/polybert_weighted_evidence/tables/weighted_generated_condition_predictions_47125.csv", "44,999 LOW and 2,126 HIGH candidates; exact PolyBERT encoding for all rows."],
            ["weighted predictions for generated MD selections", "complete", 60, "candidate", "Trajectory ID", "MY_PAPER_RELATED/polybert_weighted_evidence/tables/weighted_generated_md_selection_60.csv", "All final MD-selected candidates matched to exact baseline-refit and weighted predictions."],
            ["generated MD candidate selection", "complete", 60, "candidate", "Trajectory ID", MD_SELECTION.as_posix(), "Includes group, structure, DP, rank, and surrogate score."],
            ["generated MD candidate results", "complete", 60, "candidate", "trajectory_id", MD_CANDIDATE_RESULTS.as_posix(), "Candidate-level manuscript static cNE0 summaries for the expanded reassessment."],
            ["generated MD replica results", "complete", 180, "candidate replica", "trajectory_id + replica", MD_REPLICA_RESULTS.as_posix(), "Three production replicas per candidate; RDF first minimum and Li-TFSI O contacts >= 2."],
            ["reference reassessment selection", "complete", len(reference_selection), "trajectory", "trajectory_id", "MY_PAPER_RELATED/machine_readable/reference_selection_manifest_120.csv", "All stratified selections before execution."],
            ["reference reassessment run status", "complete", len(reference_status), "trajectory", "trajectory_id", "MY_PAPER_RELATED/machine_readable/reference_run_status_120.csv", "108 completed and 12 failed; no local paths or long error logs."],
            ["reference reassessment completed results", "complete", 108, "trajectory", "trajectory_id", REFERENCE_RESULTS.as_posix(), "Completed 353 K manuscript static cNE0 reassessments."],
            ["figure source map", "complete", len(figure_manifest), "figure-source relation", "figure + source_role", "MY_PAPER_RELATED/machine_readable/figure_source_manifest.csv", "Maps numerical sources to manuscript figures."],
        ],
        columns=["requirement", "status", "row_count", "grain", "primary_key", "path", "notes"],
    )
    inventory.to_csv(output_dir / "data_inventory.csv", index=False)
    return inventory


def build_quality_report(
    output_dir: Path,
    generated_predictions: pd.DataFrame,
    reference_selection: pd.DataFrame,
    reference_status: pd.DataFrame,
    fig8: pd.DataFrame,
) -> None:
    training = read_repo_csv(TRAINING_FOLDS)
    oof = read_repo_csv(OOF_PREDICTIONS)
    md_selection = read_repo_csv(MD_SELECTION)
    md_replicas = read_repo_csv(MD_REPLICA_RESULTS)
    weighted_conditions = read_repo_csv(WEIGHTED_CONDITION_PREDICTIONS)
    weighted_md_selection = read_repo_csv(WEIGHTED_MD_SELECTION)
    training_columns = [
        "Trajectory ID",
        "SMILES",
        "PSMILES",
        "canonical_psmiles",
        "CONDUCTIVITY",
        "fold",
    ]
    require_columns(training, set(training_columns), "training folds")
    require_columns(
        oof,
        {"Trajectory ID", "canonical_psmiles", "fold", "pred_log10_cond"},
        "OOF predictions",
    )
    canonical_fold_counts = training.groupby("canonical_psmiles")["fold"].nunique()
    fold_sizes = training.groupby("fold").size().sort_index().tolist()
    checks = [
        ("training_rows", len(training) == 6270, len(training), 6270),
        ("training_unique_ids", training["Trajectory ID"].is_unique, training["Trajectory ID"].nunique(), 6270),
        ("training_required_values_complete", not training[training_columns].isna().any().any(), int(training[training_columns].isna().sum().sum()), 0),
        ("fold_values", set(training["fold"]) == {0, 1, 2, 3}, "[0,1,2,3]", "[0,1,2,3]"),
        ("canonical_structure_groups", training["canonical_psmiles"].nunique() == 6026, training["canonical_psmiles"].nunique(), 6026),
        ("canonical_groups_crossing_folds", canonical_fold_counts.gt(1).sum() == 0, int(canonical_fold_counts.gt(1).sum()), 0),
        ("canonical_group_fold_sizes", fold_sizes == [1561, 1559, 1583, 1567], str(fold_sizes), "[1561, 1559, 1583, 1567]"),
        ("oof_rows", len(oof) == 6270, len(oof), 6270),
        ("oof_unique_ids", oof["Trajectory ID"].is_unique, oof["Trajectory ID"].nunique(), 6270),
        ("oof_predictions_complete", not oof["pred_log10_cond"].isna().any(), int(oof["pred_log10_cond"].isna().sum()), 0),
        ("oof_ids_match_training", set(oof["Trajectory ID"]) == set(training["Trajectory ID"]), len(set(oof["Trajectory ID"]).symmetric_difference(set(training["Trajectory ID"]))), 0),
        ("generated_pool_rows", len(generated_predictions) == 32611, len(generated_predictions), 32611),
        ("generated_prediction_ok_rows", generated_predictions["prediction_status"].eq("ok").sum() == 32610, int(generated_predictions["prediction_status"].eq("ok").sum()), 32610),
        ("generated_ok_predictions_complete", not generated_predictions.loc[generated_predictions["prediction_status"].eq("ok"), ["pred_log10_cond", "pred_cond", "conductivity_for_summary"]].isna().any().any(), int(generated_predictions.loc[generated_predictions["prediction_status"].eq("ok"), ["pred_log10_cond", "pred_cond", "conductivity_for_summary"]].isna().sum().sum()), 0),
        ("generated_weighted_predictions_complete", not generated_predictions.loc[generated_predictions["prediction_status"].eq("ok"), ["weighted_pred_log10_conductivity", "weighted_pred_conductivity_s_cm", "weighted_rank"]].isna().any().any(), int(generated_predictions.loc[generated_predictions["prediction_status"].eq("ok"), ["weighted_pred_log10_conductivity", "weighted_pred_conductivity_s_cm", "weighted_rank"]].isna().sum().sum()), 0),
        ("generated_documented_exclusions", generated_predictions["prediction_status"].eq("excluded").sum() == 1, int(generated_predictions["prediction_status"].eq("excluded").sum()), 1),
        ("weighted_condition_rows", len(weighted_conditions) == 47125, len(weighted_conditions), 47125),
        ("weighted_condition_key_unique", not weighted_conditions.duplicated(["condition", "condition_row_index"]).any(), int(weighted_conditions.duplicated(["condition", "condition_row_index"]).sum()), 0),
        ("weighted_condition_counts", weighted_conditions["condition"].value_counts().to_dict() == {"LOW": 44999, "HIGH": 2126}, str(weighted_conditions["condition"].value_counts().to_dict()), "{'LOW': 44999, 'HIGH': 2126}"),
        ("weighted_condition_predictions_complete", not weighted_conditions[["baseline_pred_log10_conductivity_refit", "weighted_pred_log10_conductivity", "baseline_rank", "weighted_rank"]].isna().any().any(), int(weighted_conditions[["baseline_pred_log10_conductivity_refit", "weighted_pred_log10_conductivity", "baseline_rank", "weighted_rank"]].isna().sum().sum()), 0),
        ("weighted_md_selection_rows", len(weighted_md_selection) == 60, len(weighted_md_selection), 60),
        ("weighted_md_selection_unique_ids", weighted_md_selection["Trajectory ID"].is_unique, weighted_md_selection["Trajectory ID"].nunique(), 60),
        ("md_candidate_rows", len(md_selection) == 60, len(md_selection), 60),
        ("md_replica_rows", len(md_replicas) == 180, len(md_replicas), 180),
        ("md_replica_key_unique", not md_replicas.duplicated(["trajectory_id", "replica"]).any(), int(md_replicas.duplicated(["trajectory_id", "replica"]).sum()), 0),
        ("md_three_replicas_per_candidate", md_replicas.groupby("trajectory_id")["replica"].nunique().eq(3).all(), int(md_replicas.groupby("trajectory_id")["replica"].nunique().eq(3).sum()), 60),
        ("reference_selection_rows", len(reference_selection) == 120, len(reference_selection), 120),
        ("reference_status_rows", len(reference_status) == 120, len(reference_status), 120),
        ("reference_completed_rows", reference_status["has_completed_static_cNE0"].sum() == 108, int(reference_status["has_completed_static_cNE0"].sum()), 108),
        ("reference_failed_rows", reference_status["run_status"].eq("failed").sum() == 12, int(reference_status["run_status"].eq("failed").sum()), 12),
        ("fig8_rows", len(fig8) == 108, len(fig8), 108),
        ("fig8_ids_match_completed_reference", set(fig8["trajectory_id"]) == set(reference_status.loc[reference_status["has_completed_static_cNE0"], "trajectory_id"]), len(set(fig8["trajectory_id"]).symmetric_difference(set(reference_status.loc[reference_status["has_completed_static_cNE0"], "trajectory_id"]))), 0),
    ]
    report = pd.DataFrame(checks, columns=["check", "passed", "observed", "expected"])
    report.to_csv(output_dir / "data_quality_report.csv", index=False)
    if not report["passed"].all():
        failed = report.loc[~report["passed"], "check"].tolist()
        raise ValueError(f"Machine-readable release QA failed: {failed}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_predictions = build_generated_predictions(output_dir)
    reference_selection, reference_status = build_reference_tables(
        args.reference_root.resolve(), output_dir
    )
    fig8 = build_fig8_source(args.reference_root.resolve(), output_dir)
    figure_manifest = build_figure_manifest(output_dir, len(fig8))
    build_inventory(
        output_dir,
        generated_predictions,
        reference_selection,
        reference_status,
        figure_manifest,
    )
    build_quality_report(
        output_dir,
        generated_predictions,
        reference_selection,
        reference_status,
        fig8,
    )

    print(f"Wrote machine-readable release to {output_dir}")
    print("Validated: 6,270 labels in 6,026 canonical groups; no fold overlap")
    print("Validated: 32,611 generated; 60 candidates; 180 replicas")
    print("Validated: 120 reference selections; 108 completed; 12 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
