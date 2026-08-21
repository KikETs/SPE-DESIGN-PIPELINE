#!/usr/bin/env python3
"""Build additive JCIM reproducibility indexes from archived repository data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproducibility"
STATIC = (
    ROOT
    / "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release"
)
GENERATED = STATIC / "data/generated"
REFERENCE = STATIC / "data/reference"
MACHINE = ROOT / "MY_PAPER_RELATED/machine_readable"
POLYBERT = ROOT / "MY_PAPER_RELATED/polybert_con"
WEIGHTED = ROOT / "MY_PAPER_RELATED/polybert_weighted_evidence"
SELECTION = (
    ROOT
    / "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/github_results"
    / "latest_notebook_manifest_60/sample_manifest.csv"
)


def repo_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def build_split_table() -> Path:
    source = POLYBERT / "fold_assignment.csv"
    frame = pd.read_csv(source)
    bins = pd.qcut(frame["log10_cond"], q=6, labels=False, duplicates="drop")
    group_ids = frame["canonical_psmiles"].map(
        lambda value: "sha256:" + hashlib.sha256(str(value).encode()).hexdigest()
    )
    output = pd.DataFrame(
        {
            "row_id": frame["Trajectory ID"],
            "canonical_structure": frame["canonical_psmiles"],
            "canonical_group_id": group_ids,
            "conductivity_S_cm": frame["CONDUCTIVITY"],
            "log10_conductivity_S_cm": frame["log10_cond"],
            "stratification_bin": bins.astype(int),
            "fold": frame["fold"].astype(int),
            "source_file": repo_path(source),
        }
    )
    path = OUT / "ml/splits/canonical_grouped_split_6270.csv"
    write_csv(output, path)
    return path


def build_generated_tables() -> tuple[Path, Path]:
    selection = pd.read_csv(SELECTION).rename(
        columns={"Trajectory ID": "candidate_id", "sample_group": "group"}
    )
    composition_path = GENERATED / "generated_candidate_composition_60.csv"
    composition = pd.read_csv(composition_path).rename(
        columns={"trajectory_id": "candidate_id", "sample_group": "group"}
    )
    weighted_path = WEIGHTED / "tables/weighted_generated_md_selection_60.csv"
    weighted = pd.read_csv(weighted_path).rename(
        columns={"Trajectory ID": "candidate_id", "sample_group": "group"}
    )
    replica_path = GENERATED / "generated_static_cNE0_replica_180.csv"
    replica = pd.read_csv(replica_path).rename(
        columns={"trajectory_id": "candidate_id", "sample_group": "group"}
    )

    selected = selection[
        [
            "candidate_id",
            "group",
            "SMILES",
            "PSMILES",
            "Degree of Polymerization",
            "design_condition",
            "pred_log10_cond",
            "pred_cond",
            "candidate_rank_by_pred_cond",
        ]
    ].rename(
        columns={
            "SMILES": "repeat_unit_smiles",
            "PSMILES": "repeat_unit_psmiles",
            "Degree of Polymerization": "degree_of_polymerization",
            "pred_log10_cond": "baseline_surrogate_log10_sigma_S_cm",
            "pred_cond": "baseline_surrogate_sigma_S_cm",
            "candidate_rank_by_pred_cond": "baseline_surrogate_rank",
        }
    )
    comp_fields = composition[
        [
            "candidate_id",
            "topology_route",
            "charge_neutrality_class",
            "system_q",
            "polymer_q_chain",
            "polymer_n_mol",
            "Li_count",
            "TFSI_count",
        ]
    ]
    weighted_fields = weighted[
        [
            "candidate_id",
            "weighted_pred_log10_conductivity",
            "weighted_pred_conductivity_s_cm",
            "weighted_rank",
            "selected_weighted_model",
            "deployment_fit",
            "embedding_model",
        ]
    ].rename(
        columns={
            "weighted_pred_log10_conductivity": "weighted_surrogate_log10_sigma_S_cm",
            "weighted_pred_conductivity_s_cm": "weighted_surrogate_sigma_S_cm",
            "weighted_rank": "weighted_surrogate_rank",
        }
    )
    consolidated = (
        replica.merge(selected, on=["candidate_id", "group"], validate="many_to_one")
        .merge(comp_fields, on="candidate_id", validate="many_to_one")
        .merge(weighted_fields, on="candidate_id", validate="many_to_one")
    )
    consolidated["log10_static_cNE0_S_cm"] = np.log10(
        consolidated["sigma_static_cNE0_S_cm"]
    )
    consolidated["log10_NE_S_cm"] = np.log10(consolidated["sigma_NE_S_cm"])
    consolidated["analysis_status"] = "ok"
    consolidated["source_replica_file"] = repo_path(replica_path)
    consolidated["source_selection_file"] = repo_path(SELECTION)
    replica_out = OUT / "data/generated_md/generated_md_replica_180.csv"
    write_csv(consolidated, replica_out)

    aggregates = (
        consolidated.groupby(["candidate_id", "group"], as_index=False)
        .agg(
            production_replicas=("replica", "nunique"),
            sigma_static_cNE0_mean_S_cm=("sigma_static_cNE0_S_cm", "mean"),
            sigma_static_cNE0_std_S_cm=("sigma_static_cNE0_S_cm", "std"),
            sigma_static_cNE0_min_S_cm=("sigma_static_cNE0_S_cm", "min"),
            sigma_static_cNE0_max_S_cm=("sigma_static_cNE0_S_cm", "max"),
            sigma_NE_mean_S_cm=("sigma_NE_S_cm", "mean"),
            sigma_NE_std_S_cm=("sigma_NE_S_cm", "std"),
            D_Li_mean_cm2s=("D_Li_cm2s", "mean"),
            D_TFSI_mean_cm2s=("D_TFSI_cm2s", "mean"),
            RDF_first_minimum_mean_nm=("cutoff_nm", "mean"),
            RDF_first_minimum_median_nm=("cutoff_nm", "median"),
            RDF_first_minimum_min_nm=("cutoff_nm", "min"),
            RDF_first_minimum_max_nm=("cutoff_nm", "max"),
        )
    )
    aggregates["log10_static_cNE0_mean_S_cm"] = np.log10(
        aggregates["sigma_static_cNE0_mean_S_cm"]
    )
    candidate_fields = consolidated[
        [
            "candidate_id",
            "repeat_unit_smiles",
            "repeat_unit_psmiles",
            "degree_of_polymerization",
            "design_condition",
            "baseline_surrogate_log10_sigma_S_cm",
            "baseline_surrogate_sigma_S_cm",
            "baseline_surrogate_rank",
            "weighted_surrogate_log10_sigma_S_cm",
            "weighted_surrogate_sigma_S_cm",
            "weighted_surrogate_rank",
            "topology_route",
            "charge_neutrality_class",
            "polymer_n_mol",
            "Li_count",
            "TFSI_count",
        ]
    ].drop_duplicates("candidate_id")
    candidate = aggregates.merge(candidate_fields, on="candidate_id", validate="one_to_one")
    candidate["aggregation"] = "arithmetic mean across three production replicas"
    candidate["source_replica_file"] = repo_path(replica_out)

    archived_candidate_path = GENERATED / "generated_md_results_60.csv"
    archived = pd.read_csv(archived_candidate_path).set_index("trajectory_id")
    check = candidate.set_index("candidate_id")["sigma_static_cNE0_mean_S_cm"]
    expected = archived.loc[check.index, "sigma_static_cNE0_mean_S_cm"]
    if not np.allclose(check.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=0.0):
        raise RuntimeError("Generated candidate aggregation differs from archived result")

    candidate_out = OUT / "data/generated_md/generated_md_candidate_60.csv"
    write_csv(candidate, candidate_out)
    return replica_out, candidate_out


def build_reference_table() -> Path:
    selection_path = MACHINE / "reference_selection_manifest_120.csv"
    status_path = MACHINE / "reference_run_status_120.csv"
    result_path = REFERENCE / "reference_static_cNE0_reassessment_108.csv"
    selection = pd.read_csv(selection_path).rename(
        columns={"trajectory_id": "candidate_id", "sample_group": "reference_stratum"}
    )
    status = pd.read_csv(status_path).rename(
        columns={"trajectory_id": "candidate_id", "sample_group": "reference_stratum"}
    )
    result = pd.read_csv(result_path).rename(
        columns={"trajectory_id": "candidate_id", "sample_group": "reference_stratum"}
    )
    result_fields = result[
        [
            "candidate_id",
            "sigma_static_cNE0_S_cm",
            "sigma_NE_S_cm",
            "cutoff_nm",
            "abs_log10_error",
            "protocol",
            "association_filter",
            "temperature_K",
        ]
    ]
    output = (
        selection.merge(
            status,
            on=["candidate_id", "reference_stratum"],
            validate="one_to_one",
        )
        .merge(result_fields, on="candidate_id", how="left", validate="one_to_one")
    )
    output["md_status"] = np.where(
        output["has_completed_static_cNE0"].astype(bool), "completed", "failed"
    )
    fallback_stage = output["error_phase"].combine_first(output["analysis_last_stage"])
    output["failure_stage"] = np.where(output["md_status"].eq("failed"), fallback_stage, "")
    output["log10_reference_sigma_S_cm"] = np.log10(
        output["reference_conductivity_S_cm"]
    )
    output["log10_static_cNE0_S_cm"] = output["sigma_static_cNE0_S_cm"].map(
        lambda value: math.log10(value) if pd.notna(value) and value > 0 else np.nan
    )
    output["MALogE_component"] = output["abs_log10_error"]
    output["source_selection_file"] = repo_path(selection_path)
    output["source_status_file"] = repo_path(status_path)
    output["source_completed_file"] = repo_path(result_path)
    path = OUT / "data/reference_md/reference_md_selection_status_120.csv"
    write_csv(output, path)
    return path


def parse_mdp(path: Path) -> list[tuple[str, str]]:
    parameters: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        content = raw.split(";", 1)[0].strip()
        if not content or "=" not in content:
            continue
        name, value = content.split("=", 1)
        parameters.append((name.strip(), value.strip()))
    return parameters


def build_gromacs_parameter_table() -> Path:
    examples = {
        "generated": STATIC / "examples/generated_Traj_912118_rep3/mdp",
        "reference": STATIC / "examples/reference_Traj_13430/mdp",
    }
    rows: list[dict[str, object]] = []
    for dataset, directory in examples.items():
        for path in sorted(directory.glob("*.mdp")):
            for parameter, value in parse_mdp(path):
                rows.append(
                    {
                        "stage": f"{dataset}:{path.stem}",
                        "source_mdp": repo_path(path),
                        "parameter": parameter,
                        "value": value,
                    }
                )
    output = pd.DataFrame(rows)
    path = OUT / "md/protocol/gromacs_parameters.csv"
    write_csv(output, path)
    return path


def build_seed_table() -> Path:
    selection = pd.read_csv(SELECTION)
    workflow = STATIC / "workflow/phase_scripts/gromacs_new_phase_md.py"
    rows: list[dict[str, object]] = []
    for record in selection[["Trajectory ID", "sample_group"]].itertuples(index=False):
        candidate_id = int(record[0])
        group = str(record[1])
        rows.extend(
            [
                {
                    "candidate_id": candidate_id,
                    "group": group,
                    "replica": 1,
                    "seed": "continuation_from_equilibrated_state",
                    "source_file": repo_path(workflow),
                    "status": "RECOVERED_SEMANTIC",
                },
                {
                    "candidate_id": candidate_id,
                    "group": group,
                    "replica": 2,
                    "seed": "730002",
                    "source_file": repo_path(workflow),
                    "status": "RECOVERED",
                },
                {
                    "candidate_id": candidate_id,
                    "group": group,
                    "replica": 3,
                    "seed": "730003",
                    "source_file": repo_path(workflow),
                    "status": "RECOVERED",
                },
            ]
        )
    path = OUT / "md/seeds/generated_md_replica_seeds.csv"
    write_csv(pd.DataFrame(rows), path)
    return path


def build_table_provenance(outputs: list[Path]) -> Path:
    roles = {
        "canonical_grouped_split_6270.csv": "final canonical-grouped four-fold assignment",
        "generated_md_replica_180.csv": "generated replica-level MD results",
        "generated_md_candidate_60.csv": "deterministic generated candidate aggregation",
        "reference_md_selection_status_120.csv": "reference selection, status, and completed results",
        "gromacs_parameters.csv": "parameters extracted from archived MDP files",
        "generated_md_replica_seeds.csv": "production replica seed disclosure",
    }
    rows = [
        {
            "released_table": repo_path(path),
            "role": roles[path.name],
            "generation_script": repo_path(Path(__file__)),
            "source_metadata_status": "documented",
        }
        for path in outputs
    ]
    path = OUT / "data/TABLE_PROVENANCE.csv"
    write_csv(pd.DataFrame(rows), path)
    return path


def build_manifest() -> Path:
    references = [
        ROOT / "REPRODUCIBILITY_AUDIT.md",
        ROOT / "CITATION.cff",
        ROOT / "NOTICE.md",
        ROOT / ".zenodo.json",
        ROOT / "RELEASE_NOTES_JCIM.md",
        ROOT / "JCIM_REPRODUCIBILITY_REPORT.md",
        ROOT / "scripts/zenodo_upload_representative.py",
        STATIC / "README.md",
        STATIC / "ACPYPE_INVOCATIONS.md",
        STATIC / "FALLBACK_TOPOLOGY_PROCEDURE.md",
        STATIC / "inputs/CHARGE_MODEL.md",
        STATIC / "inputs/charge_scaling_parameters.csv",
        STATIC / "external_data/representative_trajectory_files.csv",
        ROOT / "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/pysoftk/LICENSE.md",
        POLYBERT / "fold_assignment.csv",
        POLYBERT / "oof_predictions.csv",
        MACHINE / "figure_source_manifest.csv",
        MACHINE / "reference_selection_manifest_120.csv",
        MACHINE / "reference_run_status_120.csv",
    ]
    mutable_names = {"MANIFEST.csv", "zenodo_draft_state.json"}
    package_files = [
        path for path in OUT.rglob("*") if path.is_file() and path.name not in mutable_names
    ]
    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for path, scope in [(path, "package") for path in package_files] + [
        (path, "referenced_source") for path in references
    ]:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        rows.append(
            {
                "path": repo_path(path),
                "scope": scope,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "availability": "github",
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["scope", "path"])
    path = OUT / "MANIFEST.csv"
    write_csv(manifest, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Refresh MANIFEST.csv after documentation edits.",
    )
    args = parser.parse_args()
    if not args.manifest_only:
        outputs = [
            build_split_table(),
            *build_generated_tables(),
            build_reference_table(),
            build_gromacs_parameter_table(),
            build_seed_table(),
        ]
        build_table_provenance(outputs)
    manifest = build_manifest()
    print(f"wrote {repo_path(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
