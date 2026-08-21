from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "reproducibility"
STATIC = (
    ROOT
    / "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/reproducibility/static_cne0_release"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def test_final_ml_split_is_grouped_and_complete() -> None:
    data = pd.read_csv(REPRO / "ml/splits/canonical_grouped_split_6270.csv")
    assert len(data) == 6270
    assert data["row_id"].is_unique
    assert data["canonical_group_id"].nunique() == 6026
    assert sorted(data["fold"].unique().tolist()) == [0, 1, 2, 3]
    assert sorted(data["stratification_bin"].unique().tolist()) == [0, 1, 2, 3, 4, 5]
    assert data.groupby("canonical_group_id")["fold"].nunique().max() == 1


def test_generated_md_denominator_and_replica_relationship() -> None:
    data = pd.read_csv(REPRO / "data/generated_md/generated_md_replica_180.csv")
    assert len(data) == 180
    assert data["dataset"].eq("generated").all()
    assert data["candidate_id"].nunique() == 60
    assert data["replica_id"].is_unique
    assert data["analysis_status"].eq("ok").all()
    assert data.groupby("candidate_id")["replica"].apply(
        lambda values: set(values) == {1, 2, 3}
    ).all()
    group_counts = data.groupby("group").agg(
        candidates=("candidate_id", "nunique"), replicas=("replica_id", "nunique")
    )
    assert len(group_counts) == 6
    assert group_counts["candidates"].eq(10).all()
    assert group_counts["replicas"].eq(30).all()


def test_generated_candidate_aggregation_matches_archived_result() -> None:
    generated = pd.read_csv(REPRO / "data/generated_md/generated_md_candidate_60.csv")
    archived = pd.read_csv(STATIC / "data/generated/generated_md_results_60.csv")
    assert len(generated) == 60
    assert generated["candidate_id"].is_unique
    assert generated["production_replicas"].eq(3).all()
    left = generated.set_index("candidate_id")["sigma_static_cNE0_mean_S_cm"].sort_index()
    right = archived.set_index("trajectory_id")["sigma_static_cNE0_mean_S_cm"].sort_index()
    assert left.index.equals(right.index)
    assert np.allclose(left.to_numpy(), right.to_numpy(), rtol=1e-12, atol=0.0)


def test_reference_selection_preserves_failures() -> None:
    data = pd.read_csv(REPRO / "data/reference_md/reference_md_selection_status_120.csv")
    assert len(data) == 120
    assert data["candidate_id"].is_unique
    assert data["md_status"].value_counts().to_dict() == {"completed": 108, "failed": 12}
    complete = data["md_status"].eq("completed")
    assert data.loc[complete, "sigma_static_cNE0_S_cm"].notna().all()
    assert data.loc[~complete, "sigma_static_cNE0_S_cm"].isna().all()
    assert data.loc[~complete, "failure_stage"].notna().all()
    assert set(data["reference_stratum"]) == {"bottom", "middle_stratified", "top"}


def test_generated_and_reference_panels_are_not_mixed() -> None:
    generated = pd.read_csv(REPRO / "data/generated_md/generated_md_replica_180.csv")
    reference = pd.read_csv(REPRO / "data/reference_md/reference_md_selection_status_120.csv")
    assert generated["dataset"].eq("generated").all()
    assert not set(generated["candidate_id"]).intersection(reference["candidate_id"])


def test_production_replica_seed_disclosure() -> None:
    data = pd.read_csv(REPRO / "md/seeds/generated_md_replica_seeds.csv", dtype={"seed": str})
    assert len(data) == 180
    assert data.groupby("candidate_id")["replica"].apply(
        lambda values: set(values) == {1, 2, 3}
    ).all()
    assert data.loc[data["replica"].eq(1), "seed"].eq(
        "continuation_from_equilibrated_state"
    ).all()
    assert data.loc[data["replica"].eq(2), "seed"].eq("730002").all()
    assert data.loc[data["replica"].eq(3), "seed"].eq("730003").all()


def test_gromacs_parameter_table_comes_from_both_examples() -> None:
    data = pd.read_csv(REPRO / "md/protocol/gromacs_parameters.csv", dtype=str)
    required = {
        "integrator",
        "dt",
        "nsteps",
        "tcoupl",
        "ref-t",
        "tau-t",
        "pcoupl",
        "constraints",
        "constraint-algorithm",
        "coulombtype",
        "rcoulomb",
        "rvdw",
        "fourierspacing",
        "nstlist",
        "nstxout-compressed",
        "gen-vel",
    }
    assert required.issubset(set(data["parameter"]))
    generated_dt = data.loc[
        data["stage"].eq("generated:production") & data["parameter"].eq("dt"), "value"
    ].item()
    reference_dt = data.loc[
        data["stage"].eq("reference:production") & data["parameter"].eq("dt"), "value"
    ].item()
    assert generated_dt == "0.001000"
    assert reference_dt == "0.002000"
    assert data["source_mdp"].map(lambda value: (ROOT / value).is_file()).all()


def test_every_consolidated_table_has_provenance() -> None:
    provenance = pd.read_csv(REPRO / "data/TABLE_PROVENANCE.csv")
    released = {
        path.relative_to(ROOT).as_posix()
        for path in REPRO.rglob("*.csv")
        if path.name not in {"MANIFEST.csv", "TABLE_PROVENANCE.csv"}
    }
    assert set(provenance["released_table"]) == released
    assert provenance["source_metadata_status"].eq("documented").all()
    assert provenance["released_table"].map(lambda value: (ROOT / value).is_file()).all()
    assert provenance["generation_script"].map(lambda value: (ROOT / value).is_file()).all()


def test_manifest_paths_sizes_and_hashes() -> None:
    manifest = pd.read_csv(REPRO / "MANIFEST.csv")
    assert manifest["path"].is_unique
    for row in manifest.itertuples(index=False):
        path = ROOT / row.path
        assert path.is_file(), row.path
        assert path.stat().st_size == row.size_bytes, row.path
        assert digest(path) == row.sha256, row.path


def test_vendored_pysoftk_provenance_and_license_are_retained() -> None:
    provenance = REPRO / "md/inputs/PYSOFTK_PROVENANCE.md"
    license_path = (
        ROOT / "MY_PAPER_RELATED/gromacs_eval_pred_conductivity/pysoftk/LICENSE.md"
    )
    text = provenance.read_text(encoding="utf-8")
    assert "a3458567d9b61e5caf6d891861e5fa016dfe97b5" in text
    assert "43 of 44" in text
    assert "linear_polymer/linear_polymer.py" in text
    assert license_path.read_text(encoding="utf-8").startswith("BSD 3-Clause License")
