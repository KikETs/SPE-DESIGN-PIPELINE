from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[2]
GENERATED_TRAJ_ID = 912118
GENERATED_REPLICA = 3
REFERENCE_TRAJ_ID = 13430


def copy_files(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in paths:
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / source.name)


def copy_glob(source_dir: Path, pattern: str, destination: Path) -> None:
    paths = sorted(source_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched {source_dir / pattern}")
    copy_files(paths, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_grain(frame: pd.DataFrame, keys: list[str], expected_rows: int, label: str) -> None:
    if len(frame) != expected_rows:
        raise RuntimeError(f"{label}: expected {expected_rows} rows, found {len(frame)}")
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        values = frame.loc[duplicates, keys].head(10).to_dict("records")
        raise RuntimeError(f"{label}: duplicate keys {keys}: {values}")


def relative_source(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def sanitize_text_file(path: Path, replacements: dict[str, str]) -> None:
    text = path.read_text()
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text)


def normalize_gromacs_text(release_root: Path) -> None:
    for suffix in ("*.itp", "*.top", "*.pdb"):
        for path in release_root.rglob(suffix):
            lines = path.read_text().splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def build_generated_data(base: Path, release_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_root = base / "github_results" / "latest_notebook_manifest_60"
    cne_root = base / "results" / "manuscript_static_cne0_generated_180"
    manifest = pd.read_csv(result_root / "sample_manifest.csv")
    run_results = pd.read_csv(result_root / "run_results.csv")
    topology = pd.read_csv(result_root / "s12a_construction_summary" / "s12a_topology_charge_per_candidate.csv")
    replicas = pd.read_csv(cne_root / "replica_results.csv")
    agg = pd.read_csv(cne_root / "candidate_means.csv")

    require_grain(manifest, ["Trajectory ID"], 60, "generated candidate manifest")
    require_grain(replicas, ["trajectory_id", "replica"], 180, "generated static cNE0 replicas")
    if set(replicas["replica"].astype(int)) != {1, 2, 3}:
        raise RuntimeError("generated static cNE0 replicas must contain replica IDs 1, 2, and 3")
    counts = replicas.groupby("trajectory_id")["replica"].nunique()
    if not counts.eq(3).all() or len(counts) != 60:
        raise RuntimeError("generated static cNE0 must contain exactly three replicas for each of 60 candidates")
    if not pd.to_numeric(replicas["sigma_static_cNE0_S_cm"], errors="coerce").notna().all():
        raise RuntimeError("generated static cNE0 contains missing conductivity values")
    expected_protocol = (
        replicas["protocol"].eq("manuscript_static_cNE0")
        & replicas["cutoff_source"].eq("Li_TFSI_O_RDF_first_minimum")
        & replicas["association_filter"].eq("Li_TFSI_O_contacts_ge_2")
        & replicas["temperature_K"].eq(353.0)
        & replicas["analysis_frame_spacing_ps"].eq(1.0)
        & replicas["n_frames"].eq(50001)
    )
    if not expected_protocol.all():
        raise RuntimeError("generated rows contain results outside the manuscript protocol")
    generated_replica = replicas.copy()
    require_grain(agg, ["trajectory_id"], 60, "generated candidate means")
    manifest = manifest.rename(columns={"Trajectory ID": "trajectory_id", "sample_group": "sample_group"})
    topology = topology.rename(columns={"Trajectory ID": "trajectory_id"})
    results = manifest.merge(agg, on=["trajectory_id", "sample_group"], validate="one_to_one")
    results = results.merge(
        topology[["trajectory_id", "topology_route", "charge_neutrality_class", "system_q", "polymer_q_chain", "polymer_n_mol"]],
        on="trajectory_id",
        validate="one_to_one",
    )
    require_grain(results, ["trajectory_id"], 60, "generated candidate results")

    composition_columns = [
        "trajectory_id", "sample_group", "SMILES", "PSMILES", "Degree of Polymerization", "repeat_heavy_atoms",
        "estimated_polymer_heavy_atoms", "polymer_n_mol", "Density", "Molality", "topology_route",
        "charge_neutrality_class", "system_q", "polymer_q_chain",
    ]
    composition = results[composition_columns].copy()
    composition["Li_count"] = 100
    composition["TFSI_count"] = 100

    out = release_root / "data" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    composition.to_csv(out / "generated_candidate_composition_60.csv", index=False)
    results.to_csv(out / "generated_md_results_60.csv", index=False)
    generated_replica.to_csv(out / "generated_static_cNE0_replica_180.csv", index=False)
    agg.to_csv(out / "generated_static_cNE0_candidate_mean_60.csv", index=False)
    shutil.copy2(cne_root / "group_summary_candidate_level.csv", out / "generated_group_summary_candidate_level.csv")
    shutil.copy2(cne_root / "pairwise_group_tests_candidate_level.csv", out / "generated_pairwise_group_tests.csv")

    for filename in [
        "msd_fit_r2_per_replica.csv",
        "msd_fit_r2_candidate_mean.csv",
        "msd_fit_r2_summary.csv",
        "msd_fit_r2_missing_or_failed.csv",
    ]:
        shutil.copy2(result_root / "s14b_msd_fit_r2" / filename, out / filename)
    for filename in ["msd_beta_summary_s14c.csv", "msd_beta_windows.csv", "msd_beta_all_windows.csv", "msd_beta_missing_or_failed.csv"]:
        source = result_root / "s14c_msd_beta_windows" / filename
        destination = out / filename
        if filename in {"msd_beta_windows.csv", "msd_beta_all_windows.csv"}:
            beta = pd.read_csv(source)
            for column in ["analysis_summary_csv", "msd_file"]:
                beta[column] = beta[column].map(lambda value: relative_source(Path(value), base) if pd.notna(value) else value)
            beta.to_csv(destination, index=False)
        else:
            shutil.copy2(source, destination)

    return results, generated_replica


def build_reference_data(reference_base: Path, release_root: Path) -> pd.DataFrame:
    result_root = reference_base / "results"
    manifest = pd.read_csv(result_root / "sample_manifest.csv")
    run_results = pd.read_csv(result_root / "run_results.csv")
    cne = pd.read_csv(result_root / "manuscript_static_cne0_reference_108" / "replica_results.csv")
    completed_ids = set(run_results.loc[run_results["status"].eq("ok") & run_results["analysis_status"].eq("ok"), "Trajectory ID"].astype(int))
    selected = cne[cne["trajectory_id"].astype(int).isin(completed_ids)].copy()
    require_grain(selected, ["trajectory_id"], 108, "reference static cNE0")
    expected_protocol = (
        selected["protocol"].eq("manuscript_static_cNE0")
        & selected["cutoff_source"].eq("Li_TFSI_O_RDF_first_minimum")
        & selected["association_filter"].eq("Li_TFSI_O_contacts_ge_2")
        & selected["temperature_K"].eq(353.0)
        & selected["analysis_frame_spacing_ps"].eq(1.0)
        & selected["n_frames"].eq(50001)
    )
    if not expected_protocol.all():
        raise RuntimeError("reference rows contain results outside the manuscript protocol")
    manifest_public = manifest[manifest["Trajectory ID"].astype(int).isin(completed_ids)].rename(
        columns={
            "Trajectory ID": "trajectory_id",
            "sample_group": "manifest_sample_group",
            "CONDUCTIVITY": "manifest_reference_conductivity_S_cm",
            "Transference Number": "reference_transference_number",
            "Li Diffusivity": "reference_D_Li_cm2s",
            "TFSI Diffusivity": "reference_D_TFSI_cm2s",
        }
    )
    reference = selected.merge(
        manifest_public,
        on="trajectory_id",
        suffixes=("", "_manifest"),
        validate="one_to_one",
    )
    require_grain(reference, ["trajectory_id"], 108, "reference reassessment")
    if not reference["sample_group"].eq(reference["manifest_sample_group"]).all():
        raise RuntimeError("reference group labels differ between calculation and manifest")
    if not np.allclose(
        reference["selection_reference_conductivity_S_cm"],
        reference["manifest_reference_conductivity_S_cm"],
        rtol=0,
        atol=0,
    ):
        raise RuntimeError("reference conductivity labels differ between calculation and manifest")
    reference = reference.drop(
        columns=["manifest_sample_group", "manifest_reference_conductivity_S_cm"]
    )
    out = release_root / "data" / "reference"
    out.mkdir(parents=True, exist_ok=True)
    reference.to_csv(out / "reference_static_cNE0_reassessment_108.csv", index=False)
    manifest[manifest["Trajectory ID"].astype(int).isin(completed_ids)].to_csv(out / "reference_completed_manifest_108.csv", index=False)
    summary_root = result_root / "manuscript_static_cne0_reference_108"
    shutil.copy2(summary_root / "group_summary_trajectory_level.csv", out / "reference_group_summary_trajectory_level.csv")
    shutil.copy2(summary_root / "overall_statistics.csv", out / "reference_overall_statistics.csv")
    return reference


def build_example(source_base: Path, traj_id: int, destination: Path, stage: str) -> None:
    run = source_base / "runs" / f"Traj_{traj_id}"
    copy_glob(run / "mdp", "*.mdp", destination / "mdp")
    copy_glob(run / "packmol", "*.inp", destination / "packmol")
    copy_glob(run / "structures", "*.pdb", destination / "structures")
    for packmol_input in (destination / "packmol").glob("*.inp"):
        sanitize_text_file(packmol_input, {str(run / "structures"): "../structures"})
    score = run / "packmol" / "packmol_seed_scores.csv"
    if score.exists():
        copy_files([score], destination / "packmol")
    copy_files(
        [
            run / "topology" / "topol.top",
            run / "topology" / "all_atomtypes.itp",
            run / "topology" / "polymer_GMX.itp",
            run / "topology" / "polymer_clean.itp",
            run / "topology" / "posre_POL.itp",
            run / "topology" / "li_GMX.itp",
            run / "topology" / "li_clean.itp",
            run / "topology" / "tfsi_GMX.itp",
            run / "topology" / "tfsi_clean.itp",
            run / "topology" / "charge_diagnosis_atomtyping.json",
            run / "topology" / "charge_sanity_interphase.json",
        ],
        destination / "topology",
    )
    sanitize_text_file(
        destination / "topology" / "charge_sanity_interphase.json",
        {str(run / "topology") + "/": ""},
    )
    copy_files(
        [
            run / "md" / stage / f"{stage}.gro",
            run / "md" / stage / f"{stage}.tpr",
            run / "md" / "index.ndx",
        ],
        destination / "representative_structure",
    )
    analysis_out = destination / "analysis"
    analysis_out.mkdir(parents=True, exist_ok=True)
    legacy_summary = analysis_out / "conductivity_summary_htpmd_ref.csv"
    legacy_summary.unlink(missing_ok=True)
    if source_base == BASE:
        result_path = source_base / "results" / "manuscript_static_cne0_generated_180" / "replica_results.csv"
        result = pd.read_csv(result_path)
        result = result[
            result["trajectory_id"].eq(traj_id)
            & result["replica"].eq(GENERATED_REPLICA)
        ]
    else:
        result_path = source_base / "results" / "manuscript_static_cne0_reference_108" / "replica_results.csv"
        result = pd.read_csv(result_path)
        result = result[result["trajectory_id"].eq(traj_id)]
    require_grain(result, ["trajectory_id", "replica"], 1, "representative manuscript static cNE0 result")
    result.to_csv(analysis_out / "manuscript_static_cne0_result.csv", index=False)


def build_special_examples(base: Path, release_root: Path) -> None:
    fallback = base / "runs" / "Traj_912122"
    fallback_out = release_root / "examples" / "fallback_Traj_912122"
    copy_files(
        [
            fallback / "topology" / "topol.top",
            fallback / "topology" / "all_atomtypes.itp",
            fallback / "topology" / "polymer_trimer_fallback_full_GMX.itp",
            fallback / "topology" / "polymer_clean.itp",
            fallback / "topology" / "charge_diagnosis_atomtyping.json",
            fallback / "topology" / "charge_sanity_interphase.json",
        ],
        fallback_out,
    )
    sanitize_text_file(fallback_out / "charge_sanity_interphase.json", {str(fallback / "topology") + "/": ""})
    repaired = base / "runs" / "Traj_911758"
    repair_out = release_root / "examples" / "charge_repair_Traj_911758"
    copy_files(
        [
            repaired / "topology" / "topol.top",
            repaired / "topology" / "polymer_GMX.itp",
            repaired / "topology" / "polymer_clean.itp",
            repaired / "topology" / "charge_diagnosis_atomtyping.json",
            repaired / "topology" / "charge_sanity_interphase.json",
        ],
        repair_out,
    )
    sanitize_text_file(repair_out / "charge_sanity_interphase.json", {str(repaired / "topology") + "/": ""})


def build_external_manifest(base: Path, reference_base: Path, release_root: Path, hash_trajectories: bool) -> pd.DataFrame:
    records = [
        ("generated", GENERATED_TRAJ_ID, GENERATED_REPLICA, "production_rep3", base),
        ("reference_reassessment", REFERENCE_TRAJ_ID, 1, "production", reference_base),
    ]
    rows = []
    for dataset, traj_id, replica, stage, source_base in records:
        run = source_base / "runs" / f"Traj_{traj_id}"
        files = [
            (run / "md" / stage / f"{stage}.xtc", f"{dataset}_Traj_{traj_id}_rep{replica}_50ns.xtc", "trajectory"),
            (run / "md" / stage / f"{stage}.tpr", f"{dataset}_Traj_{traj_id}_rep{replica}.tpr", "run_input"),
            (run / "md" / stage / f"{stage}.gro", f"{dataset}_Traj_{traj_id}_rep{replica}_final.gro", "final_structure"),
            (run / "mdp" / (f"{stage}.mdp" if stage != "production" else "production.mdp"), f"{dataset}_Traj_{traj_id}_rep{replica}_production.mdp", "md_parameters"),
            (run / "md" / "index.ndx", f"{dataset}_Traj_{traj_id}_rep{replica}_index.ndx", "index"),
        ]
        for path, archive_name, role in files:
            if not path.exists():
                raise FileNotFoundError(path)
            digest = sha256(path) if (hash_trajectories or path.suffix != ".xtc") else "NOT_COMPUTED"
            rows.append(
                {
                    "dataset": dataset,
                    "trajectory_id": traj_id,
                    "replica": replica,
                    "duration_ns": 50.0,
                    "role": role,
                    "archive_filename": archive_name,
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                    "repository_example": f"examples/{'generated_Traj_912118_rep3' if dataset == 'generated' else 'reference_Traj_13430'}",
                }
            )
    frame = pd.DataFrame(rows)
    out = release_root / "external_data"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "representative_trajectory_files.csv", index=False)
    return frame


def write_support_files(base: Path, release_root: Path) -> None:
    versions = base / "github_results" / "latest_notebook_manifest_60" / "s12a_construction_summary"
    out = release_root / "environment"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(versions / "software_versions_60candidate.md", out / "software_versions.md")
    shutil.copy2(versions / "software_versions_60candidate.csv", out / "software_versions.csv")
    environment_bin = Path(sys.executable).resolve().parent
    for version_file in [out / "software_versions.md", out / "software_versions.csv"]:
        sanitize_text_file(
            version_file,
            {
                str(Path(sys.executable).resolve()): "python",
                str(environment_bin / "packmol"): "packmol",
                str(environment_bin / "acpype"): "acpype",
                str(base / "pysoftk"): "pysoftk/",
            },
        )

    workflow = release_root / "workflow"
    workflow.mkdir(parents=True, exist_ok=True)
    copy_files([base / "gromacs_new_pipeline_importable.py"], workflow)
    copy_glob(base / "phase_scripts", "*.py", workflow / "phase_scripts")
    copy_glob(base / "batch_utils", "*.py", workflow / "batch_utils")

    analysis = release_root / "analysis"
    copy_files(
        [
            base / "scripts" / "analysis" / "rdf_peak_tau0_replica_eval.py",
            base / "scripts" / "analysis" / "manuscript_static_cne0.py",
            base / "scripts" / "analysis" / "persistence_sweep_and_hybrid.py",
            base / "scripts" / "analysis" / "msd_fit_r2_expanded.py",
            base / "scripts" / "analysis" / "msd_beta_windows.py",
        ],
        analysis,
    )
    inputs = release_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"component": "Li", "formal_charge_e": 1.0, "charge_scale": 0.7, "scaled_charge_e": 0.7, "model": "lammps_fq07"},
            {"component": "TFSI", "formal_charge_e": -1.0, "charge_scale": 0.7, "scaled_charge_e": -0.7, "model": "lammps_fq07"},
            {"component": "polymer", "formal_charge_e": 0.0, "charge_scale": 1.0, "scaled_charge_e": 0.0, "model": "ACPYPE gas charges with neutrality repair when required"},
        ]
    ).to_csv(inputs / "charge_scaling_parameters.csv", index=False)


def write_quality_report(release_root: Path, generated_results: pd.DataFrame, generated_replicas: pd.DataFrame, reference: pd.DataFrame, external: pd.DataFrame) -> None:
    forbidden_suffixes = {".xtc", ".trr", ".edr", ".cpt"}
    forbidden_files = [path for path in release_root.rglob("*") if path.is_file() and path.suffix.lower() in forbidden_suffixes]
    text_suffixes = {".csv", ".json", ".md", ".inp", ".py", ".top", ".itp", ".mdp"}
    private_path_files = []
    for path in release_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in text_suffixes:
            text = path.read_text(errors="ignore")
            if re.search(r"/home/[^/]+/", text):
                private_path_files.append(path)
    checks = [
        ("generated_candidates", len(generated_results), 60),
        ("generated_replica_static_cNE0", len(generated_replicas), 180),
        ("generated_replicas_per_candidate_min", int(generated_replicas.groupby("trajectory_id")["replica"].nunique().min()), 3),
        ("generated_replicas_per_candidate_max", int(generated_replicas.groupby("trajectory_id")["replica"].nunique().max()), 3),
        ("reference_completed_static_cNE0", len(reference), 108),
        ("representative_datasets", external["dataset"].nunique(), 2),
        ("missing_generated_static_cNE0", int(generated_replicas["sigma_static_cNE0_S_cm"].isna().sum()), 0),
        ("missing_reference_static_cNE0", int(reference["sigma_static_cNE0_S_cm"].isna().sum()), 0),
        ("generated_temperature_not_353K", int((generated_replicas["temperature_K"] != 353.0).sum()), 0),
        ("reference_temperature_not_353K", int((reference["temperature_K"] != 353.0).sum()), 0),
        ("generated_non_manuscript_protocol", int((generated_replicas["protocol"] != "manuscript_static_cNE0").sum()), 0),
        ("reference_non_manuscript_protocol", int((reference["protocol"] != "manuscript_static_cNE0").sum()), 0),
        ("generated_non_first_minimum_cutoff", int((generated_replicas["cutoff_source"] != "Li_TFSI_O_RDF_first_minimum").sum()), 0),
        ("reference_non_first_minimum_cutoff", int((reference["cutoff_source"] != "Li_TFSI_O_RDF_first_minimum").sum()), 0),
        ("generated_wrong_association_filter", int((generated_replicas["association_filter"] != "Li_TFSI_O_contacts_ge_2").sum()), 0),
        ("reference_wrong_association_filter", int((reference["association_filter"] != "Li_TFSI_O_contacts_ge_2").sum()), 0),
        ("generated_wrong_analysis_frame_count", int((generated_replicas["n_frames"] != 50001).sum()), 0),
        ("reference_wrong_analysis_frame_count", int((reference["n_frames"] != 50001).sum()), 0),
        ("forbidden_raw_trajectory_files", len(forbidden_files), 0),
        ("files_with_private_absolute_paths", len(private_path_files), 0),
    ]
    report = pd.DataFrame(checks, columns=["check", "observed", "expected"])
    report["status"] = np.where(report["observed"].eq(report["expected"]), "PASS", "FAIL")
    report.to_csv(release_root / "data_quality_report.csv", index=False)
    if not report["status"].eq("PASS").all():
        raise RuntimeError("Release data-quality checks failed")


def write_manuscript_update(release_root: Path, generated_results: pd.DataFrame) -> None:
    generated = pd.read_csv(
        release_root / "data" / "generated" / "generated_group_summary_candidate_level.csv"
    )
    pairwise = pd.read_csv(
        release_root / "data" / "generated" / "generated_pairwise_group_tests.csv"
    )
    reference = pd.read_csv(
        release_root / "data" / "reference" / "reference_group_summary_trajectory_level.csv"
    )
    reference_stats = pd.read_csv(
        release_root / "data" / "reference" / "reference_overall_statistics.csv"
    ).iloc[0]

    lines = [
        "# Manuscript and SI static cNE0 replacement values",
        "",
        "These values supersede the RDF-peak/zero-persistence release and the generated "
        "363 K method-sweep values. They use the manuscript/SI protocol: Li-TFSI oxygen "
        "RDF first minimum, at least two oxygen contacts per retained Li-TFSI pair, "
        "353 K, 0-50 ns at 1 ps spacing, 50,001 frames, static clusters, maximum cluster "
        "size 101, and formal cluster charge z_ij = i - j.",
        "",
        "Generated candidate values are arithmetic means of three replica-level linear "
        "conductivities. Group statistics use the 10 candidate-level values in each group.",
        "",
        "## Main-text reference reassessment table",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Completed static cNE0 analyses | 108 / 120 |",
        "| Group coverage | 8 bottom, 91 middle, 9 top |",
        "| Simulation length | 50 ns |",
        "| Primary reassessment protocol | RDF first minimum + Li-TFSI O contacts >= 2 + static cNE0 |",
    ]
    reference_by_label = reference.set_index("manuscript_group")
    overall = reference_by_label.loc["Overall"]
    lines.extend(
        [
            f"| Overall MALogE (log10 scale) | {overall['mean_abs_delta_log10']:.3f} |",
            f"| Overall median static cNE0/reference-label ratio | {overall['median_static_cNE0_over_reference']:.2f} |",
        ]
    )
    for label in ["Bottom", "Middle-stratified", "Top"]:
        row = reference_by_label.loc[label]
        lines.append(
            f"| {label} MALogE / median ratio | {row['mean_abs_delta_log10']:.3f} / "
            f"{row['median_static_cNE0_over_reference']:.3g} |"
        )
    lines.extend(
        [
            "| Median RDF first-minimum cutoff | "
            f"{overall['median_RDF_first_minimum_nm']:.3f} nm overall; "
            f"{reference_by_label.loc['Bottom', 'median_RDF_first_minimum_nm']:.3f} nm bottom; "
            f"{reference_by_label.loc['Middle-stratified', 'median_RDF_first_minimum_nm']:.3f} nm middle; "
            f"{reference_by_label.loc['Top', 'median_RDF_first_minimum_nm']:.3f} nm top |",
            f"| Top-bottom median shift (log10 scale) | {reference_stats['top_bottom_median_shift_log10']:.4f} |",
            f"| Spearman rho vs reference labels | {reference_stats['spearman_rho']:.4f} |",
            f"| Kendall tau vs reference labels | {reference_stats['kendall_tau']:.4f} |",
            f"| Mann-Whitney U and exact p-value | {reference_stats['top_bottom_mann_whitney_U']:.1f}; p = {reference_stats['top_bottom_exact_two_sided_p']:.4g} |",
            "",
            "## Main-text generated-candidate table",
            "",
            "| Group | n candidates / replicas | Mean abs delta log10 | Median PolyBERT/static cNE0 | RDF cutoff / nm |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    generated_order = ["TT-top", "TT-middle", "TT-bottom", "LT-top", "LT-middle", "LT-bottom", "Overall"]
    generated_by_label = generated.set_index("manuscript_group")
    for label in generated_order:
        row = generated_by_label.loc[label]
        lines.append(
            f"| {label} | {int(row['candidates_or_trajectories'])} / {int(row['production_replicas'])} | "
            f"{row['mean_abs_delta_log10']:.3f} | {row['median_reference_over_static_cNE0']:.3f} | "
            f"{row['median_RDF_first_minimum_nm']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## SI generated-candidate diagnostics",
            "",
            "| Group | n | Mean abs delta log10 | Median abs delta log10 | Median PolyBERT/static cNE0 | Median cNE0/NE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in generated_order:
        row = generated_by_label.loc[label]
        lines.append(
            f"| {label} | {int(row['candidates_or_trajectories'])} | "
            f"{row['mean_abs_delta_log10']:.3f} | {row['median_abs_delta_log10']:.3f} | "
            f"{row['median_reference_over_static_cNE0']:.3f} | {row['median_static_cNE0_over_NE']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## SI generated-candidate log10 static cNE0",
            "",
            "Percentile bootstrap intervals use 5000 candidate-level resamples. The seed is recorded in the CSV.",
            "",
            "| Group | n | Mean log10 static cNE0 | Mean 95% CI | Median log10 static cNE0 | Median 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in generated_order[:-1]:
        row = generated_by_label.loc[label]
        lines.append(
            f"| {label} | {int(row['candidates_or_trajectories'])} | {row['mean_log10_static_cNE0']:.3f} | "
            f"{row['bootstrap_mean_log10_95CI_low']:.3f} to {row['bootstrap_mean_log10_95CI_high']:.3f} | "
            f"{row['median_log10_static_cNE0']:.3f} | "
            f"{row['bootstrap_median_log10_95CI_low']:.3f} to {row['bootstrap_median_log10_95CI_high']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## SI pairwise generated-group tests",
            "",
            "| Comparison | Median delta log10 | Fold difference | Cliff's delta | Exact p |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in pairwise.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['median_delta_log10_static_cNE0']:.3f} | "
            f"{row['fold_difference']:.3g} | {row['cliffs_delta']:.3f} | {row['exact_two_sided_p']:.3g} |"
        )

    top = generated_results.nlargest(5, "sigma_static_cNE0_mean_S_cm")
    label_map = {
        "HIGH_top": "TT-top",
        "HIGH_middle_stratified": "TT-middle",
        "HIGH_bottom": "TT-bottom",
        "LOW_top": "LT-top",
        "LOW_middle_stratified": "LT-middle",
        "LOW_bottom": "LT-bottom",
    }
    lines.extend(
        [
            "",
            "## SI top five generated candidates by corrected static cNE0",
            "",
            "| ID | Group | Repeat-unit string | Rank | PolyBERT / S cm-1 | Static cNE0 / S cm-1 |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for _, row in top.iterrows():
        lines.append(
            f"| {int(row['trajectory_id'])} | {label_map[row['sample_group']]} | {row['PSMILES']} | "
            f"{int(row['candidate_rank_by_pred_cond'])} | {row['CONDUCTIVITY']:.3e} | "
            f"{row['sigma_static_cNE0_mean_S_cm']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## Source files",
            "",
            "- `data/generated/generated_static_cNE0_replica_180.csv`",
            "- `data/generated/generated_static_cNE0_candidate_mean_60.csv`",
            "- `data/generated/generated_group_summary_candidate_level.csv`",
            "- `data/generated/generated_pairwise_group_tests.csv`",
            "- `data/reference/reference_static_cNE0_reassessment_108.csv`",
            "- `data/reference/reference_group_summary_trajectory_level.csv`",
            "- `data/reference/reference_overall_statistics.csv`",
            "- `analysis/manuscript_static_cne0.py`",
            "",
        ]
    )
    (release_root / "MANUSCRIPT_STATIC_CNE0_UPDATE.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the compact GROMACS static cNE0 reproducibility release.")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--reference-base", type=Path, required=True)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--hash-trajectories", action="store_true")
    args = parser.parse_args()
    base = args.base.resolve()
    reference_base = args.reference_base.resolve()
    release_root = (args.release_root or base / "reproducibility" / "static_cne0_release").resolve()
    release_root.mkdir(parents=True, exist_ok=True)

    generated_results, generated_replicas = build_generated_data(base, release_root)
    reference = build_reference_data(reference_base, release_root)
    build_example(base, GENERATED_TRAJ_ID, release_root / "examples" / "generated_Traj_912118_rep3", "production_rep3")
    build_example(reference_base, REFERENCE_TRAJ_ID, release_root / "examples" / "reference_Traj_13430", "production")
    build_special_examples(base, release_root)
    write_support_files(base, release_root)
    external = build_external_manifest(base, reference_base, release_root, args.hash_trajectories)
    normalize_gromacs_text(release_root)
    write_manuscript_update(release_root, generated_results)
    write_quality_report(release_root, generated_results, generated_replicas, reference, external)
    print(f"[done] release root: {release_root}")


if __name__ == "__main__":
    main()
