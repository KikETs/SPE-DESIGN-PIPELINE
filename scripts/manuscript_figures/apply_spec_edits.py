from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageOps

from scripts.manuscript_figures.common import ROOT, manuscript_spec_path, setup_liberation_sans


SVG_NS = "{http://www.w3.org/2000/svg}"
LIBERATION_FONT = "Liberation Sans,Liberation Sans_MSFontService,sans-serif"
CAMBRIA_MATH_FONT = "Cambria Math,Cambria Math_MSFontService,sans-serif"
GENERATED_POOL_SHIFT_X = 30
GENERATED_POOL_RIGHT_ENDPOINT_SHIFT_X = 14
POLYBERT_SHIFT_X = 25
RUN_RESULTS_CSV = (
    ROOT
    / "MY_PAPER_RELATED"
    / "gromacs_eval_pred_conductivity"
    / "github_results"
    / "latest_notebook_manifest_60"
    / "run_results.csv"
)
GRAPHICAL_ABSTRACT_FIG9B_SIZE = (540, 246)
MD_GROUP_ORDER = [
    "HIGH_top",
    "HIGH_middle_stratified",
    "HIGH_bottom",
    "LOW_top",
    "LOW_middle_stratified",
    "LOW_bottom",
]
MD_GROUP_COLORS = {
    "HIGH_top": "#1f5b95",
    "HIGH_middle_stratified": "#4c78a8",
    "HIGH_bottom": "#8fb7dd",
    "LOW_top": "#c64616",
    "LOW_middle_stratified": "#e67e22",
    "LOW_bottom": "#f2a51a",
}


def walk(node: dict[str, object]) -> Iterator[dict[str, object]]:
    yield node
    for child in node.get("children", []):
        yield from walk(child)  # type: ignore[arg-type]


def load_spec(name: str) -> dict[str, object]:
    return json.loads(manuscript_spec_path(name).read_text(encoding="utf-8"))


def save_spec(name: str, spec: dict[str, object]) -> None:
    manuscript_spec_path(name).write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def node(tag: str, attrs: dict[str, str], text: str = "", children: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "tag": f"{SVG_NS}{tag}",
        "attrs": attrs,
        "text": text,
        "children": children or [],
    }


def text_node(
    text: str,
    x: float,
    y: float,
    *,
    size: float,
    weight: str = "400",
    fill: str = "#000000",
    family: str = LIBERATION_FONT,
    anchor: str = "start",
) -> dict[str, object]:
    return node(
        "text",
        {
            "font-family": family,
            "font-weight": weight,
            "font-size": f"{size:g}",
            "fill": fill,
            "transform": f"translate({x:g} {y:g})",
            "text-anchor": anchor,
        },
        text,
    )


def prepend_translate(attrs: dict[str, str], dx: float, dy: float = 0) -> None:
    prefix = f"translate({dx:g} {dy:g})"
    current = str(attrs.get("transform", "")).strip()
    if current == prefix or current.startswith(f"{prefix} "):
        return
    attrs["transform"] = f"{prefix} {current}" if current else prefix


def set_leading_translate(attrs: dict[str, str], dx: float, dy: float = 0) -> None:
    prefix = f"translate({dx:g} {dy:g})"
    current = str(attrs.get("transform", "")).strip()
    current = re.sub(r"^translate\([-+0-9.]+\s+[-+0-9.]+\)\s*", "", current)
    attrs["transform"] = f"{prefix} {current}".strip()


def rect_node(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = "#FFFFFF",
    stroke: str = "#5A3471",
    stroke_width: float = 1.4,
) -> dict[str, object]:
    return node(
        "rect",
        {
            "x": f"{x:g}",
            "y": f"{y:g}",
            "width": f"{width:g}",
            "height": f"{height:g}",
            "rx": "4",
            "ry": "4",
            "fill": fill,
            "stroke": stroke,
            "stroke-width": f"{stroke_width:g}",
        },
    )


def path_node(d: str, *, fill: str = "none", stroke: str | None = None, stroke_width: float = 1.4) -> dict[str, object]:
    attrs = {"d": d, "fill": fill}
    if stroke is not None:
        attrs["stroke"] = stroke
        attrs["stroke-width"] = f"{stroke_width:g}"
        attrs["stroke-miterlimit"] = "8"
    return node("path", attrs)


def circle_path_node(
    cx: float,
    cy: float,
    r: float,
    *,
    fill: str,
    stroke: str = "#473452",
    stroke_width: float = 2.0,
) -> dict[str, object]:
    k = 0.55228475
    d = (
        f"M{cx - r:g} {cy:g}C{cx - r:g} {cy - k * r:g} "
        f"{cx - k * r:g} {cy - r:g} {cx:g} {cy - r:g} "
        f"{cx + k * r:g} {cy - r:g} {cx + r:g} {cy - k * r:g} "
        f"{cx + r:g} {cy:g} {cx + r:g} {cy + k * r:g} "
        f"{cx + k * r:g} {cy + r:g} {cx:g} {cy + r:g} "
        f"{cx - k * r:g} {cy + r:g} {cx - r:g} {cy + k * r:g} "
        f"{cx - r:g} {cy:g}Z"
    )
    return path_node(d, fill=fill, stroke=stroke, stroke_width=stroke_width)


def edit_fig1() -> None:
    spec = load_spec("fig1")
    for node in walk(spec["root"]):  # type: ignore[arg-type]
        attrs = node.get("attrs", {})
        text = node.get("text", "")
        if text in {"Rare high", "-", "conductivity", "candidates"} and attrs.get("fill") == "#006B1F":
            attrs["font-weight"] = "900"
        if attrs.get("fill") == "#006B1F" and text == "endpoint" and attrs.get("transform") == "translate(659.942 401)":
            node["text"] = "valid endpoint"
        if attrs.get("fill") == "#006B1F" and attrs.get("transform") in {
            "translate(738.275 401)",
            "translate(744.442 401)",
        }:
            node["text"] = ""
        if node.get("tag", "").endswith("path") and attrs.get("fill") == "#9DABC4" and "637.5 395" in attrs.get("d", ""):
            attrs["fill"] = "#006B1F"
            attrs["stroke"] = "#006B1F"
    save_spec("fig1", spec)


def load_fig9b_data() -> pd.DataFrame:
    if not RUN_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing required 60-candidate reassessment file: {RUN_RESULTS_CSV}")

    run_df = pd.read_csv(RUN_RESULTS_CSV).copy()
    required_cols = {
        "Trajectory ID",
        "sample_group",
        "status",
        "md_status",
        "analysis_status",
        "analysis_replica_success_count",
        "CONDUCTIVITY",
        "sigma_cNE_htpmd_S_cm_mean_pred",
        "sigma_cNE_htpmd_S_cm_std_pred",
    }
    missing = required_cols - set(run_df.columns)
    if missing:
        raise ValueError(f"Missing required 60-candidate reassessment columns: {missing}")

    for status_col in ["status", "md_status", "analysis_status"]:
        bad = run_df.loc[~run_df[status_col].eq("ok"), ["Trajectory ID", "sample_group", status_col]]
        if not bad.empty:
            raise ValueError(f"Non-ok rows found in {status_col}:\n{bad.to_string(index=False)}")

    md_df = run_df[
        [
            "Trajectory ID",
            "sample_group",
            "CONDUCTIVITY",
            "sigma_cNE_htpmd_S_cm_mean_pred",
            "sigma_cNE_htpmd_S_cm_std_pred",
            "analysis_replica_success_count",
        ]
    ].copy()
    md_df = md_df.rename(
        columns={
            "CONDUCTIVITY": "polybert_pred_cond_S_cm",
            "sigma_cNE_htpmd_S_cm_mean_pred": "md_static_cNE0_S_cm",
            "sigma_cNE_htpmd_S_cm_std_pred": "md_static_cNE0_std_S_cm",
            "analysis_replica_success_count": "n_replicas",
        }
    )

    for col in ["polybert_pred_cond_S_cm", "md_static_cNE0_S_cm", "md_static_cNE0_std_S_cm", "n_replicas"]:
        md_df[col] = pd.to_numeric(md_df[col], errors="coerce")

    md_df = md_df.dropna(subset=["sample_group", "polybert_pred_cond_S_cm", "md_static_cNE0_S_cm"]).copy()
    md_df = md_df.loc[md_df["polybert_pred_cond_S_cm"].gt(0) & md_df["md_static_cNE0_S_cm"].gt(0)].copy()
    md_df["Trajectory ID"] = md_df["Trajectory ID"].astype(int)
    md_df["log10_polybert_pred_cond"] = np.log10(md_df["polybert_pred_cond_S_cm"])
    md_df["log10_md_static_cNE0"] = np.log10(md_df["md_static_cNE0_S_cm"])

    missing_groups = sorted(set(MD_GROUP_ORDER) - set(md_df["sample_group"]))
    if missing_groups:
        raise ValueError(f"Missing expected 60-candidate sample groups: {missing_groups}")

    extra_groups = sorted(set(md_df["sample_group"]) - set(MD_GROUP_ORDER))
    if extra_groups:
        raise ValueError(f"Unexpected sample groups in 60-candidate reassessment: {extra_groups}")

    if len(md_df) != 60:
        raise ValueError(f"Expected 60 candidate-level rows, got {len(md_df)}")

    group_counts = md_df.groupby("sample_group")["Trajectory ID"].nunique().reindex(MD_GROUP_ORDER)
    if not group_counts.eq(10).all():
        raise ValueError(f"Expected 10 candidates per group, got:\n{group_counts.to_string()}")

    if not md_df["n_replicas"].eq(3).all():
        replica_counts = md_df.groupby("sample_group")["n_replicas"].value_counts(dropna=False)
        raise ValueError(f"Expected 3 successful replicas per candidate, got:\n{replica_counts.to_string()}")

    return md_df


def trim_white_margin(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    white = Image.new("RGB", rgb.size, "white")
    bbox = ImageChops.difference(rgb, white).getbbox()
    if bbox is None:
        return rgb
    return rgb.crop(bbox)


def render_graphical_abstract_fig9b() -> Image.Image:
    setup_liberation_sans()
    md_df = load_fig9b_data()
    x_col = "log10_polybert_pred_cond"
    y_col = "log10_md_static_cNE0"

    fig, ax = plt.subplots(figsize=(3.35, 1.55), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for group in MD_GROUP_ORDER:
        sub = md_df.loc[md_df["sample_group"].eq(group)].copy()
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=18,
            facecolor=MD_GROUP_COLORS[group],
            edgecolor="black",
            linewidth=0.28,
            alpha=0.95,
            zorder=3,
        )

    axis_min = min(md_df[x_col].min(), md_df[y_col].min()) - 0.12
    axis_max = max(md_df[x_col].max(), md_df[y_col].max()) + 0.12
    ax.plot([axis_min, axis_max], [axis_min, axis_max], color="0.55", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(-6.5, -2.25)
    ax.set_ylim(-6.5, -2.3)
    ax.set_xticks([-6, -5, -4, -3])
    ax.set_yticks([-6, -5, -4, -3])
    ax.tick_params(axis="both", labelsize=6, length=2.0, width=0.55, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.155, top=0.99)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=600, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return trim_white_margin(Image.open(buffer))


def fig9b_png_data_uri() -> str:
    content = render_graphical_abstract_fig9b()
    canvas = ImageOps.fit(
        content,
        GRAPHICAL_ABSTRACT_FIG9B_SIZE,
        method=Image.Resampling.LANCZOS,
        centering=(0.50, 0.50),
    )

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def graphic_children(spec: dict[str, object]) -> list[dict[str, object]]:
    root_children = spec["root"]["children"]  # type: ignore[index]
    for child in root_children:  # type: ignore[union-attr]
        if child.get("tag", "").endswith("g"):
            return child["children"]  # type: ignore[index,return-value]
    raise ValueError("Could not find graphical abstract drawing group")


def old_graphical_abstract_node(node_: dict[str, object]) -> bool:
    attrs = node_.get("attrs", {})
    text = node_.get("text", "")
    transform = attrs.get("transform")
    d = attrs.get("d", "")

    if attrs.get("data-ga-edit") == "workflow":
        return True

    old_text_transforms = {
        "translate(221.477 167)",
        "translate(233.837 172)",
        "translate(325.412 300)",
        "translate(10.1629 368)",
        "translate(98.8296 368)",
        "translate(105.996 368)",
        "translate(177.305 305)",
        "translate(12.5739 305)",
        "translate(98.6114 301)",
        "translate(168.01 328)",
        "translate(739.715 284)",
        "translate(816.169 284)",
        "translate(821.502 284)",
        "translate(492.179 566)",
        "translate(584.512 566)",
        "translate(591.679 566)",
        "translate(957.667 190)",
        "translate(971.919 489)",
    }
    if text and transform in old_text_transforms:
        return True

    if attrs.get("x") == "30" and attrs.get("y") == "277":
        return True
    if attrs.get("x") == "35" and attrs.get("y") == "267":
        return True

    old_path_prefixes = (
        "M264 215",
        "M249.5 371",
        "M264 370",
        "M28 299.552",
        "M109 299.6",
    )
    return any(str(d).startswith(prefix) for prefix in old_path_prefixes)


def graphical_abstract_replacement_nodes() -> list[dict[str, object]]:
    nodes = [
        rect_node(30, 218, 148, 46, fill="#F2F6F6", stroke="#8D6AA6", stroke_width=0.95),
        rect_node(18, 238, 168, 52, fill="#F7FBFB", stroke="#6D4E82", stroke_width=1.05),
        rect_node(6, 260, 188, 62, fill="#FFFFFF", stroke="#5A3471", stroke_width=1.25),
        text_node("[*] O C C O [*]", 100, 281, size=12.2, weight="700", fill="#1E1E1E", anchor="middle"),
        text_node("[*] C(=O) O C [*]", 100, 303, size=12.0, weight="700", fill="#1E1E1E", anchor="middle"),
        text_node("Endpoint-aware", 100, 356, size=15.5, weight="700", anchor="middle"),
        text_node("SELFIES-PSMILES corpus", 100, 378, size=15.5, weight="700", anchor="middle"),
        text_node("target", 205, 444, size=15.5, weight="700"),
        text_node("log₁₀ 𝜎", 258, 444, size=15.5, weight="700", family=CAMBRIA_MATH_FONT),
        text_node("top-tail-conductivity target", 257, 183, size=13.5, weight="700", fill="#5A3471", anchor="middle"),
        circle_path_node(257, 371, 7.5, fill="#159E9C", stroke="#473452", stroke_width=2.0),
        text_node("lower-conductivity target", 257, 416, size=13.5, weight="700", fill="#026F6C", anchor="middle"),
        text_node("latent z", 298, 280, size=14.0, weight="700", fill="#5A3471", anchor="middle"),
        text_node("TransCVAE", 395.5, 288, size=15.2, weight="700", fill="#5A3471", anchor="middle"),
        text_node("generation", 395.5, 311, size=15.2, weight="700", fill="#5A3471", anchor="middle"),
        text_node("valid endpoint", 620, 562, size=16.5, weight="700", anchor="middle"),
        text_node("generated pool", 620, 584, size=16.5, weight="700", anchor="middle"),
        text_node("PolyBERT-Ridge", 825, 278, size=13.6, weight="900", fill="#026F6C", anchor="middle"),
        text_node("prioritization", 825, 300, size=13.6, weight="900", fill="#026F6C", anchor="middle"),
        text_node("selected subset", 1060, 190, size=16.5, weight="700", anchor="middle"),
        text_node("static cNE0 MD", 1060, 477, size=16.2, weight="700", anchor="middle"),
        text_node("reassessment", 1060, 499, size=14.5, weight="700", anchor="middle"),
    ]
    for replacement in nodes:
        replacement["attrs"]["data-ga-edit"] = "workflow"  # type: ignore[index]
    return nodes


def update_graphical_abstract_workflow_layout(spec: dict[str, object]) -> None:
    children = graphic_children(spec)
    children[:] = [child for child in children if not old_graphical_abstract_node(child)]
    for child in children:
        attrs = child.get("attrs", {})
        d = str(attrs.get("d", ""))
        if d.startswith("M249.5 216"):
            attrs["fill"] = "#FA952B"
        if d.startswith("M310 260.334"):
            attrs["d"] = (
                "M330 260.334C330 251.313 337.313 244 346.334 244L444.666 244"
                "C453.687 244 461 251.313 461 260.334L461 325.666"
                "C461 334.687 453.687 342 444.666 342L346.334 342"
                "C337.313 342 330 334.687 330 325.666Z"
            )
        if d.startswith("M271 292"):
            attrs["d"] = "M271 292 314.502 292 314.502 294 271 294ZM313.168 289 321.168 293 313.168 297Z"
        if d.startswith("M453 292") or d.startswith("M473 292"):
            attrs["d"] = "M473 292 512 292 512 294 473 294ZM510.666 289 518.666 293 510.666 297Z"
        if d.startswith("M505 533"):
            prepend_translate(attrs, GENERATED_POOL_SHIFT_X)
        if attrs.get("stroke") == "#000000" and attrs.get("fill") == "none":
            try:
                first = d.split()[0]
                x = float(first[1:]) if first.startswith("M") else None
            except ValueError:
                x = None
            if x is not None and 520 <= x <= 710:
                prepend_translate(attrs, GENERATED_POOL_SHIFT_X)
            matrix_match = re.match(r"matrix\(-1\s+0\s+0\s+1\s+([-+0-9.]+)\s+([-+0-9.]+)\)", str(attrs.get("transform", "")))
            if matrix_match and 520 <= float(matrix_match.group(1)) <= 710:
                prepend_translate(attrs, GENERATED_POOL_SHIFT_X)
        if d.startswith("M729 175"):
            prepend_translate(attrs, POLYBERT_SHIFT_X)
        if (
            d.startswith("M0-1 38.0644-1") and str(attrs.get("transform", "")).endswith("884 277.135)")
        ) or d.startswith("M899 276"):
            attrs["d"] = (
                "M899 276 924 276 924 124 954 124 954 126 "
                "926 126 926 278 899 278Z"
                "M952.666 121 960.666 125 952.666 129Z"
            )
            attrs.pop("transform", None)
        transform = str(attrs.get("transform", ""))
        if transform.startswith("matrix(0.000104987 0 0 0.000104987 517 "):
            prepend_translate(attrs, GENERATED_POOL_SHIFT_X)
        if child.get("text") == "[*]" and attrs.get("fill") == "#5A3471":
            match = re.search(r"translate\(([-+0-9.]+)\s+([-+0-9.]+)\)", transform)
            if match and 500 <= float(match.group(1)) <= 710:
                base_x = float(match.group(1))
                dx = GENERATED_POOL_RIGHT_ENDPOINT_SHIFT_X if base_x > 590 else GENERATED_POOL_SHIFT_X
                set_leading_translate(attrs, dx)
        if d.startswith("M1060 507"):
            attrs["d"] = "M1060 512 1060 529.65 1058 529.65 1058 512ZM1063 528.316 1059 536.316 1055 528.316Z"

    children.extend(graphical_abstract_replacement_nodes())


def edit_graphical_abstract() -> None:
    spec = load_spec("graphical_abstract")
    data_uri = fig9b_png_data_uri()
    update_graphical_abstract_workflow_layout(spec)
    found_fig9b = False
    for node in walk(spec["root"]):  # type: ignore[arg-type]
        attrs = node.get("attrs", {})
        if attrs.get("id") == "img18":
            attrs["width"] = str(GRAPHICAL_ABSTRACT_FIG9B_SIZE[0])
            attrs["height"] = str(GRAPHICAL_ABSTRACT_FIG9B_SIZE[1])
            attrs["{http://www.w3.org/1999/xlink}href"] = data_uri
            found_fig9b = True

    if not found_fig9b:
        raise ValueError("Could not find graphical abstract image id img18")
    save_spec("graphical_abstract", spec)


def main() -> int:
    edit_fig1()
    edit_graphical_abstract()
    print("Applied manuscript figure spec edits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
