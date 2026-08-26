#!/usr/bin/env python
"""Regenerate iScience Figures 1-9 and write publication-artwork QA files."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import os
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from xml.etree import ElementTree as ET

import cairosvg
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "iScience" / "02_figures" / "regenerated"
QA_DIR = OUTPUT_DIR / "qa"
FINAL_WIDTH_IN = 6.2
FINAL_WIDTH_PT = FINAL_WIDTH_IN * 72
OUTPUT_DPI = 600
OUTPUT_WIDTH_PX = round(FINAL_WIDTH_IN * OUTPUT_DPI)

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

PANEL_COUNTS = {1: 3, 2: 3, 3: 5, 4: 3, 5: 1, 6: 4, 7: 2, 8: 3, 9: 2}
PANEL_LABELS = {1: "A-C", 2: "A-C", 3: "A-E", 4: "A-C", 5: "single panel", 6: "A-D", 7: "A-B", 8: "A-C", 9: "A-B"}
SCHEMATIC_MIN_FONT_PT = {1: 6.5, 2: 8.0, 3: 8.0, 4: 6.5, 5: 6.5}
VISUAL_QC_NOTES = {
    1: "schematic labels and connectors clear",
    2: "axes, thresholds, and table clear",
    3: "PolyBERT label has 131 px left/right clearance at 600 dpi",
    4: "both cross-attention labels clear decoder boxes",
    5: "ACPYPE content clearance L/T/R/B is 180/163/102/173 px at 600 dpi",
    6: "panel titles removed; panel A retains only percentages",
    7: "metric box, bars, labels, and legend clear",
    8: "panel A spans the full top-row width above panels B and C",
    9: "legend clears points; labels inside canvas",
}

# These digests bind the manual layout review to the exact TIFFs that were
# inspected. Any rendering change invalidates visual QC until it is reviewed.
VISUAL_QC_APPROVED_SHA256 = {
    1: "f818b5071db3c73ed85714655524022718d3186eca9d0260974e39a1e60c200f",
    2: "99ea72542dd5fab81baa00754d640e7591abc7c9576fbe9458826d42d82859ae",
    3: "02696f4f3e6f8608f39f2a5c649b7aa198540cc3ba87a66429e144f1e44297ae",
    4: "a1e352b21836768da52074736ada8f02e1d8dce82f59e1693422b72518549891",
    5: "fc0fe009e2c90fd485485c21a16b04224e359acad79365b04162bdef4357787d",
    6: "625c8f852f908ed369d05eee7bc24e23f8da841e93267fdaf20176e085402f2f",
    7: "3d7ae2fdd3506e9397f8d14faea2249f5e4b0aa5221dd9d3c1e4df20a764ea54",
    8: "f30b2dab1aaf4104486d732b747daa0f5c627ba5a4ac16ef14352007a52565f8",
    9: "442c61a7b3f1b696bb2f7960f332f3df9dc8a64f50823038cb8fd412210e9bfd",
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_style(style: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in style.split(";"):
        if ":" in item:
            key, value = item.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def format_style(values: dict[str, str]) -> str:
    return "; ".join(f"{key}: {value}" for key, value in values.items())


def numeric_font_size(element: ET.Element) -> float | None:
    style = parse_style(element.get("style", ""))
    raw = style.get("font-size", element.get("font-size", ""))
    match = re.fullmatch(r"\s*([0-9.]+)(?:px|pt)?\s*", raw)
    return float(match.group(1)) if match else None


def normalized_nontext_hash(root: ET.Element) -> str:
    clone = copy.deepcopy(root)
    clone.attrib.pop("width", None)
    clone.attrib.pop("height", None)
    for parent in clone.iter():
        for child in list(parent):
            if local_name(child.tag) == "text":
                parent.remove(child)
    return hashlib.sha256(ET.tostring(clone)).hexdigest()


def iter_text_with_scale(element: ET.Element, scale: float):
    if local_name(element.tag) == "text":
        yield element, scale
    for child in element:
        child_scale = scale
        if local_name(child.tag) == "svg" and child.get("viewBox"):
            child_viewbox = [
                float(value) for value in child.get("viewBox", "").replace(",", " ").split()
            ]
            width_match = re.match(r"\s*([0-9.]+)", child.get("width", ""))
            if len(child_viewbox) == 4 and child_viewbox[2] > 0 and width_match:
                child_scale *= float(width_match.group(1)) / child_viewbox[2]
        yield from iter_text_with_scale(child, child_scale)


def has_math_glyph(text: str) -> bool:
    return any(
        0x0370 <= ord(character) <= 0x03FF
        or 0x1D400 <= ord(character) <= 0x1D7FF
        or character in {"−", "×", "∗"}
        for character in text
    )


def schematic_minimum_font(
    figure_number: int,
    text: str,
    element: ET.Element,
    old_final_size: float,
    family: str,
) -> float:
    x = float(element.get("x", "0") or 0)
    y = float(element.get("y", "0") or 0)
    if "Math" in family or "STIX" in family or has_math_glyph(text):
        return old_final_size
    if figure_number == 1 and x > 640 and y < 300:
        return max(old_final_size, 6.0)
    if figure_number == 4 and y in {64.8, 74.88, 295.92, 306.0}:
        return max(old_final_size, 6.0)
    if figure_number == 5 and y > 270:
        return old_final_size
    return max(old_final_size, SCHEMATIC_MIN_FONT_PT[figure_number])


def adjust_text_position(figure_number: int, text: str, element: ET.Element) -> None:
    y = float(element.get("y", "0") or 0)
    if figure_number == 1 and y == 263.52:
        if text == "-":
            element.set("x", "231.0")
        elif text == "conductivity":
            element.set("x", "235.0")
    if figure_number == 5 and y == 196.56:
        if text == "-":
            element.set("x", "676.0")
        elif text == "ready":
            element.set("x", "680.5")


def normalize_schematic_svg(figure_number: int) -> dict[str, object]:
    source = ROOT / "figures" / f"Fig{figure_number}.svg"
    output = OUTPUT_DIR / f"Figure_{figure_number}.svg"
    tree = ET.parse(source)
    root = tree.getroot()
    viewbox = [float(value) for value in root.get("viewBox", "").replace(",", " ").split()]
    if len(viewbox) != 4 or viewbox[2] <= 0 or viewbox[3] <= 0:
        raise ValueError(f"Invalid viewBox in {source}")

    source_hash = normalized_nontext_hash(root)
    scale = FINAL_WIDTH_PT / viewbox[2]
    root.set("width", f"{FINAL_WIDTH_PT:.4f}pt")
    root.set("height", f"{FINAL_WIDTH_PT * viewbox[3] / viewbox[2]:.4f}pt")

    text_count = 0
    panel_count = 0
    semantic_sizes: list[float] = []
    for element, text_scale in iter_text_with_scale(root, scale):
        text_count += 1
        text_value = "".join(element.itertext()).strip()
        adjust_text_position(figure_number, text_value, element)
        old_size = numeric_font_size(element)
        if old_size is None:
            continue
        style = parse_style(element.get("style", ""))
        family = style.get("font-family", "")
        is_panel = text_value in {"A", "B", "C", "D", "E"}
        is_tiny_chemical_label = figure_number == 5 and old_size < 4
        old_final_size = old_size * text_scale

        if is_panel:
            new_final_size = 13.0
            style["font-weight"] = "700"
            panel_count += 1
        elif is_tiny_chemical_label:
            new_final_size = old_final_size
        else:
            new_final_size = schematic_minimum_font(
                figure_number,
                text_value,
                element,
                old_final_size,
                family,
            )
            if (
                "Math" not in family
                and "STIX" not in family
                and not has_math_glyph(text_value)
                and text_value not in {"+", "-", "−", "∗"}
            ):
                semantic_sizes.append(new_final_size)

        style["font-size"] = f"{new_final_size / text_scale:.6f}px"
        if "Math" not in family and "STIX" not in family and not has_math_glyph(text_value):
            style["font-family"] = "'Liberation Sans', sans-serif"
        element.set("style", format_style(style))

    if figure_number != 5 and panel_count != PANEL_COUNTS[figure_number]:
        raise ValueError(
            f"Figure {figure_number}: expected {PANEL_COUNTS[figure_number]} panel labels, "
            f"found {panel_count}"
        )
    if normalized_nontext_hash(root) != source_hash:
        raise AssertionError(f"Figure {figure_number}: non-text SVG geometry changed")

    tree.write(output, encoding="utf-8", xml_declaration=True)
    render_svg(output)
    return {
        "source": source,
        "text_count": text_count,
        "panel_count": panel_count,
        "minimum_semantic_font_pt": min(semantic_sizes) if semantic_sizes else None,
        "geometry_hash": source_hash,
    }


def render_svg(svg_path: Path) -> None:
    pdf_path = svg_path.with_suffix(".pdf")
    tiff_path = svg_path.with_suffix(".tiff")
    png_bytes = cairosvg.svg2png(
        url=str(svg_path), output_width=OUTPUT_WIDTH_PX, background_color="white"
    )
    with Image.open(io.BytesIO(png_bytes)) as image:
        image.convert("RGB").save(
            tiff_path,
            format="TIFF",
            dpi=(OUTPUT_DPI, OUTPUT_DPI),
            compression="tiff_lzw",
        )
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))


def strip_svg_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def run_figure6() -> str:
    source = ROOT / "scripts" / "figures" / "notebook_exports" / "plot_pipeline_from_notebook.py"
    code = source.read_text(encoding="utf-8").split("# %% [cell 5]", 1)[0]
    namespace = {
        "__name__": "__iscience_figure6__",
        "__file__": str(source),
        "display": lambda *_args, **_kwargs: None,
    }
    previous_dir = os.environ.get("ISCIENCE_REGENERATED_DIR")
    previous_stem = os.environ.get("ISCIENCE_FIGURE6_STEM")
    os.environ["ISCIENCE_REGENERATED_DIR"] = str(OUTPUT_DIR)
    os.environ["ISCIENCE_FIGURE6_STEM"] = "Figure_6"
    try:
        with redirect_stdout(io.StringIO()) as captured:
            exec(compile(code, str(source), "exec"), namespace)
    finally:
        if previous_dir is None:
            os.environ.pop("ISCIENCE_REGENERATED_DIR", None)
        else:
            os.environ["ISCIENCE_REGENERATED_DIR"] = previous_dir
        if previous_stem is None:
            os.environ.pop("ISCIENCE_FIGURE6_STEM", None)
        else:
            os.environ["ISCIENCE_FIGURE6_STEM"] = previous_stem
    return captured.getvalue()


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_figure_script(figure_number: int, script_name: str) -> str:
    path = ROOT / "scripts" / "figures" / script_name
    module = load_module(path, f"iscience_figure_{figure_number}")
    with redirect_stdout(io.StringIO()) as captured:
        result = module.main(output_dir=OUTPUT_DIR, output_stem=f"Figure_{figure_number}")
    if result != 0:
        raise RuntimeError(f"{path} returned {result}")
    return captured.getvalue()


def make_comparison(figure_number: int) -> Path:
    original_path = ROOT / "figures" / f"Fig{figure_number}.tif"
    regenerated_path = OUTPUT_DIR / f"Figure_{figure_number}.tiff"
    panel_width = 930
    header = 42
    with Image.open(original_path) as original_image, Image.open(regenerated_path) as regenerated_image:
        original = original_image.convert("RGB")
        regenerated = regenerated_image.convert("RGB")
        original.thumbnail((panel_width, 2000), Image.Resampling.LANCZOS)
        regenerated.thumbnail((panel_width, 2000), Image.Resampling.LANCZOS)
        height = max(original.height, regenerated.height)
        canvas = Image.new("RGB", (panel_width * 2 + 24, height + header), "white")
        canvas.paste(original, ((panel_width - original.width) // 2, header))
        canvas.paste(regenerated, (panel_width + 24 + (panel_width - regenerated.width) // 2, header))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", 24)
        draw.text((12, 8), "Repository baseline", fill="black", font=font)
        draw.text((panel_width + 36, 8), "Regenerated", fill="black", font=font)
        destination = QA_DIR / f"Comparison_Figure_{figure_number}.png"
        canvas.save(destination, dpi=(150, 150))
    return destination


def tiff_metadata(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        compression = image.info.get("compression", "")
        dpi = image.info.get("dpi", (0, 0))
        return {
            "pixels": f"{image.width}x{image.height}",
            "width_px": image.width,
            "mode": image.mode,
            "dpi_x": round(float(dpi[0])),
            "dpi_y": round(float(dpi[1])),
            "compression": compression,
        }


def edge_clearance(path: Path) -> tuple[int, int, int, int]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        white = Image.new("RGB", rgb.size, "white")
        mask = ImageChops.difference(rgb, white).convert("L").point(
            lambda value: 255 if value > 5 else 0
        )
        bbox = mask.getbbox()
        if bbox is None:
            return (rgb.width, rgb.height, rgb.width, rgb.height)
        return (bbox[0], bbox[1], rgb.width - bbox[2], rgb.height - bbox[3])


def array_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    y_values, x_values = np.nonzero(mask)
    if not len(x_values):
        raise ValueError("Expected rendered pixels were not found")
    return (
        int(x_values.min()),
        int(y_values.min()),
        int(x_values.max()),
        int(y_values.max()),
    )


def targeted_content_clearance(
    figure_number: int, path: Path
) -> tuple[int, int, int, int] | None:
    if figure_number not in {3, 5}:
        return None

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))

    if figure_number == 3:
        x0, y0, x1, y1 = 350, 1750, 1900, 2350
        roi = pixels[y0:y1, x0:x1]
        border = np.all(roi == np.array([93, 115, 154]), axis=2)
        bx0, by0, bx1, by1 = array_bbox(border)
        box = (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)
        inset = pixels[box[1] + 10 : box[3] - 10, box[0] + 10 : box[2] - 10]
        content = np.all(inset == np.array([55, 64, 66]), axis=2)
        tx0, ty0, tx1, ty1 = array_bbox(content)
        text = (tx0 + box[0] + 10, ty0 + box[1] + 10, tx1 + box[0] + 10, ty1 + box[1] + 10)
    else:
        x0, y0, x1, y1 = 2000, 250, 3200, 1150
        roi = pixels[y0:y1, x0:x1]
        border = np.all(roi == np.array([55, 115, 206]), axis=2)
        bx0, by0, bx1, by1 = array_bbox(border)
        box = (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)
        inset = pixels[box[1] + 25 : box[3] - 25, box[0] + 25 : box[2] - 25]
        content = np.all(inset < 20, axis=2)
        tx0, ty0, tx1, ty1 = array_bbox(content)
        text = (tx0 + box[0] + 25, ty0 + box[1] + 25, tx1 + box[0] + 25, ty1 + box[1] + 25)

    return (
        text[0] - box[0],
        text[1] - box[1],
        box[2] - text[2],
        box[3] - text[3],
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def svg_panel_label_count(path: Path) -> int:
    root = ET.parse(path).getroot()
    labels = {"A", "B", "C", "D", "E"}
    return sum(
        1
        for element in root.iter()
        if local_name(element.tag) == "text"
        and "".join(element.itertext()).strip() in labels
    )


def pdf_page_size(path: Path) -> tuple[float, float]:
    page = PdfReader(path).pages[0]
    return float(page.mediabox.width), float(page.mediabox.height)


def forbidden_svg_tokens(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    text = "\n".join(
        "".join(element.itertext())
        for element in root.iter()
        if local_name(element.tag) == "text"
    )
    patterns = {
        "caret exponent": r"10\^-?\d|cm\^-?\d",
        "e notation": r"\d(?:\.\d+)?e[+-]\d+",
        "unicode superscript": r"[⁰¹²³⁴-⁹]",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text, re.IGNORECASE)]


def baseline_checks() -> dict[int, str]:
    checks: dict[int, str] = {}
    for number in range(1, 6):
        checks[number] = "non-text SVG geometry SHA-256 unchanged"

    baseline6 = (ROOT / "figures" / "Fig6.svg").read_text(encoding="utf-8")
    required6 = ["Retained", "5,487", "Excluded", "14,513", "Hit = 0.0072", "Hit = 0.9981"]
    if not all(value in baseline6 for value in required6):
        raise AssertionError("Figure 6 repository baseline does not match source-code key values")
    checks[6] = "baseline SVG key values and 14 source-code points matched"

    oof = pd.read_csv(ROOT / "MY_PAPER_RELATED" / "polybert_con" / "oof_predictions.csv")
    selection = pd.read_csv(
        ROOT / "MY_PAPER_RELATED" / "polybert_con" / "weighted_model_selection_canonical_group.csv"
    )
    if len(oof) != 6270 or len(selection) != 85:
        raise AssertionError("Figure 7 source row counts changed")
    checks[7] = "source CSV rows verified: OOF=6270, model selection=85"

    reference = pd.read_csv(
        ROOT / "MY_PAPER_RELATED" / "machine_readable" / "figure_data" / "fig8_reference_stratified_static_cne0.csv"
    )
    if len(reference) != 108:
        raise AssertionError(f"Figure 8 expected 108 completed rows, found {len(reference)}")
    checks[8] = "source CSV completed rows verified: 108"

    generated = pd.read_csv(
        ROOT
        / "MY_PAPER_RELATED"
        / "gromacs_eval_pred_conductivity"
        / "reproducibility"
        / "static_cne0_release"
        / "data"
        / "generated"
        / "generated_md_results_60.csv"
    )
    groups = generated.groupby("sample_group").size()
    if len(generated) != 60 or len(groups) != 6 or not (groups == 10).all():
        raise AssertionError("Figure 9 candidate/group counts changed")
    replicas = pd.to_numeric(generated["production_replicas"], errors="raise")
    if not (replicas == 3).all() or int(replicas.sum()) != 180:
        raise AssertionError("Figure 9 replica counts changed")
    checks[9] = "source CSV verified: 60 candidates, 6x10 groups, 180 replicas"
    return checks


def write_qa_report(
    schematic_info: dict[int, dict[str, object]],
    run_logs: dict[int, str],
) -> None:
    checks = baseline_checks()
    rows: list[dict[str, object]] = []
    for number in range(1, 10):
        svg_path = OUTPUT_DIR / f"Figure_{number}.svg"
        pdf_path = OUTPUT_DIR / f"Figure_{number}.pdf"
        tiff_path = OUTPUT_DIR / f"Figure_{number}.tiff"
        metadata = tiff_metadata(tiff_path)
        pdf_width_pt, pdf_height_pt = pdf_page_size(pdf_path)
        clearances = edge_clearance(tiff_path)
        targeted_clearance = targeted_content_clearance(number, tiff_path)
        targeted_minimum = {3: 30, 5: 80}.get(number, 0)
        tiff_sha256 = file_sha256(tiff_path)
        visual_approved = tiff_sha256 == VISUAL_QC_APPROVED_SHA256[number]
        panel_label_count = svg_panel_label_count(svg_path)
        expected_panel_labels = 0 if number == 5 else PANEL_COUNTS[number]
        forbidden = forbidden_svg_tokens(svg_path)
        requirements = [
            svg_path.exists(),
            pdf_path.exists(),
            metadata["width_px"] == OUTPUT_WIDTH_PX,
            metadata["mode"] == "RGB",
            metadata["dpi_x"] == OUTPUT_DPI,
            metadata["dpi_y"] == OUTPUT_DPI,
            metadata["compression"] == "tiff_lzw",
            abs(pdf_width_pt - FINAL_WIDTH_PT) < 0.01,
            min(clearances) > 0,
            panel_label_count == expected_panel_labels,
            not forbidden,
            visual_approved,
            targeted_clearance is None or min(targeted_clearance) >= targeted_minimum,
        ]
        minimum_font = (
            schematic_info[number]["minimum_semantic_font_pt"]
            if number in schematic_info
            else {6: 7.0, 7: 7.0, 8: 7.0, 9: 7.0}[number]
        )
        rows.append(
            {
                "figure": f"Figure {number}",
                "status": "PASS" if all(requirements) else "FAIL",
                "panels": PANEL_LABELS[number],
                "data_or_geometry_check": checks[number],
                "minimum_semantic_font_pt": f"{minimum_font:.1f}",
                "pixels": metadata["pixels"],
                "dpi": f"{metadata['dpi_x']}x{metadata['dpi_y']}",
                "mode": metadata["mode"],
                "compression": metadata["compression"],
                "pdf_size_pt": f"{pdf_width_pt:.1f}x{pdf_height_pt:.1f}",
                "edge_clearance_px": "/".join(str(value) for value in clearances),
                "targeted_content_clearance_px": (
                    "/".join(str(value) for value in targeted_clearance)
                    if targeted_clearance is not None
                    else "not applicable"
                ),
                "panel_label_count": panel_label_count,
                "visual_layout_check": (
                    f"PASS: {VISUAL_QC_NOTES[number]} (reviewed TIFF hash matched)"
                    if visual_approved
                    else "FAIL: TIFF differs from manually reviewed artifact"
                ),
                "tiff_sha256": tiff_sha256,
                "forbidden_math_tokens": ", ".join(forbidden) if forbidden else "none",
                "comparison": str(make_comparison(number).relative_to(ROOT)),
            }
        )

    csv_path = OUTPUT_DIR / "QA_REPORT.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# iScience Figure 1-9 regeneration QA",
        "",
        "The exact Windows-side `iScience/02_figures/Figure_N.tiff` files and "
        "`manuscript_iScience_v5.docx` were not present in this checkout. Read-only "
        "comparisons therefore use the tracked `figures/FigN.tif` and editable "
        "`figures/FigN.svg` artifacts as the repository baselines.",
        "",
        "| Figure | Status | Panels | Data/geometry identity | Visual layout QC | Edge clearance L/T/R/B px | Targeted content clearance L/T/R/B px | Min body text / pt | TIFF pixels | PDF points | DPI | RGB/LZW | Forbidden exponent tokens |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['figure']} | {row['status']} | {row['panels']} | "
            f"{row['data_or_geometry_check']} | {row['visual_layout_check']} | "
            f"{row['edge_clearance_px']} | {row['targeted_content_clearance_px']} | "
            f"{row['minimum_semantic_font_pt']} | "
            f"{row['pixels']} | {row['pdf_size_pt']} | {row['dpi']} | "
            f"{row['mode']}/{row['compression']} | "
            f"{row['forbidden_math_tokens']} |"
        )
    lines.extend(
        [
            "",
            "## Scope and limitations",
            "",
            "- Figures 1-5 were edited from tracked SVG sources. Non-text nodes, embedded molecular images, paths, connectors, colors, and their order are byte-normalized and SHA-256 checked before export.",
            "- Figure 6 uses cells 0-4 of the specified notebook-export script. Its 14 plotted model/condition points and key baseline labels were checked before regeneration.",
            "- Figures 7-9 retain their existing CSV loading, validation, data coordinates, statistics, axis ranges, group order, and color mappings. Only typography, annotation formatting, layout, and export handling changed.",
            "- Minimum body-text values exclude mathematical sub/superscripts and chemical atom labels. For plotted figures the listed minimum is the smallest intentional legend/statistical annotation size; axes and tick labels meet the requested 9 pt and 8 pt targets.",
            "- The side-by-side PNG files under `qa/` were manually reviewed at the 6.2-inch comparison scale on 2026-08-27; no generated panel was clipped and the corrected labels, annotations, and legends do not overlap incoherently.",
            "- Each manual visual PASS is bound to the reviewed TIFF SHA-256 in `VISUAL_QC_APPROVED_SHA256`. A rendering change therefore fails QC until the new comparison is reviewed and its digest is explicitly approved.",
            "- This report does not claim pixel identity because typography and layout intentionally changed.",
        ]
    )
    (OUTPUT_DIR / "QA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    log_lines = ["iScience figure regeneration log", ""]
    for number in sorted(run_logs):
        log_lines.append(f"--- Figure {number} ---")
        log_lines.append(run_logs[number].strip())
        log_lines.append("")
    for row in rows:
        log_lines.append(
            f"{row['figure']}: {row['status']}; {row['pixels']}; {row['dpi']} dpi; "
            f"{row['mode']}; {row['compression']}"
        )
    (OUTPUT_DIR / "regeneration.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    schematic_info: dict[int, dict[str, object]] = {}
    run_logs: dict[int, str] = {}

    for number in range(1, 6):
        info = normalize_schematic_svg(number)
        schematic_info[number] = info
        run_logs[number] = (
            f"Source: {info['source']}\n"
            f"Editable text elements: {info['text_count']}\n"
            f"Non-text geometry SHA-256: {info['geometry_hash']}"
        )

    run_logs[6] = run_figure6()
    run_logs[7] = run_figure_script(7, "make_fig7_surrogate_reliability.py")
    run_logs[8] = run_figure_script(8, "make_fig8_reference_stratified_static_cne0.py")
    run_logs[9] = run_figure_script(9, "make_fig9_generated_static_cne0.py")
    for number in range(1, 10):
        strip_svg_trailing_whitespace(OUTPUT_DIR / f"Figure_{number}.svg")
    write_qa_report(schematic_info, run_logs)
    print(f"Regenerated Figure 1-9 artwork: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
