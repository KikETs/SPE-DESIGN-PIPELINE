from __future__ import annotations

import base64
import io
import json
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from PIL import Image

from scripts.manuscript_figures.common import (
    FIGURES_DIR,
    ROOT,
    SVG_CANVAS_DPI,
    content_bbox_inches,
    display_or_print,
    figure_row,
    manuscript_spec_path,
    save_tif_600dpi,
    setup_liberation_sans,
)
from scripts.manuscript_figures.plot_ipynb_runner import (
    FIGURE_CELLS,
    SOURCE_OUTPUTS,
    _exec_cell,
    _repo_cwd,
    _setup_namespace,
    cleanup_source_outputs,
)
from scripts.manuscript_figures.svg_rebuilder import SvgRebuilder, element_from_spec


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
XLINK_HREF = f"{{{XLINK_NS}}}href"
FIG2_SOURCE_STEM = ROOT / "figures" / "Figure_raw_training_label_distribution_ecdf"
CONTENT_PADDING_PIXELS = 10

# Slide 2 stores the original A/B plot as one grouped object in this slot.
AB_SLOT_X = 186.0
AB_SLOT_Y = 36.0804
AB_SLOT_WIDTH = 906.0
AB_SLOT_HEIGHT = 392.222 * 1.00157

C_CHILD_START = 1
C_CHILD_STOP = 26


ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def main(display_inline: bool = False) -> dict[str, str]:
    setup_liberation_sans()
    source_svg, source_tif = generate_existing_ab_sources()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = FIGURES_DIR / "Fig2.svg"
    tif_path = FIGURES_DIR / "Fig2.tif"

    render_root = raster_ab_render_root(source_tif)
    renderer = SvgRebuilder(render_root)
    fig = renderer.render()
    bbox_inches = content_bbox_inches(fig, padding_px=CONTENT_PADDING_PIXELS)
    save_tif_600dpi(fig, tif_path, bbox_inches=bbox_inches)
    view_box = bbox_inches_to_svg_viewbox(fig, bbox_inches)
    plt.close(fig)

    write_vector_svg(source_svg, svg_path, view_box)
    cleanup_source_outputs("Fig2")

    row = figure_row("Fig2", svg_path, tif_path)
    return display_or_print(row, display_inline=display_inline)


def generate_existing_ab_sources() -> tuple[Path, Path]:
    namespace = _setup_namespace()
    with _repo_cwd():
        _exec_cell(FIGURE_CELLS["Fig2"], namespace)

    source_svg = FIG2_SOURCE_STEM.with_suffix(".svg")
    source_tif = FIG2_SOURCE_STEM.with_suffix(".tif")
    missing = [path for path in (source_svg, source_tif) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Fig2 A/B source output(s): " + ", ".join(str(path) for path in missing))
    return source_svg, source_tif


def raster_ab_render_root(source_tif: Path) -> ET.Element:
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": "1280",
            "height": "720",
            "overflow": "hidden",
        },
    )
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {"x": "0", "y": "0", "width": "1280", "height": "720", "fill": "#FFFFFF"})
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}image",
        {
            "x": format_float(AB_SLOT_X),
            "y": format_float(AB_SLOT_Y),
            "width": format_float(AB_SLOT_WIDTH),
            "height": format_float(AB_SLOT_HEIGHT),
            XLINK_HREF: image_data_url(source_tif),
            "preserveAspectRatio": "none",
        },
    )
    for child_spec in fig2_c_child_specs():
        root.append(element_from_spec(child_spec))
    return root


def write_vector_svg(source_svg: Path, dest_svg: Path, view_box: tuple[float, float, float, float] | None) -> None:
    source_root = ET.parse(source_svg).getroot()
    source_view_box = source_root.attrib.get("viewBox")
    if not source_view_box:
        raise ValueError(f"Missing viewBox in {source_svg}")

    if view_box is None:
        view_box = (0.0, 0.0, 1280.0, 720.0)
    vx, vy, vw, vh = view_box

    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": format_float(vw),
            "height": format_float(vh),
            "viewBox": " ".join(format_float(value) for value in view_box),
            "overflow": "hidden",
            "version": "1.1",
        },
    )
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {
            "x": format_float(vx),
            "y": format_float(vy),
            "width": format_float(vw),
            "height": format_float(vh),
            "fill": "#FFFFFF",
        },
    )

    nested = ET.SubElement(
        root,
        f"{{{SVG_NS}}}svg",
        {
            "x": format_float(AB_SLOT_X),
            "y": format_float(AB_SLOT_Y),
            "width": format_float(AB_SLOT_WIDTH),
            "height": format_float(AB_SLOT_HEIGHT),
            "viewBox": source_view_box,
            "preserveAspectRatio": "none",
            "overflow": "visible",
        },
    )
    for child in list(source_root):
        if local_name(child.tag) == "metadata":
            continue
        nested.append(deepcopy(child))

    for child_spec in fig2_c_child_specs():
        root.append(element_from_spec(child_spec))

    dest_svg.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(dest_svg, encoding="utf-8", xml_declaration=True)


def fig2_c_child_specs() -> list[dict[str, object]]:
    spec_path = manuscript_spec_path("fig2")
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    group = spec["root"]["children"][1]  # type: ignore[index]
    return list(group["children"][C_CHILD_START:C_CHILD_STOP])  # type: ignore[index]


def image_data_url(source_tif: Path) -> str:
    with Image.open(source_tif) as image:
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def bbox_inches_to_svg_viewbox(fig: plt.Figure, bbox_inches: Bbox | None) -> tuple[float, float, float, float] | None:
    if bbox_inches is None:
        return None
    fig_width, fig_height = fig.get_size_inches()
    x0 = bbox_inches.x0 * SVG_CANVAS_DPI
    x1 = bbox_inches.x1 * SVG_CANVAS_DPI
    y0 = (fig_height - bbox_inches.y1) * SVG_CANVAS_DPI
    y1 = (fig_height - bbox_inches.y0) * SVG_CANVAS_DPI
    return (x0, y0, x1 - x0, y1 - y0)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def format_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return re.sub(r"^-0$", "0", text)

