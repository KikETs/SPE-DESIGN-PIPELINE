from __future__ import annotations

import base64
import io
import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import path as mpath
from matplotlib import transforms as mtransforms
from PIL import Image

from scripts.manuscript_figures.common import (
    FIGURES_DIR,
    FONT_FAMILY,
    MATH_FALLBACK_FONT_SCALE,
    SPEC_DIR,
    POINTS_PER_SVG_UNIT,
    SVG_CANVAS_DPI,
    content_bbox_inches,
    display_or_print,
    figure_row,
    math_font_family,
    manuscript_spec_path,
    manuscript_svg_path,
    save_tif_600dpi,
    setup_liberation_sans,
    using_math_fallback,
)


XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
COMMAND_RE = re.compile(r"[MmZzLlHhVvCcSsQqTtAa]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
TRANSFORM_RE = re.compile(r"([A-Za-z]+)\(([^)]*)\)")
SVG_CONTENT_PADDING_PIXELS = 10


@dataclass(frozen=True)
class RenderState:
    transform: np.ndarray
    clip_id: str | None = None
    clip_transform: np.ndarray | None = None


def rebuild_slide_figure(slide_number: int, output_name: str, display_inline: bool = False) -> dict[str, str]:
    source_svg = manuscript_svg_path(slide_number)
    if not source_svg.exists():
        raise FileNotFoundError(source_svg)

    setup_liberation_sans()
    renderer = SvgRebuilder(source_svg)
    fig = renderer.render()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = FIGURES_DIR / f"{output_name}.svg"
    tif_path = FIGURES_DIR / f"{output_name}.tif"
    bbox_inches = content_bbox_inches(fig, padding_px=SVG_CONTENT_PADDING_PIXELS)
    fig.savefig(svg_path, facecolor="white", bbox_inches=bbox_inches, pad_inches=0.0)
    save_tif_600dpi(fig, tif_path, bbox_inches=bbox_inches)
    plt.close(fig)

    row = figure_row(output_name, svg_path, tif_path)
    return display_or_print(row, display_inline=display_inline)


def rebuild_spec_figure(spec_name: str, output_name: str, display_inline: bool = False) -> dict[str, str]:
    spec_path = manuscript_spec_path(spec_name)
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)

    setup_liberation_sans()
    renderer = SvgRebuilder(root_from_spec(json.loads(spec_path.read_text(encoding="utf-8"))))
    fig = renderer.render()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = FIGURES_DIR / f"{output_name}.svg"
    tif_path = FIGURES_DIR / f"{output_name}.tif"
    bbox_inches = content_bbox_inches(fig, padding_px=SVG_CONTENT_PADDING_PIXELS)
    fig.savefig(svg_path, facecolor="white", bbox_inches=bbox_inches, pad_inches=0.0)
    save_tif_600dpi(fig, tif_path, bbox_inches=bbox_inches)
    plt.close(fig)

    row = figure_row(output_name, svg_path, tif_path)
    return display_or_print(row, display_inline=display_inline)


def rebuild_spec_figure_tif_only(spec_name: str, output_name: str, display_inline: bool = False) -> dict[str, str]:
    spec_path = manuscript_spec_path(spec_name)
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)

    setup_liberation_sans()
    renderer = SvgRebuilder(root_from_spec(json.loads(spec_path.read_text(encoding="utf-8"))))
    fig = renderer.render()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    svg_path = FIGURES_DIR / f"{output_name}.svg"
    tif_path = FIGURES_DIR / f"{output_name}.tif"
    bbox_inches = content_bbox_inches(fig, padding_px=SVG_CONTENT_PADDING_PIXELS)
    save_tif_600dpi(fig, tif_path, bbox_inches=bbox_inches)
    plt.close(fig)
    if svg_path.exists():
        svg_path.unlink()

    row = {
        "figure": output_name,
        "tif": str(tif_path.relative_to(FIGURES_DIR.parent)),
    }
    print(f"{row['figure']}: {row['tif']}")
    if display_inline:
        from IPython.display import Image as IPythonImage
        from IPython.display import display

        preview = Image.open(tif_path).convert("RGB")
        preview.thumbnail((1400, 900), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        preview.save(buffer, format="PNG")
        display(IPythonImage(data=buffer.getvalue()))
    return row


def write_slide_spec(slide_number: int, spec_name: str) -> Path:
    source_svg = manuscript_svg_path(slide_number)
    if not source_svg.exists():
        raise FileNotFoundError(source_svg)
    root = ET.parse(source_svg).getroot()
    spec = {
        "schema": "manuscript_figure_svg_element_tree_v1",
        "source": {
            "slide_number": slide_number,
            "source_svg_name": source_svg.name,
        },
        "root": element_to_spec(root),
    }
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = manuscript_spec_path(spec_name)
    out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def element_to_spec(element: ET.Element) -> dict[str, object]:
    return {
        "tag": element.tag,
        "attrs": dict(element.attrib),
        "text": element.text or "",
        "children": [element_to_spec(child) for child in element],
    }


def root_from_spec(spec: dict[str, object]) -> ET.Element:
    if spec.get("schema") != "manuscript_figure_svg_element_tree_v1":
        raise ValueError(f"Unsupported manuscript figure spec schema: {spec.get('schema')!r}")
    return element_from_spec(spec["root"])  # type: ignore[arg-type]


def element_from_spec(spec: dict[str, object]) -> ET.Element:
    element = ET.Element(str(spec["tag"]), {str(k): str(v) for k, v in dict(spec.get("attrs", {})).items()})
    text = spec.get("text")
    if text:
        element.text = str(text)
    for child_spec in spec.get("children", []):
        element.append(element_from_spec(child_spec))  # type: ignore[arg-type]
    return element


class SvgRebuilder:
    def __init__(self, source: Path | ET.Element):
        self.svg_path = source if isinstance(source, Path) else None
        self.root = ET.parse(source).getroot() if isinstance(source, Path) else source
        self.width = parse_length(self.root.attrib.get("width", "1280"))
        self.height = parse_length(self.root.attrib.get("height", "720"))
        self.defs = self._collect_defs()
        self.gradients = self._collect_gradients()
        self.fig: plt.Figure | None = None
        self.ax: plt.Axes | None = None
        self.zorder = 1

    def render(self) -> plt.Figure:
        self.fig = plt.figure(figsize=(self.width / SVG_CANVAS_DPI, self.height / SVG_CANVAS_DPI), dpi=SVG_CANVAS_DPI)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(self.height, 0)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.axis("off")
        self.fig.patch.set_facecolor("white")

        state = RenderState(transform=np.eye(3), clip_id=None, clip_transform=None)
        for child in self.root:
            if local_name(child.tag) == "defs":
                continue
            self._render_element(child, state)
        return self.fig

    def _collect_defs(self) -> dict[str, ET.Element]:
        defs: dict[str, ET.Element] = {}
        for element in self.root.iter():
            element_id = element.attrib.get("id")
            if element_id:
                defs[element_id] = element
        return defs

    def _collect_gradients(self) -> dict[str, dict[str, object]]:
        gradients: dict[str, dict[str, object]] = {}
        for element in self.root.iter():
            if local_name(element.tag) != "linearGradient":
                continue
            gradient_id = element.attrib.get("id")
            if not gradient_id:
                continue
            stops = []
            for stop in element:
                if local_name(stop.tag) != "stop":
                    continue
                stops.append(
                    (
                        parse_offset(stop.attrib.get("offset", "0")),
                        stop.attrib.get("stop-color", "#000000"),
                    )
                )
            gradients[gradient_id] = {
                "x1": parse_length(element.attrib.get("x1", "0")),
                "y1": parse_length(element.attrib.get("y1", "0")),
                "x2": parse_length(element.attrib.get("x2", "1")),
                "y2": parse_length(element.attrib.get("y2", "0")),
                "spread": element.attrib.get("spreadMethod", "pad"),
                "stops": sorted(stops),
            }
        return gradients

    def _render_element(self, element: ET.Element, state: RenderState) -> None:
        tag = local_name(element.tag)
        if tag in {"defs", "clipPath", "linearGradient", "stop"}:
            return

        local_transform = parse_transform(element.attrib.get("transform"))
        next_transform = state.transform @ local_transform
        own_clip_id = parse_url_id(element.attrib.get("clip-path"))
        next_state = RenderState(
            transform=next_transform,
            clip_id=own_clip_id or state.clip_id,
            clip_transform=next_transform if own_clip_id else state.clip_transform,
        )

        if tag == "g":
            for child in element:
                self._render_element(child, next_state)
            return
        if tag == "path":
            self._draw_path(element, next_state)
        elif tag == "rect":
            self._draw_rect(element, next_state)
        elif tag == "text":
            self._draw_text(element, next_state)
        elif tag == "image":
            self._draw_image(element, next_state)
        elif tag == "use":
            self._draw_use(element, next_state)

        for child in element:
            self._render_element(child, next_state)

    @property
    def _axes(self) -> plt.Axes:
        if self.ax is None:
            raise RuntimeError("render() has not initialized axes")
        return self.ax

    def _next_zorder(self) -> int:
        self.zorder += 1
        return self.zorder

    def _clip_patch(self, clip_id: str | None, transform: np.ndarray) -> mpatches.PathPatch | None:
        if not clip_id:
            return None
        clip_element = self.defs.get(clip_id)
        if clip_element is None:
            return None
        paths: list[mpath.Path] = []
        for child in clip_element:
            tag = local_name(child.tag)
            if tag == "rect":
                paths.append(rect_path(child))
            elif tag == "path" and child.attrib.get("d"):
                paths.append(parse_path_data(child.attrib["d"]))
        if not paths:
            return None
        compound = compound_path(paths)
        return mpatches.PathPatch(
            compound,
            transform=affine(transform) + self._axes.transData,
            facecolor="none",
            edgecolor="none",
        )

    def _apply_clip(self, artist, state: RenderState) -> None:
        clip_patch = self._clip_patch(state.clip_id, state.clip_transform if state.clip_transform is not None else state.transform)
        if clip_patch is not None:
            artist.set_clip_path(clip_patch)

    def _draw_path(self, element: ET.Element, state: RenderState) -> None:
        d = element.attrib.get("d")
        if not d:
            return
        svg_path = parse_path_data(d)
        style = element_style(element)
        facecolor, edgecolor, linewidth = colors_and_linewidth(style)
        gradient_id = parse_url_id(style.get("fill"))
        transform = affine(state.transform) + self._axes.transData

        if gradient_id:
            outline = mpatches.PathPatch(svg_path, transform=transform, facecolor="none", edgecolor="none")
            image = self._gradient_image(gradient_id)
            if image is not None:
                artist = self._axes.imshow(
                    image,
                    extent=(0, self.width, self.height, 0),
                    origin="upper",
                    interpolation="bicubic",
                    zorder=self._next_zorder(),
                )
                artist.set_clip_path(outline)
                self._apply_clip(artist, state)
            if edgecolor != "none" and linewidth > 0:
                patch = mpatches.PathPatch(
                    svg_path,
                    transform=transform,
                    facecolor="none",
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    joinstyle=style.get("stroke-linejoin", "miter"),
                    capstyle=capstyle(style.get("stroke-linecap", "butt")),
                    zorder=self._next_zorder(),
                )
                self._style_dash(patch, style)
                self._apply_clip(patch, state)
                self._axes.add_patch(patch)
            return

        patch = mpatches.PathPatch(
            svg_path,
            transform=transform,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            joinstyle=style.get("stroke-linejoin", "miter"),
            capstyle=capstyle(style.get("stroke-linecap", "butt")),
            zorder=self._next_zorder(),
        )
        self._style_dash(patch, style)
        self._apply_clip(patch, state)
        self._axes.add_patch(patch)

    def _draw_rect(self, element: ET.Element, state: RenderState) -> None:
        style = element_style(element)
        facecolor, edgecolor, linewidth = colors_and_linewidth(style)
        if facecolor == "none" and edgecolor == "none":
            return
        patch = mpatches.PathPatch(
            rect_path(element),
            transform=affine(state.transform) + self._axes.transData,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=self._next_zorder(),
        )
        self._style_dash(patch, style)
        self._apply_clip(patch, state)
        self._axes.add_patch(patch)

    def _draw_text(self, element: ET.Element, state: RenderState) -> None:
        style = element_style(element)
        fill = style.get("fill", "#000000")
        if fill == "none":
            return
        fill_alpha = parse_float(style.get("fill-opacity", "1"), default=1.0)
        if fill_alpha <= 0:
            return
        raw_text = "".join(element.itertext())
        is_math = "Cambria Math" in style.get("font-family", "")
        text = raw_text if is_math and not using_math_fallback() else unicodedata.normalize("NFKC", raw_text)
        if not text:
            return
        fontsize = parse_length(style.get("font-size", "12")) * POINTS_PER_SVG_UNIT
        if is_math and using_math_fallback():
            fontsize *= MATH_FALLBACK_FONT_SCALE
        fontweight = "bold" if style.get("font-weight") in {"700", "bold", "800", "900"} else "normal"
        color = color_with_alpha(fill, fill_alpha)
        text_anchor = style.get("text-anchor", "start")
        ha = {"middle": "center", "end": "right"}.get(text_anchor, "left")
        self._axes.text(
            0,
            0,
            text,
            transform=affine(state.transform) + self._axes.transData,
            fontfamily=math_font_family() if is_math else FONT_FAMILY,
            fontweight=fontweight,
            fontsize=fontsize,
            color=color,
            ha=ha,
            va="baseline",
            rotation_mode="anchor",
            zorder=self._next_zorder(),
        )

    def _draw_image(self, element: ET.Element, state: RenderState) -> None:
        image = decode_image_element(element)
        if image is None:
            return
        x = parse_length(element.attrib.get("x", "0"))
        y = parse_length(element.attrib.get("y", "0"))
        width = parse_length(element.attrib.get("width", str(image.width)))
        height = parse_length(element.attrib.get("height", str(image.height)))
        image_transform = state.transform @ translate_matrix(x, y)
        artist = self._axes.imshow(
            np.asarray(image),
            extent=(0, width, height, 0),
            origin="upper",
            transform=affine(image_transform) + self._axes.transData,
            interpolation="lanczos",
            zorder=self._next_zorder(),
        )
        self._apply_clip(artist, state)

    def _draw_use(self, element: ET.Element, state: RenderState) -> None:
        href = element.attrib.get(XLINK_HREF) or element.attrib.get("href")
        ref_id = href[1:] if href and href.startswith("#") else None
        if not ref_id:
            return
        target = self.defs.get(ref_id)
        if target is None:
            return
        target_tag = local_name(target.tag)
        if target_tag == "image":
            self._draw_image(target, state)
        elif target_tag == "path":
            self._draw_path(target, state)
        elif target_tag == "rect":
            self._draw_rect(target, state)

    def _style_dash(self, patch: mpatches.PathPatch, style: dict[str, str]) -> None:
        dash = style.get("stroke-dasharray")
        if not dash or dash == "none":
            return
        values = [parse_length(part) * POINTS_PER_SVG_UNIT for part in re.split(r"[,\s]+", dash.strip()) if part]
        if values:
            patch.set_linestyle((0, values))

    def _gradient_image(self, gradient_id: str) -> np.ndarray | None:
        gradient = self.gradients.get(gradient_id)
        if gradient is None:
            return None
        stops = gradient.get("stops") or [(0.0, "#000000"), (1.0, "#000000")]
        stops = list(stops)  # type: ignore[arg-type]
        if len(stops) == 1:
            stops = [(0.0, stops[0][1]), (1.0, stops[0][1])]

        x1, y1 = float(gradient["x1"]), float(gradient["y1"])
        x2, y2 = float(gradient["x2"]), float(gradient["y2"])
        xx, yy = np.meshgrid(np.linspace(0, self.width, int(self.width)), np.linspace(0, self.height, int(self.height)))
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        t = np.zeros_like(xx) if denom == 0 else ((xx - x1) * dx + (yy - y1) * dy) / denom
        if gradient.get("spread") == "reflect":
            t = np.abs((t % 2) - 1)
        else:
            t = np.clip(t, 0, 1)

        offsets = np.array([offset for offset, _ in stops], dtype=float)
        rgba_stops = np.array([mcolors.to_rgba(color) for _, color in stops], dtype=float)
        channels = [np.interp(t, offsets, rgba_stops[:, idx]) for idx in range(4)]
        return np.dstack(channels)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_length(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    stripped = str(value).strip()
    if stripped.endswith("%"):
        stripped = stripped[:-1]
    match = re.match(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", stripped)
    return float(match.group(0)) if match else 0.0


def parse_float(value: str | float | int | None, default: float = 0.0) -> float:
    try:
        return parse_length(value)
    except Exception:
        return default


def parse_offset(value: str) -> float:
    value = value.strip()
    if value.endswith("%"):
        return float(value[:-1]) / 100.0
    return float(value)


def element_style(element: ET.Element) -> dict[str, str]:
    style: dict[str, str] = {
        "fill": element.attrib.get("fill", "#000000"),
        "stroke": element.attrib.get("stroke", "none"),
        "stroke-width": element.attrib.get("stroke-width", "1"),
    }
    for key, value in element.attrib.items():
        name = local_name(key)
        if name in {
            "fill",
            "stroke",
            "stroke-width",
            "fill-opacity",
            "stroke-opacity",
            "stroke-dasharray",
            "stroke-linecap",
            "stroke-linejoin",
            "font-size",
            "font-family",
            "font-weight",
            "text-anchor",
        }:
            style[name] = value
    return style


def color_with_alpha(color: str, alpha: float = 1.0):
    if color == "none" or color.startswith("url("):
        return "none"
    try:
        rgba = mcolors.to_rgba(color)
    except ValueError:
        rgba = (0, 0, 0, 1)
    return rgba[:3] + (rgba[3] * alpha,)


def colors_and_linewidth(style: dict[str, str]):
    fill = style.get("fill", "#000000")
    stroke = style.get("stroke", "none")
    fill_alpha = parse_float(style.get("fill-opacity", "1"), default=1.0)
    stroke_alpha = parse_float(style.get("stroke-opacity", "1"), default=1.0)
    facecolor = "none" if fill == "none" or fill_alpha <= 0 or fill.startswith("url(") else color_with_alpha(fill, fill_alpha)
    edgecolor = "none" if stroke == "none" or stroke_alpha <= 0 else color_with_alpha(stroke, stroke_alpha)
    linewidth = parse_length(style.get("stroke-width", "1")) * POINTS_PER_SVG_UNIT
    return facecolor, edgecolor, linewidth


def capstyle(value: str) -> str:
    return {"square": "projecting"}.get(value, value)


def parse_url_id(value: str | None) -> str | None:
    if not value:
        return None
    match = re.match(r"url\(#([^)]+)\)", value.strip())
    return match.group(1) if match else None


def translate_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def scale_matrix(sx: float, sy: float | None = None) -> np.ndarray:
    sy = sx if sy is None else sy
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])


def rotate_matrix(angle_deg: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    angle = math.radians(angle_deg)
    rot = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return translate_matrix(cx, cy) @ rot @ translate_matrix(-cx, -cy)


def parse_transform(value: str | None) -> np.ndarray:
    if not value:
        return np.eye(3)
    matrix = np.eye(3)
    for name, raw_args in TRANSFORM_RE.findall(value):
        args = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", raw_args)]
        if name == "matrix" and len(args) == 6:
            a, b, c, d, e, f = args
            local = np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])
        elif name == "translate":
            local = translate_matrix(args[0], args[1] if len(args) > 1 else 0.0)
        elif name == "scale":
            local = scale_matrix(args[0], args[1] if len(args) > 1 else None)
        elif name == "rotate":
            local = rotate_matrix(args[0], args[1] if len(args) > 2 else 0.0, args[2] if len(args) > 2 else 0.0)
        else:
            local = np.eye(3)
        matrix = matrix @ local
    return matrix


def affine(matrix: np.ndarray) -> mtransforms.Affine2D:
    return mtransforms.Affine2D.from_values(
        matrix[0, 0],
        matrix[1, 0],
        matrix[0, 1],
        matrix[1, 1],
        matrix[0, 2],
        matrix[1, 2],
    )


def rect_path(element: ET.Element) -> mpath.Path:
    x = parse_length(element.attrib.get("x", "0"))
    y = parse_length(element.attrib.get("y", "0"))
    width = parse_length(element.attrib.get("width", "0"))
    height = parse_length(element.attrib.get("height", "0"))
    vertices = [(x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)]
    codes = [mpath.Path.MOVETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    return mpath.Path(vertices, codes)


def compound_path(paths: Iterable[mpath.Path]) -> mpath.Path:
    vertices = []
    codes = []
    for path in paths:
        vertices.extend(path.vertices.tolist())
        codes.extend(path.codes.tolist())
    return mpath.Path(vertices, codes)


def parse_path_data(d: str) -> mpath.Path:
    tokens = COMMAND_RE.findall(d.replace(",", " "))
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    idx = 0
    command = None
    current = (0.0, 0.0)
    start = (0.0, 0.0)

    def is_command(token: str) -> bool:
        return len(token) == 1 and token.isalpha()

    def has_numbers(count: int) -> bool:
        return idx + count <= len(tokens) and not any(is_command(tokens[idx + j]) for j in range(count))

    def read_float() -> float:
        nonlocal idx
        value = float(tokens[idx])
        idx += 1
        return value

    def add_point(x: float, y: float, code: int) -> None:
        nonlocal current, start
        vertices.append((x, y))
        codes.append(code)
        current = (x, y)
        if code == mpath.Path.MOVETO:
            start = (x, y)

    while idx < len(tokens):
        if is_command(tokens[idx]):
            command = tokens[idx]
            idx += 1
        if command is None:
            break

        relative = command.islower()
        cmd = command.upper()
        if cmd == "M":
            first = True
            while has_numbers(2):
                x, y = read_float(), read_float()
                if relative:
                    x += current[0]
                    y += current[1]
                add_point(x, y, mpath.Path.MOVETO if first else mpath.Path.LINETO)
                first = False
            command = "l" if relative else "L"
        elif cmd == "L":
            while has_numbers(2):
                x, y = read_float(), read_float()
                if relative:
                    x += current[0]
                    y += current[1]
                add_point(x, y, mpath.Path.LINETO)
        elif cmd == "H":
            while has_numbers(1):
                x = read_float()
                if relative:
                    x += current[0]
                add_point(x, current[1], mpath.Path.LINETO)
        elif cmd == "V":
            while has_numbers(1):
                y = read_float()
                if relative:
                    y += current[1]
                add_point(current[0], y, mpath.Path.LINETO)
        elif cmd == "C":
            while has_numbers(6):
                points = [(read_float(), read_float()) for _ in range(3)]
                if relative:
                    points = [(x + current[0], y + current[1]) for x, y in points]
                vertices.extend(points)
                codes.extend([mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4])
                current = points[-1]
        elif cmd == "Z":
            vertices.append(start)
            codes.append(mpath.Path.CLOSEPOLY)
            current = start
        else:
            while idx < len(tokens) and not is_command(tokens[idx]):
                idx += 1
    if not vertices:
        vertices = [(0.0, 0.0)]
        codes = [mpath.Path.MOVETO]
    return mpath.Path(vertices, codes)


def decode_image_element(element: ET.Element) -> Image.Image | None:
    href = element.attrib.get(XLINK_HREF) or element.attrib.get("href")
    if not href or not href.startswith("data:image/"):
        return None
    _, payload = href.split(",", 1)
    raw = base64.b64decode(payload)
    return Image.open(io.BytesIO(raw)).convert("RGBA")
