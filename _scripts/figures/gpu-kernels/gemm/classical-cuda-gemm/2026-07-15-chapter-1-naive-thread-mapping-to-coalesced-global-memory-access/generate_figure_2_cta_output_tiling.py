#!/usr/bin/env python3
"""Generate Figure 2 for the naive CUDA GEMM chapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

POST_PATH = Path(
    "gpu-kernels/gemm/classical-cuda-gemm/"
    "2026-07-15-chapter-1-naive-thread-mapping-to-coalesced-global-memory-access.md"
)
OUTPUT_NAME = "figure-2-cta-output-tiling.svg"

CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 700

OVERVIEW_ROWS = 12
OVERVIEW_COLUMNS = 16
CTA_SIZE = 4
OVERVIEW_CELL_SIZE = 34
DETAIL_CELL_SIZE = 92
SELECTED_CTA = (1, 3)


def svg_tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def add_element(parent: ET.Element, name: str, **attributes: object) -> ET.Element:
    normalized = {
        key.removesuffix("_").replace("_", "-"): str(value)
        for key, value in attributes.items()
    }
    return ET.SubElement(parent, svg_tag(name), normalized)


def add_text(
    parent: ET.Element,
    x: float,
    y: float,
    value: str,
    css_class: str,
    *,
    anchor: str = "middle",
    transform: str | None = None,
) -> ET.Element:
    attributes: dict[str, object] = {
        "x": x,
        "y": y,
        "class": css_class,
        "text-anchor": anchor,
        "dominant-baseline": "middle",
    }
    if transform is not None:
        attributes["transform"] = transform
    node = add_element(parent, "text", **attributes)
    node.text = value
    return node


def draw_horizontal_index_axis(
    parent: ET.Element,
    *,
    x: float,
    y: float,
    count: int,
    spacing: float,
    label: str,
) -> None:
    width = count * spacing
    line_y = y + 44
    add_text(parent, x + width / 2, y, label, "axis-label")
    add_element(
        parent,
        "line",
        x1=x,
        y1=line_y,
        x2=x + width,
        y2=line_y,
        class_="axis-line",
    )
    for index in range(count):
        center_x = x + (index + 0.5) * spacing
        add_text(parent, center_x, y + 27, str(index), "axis-index")
        add_element(
            parent,
            "line",
            x1=center_x,
            y1=line_y - 5,
            x2=center_x,
            y2=line_y + 6,
            class_="axis-line",
        )


def draw_vertical_index_axis(
    parent: ET.Element,
    *,
    x: float,
    y: float,
    count: int,
    spacing: float,
    label: str,
) -> None:
    height = count * spacing
    line_x = x - 12
    label_x = x - 62
    index_x = x - 31
    center_y = y + height / 2
    add_text(
        parent,
        label_x,
        center_y,
        label,
        "axis-label",
        transform=f"rotate(90 {label_x} {center_y})",
    )
    add_element(
        parent,
        "line",
        x1=line_x,
        y1=y,
        x2=line_x,
        y2=y + height,
        class_="axis-line",
    )
    for index in range(count):
        item_y = y + (index + 0.5) * spacing
        add_text(parent, index_x, item_y, str(index), "axis-index")
        add_element(
            parent,
            "line",
            x1=line_x - 5,
            y1=item_y,
            x2=line_x + 6,
            y2=item_y,
            class_="axis-line",
        )


def draw_output_overview(
    parent: ET.Element,
    *,
    x: float,
    y: float,
) -> tuple[float, float, float]:
    group = add_element(parent, "g", id="output-matrix-overview")
    selected_cta_row, selected_cta_column = SELECTED_CTA

    for row in range(OVERVIEW_ROWS):
        for column in range(OVERVIEW_COLUMNS):
            cta_row = row // CTA_SIZE
            cta_column = column // CTA_SIZE
            css_class = "overview-cell"
            if (cta_row, cta_column) == SELECTED_CTA:
                css_class += " selected-tile-cell"
            add_element(
                group,
                "rect",
                x=x + column * OVERVIEW_CELL_SIZE,
                y=y + row * OVERVIEW_CELL_SIZE,
                width=OVERVIEW_CELL_SIZE,
                height=OVERVIEW_CELL_SIZE,
                class_=css_class,
            )

    cta_rows = OVERVIEW_ROWS // CTA_SIZE
    cta_columns = OVERVIEW_COLUMNS // CTA_SIZE
    cta_pixel_size = CTA_SIZE * OVERVIEW_CELL_SIZE

    for cta_row in range(cta_rows):
        for cta_column in range(cta_columns):
            is_selected = (cta_row, cta_column) == SELECTED_CTA
            tile_x = x + cta_column * cta_pixel_size
            tile_y = y + cta_row * cta_pixel_size
            css_class = "cta-boundary"
            if is_selected:
                css_class += " selected-cta-boundary"

            add_element(
                group,
                "rect",
                x=tile_x,
                y=tile_y,
                width=cta_pixel_size,
                height=cta_pixel_size,
                class_=css_class,
            )

            label_width = 68
            label = f"({cta_row}, {cta_column})"
            label_x = tile_x + (cta_pixel_size - label_width) / 2
            label_y = tile_y + (cta_pixel_size - 32) / 2
            label_class = "cta-label-background"
            text_class = "cta-label"
            if is_selected:
                label_class += " selected-cta-label-background"
                text_class += " selected-cta-label"

            add_element(
                group,
                "rect",
                x=label_x,
                y=label_y,
                width=label_width,
                height=32,
                rx=9,
                class_=label_class,
            )
            add_text(
                group,
                tile_x + cta_pixel_size / 2,
                tile_y + cta_pixel_size / 2,
                label,
                text_class,
            )

    matrix_width = OVERVIEW_COLUMNS * OVERVIEW_CELL_SIZE
    matrix_height = OVERVIEW_ROWS * OVERVIEW_CELL_SIZE
    add_element(
        group,
        "rect",
        x=x,
        y=y,
        width=matrix_width,
        height=matrix_height,
        class_="matrix-outline",
    )

    selected_x = x + selected_cta_column * cta_pixel_size
    selected_y = y + selected_cta_row * cta_pixel_size
    return selected_x + cta_pixel_size, selected_y, selected_y + cta_pixel_size


def draw_cta_detail(parent: ET.Element, *, x: float, y: float) -> None:
    group = add_element(parent, "g", id="selected-cta-detail")

    for row in range(CTA_SIZE):
        for column in range(CTA_SIZE):
            cell_x = x + column * DETAIL_CELL_SIZE
            cell_y = y + row * DETAIL_CELL_SIZE
            add_element(
                group,
                "rect",
                x=cell_x,
                y=cell_y,
                width=DETAIL_CELL_SIZE,
                height=DETAIL_CELL_SIZE,
                class_="detail-cell",
            )
            add_text(
                group,
                cell_x + DETAIL_CELL_SIZE / 2,
                cell_y + DETAIL_CELL_SIZE / 2 - 14,
                "thread",
                "thread-label",
            )
            add_text(
                group,
                cell_x + DETAIL_CELL_SIZE / 2,
                cell_y + DETAIL_CELL_SIZE / 2 + 17,
                f"({row}, {column})",
                "thread-coordinate",
            )

    detail_size = CTA_SIZE * DETAIL_CELL_SIZE
    add_element(
        group,
        "rect",
        x=x,
        y=y,
        width=detail_size,
        height=detail_size,
        class_="detail-outline",
    )


def add_explanation_tag(
    parent: ET.Element,
    *,
    x: float,
    y: float,
    width: float,
    label: str,
) -> None:
    add_element(
        parent,
        "rect",
        x=x,
        y=y,
        width=width,
        height=58,
        rx=14,
        class_="explanation-tag",
    )
    add_text(parent, x + width / 2, y + 29, label, "explanation-label")


def find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "_posts").is_dir() and (candidate / "assets" / "img").is_dir():
            return candidate
    raise RuntimeError("Could not locate the repository root")


def default_output_path() -> Path:
    repository_root = find_repository_root(Path(__file__).resolve().parent)
    return repository_root / "assets" / "img" / POST_PATH.with_suffix("") / OUTPUT_NAME


def build_svg() -> ET.Element:
    root = ET.Element(
        svg_tag("svg"),
        {
            "viewBox": f"0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}",
            "width": str(CANVAS_WIDTH // 2),
            "height": str(CANVAS_HEIGHT // 2),
            "role": "img",
            "aria-labelledby": "figure-title figure-description",
        },
    )

    title = add_element(root, "title", id="figure-title")
    title.text = "Partitioning the GEMM output matrix among CUDA thread blocks"
    description = add_element(root, "desc", id="figure-description")
    description.text = (
        "Kernel 13 maps blockIdx.x and threadIdx.x to output rows, and blockIdx.y "
        "and threadIdx.y to output columns. The output matrix D is divided into "
        "tiles, and one tile is enlarged as an illustrative four-by-four thread "
        "block where each in-bounds thread owns one output element."
    )

    definitions = add_element(root, "defs")
    style = add_element(definitions, "style")
    style.text = """
      :root {
        color-scheme: light dark;
        --background: #fafaf9;
        --text: #172033;
        --muted-text: #475569;
        --cell: #f1f5f9;
        --grid: #94a3b8;
        --tile-boundary: #475569;
        --d-accent: #cc79a7;
        --d-fill: #f7dceb;
        --d-text: #9d4778;
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --background: #111827;
          --text: #f8fafc;
          --muted-text: #cbd5e1;
          --cell: #1f2937;
          --grid: #64748b;
          --tile-boundary: #cbd5e1;
          --d-fill: #552f48;
          --d-text: #f0a8d0;
        }
      }

      text {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
          "Segoe UI", sans-serif;
      }

      .background { fill: var(--background); }
      .panel-title { font-size: 38px; font-weight: 750; fill: var(--text); }
      .panel-title-d { fill: var(--d-text); }
      .panel-subtitle { font-size: 25px; font-weight: 550; fill: var(--muted-text); }
      .overview-cell { fill: var(--cell); stroke: var(--grid); stroke-width: 1; }
      .selected-tile-cell { fill: var(--d-fill); }
      .cta-boundary { fill: none; stroke: var(--tile-boundary); stroke-width: 3; }
      .selected-cta-boundary { stroke: var(--d-accent); stroke-width: 5; }
      .matrix-outline { fill: none; stroke: var(--tile-boundary); stroke-width: 4; }
      .cta-label-background {
        fill: var(--background);
        stroke: var(--tile-boundary);
        stroke-width: 1.5;
      }
      .selected-cta-label-background {
        fill: var(--d-fill);
        stroke: var(--d-accent);
        stroke-width: 2.5;
      }
      .cta-label { font-size: 19px; font-weight: 700; fill: var(--muted-text); }
      .selected-cta-label { font-size: 19px; fill: var(--d-text); }
      .axis-line { stroke: var(--muted-text); stroke-width: 2; }
      .axis-label { font-size: 19px; font-weight: 700; fill: var(--muted-text); }
      .axis-index {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 18px;
        font-weight: 700;
        fill: var(--muted-text);
      }
      .connector {
        fill: none;
        stroke: var(--d-accent);
        stroke-width: 4;
        stroke-dasharray: 11 10;
      }
      .zoom-label-background {
        fill: var(--background);
        stroke: var(--d-accent);
        stroke-width: 2.5;
      }
      .zoom-label { font-size: 21px; font-weight: 700; fill: var(--d-text); }
      .detail-cell { fill: var(--d-fill); stroke: var(--d-accent); stroke-width: 2; }
      .detail-outline { fill: none; stroke: var(--d-accent); stroke-width: 5; }
      .thread-label { font-size: 19px; font-weight: 750; fill: var(--d-text); }
      .thread-coordinate {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 18px;
        font-weight: 550;
        fill: var(--muted-text);
      }
      .explanation-tag {
        fill: var(--d-fill);
        stroke: var(--d-accent);
        stroke-width: 2.5;
      }
      .explanation-label { font-size: 22px; font-weight: 700; fill: var(--d-text); }
    """

    add_element(
        root,
        "rect",
        x=0,
        y=0,
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        rx=24,
        class_="background",
    )

    overview_x = 80
    overview_y = 170
    overview_width = OVERVIEW_COLUMNS * OVERVIEW_CELL_SIZE
    overview_height = OVERVIEW_ROWS * OVERVIEW_CELL_SIZE
    overview_center_x = overview_x + overview_width / 2

    detail_x = 1080
    detail_y = 170
    detail_size = CTA_SIZE * DETAIL_CELL_SIZE
    detail_center_x = detail_x + detail_size / 2

    add_text(
        root,
        overview_center_x,
        47,
        "Output matrix D",
        "panel-title panel-title-d",
    )
    add_text(
        root,
        overview_center_x,
        86,
        "(M × N), partitioned into output tiles",
        "panel-subtitle",
    )
    add_text(root, detail_center_x, 47, "One CTA / thread block", "panel-title")
    add_text(
        root,
        detail_center_x,
        86,
        "illustrative 4 × 4 threads",
        "panel-subtitle",
    )

    cta_pixel_size = CTA_SIZE * OVERVIEW_CELL_SIZE
    draw_horizontal_index_axis(
        root,
        x=overview_x,
        y=111,
        count=OVERVIEW_COLUMNS // CTA_SIZE,
        spacing=cta_pixel_size,
        label="blockIdx.y  →  output tile columns",
    )
    draw_vertical_index_axis(
        root,
        x=overview_x,
        y=overview_y,
        count=OVERVIEW_ROWS // CTA_SIZE,
        spacing=cta_pixel_size,
        label="blockIdx.x  →  output tile rows",
    )
    draw_horizontal_index_axis(
        root,
        x=detail_x,
        y=111,
        count=CTA_SIZE,
        spacing=DETAIL_CELL_SIZE,
        label="threadIdx.y  →  output columns",
    )
    draw_vertical_index_axis(
        root,
        x=detail_x,
        y=detail_y,
        count=CTA_SIZE,
        spacing=DETAIL_CELL_SIZE,
        label="threadIdx.x  →  output rows",
    )

    selected_right_x, selected_top_y, selected_bottom_y = draw_output_overview(
        root,
        x=overview_x,
        y=overview_y,
    )

    add_element(
        root,
        "line",
        x1=selected_right_x,
        y1=selected_top_y,
        x2=detail_x,
        y2=detail_y,
        class_="connector",
    )
    add_element(
        root,
        "line",
        x1=selected_right_x,
        y1=selected_bottom_y,
        x2=detail_x,
        y2=detail_y + detail_size,
        class_="connector",
    )
    add_element(
        root,
        "rect",
        x=805,
        y=334,
        width=126,
        height=42,
        rx=12,
        class_="zoom-label-background",
    )
    add_text(root, 868, 355, "zoom in", "zoom-label")

    draw_cta_detail(root, x=detail_x, y=detail_y)

    add_explanation_tag(
        root,
        x=100,
        y=607,
        width=500,
        label="one CTA  →  one output tile of D",
    )
    add_explanation_tag(
        root,
        x=970,
        y=607,
        width=540,
        label="one in-bounds thread  →  one D[m, n] element",
    )

    return root


def write_svg(output_path: Path) -> None:
    root = build_svg()
    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )
    output_path.write_text(document, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="SVG output path (defaults to the post-aligned assets/img directory)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    write_svg(arguments.output.resolve())
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
