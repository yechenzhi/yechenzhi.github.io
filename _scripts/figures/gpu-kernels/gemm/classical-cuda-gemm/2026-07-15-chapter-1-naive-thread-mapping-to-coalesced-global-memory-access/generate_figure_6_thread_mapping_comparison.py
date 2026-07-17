#!/usr/bin/env python3
"""Generate Figure 6 for the naive CUDA GEMM chapter."""

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
OUTPUT_NAME = "figure-6-thread-mapping-comparison.svg"

CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 700
DISPLAY_WIDTH = 800

PANEL_Y = 132
PANEL_WIDTH = 740
PANEL_HEIGHT = 544
PANEL_X_POSITIONS = (40, 820)

CELL_WIDTH = 100
CELL_HEIGHT = 64
GRID_SIZE = 4
GRID_Y = 300


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
        "text_anchor": anchor,
        "dominant_baseline": "middle",
    }
    if transform is not None:
        attributes["transform"] = transform
    node = add_element(parent, "text", **attributes)
    node.text = value
    return node


def draw_panel_frame(
    parent: ET.Element,
    *,
    panel_x: float,
    title: str,
    subtitle: str,
) -> None:
    add_element(
        parent,
        "rect",
        x=panel_x,
        y=PANEL_Y,
        width=PANEL_WIDTH,
        height=PANEL_HEIGHT,
        rx=22,
        class_="panel",
    )
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        170,
        title,
        "panel-title",
    )
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        210,
        subtitle,
        "panel-subtitle",
    )


def draw_matrix_grid(
    parent: ET.Element,
    *,
    group_id: str,
    grid_x: float,
    selected_row: int | None,
    selected_column: int | None,
    omitted_row: int | None = None,
    omitted_column: int | None = None,
) -> ET.Element:
    group = add_element(parent, "g", id=group_id)

    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            css_class = "matrix-cell"
            if row == omitted_row or column == omitted_column:
                css_class += " omitted-cell"
            if row == selected_row or column == selected_column:
                css_class += " selected-cell"

            add_element(
                group,
                "rect",
                x=grid_x + column * CELL_WIDTH,
                y=GRID_Y + row * CELL_HEIGHT,
                width=CELL_WIDTH,
                height=CELL_HEIGHT,
                class_=css_class,
            )

    add_element(
        group,
        "rect",
        x=grid_x,
        y=GRID_Y,
        width=GRID_SIZE * CELL_WIDTH,
        height=GRID_SIZE * CELL_HEIGHT,
        class_="matrix-outline",
    )
    return group


def draw_kernel_13_panel(parent: ET.Element, *, panel_x: float) -> None:
    grid_x = panel_x + (PANEL_WIDTH - GRID_SIZE * CELL_WIDTH) / 2
    selected_column = 2
    omitted_row = 2
    row_labels = ("m", "m + 1", "⋮", "m + 31")
    thread_labels = ("thread 0", "thread 1", "⋮", "thread 31")

    draw_panel_frame(
        parent,
        panel_x=panel_x,
        title="Kernel 13",
        subtitle="threadIdx.x selects the output row",
    )
    add_text(
        parent,
        grid_x + GRID_SIZE * CELL_WIDTH / 2,
        251,
        "output columns",
        "axis-title",
    )
    add_text(
        parent,
        grid_x - 108,
        GRID_Y + GRID_SIZE * CELL_HEIGHT / 2,
        "output rows",
        "axis-title",
        transform=(
            f"rotate(-90 {grid_x - 108} "
            f"{GRID_Y + GRID_SIZE * CELL_HEIGHT / 2})"
        ),
    )

    group = draw_matrix_grid(
        parent,
        group_id="kernel-13-d-tile",
        grid_x=grid_x,
        selected_row=None,
        selected_column=selected_column,
        omitted_row=omitted_row,
    )

    selected_x = grid_x + (selected_column + 0.5) * CELL_WIDTH
    add_text(group, selected_x, 279, "n", "selected-index")
    for row, (row_label, thread_label) in enumerate(
        zip(row_labels, thread_labels, strict=True)
    ):
        cell_y = GRID_Y + (row + 0.5) * CELL_HEIGHT
        label_class = "index-label"
        thread_class = "thread-label"
        if row == omitted_row:
            label_class += " ellipsis"
            thread_class += " ellipsis"
        add_text(group, grid_x - 22, cell_y, row_label, label_class, anchor="end")
        add_text(group, selected_x, cell_y, thread_label, thread_class)

    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        590,
        "thread ℓ  →  D[m + ℓ, n]",
        "mapping-formula",
    )
    add_element(
        parent,
        "rect",
        x=panel_x + 174,
        y=619,
        width=392,
        height=38,
        rx=12,
        class_="pattern-tag",
    )
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        638,
        "one column · stride N FP32",
        "pattern-label",
    )


def draw_kernel_14_panel(parent: ET.Element, *, panel_x: float) -> None:
    grid_x = panel_x + (PANEL_WIDTH - GRID_SIZE * CELL_WIDTH) / 2
    selected_row = 1
    omitted_column = 2
    column_labels = ("n", "n + 1", "…", "n + 31")
    thread_labels = ("thread 0", "thread 1", "…", "thread 31")

    draw_panel_frame(
        parent,
        panel_x=panel_x,
        title="Kernel 14",
        subtitle="threadIdx.x selects the output column",
    )
    add_text(
        parent,
        grid_x + GRID_SIZE * CELL_WIDTH / 2,
        251,
        "output columns",
        "axis-title",
    )
    add_text(
        parent,
        grid_x - 108,
        GRID_Y + GRID_SIZE * CELL_HEIGHT / 2,
        "output rows",
        "axis-title",
        transform=(
            f"rotate(-90 {grid_x - 108} "
            f"{GRID_Y + GRID_SIZE * CELL_HEIGHT / 2})"
        ),
    )

    group = draw_matrix_grid(
        parent,
        group_id="kernel-14-d-tile",
        grid_x=grid_x,
        selected_row=selected_row,
        selected_column=None,
        omitted_column=omitted_column,
    )

    selected_y = GRID_Y + (selected_row + 0.5) * CELL_HEIGHT
    add_text(group, grid_x - 22, selected_y, "m", "selected-index", anchor="end")
    for column, (column_label, thread_label) in enumerate(
        zip(column_labels, thread_labels, strict=True)
    ):
        cell_x = grid_x + (column + 0.5) * CELL_WIDTH
        label_class = "index-label"
        thread_class = "thread-label"
        if column == omitted_column:
            label_class += " ellipsis"
            thread_class += " ellipsis"
        add_text(group, cell_x, 279, column_label, label_class)
        add_text(group, cell_x, selected_y, thread_label, thread_class)

    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        590,
        "thread ℓ  →  D[m, n + ℓ]",
        "mapping-formula",
    )
    add_element(
        parent,
        "rect",
        x=panel_x + 174,
        y=619,
        width=392,
        height=38,
        rx=12,
        class_="pattern-tag",
    )
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        638,
        "one row · consecutive FP32",
        "pattern-label",
    )


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
            "width": str(DISPLAY_WIDTH),
            "height": str(round(DISPLAY_WIDTH * CANVAS_HEIGHT / CANVAS_WIDTH)),
            "role": "img",
            "aria-labelledby": "figure-title figure-description",
        },
    )

    title = add_element(root, "title", id="figure-title")
    title.text = "Kernel 13 and Kernel 14 thread-to-output mappings"
    description = add_element(root, "desc", id="figure-description")
    description.text = (
        "The same warp membership is used in both kernels: threadIdx.y is fixed "
        "at w and threadIdx.x equals ell for ell zero through 31. In Kernel 13, those "
        "threads map vertically to D m plus ell comma n. In Kernel 14, they map "
        "horizontally to D m comma n plus ell."
    )

    definitions = add_element(root, "defs")
    style = add_element(definitions, "style")
    style.text = """
      :root {
        color-scheme: light dark;
        --background: #fafaf9;
        --panel: #ffffff;
        --text: #172033;
        --muted-text: #475569;
        --cell: #f1f5f9;
        --grid: #64748b;
        --banner: #eef2f7;
        --banner-border: #94a3b8;
        --d-accent: #cc79a7;
        --d-fill: #f7dceb;
        --d-text: #8f386d;
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --background: #111827;
          --panel: #172033;
          --text: #f8fafc;
          --muted-text: #cbd5e1;
          --cell: #1f2937;
          --grid: #94a3b8;
          --banner: #1f2937;
          --banner-border: #64748b;
          --d-fill: #552f48;
          --d-text: #f0a8d0;
        }
      }

      text {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
          "Segoe UI", sans-serif;
      }

      .background { fill: var(--background); }
      .membership-banner {
        fill: var(--banner);
        stroke: var(--banner-border);
        stroke-width: 2;
      }
      .membership-title { font-size: 27px; font-weight: 780; fill: var(--text); }
      .membership-detail {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 22px;
        font-weight: 650;
        fill: var(--muted-text);
      }
      .panel { fill: var(--panel); stroke: var(--grid); stroke-width: 2; }
      .panel-title { font-size: 34px; font-weight: 800; fill: var(--d-text); }
      .panel-subtitle { font-size: 23px; font-weight: 670; fill: var(--text); }
      .axis-title { font-size: 19px; font-weight: 650; fill: var(--muted-text); }
      .matrix-cell { fill: var(--cell); stroke: var(--grid); stroke-width: 1.5; }
      .omitted-cell { stroke-dasharray: 6 5; }
      .selected-cell {
        fill: var(--d-fill);
        stroke: var(--d-accent);
        stroke-width: 3;
      }
      .matrix-outline { fill: none; stroke: var(--grid); stroke-width: 3; }
      .selected-index { font-size: 22px; font-weight: 820; fill: var(--d-text); }
      .index-label { font-size: 19px; font-weight: 720; fill: var(--muted-text); }
      .thread-label {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 18px;
        font-weight: 760;
        fill: var(--d-text);
      }
      .ellipsis { font-size: 25px; font-weight: 800; }
      .mapping-formula {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 22px;
        font-weight: 760;
        fill: var(--text);
      }
      .pattern-tag {
        fill: var(--d-fill);
        stroke: var(--d-accent);
        stroke-width: 2;
      }
      .pattern-label { font-size: 20px; font-weight: 720; fill: var(--d-text); }
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
    add_element(
        root,
        "rect",
        x=150,
        y=18,
        width=1300,
        height=92,
        rx=18,
        class_="membership-banner",
    )
    add_text(
        root,
        CANVAS_WIDTH / 2,
        47,
        "Warp membership is unchanged",
        "membership-title",
    )
    add_text(
        root,
        CANVAS_WIDTH / 2,
        81,
        "fixed threadIdx.y = w;  threadIdx.x = ℓ = 0, 1, …, 31",
        "membership-detail",
    )

    draw_kernel_13_panel(root, panel_x=PANEL_X_POSITIONS[0])
    draw_kernel_14_panel(root, panel_x=PANEL_X_POSITIONS[1])
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
