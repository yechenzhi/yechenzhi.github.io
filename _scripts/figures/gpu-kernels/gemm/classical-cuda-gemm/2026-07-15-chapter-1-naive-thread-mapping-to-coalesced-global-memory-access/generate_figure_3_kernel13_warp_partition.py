#!/usr/bin/env python3
"""Generate Figure 3 for the naive CUDA GEMM chapter."""

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
OUTPUT_NAME = "figure-3-kernel13-warp-partition.svg"

CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 700
DISPLAY_WIDTH = 800
CELL_SIZE = 90
DISPLAYED_X_VALUES = ("0", "1", "⋮", "31")
DISPLAYED_Y_VALUES = ("0", "1", "⋯", "31")
SELECTED_COLUMN = 3


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
    transform: str | None = None,
) -> None:
    attributes: dict[str, object] = {
        "x": x,
        "y": y,
        "class": css_class,
        "text-anchor": "middle",
        "dominant-baseline": "middle",
    }
    if transform is not None:
        attributes["transform"] = transform
    node = add_element(parent, "text", **attributes)
    node.text = value


def coordinate_label(x_value: str, y_value: str) -> tuple[str, str]:
    if x_value == "⋮" and y_value == "⋯":
        return "⋱", "ellipsis"
    if x_value == "⋮":
        return "⋮", "ellipsis"
    if y_value == "⋯":
        return "⋯", "ellipsis"
    return f"({x_value}, {y_value})", "coordinate"


def draw_thread_block(parent: ET.Element, *, x: float, y: float) -> None:
    group = add_element(parent, "g", id="thread-block")
    grid_size = len(DISPLAYED_X_VALUES) * CELL_SIZE

    for column, y_value in enumerate(DISPLAYED_Y_VALUES):
        is_selected = column == SELECTED_COLUMN
        is_omitted = y_value == "⋯"
        column_x = x + column * CELL_SIZE

        for row, x_value in enumerate(DISPLAYED_X_VALUES):
            cell_y = y + row * CELL_SIZE
            cell_class = "thread-cell"
            if is_selected:
                cell_class += " selected-cell"
            elif is_omitted:
                cell_class += " omitted-cell"

            add_element(
                group,
                "rect",
                x=column_x,
                y=cell_y,
                width=CELL_SIZE,
                height=CELL_SIZE,
                class_=cell_class,
            )
            label, label_class = coordinate_label(x_value, y_value)
            if is_selected and label_class == "coordinate":
                label_class += " selected-coordinate"
            add_text(
                group,
                column_x + CELL_SIZE / 2,
                cell_y + CELL_SIZE / 2,
                label,
                label_class,
            )

        boundary_class = "column-boundary"
        header_class = "column-header"
        header_text_class = "column-value"
        if is_selected:
            boundary_class += " selected-boundary"
            header_class += " selected-header"
            header_text_class += " selected-value"
        elif is_omitted:
            boundary_class += " omitted-boundary"
            header_class += " omitted-header"

        add_element(
            group,
            "rect",
            x=column_x,
            y=y,
            width=CELL_SIZE,
            height=grid_size,
            class_=boundary_class,
        )
        add_element(
            group,
            "rect",
            x=column_x + 18,
            y=y - 42,
            width=CELL_SIZE - 36,
            height=30,
            rx=8,
            class_=header_class,
        )
        add_text(
            group,
            column_x + CELL_SIZE / 2,
            y - 27,
            y_value,
            header_text_class,
        )

    add_element(
        group,
        "rect",
        x=x,
        y=y,
        width=grid_size,
        height=grid_size,
        class_="block-outline",
    )


def draw_axes(parent: ET.Element, *, x: float, y: float) -> None:
    grid_size = len(DISPLAYED_X_VALUES) * CELL_SIZE
    axis_x = x - 18

    add_text(parent, x + grid_size / 2, y - 82, "threadIdx.y", "axis-label")
    add_element(
        parent,
        "line",
        x1=axis_x,
        y1=y,
        x2=axis_x,
        y2=y + grid_size,
        class_="axis-line",
    )
    add_text(
        parent,
        x - 91,
        y + grid_size / 2,
        "threadIdx.x",
        "axis-label",
        transform=f"rotate(90 {x - 91} {y + grid_size / 2})",
    )

    for row, x_value in enumerate(DISPLAYED_X_VALUES):
        row_y = y + (row + 0.5) * CELL_SIZE
        add_element(
            parent,
            "line",
            x1=axis_x - 6,
            y1=row_y,
            x2=axis_x + 6,
            y2=row_y,
            class_="axis-line",
        )
        add_text(parent, x - 48, row_y, x_value, "axis-value")


def draw_warp_bracket(parent: ET.Element, *, x: float, y: float) -> None:
    grid_size = len(DISPLAYED_X_VALUES) * CELL_SIZE
    add_element(
        parent,
        "path",
        d=(
            f"M {x - 15} {y} H {x} "
            f"V {y + grid_size} H {x - 15}"
        ),
        class_="warp-bracket",
    )
    add_text(
        parent,
        x + 43,
        y + grid_size / 2,
        "32 threads = one warp",
        "warp-callout",
        transform=f"rotate(90 {x + 43} {y + grid_size / 2})",
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
    title.text = "Warp formation in Kernel 13"
    description = add_element(root, "desc", id="figure-description")
    description.text = (
        "A schematic of Kernel 13's 32 by 32 thread block. Rows show "
        "threadIdx.x values zero, one, omitted values, and 31. Columns show "
        "threadIdx.y values zero, one, omitted values, and 31. The column at "
        "threadIdx.y equal to 31 is highlighted as one 32-thread warp. The "
        "complete block contains 32 warps."
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
        --boundary: #475569;
        --accent: #cc79a7;
        --accent-fill: #f7dceb;
        --accent-text: #9d4778;
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --background: #111827;
          --text: #f8fafc;
          --muted-text: #cbd5e1;
          --cell: #1f2937;
          --grid: #64748b;
          --boundary: #cbd5e1;
          --accent-fill: #552f48;
          --accent-text: #f0a8d0;
        }
      }

      text {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
          "Segoe UI", sans-serif;
      }

      .background { fill: var(--background); }
      .thread-cell { fill: var(--cell); stroke: var(--grid); stroke-width: 1.5; }
      .selected-cell { fill: var(--accent-fill); stroke: var(--accent); }
      .omitted-cell { fill: var(--background); stroke-dasharray: 4 4; }
      .coordinate {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 19px;
        font-weight: 600;
        fill: var(--muted-text);
      }
      .selected-coordinate { fill: var(--accent-text); font-weight: 750; }
      .ellipsis { font-size: 34px; font-weight: 700; fill: var(--muted-text); }
      .column-boundary { fill: none; stroke: var(--boundary); stroke-width: 3; }
      .selected-boundary { stroke: var(--accent); stroke-width: 5; }
      .omitted-boundary { stroke: var(--grid); stroke-width: 2; stroke-dasharray: 7 7; }
      .block-outline { fill: none; stroke: var(--boundary); stroke-width: 4; }
      .column-header {
        fill: var(--background);
        stroke: var(--boundary);
        stroke-width: 1.8;
      }
      .selected-header {
        fill: var(--accent-fill);
        stroke: var(--accent);
        stroke-width: 2.5;
      }
      .omitted-header { stroke: none; }
      .column-value { font-size: 18px; font-weight: 750; fill: var(--muted-text); }
      .selected-value { fill: var(--accent-text); }
      .axis-line { stroke: var(--muted-text); stroke-width: 2; }
      .axis-label { font-size: 20px; font-weight: 700; fill: var(--muted-text); }
      .axis-value {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 19px;
        font-weight: 700;
        fill: var(--muted-text);
      }
      .warp-bracket {
        fill: none;
        stroke: var(--accent);
        stroke-width: 4;
        stroke-linecap: round;
        stroke-linejoin: round;
      }
      .warp-callout { font-size: 20px; font-weight: 750; fill: var(--accent-text); }
      .bottom-label { font-size: 20px; font-weight: 700; fill: var(--accent-text); }
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
    grid_x = 625
    grid_y = 165
    draw_axes(root, x=grid_x, y=grid_y)
    draw_thread_block(root, x=grid_x, y=grid_y)
    draw_warp_bracket(
        root,
        x=grid_x + len(DISPLAYED_Y_VALUES) * CELL_SIZE + 20,
        y=grid_y,
    )

    add_text(
        root,
        CANVAS_WIDTH / 2,
        620,
        "32 × 32 threads  →  32 warps",
        "bottom-label",
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
