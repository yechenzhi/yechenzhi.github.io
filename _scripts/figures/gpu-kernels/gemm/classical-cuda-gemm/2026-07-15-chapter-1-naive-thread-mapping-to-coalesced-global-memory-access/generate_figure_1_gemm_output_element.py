#!/usr/bin/env python3
"""Generate Figure 1 for the naive CUDA GEMM chapter."""

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
OUTPUT_NAME = "figure-1-gemm-output-element.svg"

CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 650
CELL_SIZE = 46


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


def draw_matrix(
    parent: ET.Element,
    *,
    matrix_id: str,
    x: float,
    y: float,
    rows: int,
    columns: int,
    highlight_kind: str,
    highlight_index: int | tuple[int, int],
) -> None:
    group = add_element(parent, "g", id=matrix_id)

    for row in range(rows):
        for column in range(columns):
            is_highlighted = False
            if highlight_kind == "row":
                is_highlighted = row == highlight_index
            elif highlight_kind == "column":
                is_highlighted = column == highlight_index
            elif highlight_kind == "cell":
                is_highlighted = (row, column) == highlight_index

            css_class = "matrix-cell"
            if is_highlighted:
                css_class += f" highlight-{matrix_id.lower()}"

            add_element(
                group,
                "rect",
                x=x + column * CELL_SIZE,
                y=y + row * CELL_SIZE,
                width=CELL_SIZE,
                height=CELL_SIZE,
                class_=css_class,
            )

    add_element(
        group,
        "rect",
        x=x,
        y=y,
        width=columns * CELL_SIZE,
        height=rows * CELL_SIZE,
        class_="matrix-outline",
    )


def draw_horizontal_dimension(
    parent: ET.Element, x_start: float, x_end: float, y: float, label: str
) -> None:
    add_element(parent, "line", x1=x_start, y1=y, x2=x_end, y2=y, class_="dimension-line")
    add_element(parent, "line", x1=x_start, y1=y - 7, x2=x_start, y2=y + 7, class_="dimension-line")
    add_element(parent, "line", x1=x_end, y1=y - 7, x2=x_end, y2=y + 7, class_="dimension-line")
    add_text(parent, (x_start + x_end) / 2, y + 28, label, "dimension-label")


def draw_vertical_dimension(
    parent: ET.Element,
    x: float,
    y_start: float,
    y_end: float,
    label: str,
    *,
    label_side: str = "left",
) -> None:
    add_element(parent, "line", x1=x, y1=y_start, x2=x, y2=y_end, class_="dimension-line")
    add_element(parent, "line", x1=x - 7, y1=y_start, x2=x + 7, y2=y_start, class_="dimension-line")
    add_element(parent, "line", x1=x - 7, y1=y_end, x2=x + 7, y2=y_end, class_="dimension-line")
    center_y = (y_start + y_end) / 2
    label_x = x - 26 if label_side == "left" else x + 26
    add_text(
        parent,
        label_x,
        center_y,
        label,
        "dimension-label",
        transform=f"rotate(-90 {label_x} {center_y})",
    )


def add_tag(
    parent: ET.Element,
    *,
    x: float,
    y: float,
    width: float,
    label: str,
    kind: str,
) -> None:
    add_element(
        parent,
        "rect",
        x=x,
        y=y,
        width=width,
        height=54,
        rx=13,
        class_=f"tag tag-{kind}",
    )
    add_text(parent, x + width / 2, y + 27, label, f"tag-label tag-label-{kind}")


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
    title.text = "Computing one output element in matrix multiplication"
    description = add_element(root, "desc", id="figure-description")
    description.text = (
        "Matrix A times matrix B equals matrix D. Row m of A is highlighted in blue, "
        "column n of B in orange, and their output element D m n in purple."
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
        --grid: #64748b;
        --a-accent: #0072b2;
        --a-fill: #d9eef9;
        --a-text: #005f94;
        --b-accent: #e69f00;
        --b-fill: #fff0c2;
        --b-text: #8a5f00;
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
          --grid: #94a3b8;
          --a-fill: #123f59;
          --a-text: #67c7f4;
          --b-fill: #55400c;
          --b-text: #ffd166;
          --d-fill: #552f48;
          --d-text: #f0a8d0;
        }
      }

      text {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
          "Segoe UI", sans-serif;
      }

      .background { fill: var(--background); }
      .matrix-cell { fill: var(--cell); stroke: var(--grid); stroke-width: 1.5; }
      .matrix-outline { fill: none; stroke: var(--grid); stroke-width: 3; }
      .highlight-a { fill: var(--a-fill); stroke: var(--a-accent); stroke-width: 3; }
      .highlight-b { fill: var(--b-fill); stroke: var(--b-accent); stroke-width: 3; }
      .highlight-d { fill: var(--d-fill); stroke: var(--d-accent); stroke-width: 4; }
      .matrix-name { font-size: 44px; font-weight: 750; fill: var(--text); }
      .matrix-name-a { fill: var(--a-text); }
      .matrix-name-b { fill: var(--b-text); }
      .matrix-name-d { fill: var(--d-text); }
      .matrix-shape { font-size: 28px; font-weight: 550; fill: var(--muted-text); }
      .operator { font-size: 68px; font-weight: 400; fill: var(--text); }
      .dimension-line { stroke: var(--muted-text); stroke-width: 2; }
      .dimension-label { font-size: 27px; font-weight: 650; fill: var(--muted-text); }
      .index-label { font-size: 25px; font-weight: 800; }
      .index-a { fill: var(--a-text); }
      .index-b { fill: var(--b-text); }
      .index-d { fill: var(--d-text); }
      .tag { stroke-width: 2.5; }
      .tag-a { fill: var(--a-fill); stroke: var(--a-accent); }
      .tag-b { fill: var(--b-fill); stroke: var(--b-accent); }
      .tag-d { fill: var(--d-fill); stroke: var(--d-accent); }
      .tag-label { font-size: 25px; font-weight: 700; }
      .tag-label-a { fill: var(--a-text); }
      .tag-label-b { fill: var(--b-text); }
      .tag-label-d { fill: var(--d-text); }
      .tag-operator { font-size: 42px; font-weight: 650; fill: var(--text); }
    """

    add_element(root, "rect", x=0, y=0, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, rx=24, class_="background")

    a_x, a_y, a_rows, a_columns = 90, 158, 6, 8
    b_x, b_y, b_rows, b_columns = 650, 112, 8, 6
    d_x, d_y, d_rows, d_columns = 1150, 158, 6, 6
    selected_row = 3
    selected_column = 3

    a_center_x = a_x + a_columns * CELL_SIZE / 2
    b_center_x = b_x + b_columns * CELL_SIZE / 2
    d_center_x = d_x + d_columns * CELL_SIZE / 2

    add_text(root, a_center_x, 49, "A", "matrix-name matrix-name-a")
    add_text(root, a_center_x, 86, "(M × K)", "matrix-shape")
    add_text(root, b_center_x, 49, "B", "matrix-name matrix-name-b")
    add_text(root, b_center_x, 86, "(K × N)", "matrix-shape")
    add_text(root, d_center_x, 49, "D", "matrix-name matrix-name-d")
    add_text(root, d_center_x, 86, "(M × N)", "matrix-shape")

    draw_matrix(
        root,
        matrix_id="A",
        x=a_x,
        y=a_y,
        rows=a_rows,
        columns=a_columns,
        highlight_kind="row",
        highlight_index=selected_row,
    )
    draw_matrix(
        root,
        matrix_id="B",
        x=b_x,
        y=b_y,
        rows=b_rows,
        columns=b_columns,
        highlight_kind="column",
        highlight_index=selected_column,
    )
    draw_matrix(
        root,
        matrix_id="D",
        x=d_x,
        y=d_y,
        rows=d_rows,
        columns=d_columns,
        highlight_kind="cell",
        highlight_index=(selected_row, selected_column),
    )

    add_text(root, 550, 296, "×", "operator")
    add_text(root, 1038, 296, "=", "operator")

    a_selected_y = a_y + (selected_row + 0.5) * CELL_SIZE
    b_selected_x = b_x + (selected_column + 0.5) * CELL_SIZE
    d_selected_y = d_y + (selected_row + 0.5) * CELL_SIZE
    d_selected_x = d_x + (selected_column + 0.5) * CELL_SIZE
    add_text(root, a_x - 13, a_selected_y, "m", "index-label index-a", anchor="end")
    add_text(root, b_selected_x, b_y + CELL_SIZE / 2, "n", "index-label index-b")
    add_text(root, d_x - 13, d_selected_y, "m", "index-label index-d", anchor="end")
    add_text(root, d_selected_x, d_y - 15, "n", "index-label index-d")

    draw_horizontal_dimension(root, a_x, a_x + a_columns * CELL_SIZE, a_y + a_rows * CELL_SIZE + 20, "K")
    draw_vertical_dimension(root, a_x - 40, a_y, a_y + a_rows * CELL_SIZE, "M")
    draw_horizontal_dimension(root, b_x, b_x + b_columns * CELL_SIZE, b_y + b_rows * CELL_SIZE + 20, "N")
    draw_vertical_dimension(
        root,
        b_x + b_columns * CELL_SIZE + 24,
        b_y,
        b_y + b_rows * CELL_SIZE,
        "K",
        label_side="right",
    )
    draw_horizontal_dimension(root, d_x, d_x + d_columns * CELL_SIZE, d_y + d_rows * CELL_SIZE + 20, "N")
    draw_vertical_dimension(
        root,
        d_x + d_columns * CELL_SIZE + 24,
        d_y,
        d_y + d_rows * CELL_SIZE,
        "M",
        label_side="right",
    )

    tag_y = 578
    add_tag(root, x=150, y=tag_y, width=350, label="row m:  A[m, :]", kind="a")
    add_text(root, 555, tag_y + 27, "·", "tag-operator")
    add_tag(root, x=610, y=tag_y, width=370, label="column n:  B[:, n]", kind="b")
    add_text(root, 1035, tag_y + 27, "=", "tag-operator")
    add_tag(root, x=1090, y=tag_y, width=360, label="output:  D[m, n]", kind="d")

    return root


def write_svg(output_path: Path) -> None:
    root = build_svg()
    ET.indent(root, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"
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
