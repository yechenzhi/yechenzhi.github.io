#!/usr/bin/env python3
"""Generate Figure 4 for the naive CUDA GEMM chapter."""

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
OUTPUT_NAME = "figure-4-kernel13-warp-accesses.svg"

CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 700
DISPLAY_WIDTH = 800
CELL_SIZE = 60
WARP_CELL_WIDTH = 140
DISPLAYED_THREADS = ("0", "1", "⋮", "31")
OMITTED_ROW = 2


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
) -> ET.Element:
    node = add_element(
        parent,
        "text",
        x=x,
        y=y,
        class_=css_class,
        text_anchor=anchor,
        dominant_baseline="middle",
    )
    node.text = value
    return node


def draw_phase_header(
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
        height=54,
        rx=15,
        class_="phase-header",
    )
    add_text(parent, x + width / 2, y + 27, label, "phase-label")


def draw_warp_strip(parent: ET.Element, *, x: float, y: float) -> None:
    group = add_element(parent, "g", id="selected-warp")

    for row, thread in enumerate(DISPLAYED_THREADS):
        cell_y = y + row * CELL_SIZE
        css_class = "warp-cell"
        if row == OMITTED_ROW:
            css_class += " omitted-warp-cell"
        add_element(
            group,
            "rect",
            x=x,
            y=cell_y,
            width=WARP_CELL_WIDTH,
            height=CELL_SIZE,
            class_=css_class,
        )
        label = "⋮" if row == OMITTED_ROW else f"x = {thread}"
        label_class = "warp-ellipsis" if row == OMITTED_ROW else "warp-thread"
        add_text(
            group,
            x + WARP_CELL_WIDTH / 2,
            cell_y + CELL_SIZE / 2,
            label,
            label_class,
        )

    add_element(
        group,
        "rect",
        x=x,
        y=y,
        width=WARP_CELL_WIDTH,
        height=len(DISPLAYED_THREADS) * CELL_SIZE,
        class_="warp-outline",
    )


def draw_compressed_matrix(
    parent: ET.Element,
    *,
    matrix_id: str,
    x: float,
    y: float,
    columns: int,
    selected_column: int,
    highlight_kind: str,
    selected_index_label: str,
) -> None:
    group = add_element(parent, "g", id=f"matrix-{matrix_id.lower()}")
    rows = len(DISPLAYED_THREADS)

    for row in range(rows):
        for column in range(columns):
            css_class = "matrix-cell"
            if row == OMITTED_ROW:
                css_class += " omitted-matrix-cell"
            if column == selected_column:
                css_class += f" highlight-{highlight_kind}"

            cell_x = x + column * CELL_SIZE
            cell_y = y + row * CELL_SIZE
            add_element(
                group,
                "rect",
                x=cell_x,
                y=cell_y,
                width=CELL_SIZE,
                height=CELL_SIZE,
                class_=css_class,
            )

            if row == OMITTED_ROW:
                text_class = "matrix-ellipsis"
                if column == selected_column:
                    text_class += f" text-{highlight_kind}"
                add_text(
                    group,
                    cell_x + CELL_SIZE / 2,
                    cell_y + CELL_SIZE / 2,
                    "⋮",
                    text_class,
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
    add_text(
        group,
        x + (selected_column + 0.5) * CELL_SIZE,
        y - 20,
        selected_index_label,
        f"index-label text-{highlight_kind}",
    )


def draw_b_matrix(
    parent: ET.Element,
    *,
    x: float,
    y: float,
    rows: int,
    columns: int,
    selected_row: int,
    selected_column: int,
) -> None:
    group = add_element(parent, "g", id="matrix-b")

    for row in range(rows):
        for column in range(columns):
            css_class = "matrix-cell"
            if row == selected_row and column == selected_column:
                css_class += " highlight-b"

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

    selected_x = x + (selected_column + 0.5) * CELL_SIZE
    selected_y = y + (selected_row + 0.5) * CELL_SIZE
    add_text(group, selected_x, y - 20, "n", "index-label text-b")
    add_text(group, x - 18, selected_y, "k", "index-label text-b", anchor="end")
    add_text(group, selected_x, selected_y, "×32", "same-element-label")


def draw_access_tag(
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
        height=56,
        rx=14,
        class_=f"access-tag tag-{kind}",
    )
    add_text(parent, x + width / 2, y + 28, label, f"tag-label text-{kind}")


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
    title.text = "Elements accessed by one warp in Kernel 13"
    description = add_element(root, "desc", id="figure-description")
    description.text = (
        "The selected warp is represented by threads zero, one, omitted threads, "
        "and thread 31. During one fixed k iteration, the threads load 32 elements "
        "from one column of A while all 32 load the same element B k n. After all K "
        "iterations, they store 32 elements in one column of D."
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
      .phase-header { fill: var(--cell); stroke: var(--grid); stroke-width: 2; }
      .phase-label { font-size: 26px; font-weight: 750; fill: var(--text); }
      .matrix-name { font-size: 38px; font-weight: 760; }
      .matrix-shape { font-size: 24px; font-weight: 550; fill: var(--muted-text); }
      .text-a { fill: var(--a-text); }
      .text-b { fill: var(--b-text); }
      .text-d { fill: var(--d-text); }
      .matrix-cell { fill: var(--cell); stroke: var(--grid); stroke-width: 1.5; }
      .omitted-matrix-cell { stroke-dasharray: 5 5; }
      .matrix-outline { fill: none; stroke: var(--grid); stroke-width: 3; }
      .highlight-a { fill: var(--a-fill); stroke: var(--a-accent); stroke-width: 3; }
      .highlight-b { fill: var(--b-fill); stroke: var(--b-accent); stroke-width: 3; }
      .highlight-d { fill: var(--d-fill); stroke: var(--d-accent); stroke-width: 3; }
      .matrix-ellipsis { font-size: 31px; font-weight: 750; fill: var(--muted-text); }
      .index-label { font-size: 25px; font-weight: 800; }
      .warp-cell { fill: var(--d-fill); stroke: var(--d-accent); stroke-width: 2; }
      .omitted-warp-cell { stroke-dasharray: 5 5; }
      .warp-outline { fill: none; stroke: var(--d-accent); stroke-width: 4; }
      .warp-thread {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 22px;
        font-weight: 750;
        fill: var(--d-text);
      }
      .warp-ellipsis { font-size: 32px; font-weight: 750; fill: var(--d-text); }
      .same-element-label { font-size: 24px; font-weight: 850; fill: var(--b-text); }
      .operator { font-size: 30px; font-weight: 720; fill: var(--muted-text); }
      .flow-arrow { font-size: 52px; font-weight: 500; fill: var(--muted-text); }
      .access-tag { stroke-width: 2.5; }
      .tag-a { fill: var(--a-fill); stroke: var(--a-accent); }
      .tag-b { fill: var(--b-fill); stroke: var(--b-accent); }
      .tag-d { fill: var(--d-fill); stroke: var(--d-accent); }
      .tag-label { font-size: 23px; font-weight: 700; }
      .warp-note { font-size: 22px; font-weight: 680; fill: var(--d-text); }
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

    warp_x, warp_y = 42, 220
    a_x, a_y = 250, 220
    b_x, b_y = 690, 205
    d_x, d_y = 1170, 220
    compressed_rows = len(DISPLAYED_THREADS)
    compressed_columns = 5
    selected_column = 2

    draw_phase_header(
        root,
        x=205,
        y=34,
        width=805,
        label="one fixed k iteration",
    )
    draw_phase_header(
        root,
        x=1135,
        y=34,
        width=370,
        label="after all K iterations",
    )

    add_text(root, warp_x + WARP_CELL_WIDTH / 2, 132, "one warp", "matrix-name text-d")
    add_text(root, warp_x + WARP_CELL_WIDTH / 2, 170, "fixed threadIdx.y", "matrix-shape")
    add_text(root, a_x + compressed_columns * CELL_SIZE / 2, 132, "load A", "matrix-name text-a")
    add_text(root, a_x + compressed_columns * CELL_SIZE / 2, 170, "(M × K)", "matrix-shape")
    add_text(root, b_x + 2.5 * CELL_SIZE, 132, "load B", "matrix-name text-b")
    add_text(root, b_x + 2.5 * CELL_SIZE, 170, "(K × N)", "matrix-shape")
    add_text(root, d_x + compressed_columns * CELL_SIZE / 2, 132, "store D", "matrix-name text-d")
    add_text(root, d_x + compressed_columns * CELL_SIZE / 2, 170, "(M × N)", "matrix-shape")

    draw_warp_strip(root, x=warp_x, y=warp_y)
    draw_compressed_matrix(
        root,
        matrix_id="A",
        x=a_x,
        y=a_y,
        columns=compressed_columns,
        selected_column=selected_column,
        highlight_kind="a",
        selected_index_label="k",
    )
    draw_b_matrix(
        root,
        x=b_x,
        y=b_y,
        rows=5,
        columns=5,
        selected_row=2,
        selected_column=2,
    )
    draw_compressed_matrix(
        root,
        matrix_id="D",
        x=d_x,
        y=d_y,
        columns=compressed_columns,
        selected_column=selected_column,
        highlight_kind="d",
        selected_index_label="n",
    )

    add_text(root, 215, 340, "→", "flow-arrow")
    add_text(root, 620, 340, "and", "operator")
    add_text(root, 1080, 340, "→", "flow-arrow")

    add_text(
        root,
        warp_x + WARP_CELL_WIDTH / 2,
        510,
        "x = 0, 1, …, 31",
        "warp-note",
    )
    draw_access_tag(
        root,
        x=220,
        y=540,
        width=360,
        label="32 A elements at fixed k",
        kind="a",
    )
    draw_access_tag(
        root,
        x=640,
        y=540,
        width=400,
        label="all 32 threads load  B[k, n]",
        kind="b",
    )
    draw_access_tag(
        root,
        x=1120,
        y=540,
        width=400,
        label="32 D elements at fixed n",
        kind="d",
    )

    add_text(
        root,
        CANVAS_WIDTH / 2,
        635,
        "ℓ = 0, 1, …, 31",
        "matrix-shape",
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
