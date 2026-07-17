#!/usr/bin/env python3
"""Generate Figure 5 for the naive CUDA GEMM chapter."""

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
OUTPUT_NAME = "figure-5-global-memory-sectors.svg"

CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 700
DISPLAY_WIDTH = 800

PANEL_WIDTH = 490
PANEL_X_POSITIONS = (35, 555, 1075)
LANE_VALUES = ("0", "1", "…", "31")
LANE_CELL_WIDTH = 82
LANE_CELL_HEIGHT = 48
LANE_CELL_GAP = 18
SECTOR_WIDTH = 96
SECTOR_HEIGHT = 82
WORD_COUNT = 8


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


def add_arrow_marker(
    definitions: ET.Element,
    *,
    marker_id: str,
    css_class: str,
) -> None:
    marker = add_element(
        definitions,
        "marker",
        id=marker_id,
        viewBox="0 0 8 7",
        refX=7,
        refY=3.5,
        markerWidth=8,
        markerHeight=8,
        orient="auto-start-reverse",
    )
    add_element(
        marker,
        "path",
        d="M 0 0 L 8 3.5 L 0 7 Z",
        class_=css_class,
    )


def draw_panel_header(
    parent: ET.Element,
    *,
    x: float,
    label: str,
    kind: str,
) -> None:
    add_element(
        parent,
        "rect",
        x=x + 15,
        y=28,
        width=PANEL_WIDTH - 30,
        height=58,
        rx=16,
        class_=f"panel-header header-{kind}",
    )
    add_text(
        parent,
        x + PANEL_WIDTH / 2,
        57,
        label,
        f"panel-title text-{kind}",
    )


def lane_strip_start(panel_x: float) -> float:
    strip_width = (
        len(LANE_VALUES) * LANE_CELL_WIDTH
        + (len(LANE_VALUES) - 1) * LANE_CELL_GAP
    )
    return panel_x + (PANEL_WIDTH - strip_width) / 2


def lane_center(panel_x: float, lane_index: int) -> float:
    return (
        lane_strip_start(panel_x)
        + lane_index * (LANE_CELL_WIDTH + LANE_CELL_GAP)
        + LANE_CELL_WIDTH / 2
    )


def draw_lane_strip(parent: ET.Element, *, panel_x: float, kind: str) -> None:
    start_x = lane_strip_start(panel_x)
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        111,
        "one warp: 32 lanes",
        "lane-strip-label",
    )

    for index, lane in enumerate(LANE_VALUES):
        cell_x = start_x + index * (LANE_CELL_WIDTH + LANE_CELL_GAP)
        css_class = f"lane-cell lane-{kind}"
        if lane == "…":
            css_class += " omitted-lane"
        add_element(
            parent,
            "rect",
            x=cell_x,
            y=133,
            width=LANE_CELL_WIDTH,
            height=LANE_CELL_HEIGHT,
            rx=9,
            class_=css_class,
        )
        label = "…" if lane == "…" else f"lane {lane}"
        add_text(
            parent,
            cell_x + LANE_CELL_WIDTH / 2,
            157,
            label,
            f"lane-value text-{kind}",
        )


def draw_sector(
    parent: ET.Element,
    *,
    x: float,
    y: float,
    label: str,
    kind: str,
    selected_words: tuple[int, ...],
) -> None:
    add_element(
        parent,
        "rect",
        x=x,
        y=y,
        width=SECTOR_WIDTH,
        height=SECTOR_HEIGHT,
        rx=10,
        class_=f"sector sector-{kind}",
    )
    add_text(
        parent,
        x + SECTOR_WIDTH / 2,
        y + 12,
        label,
        f"sector-label text-{kind}",
    )

    word_width = SECTOR_WIDTH / WORD_COUNT
    word_y = y + 23
    word_height = 38
    for word in range(WORD_COUNT):
        css_class = "word-cell"
        if word in selected_words:
            css_class += f" selected-word selected-{kind}"
        add_element(
            parent,
            "rect",
            x=x + word * word_width,
            y=word_y,
            width=word_width,
            height=word_height,
            class_=css_class,
        )

    add_text(
        parent,
        x + SECTOR_WIDTH / 2,
        y + SECTOR_HEIGHT - 10,
        "8 × FP32",
        "sector-footnote",
    )


def draw_connector(
    parent: ET.Element,
    *,
    start_x: float,
    end_x: float,
    end_y: float,
    kind: str,
    dashed: bool = False,
) -> None:
    css_class = f"request-arrow arrow-{kind}"
    if dashed:
        css_class += " dashed-arrow"
    add_element(
        parent,
        "path",
        d=(
            f"M {start_x} 181 "
            f"C {start_x} 230, {end_x} 242, {end_x} {end_y}"
        ),
        class_=css_class,
        marker_end=f"url(#arrow-{kind})",
    )


def draw_result_tag(
    parent: ET.Element,
    *,
    panel_x: float,
    count: str,
    detail: str,
    kind: str,
) -> None:
    add_element(
        parent,
        "rect",
        x=panel_x + 48,
        y=495,
        width=PANEL_WIDTH - 96,
        height=64,
        rx=15,
        class_=f"result-tag result-{kind}",
    )
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        521,
        count,
        f"result-count text-{kind}",
    )
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        546,
        detail,
        "result-detail",
    )


def draw_consecutive_panel(parent: ET.Element, *, panel_x: float) -> None:
    kind = "consecutive"
    draw_panel_header(
        parent,
        x=panel_x,
        label="Consecutive addresses",
        kind=kind,
    )
    draw_lane_strip(parent, panel_x=panel_x, kind=kind)
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        217,
        "32 consecutive FP32 addresses",
        "mapping-label",
    )

    sector_y = 330
    start_x = panel_x + 43
    gap = 8
    sector_x_positions = tuple(
        start_x + index * (SECTOR_WIDTH + gap) for index in range(4)
    )
    target_positions = (
        sector_x_positions[0] + 6,
        sector_x_positions[0] + 18,
        sector_x_positions[2] + 6,
        sector_x_positions[3] + 90,
    )

    for index, (start, end) in enumerate(
        zip(
            (lane_center(panel_x, lane) for lane in range(4)),
            target_positions,
            strict=True,
        )
    ):
        draw_connector(
            parent,
            start_x=start,
            end_x=end,
            end_y=sector_y - 9,
            kind=kind,
            dashed=index == 2,
        )

    for sector, sector_x in enumerate(sector_x_positions):
        draw_sector(
            parent,
            x=sector_x,
            y=sector_y,
            label=f"sector {sector}",
            kind=kind,
            selected_words=tuple(range(WORD_COUNT)),
        )

    draw_result_tag(
        parent,
        panel_x=panel_x,
        count="4 × 32-B sectors",
        detail="aligned 128-B span",
        kind=kind,
    )
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        591,
        "aligned FP32 case",
        "panel-note",
    )


def draw_same_address_panel(parent: ET.Element, *, panel_x: float) -> None:
    kind = "broadcast"
    draw_panel_header(
        parent,
        x=panel_x,
        label="Same address / broadcast",
        kind=kind,
    )
    draw_lane_strip(parent, panel_x=panel_x, kind=kind)
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        217,
        "all lanes request one FP32 address",
        "mapping-label",
    )

    sector_y = 330
    sector_x = panel_x + (PANEL_WIDTH - SECTOR_WIDTH) / 2
    word_width = SECTOR_WIDTH / WORD_COUNT
    selected_word = 4
    target_x = sector_x + (selected_word + 0.5) * word_width

    for index in range(4):
        draw_connector(
            parent,
            start_x=lane_center(panel_x, index),
            end_x=target_x,
            end_y=sector_y - 9,
            kind=kind,
            dashed=index == 2,
        )

    draw_sector(
        parent,
        x=sector_x,
        y=sector_y,
        label="one sector",
        kind=kind,
        selected_words=(selected_word,),
    )
    add_text(
        parent,
        target_x,
        sector_y + 42,
        "×32",
        "broadcast-count",
    )

    draw_result_tag(
        parent,
        panel_x=panel_x,
        count="1 × 32-B sector",
        detail="one requested FP32 value",
        kind=kind,
    )
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        591,
        "same value returned to all lanes",
        "panel-note",
    )


def draw_strided_panel(parent: ET.Element, *, panel_x: float) -> None:
    kind = "strided"
    draw_panel_header(
        parent,
        x=panel_x,
        label="Large stride",
        kind=kind,
    )
    draw_lane_strip(parent, panel_x=panel_x, kind=kind)
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        217,
        "each address lies in a different sector",
        "mapping-label",
    )

    sector_y = 330
    sector_x_positions = (
        panel_x + 38,
        panel_x + 164,
        panel_x + 356,
    )
    sector_labels = ("sector 0", "sector 1", "sector 31")
    selected_words = (1, 5, 2)
    target_positions = tuple(
        sector_x
        + (selected_word + 0.5) * (SECTOR_WIDTH / WORD_COUNT)
        for sector_x, selected_word in zip(
            sector_x_positions,
            selected_words,
            strict=True,
        )
    )

    representative_targets: tuple[float, ...] = (
        target_positions[0],
        target_positions[1],
        panel_x + 310,
        target_positions[2],
    )
    for index, target_x in enumerate(representative_targets):
        draw_connector(
            parent,
            start_x=lane_center(panel_x, index),
            end_x=target_x,
            end_y=sector_y - 9,
            kind=kind,
            dashed=index == 2,
        )

    for sector_x, label, selected_word in zip(
        sector_x_positions,
        sector_labels,
        selected_words,
        strict=True,
    ):
        draw_sector(
            parent,
            x=sector_x,
            y=sector_y,
            label=label,
            kind=kind,
            selected_words=(selected_word,),
        )

    add_text(
        parent,
        panel_x + 310,
        sector_y + SECTOR_HEIGHT / 2,
        "…",
        "sector-ellipsis",
    )

    draw_result_tag(
        parent,
        panel_x=panel_x,
        count="32 × 32-B sectors",
        detail="one requested FP32 value per sector",
        kind=kind,
    )
    add_text(
        parent,
        panel_x + PANEL_WIDTH / 2,
        591,
        "sufficiently large stride",
        "panel-note",
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
    title.text = "Global-memory sectors requested by one warp"
    description = add_element(root, "desc", id="figure-description")
    description.text = (
        "Three panels compare the 32 FP32 addresses from one warp memory "
        "instruction. Thirty-two aligned consecutive values occupy four "
        "32-byte sectors. Same-address requests occupy one sector and return "
        "one value to all lanes. A sufficiently large stride can place one "
        "requested value in each of 32 different sectors."
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
        --consecutive-accent: #0072b2;
        --consecutive-fill: #d9eef9;
        --consecutive-text: #005f94;
        --broadcast-accent: #e69f00;
        --broadcast-fill: #fff0c2;
        --broadcast-text: #8a5f00;
        --strided-accent: #cc79a7;
        --strided-fill: #f7dceb;
        --strided-text: #9d4778;
      }

      @media (prefers-color-scheme: dark) {
        :root {
          --background: #111827;
          --text: #f8fafc;
          --muted-text: #cbd5e1;
          --cell: #1f2937;
          --grid: #94a3b8;
          --consecutive-fill: #123f59;
          --consecutive-text: #67c7f4;
          --broadcast-fill: #55400c;
          --broadcast-text: #ffd166;
          --strided-fill: #552f48;
          --strided-text: #f0a8d0;
        }
      }

      text {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
          "Segoe UI", sans-serif;
      }

      .background { fill: var(--background); }
      .panel-separator {
        stroke: var(--grid);
        stroke-width: 1.5;
        stroke-dasharray: 5 8;
        opacity: 0.75;
      }
      .panel-header { stroke-width: 2.5; }
      .header-consecutive {
        fill: var(--consecutive-fill);
        stroke: var(--consecutive-accent);
      }
      .header-broadcast {
        fill: var(--broadcast-fill);
        stroke: var(--broadcast-accent);
      }
      .header-strided {
        fill: var(--strided-fill);
        stroke: var(--strided-accent);
      }
      .panel-title { font-size: 29px; font-weight: 760; }
      .text-consecutive { fill: var(--consecutive-text); }
      .text-broadcast { fill: var(--broadcast-text); }
      .text-strided { fill: var(--strided-text); }
      .lane-strip-label {
        font-size: 21px;
        font-weight: 650;
        fill: var(--muted-text);
      }
      .lane-cell { stroke-width: 2; }
      .lane-consecutive {
        fill: var(--consecutive-fill);
        stroke: var(--consecutive-accent);
      }
      .lane-broadcast {
        fill: var(--broadcast-fill);
        stroke: var(--broadcast-accent);
      }
      .lane-strided {
        fill: var(--strided-fill);
        stroke: var(--strided-accent);
      }
      .omitted-lane { stroke-dasharray: 5 5; }
      .lane-value {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 19px;
        font-weight: 750;
      }
      .mapping-label {
        font-size: 21px;
        font-weight: 650;
        fill: var(--text);
      }
      .request-arrow {
        fill: none;
        stroke-width: 2.5;
        stroke-linecap: round;
        opacity: 0.88;
      }
      .arrow-consecutive { stroke: var(--consecutive-accent); }
      .arrow-broadcast { stroke: var(--broadcast-accent); }
      .arrow-strided { stroke: var(--strided-accent); }
      .dashed-arrow { stroke-dasharray: 6 6; }
      .marker-consecutive { fill: var(--consecutive-accent); }
      .marker-broadcast { fill: var(--broadcast-accent); }
      .marker-strided { fill: var(--strided-accent); }
      .sector { stroke-width: 3; }
      .sector-consecutive {
        fill: var(--consecutive-fill);
        stroke: var(--consecutive-accent);
      }
      .sector-broadcast {
        fill: var(--broadcast-fill);
        stroke: var(--broadcast-accent);
      }
      .sector-strided {
        fill: var(--strided-fill);
        stroke: var(--strided-accent);
      }
      .sector-label { font-size: 17px; font-weight: 750; }
      .word-cell {
        fill: var(--cell);
        stroke: var(--grid);
        stroke-width: 1;
      }
      .selected-consecutive { fill: var(--consecutive-fill); }
      .selected-broadcast { fill: var(--broadcast-fill); }
      .selected-strided { fill: var(--strided-fill); }
      .selected-word { stroke-width: 2; }
      .selected-consecutive { stroke: var(--consecutive-accent); }
      .selected-broadcast { stroke: var(--broadcast-accent); }
      .selected-strided { stroke: var(--strided-accent); }
      .sector-footnote {
        font-size: 15px;
        font-weight: 650;
        fill: var(--muted-text);
      }
      .broadcast-count {
        font-size: 16px;
        font-weight: 850;
        fill: var(--broadcast-text);
      }
      .sector-ellipsis {
        font-size: 38px;
        font-weight: 760;
        fill: var(--muted-text);
      }
      .result-tag { stroke-width: 2.5; }
      .result-consecutive {
        fill: var(--consecutive-fill);
        stroke: var(--consecutive-accent);
      }
      .result-broadcast {
        fill: var(--broadcast-fill);
        stroke: var(--broadcast-accent);
      }
      .result-strided {
        fill: var(--strided-fill);
        stroke: var(--strided-accent);
      }
      .result-count { font-size: 24px; font-weight: 800; }
      .result-detail {
        font-size: 18px;
        font-weight: 600;
        fill: var(--muted-text);
      }
      .panel-note {
        font-size: 19px;
        font-weight: 650;
        fill: var(--muted-text);
      }
      .bottom-rule {
        stroke: var(--grid);
        stroke-width: 2;
        opacity: 0.7;
      }
      .bottom-label {
        font-size: 21px;
        font-weight: 700;
        fill: var(--text);
      }
    """

    add_arrow_marker(
        definitions,
        marker_id="arrow-consecutive",
        css_class="marker-consecutive",
    )
    add_arrow_marker(
        definitions,
        marker_id="arrow-broadcast",
        css_class="marker-broadcast",
    )
    add_arrow_marker(
        definitions,
        marker_id="arrow-strided",
        css_class="marker-strided",
    )

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
    for separator_x in (540, 1060):
        add_element(
            root,
            "line",
            x1=separator_x,
            y1=105,
            x2=separator_x,
            y2=606,
            class_="panel-separator",
        )

    draw_consecutive_panel(root, panel_x=PANEL_X_POSITIONS[0])
    draw_same_address_panel(root, panel_x=PANEL_X_POSITIONS[1])
    draw_strided_panel(root, panel_x=PANEL_X_POSITIONS[2])

    add_element(
        root,
        "line",
        x1=95,
        y1=625,
        x2=1505,
        y2=625,
        class_="bottom-rule",
    )
    add_text(
        root,
        CANVAS_WIDTH / 2,
        658,
        "Addresses in one sector share a 32-B transaction; different sectors need separate transactions.",
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
