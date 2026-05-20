import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


INPUT_DIR = Path("werewolf-results/analysis")
OUTPUT_SVG = INPUT_DIR / "three_metric_comparison_tables.svg"
OUTPUT_PNG = INPUT_DIR / "three_metric_comparison_tables.png"
MODES = ["baseline", "targeted", "prior"]
MODE_LABELS = {
    "baseline": "Baseline",
    "targeted": "Targeted",
    "prior": "Prior",
}
MODEL_LABELS = {
    "gemini": "Gemini",
    "gpt5_nano": "GPT-5 nano",
    "qwen": "Qwen",
}
TABLE_SPECS = [
    ("Win rate", "win_rate_matrix.csv", "percent"),
    ("Avg votes on Werewolves", "avg_votes_on_werewolves_matrix.csv", "float"),
    ("Werewolf vote share", "avg_werewolf_vote_share_matrix.csv", "percent"),
]

BG = "#ffffff"
TEXT = "#111827"
MUTED = "#4b5563"
GRID = "#d1d5db"
HEADER_BG = "#eef2ff"
SUBHEADER_BG = "#f8fafc"
TITLE_BG = "#e0e7ff"
MODE_COLORS = {
    "baseline": "#cbd5e1",
    "targeted": "#93c5fd",
    "prior": "#c4b5fd",
}


def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def read_metric_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def format_value(value, style):
    if value in {None, "", "n/a"}:
        return "n/a"
    number = float(value)
    if style == "percent":
        return f"{number * 100:.1f}%"
    return f"{number:.3f}".rstrip("0").rstrip(".")


def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_rect(x, y, width, height, fill, stroke=GRID, stroke_width=1, rx=0):
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx}"/>'
    )


def svg_text(x, y, text, size=20, fill=TEXT, anchor="start", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{escape_xml(text)}</text>'
    )


def build_tables():
    tables = []
    for title, filename, style in TABLE_SPECS:
        rows = read_metric_csv(INPUT_DIR / filename)
        cleaned_rows = []
        for row in rows:
            cleaned_rows.append(
                {
                    "model": row["model_family"],
                    "values": {mode: row.get(mode) for mode in MODES},
                }
            )
        tables.append((title, style, cleaned_rows))
    return tables


def draw_png(tables):
    width = 1600
    margin_x = 70
    margin_y = 60
    title_h = 86
    section_gap = 34
    table_title_h = 48
    header_h = 48
    row_h = 54
    label_w = 250
    col_w = 320
    total_table_w = label_w + len(MODES) * col_w
    total_height = (
        margin_y * 2
        + title_h
        + len(tables) * (table_title_h + header_h + 3 * row_h)
        + (len(tables) - 1) * section_gap
    )

    image = Image.new("RGB", (width, total_height), BG)
    draw = ImageDraw.Draw(image)

    font_title = load_font(34, bold=True)
    font_subtitle = load_font(18, bold=False)
    font_table_title = load_font(23, bold=True)
    font_header = load_font(19, bold=True)
    font_body = load_font(20, bold=False)

    y = margin_y
    draw.rounded_rectangle(
        (margin_x, y, margin_x + total_table_w, y + title_h),
        radius=10,
        fill=TITLE_BG,
        outline=GRID,
        width=1,
    )
    draw.text((margin_x + 24, y + 18), "Werewolf Results Comparison", fill=TEXT, font=font_title)
    draw.text(
        (margin_x + 24, y + 52),
        "Three persuasion modes across Gemini, GPT-5 nano, and Qwen",
        fill=MUTED,
        font=font_subtitle,
    )
    y += title_h + 24

    for title, style, rows in tables:
        draw.rounded_rectangle(
            (margin_x, y, margin_x + total_table_w, y + table_title_h),
            radius=8,
            fill=HEADER_BG,
            outline=GRID,
            width=1,
        )
        draw.text((margin_x + 18, y + 13), title, fill=TEXT, font=font_table_title)
        y += table_title_h

        draw.rectangle((margin_x, y, margin_x + label_w, y + header_h), fill=SUBHEADER_BG, outline=GRID, width=1)
        draw.text((margin_x + 18, y + 13), "Model", fill=TEXT, font=font_header)
        x = margin_x + label_w
        for mode in MODES:
            draw.rectangle((x, y, x + col_w, y + header_h), fill=MODE_COLORS[mode], outline=GRID, width=1)
            bbox = draw.textbbox((0, 0), MODE_LABELS[mode], font=font_header)
            text_w = bbox[2] - bbox[0]
            draw.text((x + (col_w - text_w) / 2, y + 13), MODE_LABELS[mode], fill=TEXT, font=font_header)
            x += col_w
        y += header_h

        for row in rows:
            draw.rectangle((margin_x, y, margin_x + label_w, y + row_h), fill=BG, outline=GRID, width=1)
            draw.text((margin_x + 18, y + 15), MODEL_LABELS.get(row["model"], row["model"]), fill=TEXT, font=font_body)
            x = margin_x + label_w
            for mode in MODES:
                draw.rectangle((x, y, x + col_w, y + row_h), fill=BG, outline=GRID, width=1)
                value_text = format_value(row["values"][mode], style)
                bbox = draw.textbbox((0, 0), value_text, font=font_body)
                text_w = bbox[2] - bbox[0]
                draw.text((x + (col_w - text_w) / 2, y + 15), value_text, fill=TEXT, font=font_body)
                x += col_w
            y += row_h

        y += section_gap

    image.save(OUTPUT_PNG, dpi=(300, 300))


def draw_svg(tables):
    width = 1600
    margin_x = 70
    margin_y = 60
    title_h = 86
    section_gap = 34
    table_title_h = 48
    header_h = 48
    row_h = 54
    label_w = 250
    col_w = 320
    total_table_w = label_w + len(MODES) * col_w
    total_height = (
        margin_y * 2
        + title_h
        + len(tables) * (table_title_h + header_h + 3 * row_h)
        + (len(tables) - 1) * section_gap
    )

    elements = [svg_rect(0, 0, width, total_height, BG, stroke=BG)]
    y = margin_y
    elements.append(svg_rect(margin_x, y, total_table_w, title_h, TITLE_BG, rx=10))
    elements.append(svg_text(margin_x + 24, y + 38, "Werewolf Results Comparison", size=34, weight="bold"))
    elements.append(
        svg_text(
            margin_x + 24,
            y + 66,
            "Three persuasion modes across Gemini, GPT-5 nano, and Qwen",
            size=18,
            fill=MUTED,
        )
    )
    y += title_h + 24

    for title, style, rows in tables:
        elements.append(svg_rect(margin_x, y, total_table_w, table_title_h, HEADER_BG, rx=8))
        elements.append(svg_text(margin_x + 18, y + 30, title, size=23, weight="bold"))
        y += table_title_h

        elements.append(svg_rect(margin_x, y, label_w, header_h, SUBHEADER_BG))
        elements.append(svg_text(margin_x + 18, y + 30, "Model", size=19, weight="bold"))
        x = margin_x + label_w
        for mode in MODES:
            elements.append(svg_rect(x, y, col_w, header_h, MODE_COLORS[mode]))
            elements.append(svg_text(x + col_w / 2, y + 30, MODE_LABELS[mode], size=19, anchor="middle", weight="bold"))
            x += col_w
        y += header_h

        for row in rows:
            elements.append(svg_rect(margin_x, y, label_w, row_h, BG))
            elements.append(svg_text(margin_x + 18, y + 33, MODEL_LABELS.get(row["model"], row["model"]), size=20))
            x = margin_x + label_w
            for mode in MODES:
                value_text = format_value(row["values"][mode], style)
                elements.append(svg_rect(x, y, col_w, row_h, BG))
                elements.append(svg_text(x + col_w / 2, y + 33, value_text, size=20, anchor="middle"))
                x += col_w
            y += row_h
        y += section_gap

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}">'
    ]
    svg.extend(elements)
    svg.append("</svg>")
    OUTPUT_SVG.write_text("".join(svg), encoding="utf-8")


def main():
    tables = build_tables()
    draw_png(tables)
    draw_svg(tables)
    print(f"Wrote {OUTPUT_PNG}")
    print(f"Wrote {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
