import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont


DEFAULT_RESULTS_DIR = Path("werewolf-results")

MODELS = ["gpt5_nano", "gemini", "qwen"]
MODES = ["normal", "targeted", "prior"]

METRIC_SPECS = [
    {
        "key": "werewolf_win_rate",
        "filename": "win_rate_normal_targeted_prior.png",
        "title": "Werewolf Win Rate",
        "subtitle": "Normal vs targeted vs prior across models",
        "format": "percent4",
    },
    {
        "key": "avg_votes_on_werewolves",
        "filename": "avg_votes_on_werewolves_normal_targeted_prior.png",
        "title": "Average Votes on Werewolves",
        "subtitle": "Lower is better for the Werewolf team",
        "format": "float4",
    },
    {
        "key": "avg_werewolf_vote_share",
        "filename": "avg_werewolf_vote_share_normal_targeted_prior.png",
        "title": "Average Werewolf Vote Share",
        "subtitle": "Share of all votes that landed on Werewolves",
        "format": "percent4",
    },
]

BG = "#ffffff"
TITLE_BG = "#e0e7ff"
ROW_LABEL_BG = "#f8fafc"
GRID = "#d1d5db"
TEXT = "#111827"
MUTED = "#4b5563"
MODE_COLORS = {
    "normal": "#cbd5e1",
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


FONT_TITLE = load_font(32, bold=True)
FONT_SUBTITLE = load_font(18, bold=False)
FONT_HEADER = load_font(22, bold=True)
FONT_BODY = load_font(22, bold=False)


def infer_model_family(summary):
    model_mode = str(summary.get("model_mode") or "").lower()
    model = str(summary.get("model") or "").lower()
    combined = f"{model_mode} {model}"
    if "qwen" in combined:
        return "qwen"
    if "gemini" in combined:
        return "gemini"
    if "gpt" in combined or "openai" in combined:
        return "gpt5_nano"
    if "grok" in combined:
        return "grok"
    return "unknown"


def infer_mode(summary):
    if summary.get("leader_prior_personality_werewolf_persuasion"):
        return "prior"
    if summary.get("targeted_werewolf_persuasion"):
        return "targeted"
    return "normal"


def paired_games_path(summary_path):
    stem = summary_path.stem
    parts = stem.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        candidate = summary_path.with_name(f"{parts[0]}-games-{parts[1]}.json")
        if candidate.exists():
            return candidate
    candidate = summary_path.with_name(f"{stem}-games.json")
    if candidate.exists():
        return candidate
    return None


def load_summary_runs(results_dir):
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if "-games" in path.stem or path.parent.name == "analysis":
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict) or "summary" not in data or "games" not in data:
            continue

        summary = data["summary"]
        model_family = infer_model_family(summary)
        mode = infer_mode(summary)
        if model_family not in MODELS or mode not in MODES:
            continue

        completed_games = [game for game in data.get("games", []) if game.get("status") == "completed"]
        ww_vote_totals = []
        ww_vote_shares = []
        for game in completed_games:
            votes = game.get("votes") or {}
            werewolf_names = {
                player.get("name")
                for player in game.get("players", [])
                if player.get("card") == "Werewolf"
            }
            ww_votes = sum(votes.get(name, 0) for name in werewolf_names)
            total_votes = sum(votes.values())
            ww_vote_totals.append(ww_votes)
            ww_vote_shares.append((ww_votes / total_votes) if total_votes else 0.0)

        rows.append(
            {
                "path": path.name,
                "model_family": model_family,
                "mode": mode,
                "werewolf_win_rate": float(summary.get("werewolf_win_rate", 0.0)),
                "avg_votes_on_werewolves": mean(ww_vote_totals) if ww_vote_totals else 0.0,
                "avg_werewolf_vote_share": mean(ww_vote_shares) if ww_vote_shares else 0.0,
                "games_file": str(paired_games_path(path) or ""),
            }
        )
    return rows


def aggregate_metric_tables(rows):
    metric_tables = {}
    for metric in [spec["key"] for spec in METRIC_SPECS]:
        table_rows = []
        for model in MODELS:
            aggregated = []
            for mode in MODES:
                values = [
                    row[metric]
                    for row in rows
                    if row["model_family"] == model and row["mode"] == mode
                ]
                aggregated.append(round(mean(values), 4) if values else 0.0)
            table_rows.append((model, *aggregated))
        metric_tables[metric] = table_rows
    return metric_tables


def format_value(value, value_format):
    if value_format == "percent4":
        return f"{value * 100:.1f}%"
    return f"{value:.4f}"


def draw_table_image(title, subtitle, value_format, rows):
    width = 1200
    height = 420
    margin = 46
    title_h = 92
    header_h = 60
    row_h = 72
    label_w = 250
    col_w = 286

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (margin, margin, width - margin, margin + title_h),
        radius=16,
        fill=TITLE_BG,
        outline=GRID,
        width=1,
    )
    draw.text((margin + 24, margin + 18), title, fill=TEXT, font=FONT_TITLE)
    draw.text((margin + 24, margin + 56), subtitle, fill=MUTED, font=FONT_SUBTITLE)

    top = margin + title_h + 28
    left = margin

    draw.rectangle((left, top, left + label_w, top + header_h), fill=ROW_LABEL_BG, outline=GRID, width=1)
    draw.text((left + 20, top + 16), "Model", fill=TEXT, font=FONT_HEADER)

    x = left + label_w
    for mode in MODES:
        draw.rectangle((x, top, x + col_w, top + header_h), fill=MODE_COLORS[mode], outline=GRID, width=1)
        label = mode.title()
        bbox = draw.textbbox((0, 0), label, font=FONT_HEADER)
        text_w = bbox[2] - bbox[0]
        draw.text((x + (col_w - text_w) / 2, top + 16), label, fill=TEXT, font=FONT_HEADER)
        x += col_w

    y = top + header_h
    for model, normal, targeted, prior in rows:
        draw.rectangle((left, y, left + label_w, y + row_h), fill=ROW_LABEL_BG, outline=GRID, width=1)
        draw.text((left + 20, y + 21), model, fill=TEXT, font=FONT_BODY)

        x = left + label_w
        for value in [normal, targeted, prior]:
            draw.rectangle((x, y, x + col_w, y + row_h), fill=BG, outline=GRID, width=1)
            value_text = format_value(value, value_format)
            bbox = draw.textbbox((0, 0), value_text, font=FONT_BODY)
            text_w = bbox[2] - bbox[0]
            draw.text((x + (col_w - text_w) / 2, y + 21), value_text, fill=TEXT, font=FONT_BODY)
            x += col_w
        y += row_h

    return image


def draw_combined_image(metric_tables):
    section_images = []
    for spec in METRIC_SPECS:
        section_images.append(
            draw_table_image(
                spec["title"],
                spec["subtitle"],
                spec["format"],
                metric_tables[spec["key"]],
            )
        )

    width = section_images[0].width
    gap = 28
    outer_margin = 30
    total_height = outer_margin * 2 + sum(image.height for image in section_images) + gap * (len(section_images) - 1)
    combined = Image.new("RGB", (width, total_height), BG)

    y = outer_margin
    for image in section_images:
        combined.paste(image, (0, y))
        y += image.height + gap

    return combined


def write_summary_files(rows, metric_tables, summary_json_path, summary_md_path):
    summary_payload = {
        "runs_used": rows,
        "tables": {
            metric: [
                {
                    "model": model,
                    "normal": normal,
                    "targeted": targeted,
                    "prior": prior,
                }
                for model, normal, targeted, prior in table_rows
            ]
            for metric, table_rows in metric_tables.items()
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    lines = ["# Three Mode Tables", ""]
    for spec in METRIC_SPECS:
        lines.append(f"## {spec['title']}")
        lines.append("")
        lines.append("| model | normal | targeted | prior |")
        lines.append("|---|---:|---:|---:|")
        for model, normal, targeted, prior in metric_tables[spec["key"]]:
            lines.append(
                f"| {model} | {normal:.4f} | {targeted:.4f} | {prior:.4f} |"
            )
        lines.append("")
    lines.append("## Runs Used")
    lines.append("")
    for row in rows:
        lines.append(
            f"- {row['path']} ({row['model_family']} / {row['mode']})"
        )
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Render the three-mode comparison tables from the current results directory.")
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing experiment summary JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory where rendered tables and audit files will be written. Defaults to <results-dir>/analysis.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / "three_mode_tables_current.json"
    summary_md_path = output_dir / "three_mode_tables_current.md"
    combined_png = output_dir / "three_mode_tables_combined.png"

    rows = load_summary_runs(results_dir)
    metric_tables = aggregate_metric_tables(rows)
    write_summary_files(rows, metric_tables, summary_json_path, summary_md_path)

    for spec in METRIC_SPECS:
        image = draw_table_image(
            spec["title"],
            spec["subtitle"],
            spec["format"],
            metric_tables[spec["key"]],
        )
        output_path = output_dir / spec["filename"]
        image.save(output_path, dpi=(300, 300))
        print(f"Wrote {output_path}")

    combined = draw_combined_image(metric_tables)
    combined.save(combined_png, dpi=(300, 300))
    print(f"Wrote {combined_png}")

    print(f"Wrote {summary_json_path}")
    print(f"Wrote {summary_md_path}")


if __name__ == "__main__":
    main()
