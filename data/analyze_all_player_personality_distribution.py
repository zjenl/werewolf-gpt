import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT = "data/filtered_labeled_games_50.json"
DEFAULT_OUTPUT_JSON = "data/all_player_personality_distribution_50.json"
DEFAULT_OUTPUT_TEXT = "data/all_player_personality_distribution_50.txt"
DEFAULT_OUTPUT_PNG = "data/all_player_personality_distribution_50.png"

TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
LEVELS = ["low", "moderate", "high"]

BG = "#f8fafc"
PANEL_BG = "#ffffff"
TITLE_BG = "#e0e7ff"
TEXT = "#111827"
MUTED = "#4b5563"
GRID = "#d1d5db"
LEVEL_COLORS = {
    "low": "#fca5a5",
    "moderate": "#93c5fd",
    "high": "#86efac",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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


FONT_TITLE = load_font(34, bold=True)
FONT_SUBTITLE = load_font(18, bold=False)
FONT_PANEL = load_font(24, bold=True)
FONT_BODY = load_font(18, bold=False)
FONT_SMALL = load_font(15, bold=False)
FONT_LABEL = load_font(18, bold=True)
FONT_METRIC = load_font(30, bold=True)


def percent(value):
    return f"{value * 100:.1f}%"


def rounded_panel(draw, box, fill=PANEL_BG, outline=GRID, radius=14):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)


def analyze_distribution(games):
    trait_counts = {trait: Counter() for trait in TRAITS}
    unique_players = []

    for game in games:
        for profile in game.get("player_profiles", []):
            unique_players.append((game.get("Game_ID"), profile.get("player")))
            personalities = profile.get("personalities", {})
            for trait in TRAITS:
                level = personalities.get(trait)
                if level in LEVELS:
                    trait_counts[trait][level] += 1

    total_profiles = len(unique_players)
    trait_distribution = {}
    skew_flags = {}
    for trait, counts in trait_counts.items():
        trait_total = sum(counts.values())
        trait_distribution[trait] = {
            level: {
                "count": counts.get(level, 0),
                "proportion": (counts.get(level, 0) / trait_total) if trait_total else 0.0,
            }
            for level in LEVELS
        }
        max_level, max_count = max(counts.items(), key=lambda item: item[1]) if counts else ("none", 0)
        skew_flags[trait] = {
            "dominant_level": max_level,
            "dominant_share": (max_count / trait_total) if trait_total else 0.0,
            "is_potentially_skewed": (max_count / trait_total) >= 0.6 if trait_total else False,
        }

    return {
        "games_analyzed": len(games),
        "player_profile_count": total_profiles,
        "trait_distribution": trait_distribution,
        "skew_flags": skew_flags,
    }


def write_text(path, summary):
    lines = [
        "All-player Big Five distribution",
        "===============================",
        "",
        f"Games analyzed: {summary['games_analyzed']}",
        f"Player profiles analyzed: {summary['player_profile_count']}",
        "",
    ]
    for trait in TRAITS:
        lines.append(trait.title())
        dist = summary["trait_distribution"][trait]
        for level in LEVELS:
            lines.append(
                f"- {level}: {dist[level]['count']} ({percent(dist[level]['proportion'])})"
            )
        skew = summary["skew_flags"][trait]
        lines.append(
            f"- dominant level: {skew['dominant_level']} ({percent(skew['dominant_share'])}); "
            f"potentially skewed: {skew['is_potentially_skewed']}"
        )
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def draw_distribution_figure(summary, output_path):
    width = 1600
    height = 1100
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    rounded_panel(draw, (50, 40, width - 50, 140), fill=TITLE_BG, radius=18)
    draw.text((78, 62), "Big Five Distribution Across All Players", fill=TEXT, font=FONT_TITLE)
    draw.text(
        (78, 102),
        "Distribution of low / moderate / high personality labels in the labeled human dataset",
        fill=MUTED,
        font=FONT_SUBTITLE,
    )

    stats = [
        ("Games", str(summary["games_analyzed"])),
        ("Player profiles", str(summary["player_profile_count"])),
        ("Traits", str(len(TRAITS))),
    ]
    stat_x = 55
    for label, value in stats:
        rounded_panel(draw, (stat_x, 170, stat_x + 220, 292))
        draw.text((stat_x + 18, 188), label, fill=MUTED, font=FONT_BODY)
        draw.text((stat_x + 18, 228), value, fill=TEXT, font=FONT_METRIC)
        stat_x += 240

    legend_x = width - 360
    legend_y = 192
    for idx, level in enumerate(LEVELS):
        lx = legend_x + idx * 110
        draw.rounded_rectangle((lx, legend_y, lx + 24, legend_y + 24), radius=6, fill=LEVEL_COLORS[level], outline=LEVEL_COLORS[level])
        draw.text((lx + 32, legend_y + 2), level.title(), fill=MUTED, font=FONT_BODY)

    panel_x = 50
    panel_y = 330
    panel_w = width - 100
    panel_h = 700
    rounded_panel(draw, (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h))
    draw.text((panel_x + 22, panel_y + 20), "Trait-level distributions", fill=TEXT, font=FONT_PANEL)
    draw.text((panel_x + 22, panel_y + 54), "Each bar shows the proportion of low / moderate / high labels for one trait", fill=MUTED, font=FONT_SMALL)

    bar_left = panel_x + 240
    bar_right = panel_x + panel_w - 180
    bar_w = bar_right - bar_left
    bar_h = 44
    top = panel_y + 120
    row_gap = 110

    for idx, trait in enumerate(TRAITS):
        row_y = top + idx * row_gap
        draw.text((panel_x + 24, row_y + 9), trait.title(), fill=TEXT, font=FONT_LABEL)
        draw.rounded_rectangle((bar_left, row_y, bar_right, row_y + bar_h), radius=10, fill="#eef2f7", outline="#eef2f7")
        running_x = bar_left
        for level in LEVELS:
            proportion = summary["trait_distribution"][trait][level]["proportion"]
            count = summary["trait_distribution"][trait][level]["count"]
            seg_w = bar_w * proportion
            if seg_w <= 0:
                continue
            draw.rounded_rectangle((running_x, row_y, running_x + seg_w, row_y + bar_h), radius=10, fill=LEVEL_COLORS[level], outline=LEVEL_COLORS[level])
            if seg_w > 85:
                draw.text((running_x + 10, row_y + 12), f"{count}", fill=TEXT, font=FONT_SMALL)
            running_x += seg_w

        skew = summary["skew_flags"][trait]
        draw.text(
            (bar_right + 20, row_y + 4),
            f"{skew['dominant_level']} / {percent(skew['dominant_share'])}",
            fill=MUTED,
            font=FONT_SMALL,
        )
        draw.text(
            (bar_right + 20, row_y + 24),
            "skewed" if skew["is_potentially_skewed"] else "balanced-ish",
            fill="#dc2626" if skew["is_potentially_skewed"] else "#059669",
            font=FONT_SMALL,
        )

    image.save(output_path, dpi=(300, 300))


def main():
    parser = argparse.ArgumentParser(description="Analyze the Big Five distribution across all labeled player profiles.")
    parser.add_argument("--input-file", default=DEFAULT_INPUT)
    parser.add_argument("--json-output", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--text-output", default=DEFAULT_OUTPUT_TEXT)
    parser.add_argument("--png-output", default=DEFAULT_OUTPUT_PNG)
    args = parser.parse_args()

    games = load_json(args.input_file)
    summary = analyze_distribution(games)
    write_json(args.json_output, summary)
    write_text(args.text_output, summary)
    draw_distribution_figure(summary, args.png_output)

    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.text_output}")
    print(f"Wrote {args.png_output}")


if __name__ == "__main__":
    main()
