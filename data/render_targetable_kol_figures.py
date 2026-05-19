import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


INPUT_SUMMARY = Path("data/targetable_kol_attribute_summary_50.json")
OUTPUT_DIR = Path("data/targetable_kol_figures_50")

BG = "#f8fafc"
PANEL_BG = "#ffffff"
TITLE_BG = "#e0e7ff"
SUBTLE_BG = "#eef2ff"
TEXT = "#111827"
MUTED = "#4b5563"
GRID = "#d1d5db"

TRAIT_COLORS = {
    "low": "#fca5a5",
    "moderate": "#93c5fd",
    "high": "#86efac",
}

SERIES_COLORS = [
    "#2563eb",
    "#7c3aed",
    "#059669",
    "#ea580c",
    "#dc2626",
    "#0891b2",
    "#4f46e5",
    "#65a30d",
]

LINGUISTIC_KEYS = [
    ("question_utterance_rate", "Questions"),
    ("self_reference_utterance_rate", "Self-reference"),
    ("you_reference_utterance_rate", "You-reference"),
    ("agreement_utterance_rate", "Agreement"),
    ("disagreement_utterance_rate", "Disagreement"),
    ("evidence_utterance_rate", "Evidence"),
    ("inclusive_utterance_rate", "Inclusive"),
    ("hedge_utterance_rate", "Hedging"),
]


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
FONT_LABEL = load_font(18, bold=True)
FONT_BODY = load_font(18, bold=False)
FONT_SMALL = load_font(15, bold=False)
FONT_METRIC = load_font(30, bold=True)


def percent_string(value):
    return f"{value * 100:.1f}%"


def rounded_panel(draw, box, fill=PANEL_BG, outline=GRID, radius=14):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)


def draw_title_block(draw, x, y, width, title, subtitle):
    rounded_panel(draw, (x, y, x + width, y + 92), fill=TITLE_BG, radius=16)
    draw.text((x + 24, y + 16), title, fill=TEXT, font=FONT_TITLE)
    draw.text((x + 24, y + 56), subtitle, fill=MUTED, font=FONT_SUBTITLE)


def draw_stat_card(draw, x, y, width, height, label, value, note=None):
    rounded_panel(draw, (x, y, x + width, y + height))
    draw.text((x + 18, y + 16), label, fill=MUTED, font=FONT_BODY)
    draw.text((x + 18, y + 48), value, fill=TEXT, font=FONT_METRIC)
    if note:
        draw.text((x + 18, y + height - 24), note, fill=MUTED, font=FONT_SMALL)


def draw_big5_panel(draw, x, y, width, height, summary):
    rounded_panel(draw, (x, y, x + width, y + height))
    draw.text((x + 20, y + 18), "Big Five Distribution", fill=TEXT, font=FONT_PANEL)
    draw.text((x + 20, y + 50), "Counts among KOLs who were top-targeted and did not vote Werewolf", fill=MUTED, font=FONT_SMALL)

    traits = list(summary["trait_value_counts"].keys())
    total = summary["selected_player_count"]
    bar_left = x + 190
    bar_right = x + width - 30
    bar_w = bar_right - bar_left
    bar_h = 30
    top = y + 95
    row_gap = 50

    legend_x = x + width - 240
    legend_y = y + 18
    for idx, label in enumerate(["low", "moderate", "high"]):
        lx = legend_x + idx * 74
        draw.rounded_rectangle((lx, legend_y, lx + 18, legend_y + 18), radius=4, fill=TRAIT_COLORS[label], outline=TRAIT_COLORS[label])
        draw.text((lx + 24, legend_y - 1), label.title(), fill=MUTED, font=FONT_SMALL)

    for idx, trait in enumerate(traits):
        row_y = top + idx * row_gap
        draw.text((x + 20, row_y + 4), trait.title(), fill=TEXT, font=FONT_BODY)
        draw.rounded_rectangle((bar_left, row_y, bar_right, row_y + bar_h), radius=8, fill="#f1f5f9", outline="#f1f5f9")
        running_x = bar_left
        counts = summary["trait_value_counts"][trait]
        for label in ["low", "moderate", "high"]:
            count = counts.get(label, 0)
            seg_w = bar_w * (count / total if total else 0)
            if seg_w <= 0:
                continue
            draw.rounded_rectangle((running_x, row_y, running_x + seg_w, row_y + bar_h), radius=8, fill=TRAIT_COLORS[label], outline=TRAIT_COLORS[label])
            if seg_w > 48:
                draw.text((running_x + 8, row_y + 6), str(count), fill=TEXT, font=FONT_SMALL)
            running_x += seg_w
        draw.text((bar_right + 10, row_y + 4), f"n={sum(counts.values())}", fill=MUTED, font=FONT_SMALL)


def draw_bar_panel(draw, x, y, width, height, title, subtitle, items, color_cycle=None, percent_mode=False):
    rounded_panel(draw, (x, y, x + width, y + height))
    draw.text((x + 20, y + 18), title, fill=TEXT, font=FONT_PANEL)
    draw.text((x + 20, y + 50), subtitle, fill=MUTED, font=FONT_SMALL)

    top = y + 92
    left = x + 165
    right = x + width - 35
    plot_w = right - left
    row_h = 34
    gap = 12
    max_value = max(value for _, value in items) if items else 1

    for idx, (label, value) in enumerate(items):
        row_y = top + idx * (row_h + gap)
        draw.text((x + 20, row_y + 6), label, fill=TEXT, font=FONT_BODY)
        draw.rounded_rectangle((left, row_y, right, row_y + row_h), radius=8, fill="#f1f5f9", outline="#f1f5f9")
        bar_w = plot_w * (value / max_value if max_value else 0)
        color = (color_cycle or SERIES_COLORS)[idx % len(color_cycle or SERIES_COLORS)]
        draw.rounded_rectangle((left, row_y, left + bar_w, row_y + row_h), radius=8, fill=color, outline=color)
        label_text = percent_string(value) if percent_mode else str(value)
        draw.text((right - 82, row_y + 6), label_text, fill=TEXT, font=FONT_SMALL)


def draw_top_profiles_panel(draw, x, y, width, height, summary):
    rounded_panel(draw, (x, y, x + width, y + height))
    draw.text((x + 20, y + 18), "Most Common Joint Trait Profiles", fill=TEXT, font=FONT_PANEL)
    draw.text((x + 20, y + 50), "Top recurring combinations inside the selected leader subgroup", fill=MUTED, font=FONT_SMALL)

    rows = summary["top_joint_trait_profiles"][:5]
    row_y = y + 92
    for row in rows:
        rounded_panel(draw, (x + 18, row_y, x + width - 18, row_y + 48), fill=SUBTLE_BG, outline=SUBTLE_BG, radius=10)
        draw.text((x + 34, row_y + 14), compact_trait_profile(row["trait_profile"]), fill=TEXT, font=FONT_SMALL)
        draw.text((x + width - 64, row_y + 14), f"x{row['count']}", fill=MUTED, font=FONT_LABEL)
        row_y += 60


def compact_trait_profile(profile_text):
    mapping = {
        "openness": "O",
        "conscientiousness": "C",
        "extraversion": "E",
        "agreeableness": "A",
        "neuroticism": "N",
        "moderate": "mod",
    }
    parts = []
    for chunk in profile_text.split("; "):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        parts.append(f"{mapping.get(key, key[:1].upper())}={mapping.get(value, value)}")
    return "  ".join(parts)


def draw_dashboard(summary):
    width = 1800
    height = 1580
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    draw_title_block(
        draw,
        54,
        44,
        width - 108,
        "Influenceable Opinion Leaders",
        "Trait and linguistic profile for KOLs most targeted by the Werewolf and not voting Werewolf",
    )

    draw_stat_card(draw, 54, 158, 250, 128, "Selected leaders", str(summary["selected_player_count"]), "Across filtered labeled games")
    draw_stat_card(draw, 322, 158, 250, 128, "Games represented", str(summary["games_represented"]), "Games contributing at least one selected KOL")
    draw_stat_card(
        draw,
        590,
        158,
        250,
        128,
        "Mean leader count",
        f"{summary['discussion_leader_count_stats']['mean']:.2f}",
        f"Median {summary['discussion_leader_count_stats']['median']}",
    )
    draw_stat_card(
        draw,
        858,
        158,
        250,
        128,
        "Mean target count",
        f"{summary['werewolf_target_count_stats']['mean']:.2f}",
        f"Median {summary['werewolf_target_count_stats']['median']}",
    )
    draw_stat_card(
        draw,
        1126,
        158,
        330,
        128,
        "Dominant strategy",
        next(iter(summary["most_used_strategy_counts"].keys())),
        f"{next(iter(summary['most_used_strategy_counts'].values()))} players",
    )
    draw_stat_card(
        draw,
        1474,
        158,
        272,
        128,
        "Mean words / utterance",
        f"{summary['linguistic_marker_summary']['avg_words_per_utterance']['mean']:.2f}",
        "Concise but highly interactive style",
    )

    draw_big5_panel(draw, 54, 318, 820, 360, summary)

    strategy_items = list(summary["most_used_strategy_counts"].items())[:5]
    draw_bar_panel(
        draw,
        896,
        318,
        410,
        360,
        "Most Used Strategies",
        "Strategy labels from the selected leaders' own utterances",
        strategy_items,
    )

    role_items = list(summary["start_role_counts"].items())[:6]
    draw_bar_panel(
        draw,
        1328,
        318,
        418,
        360,
        "Start Roles",
        "Role composition of the selected subgroup",
        role_items,
    )

    linguistic_items = [
        (label, summary["linguistic_marker_summary"][key]["mean"])
        for key, label in LINGUISTIC_KEYS
    ]
    draw_bar_panel(
        draw,
        54,
        708,
        880,
        620,
        "Linguistic Marker Profile",
        "Mean utterance-level rates from rule-based markers; higher means more common in their speech",
        linguistic_items,
        color_cycle=SERIES_COLORS,
        percent_mode=True,
    )

    top_word_items = [(row["word"], row["count"]) for row in summary["top_words_across_selected_players"][:10]]
    draw_bar_panel(
        draw,
        956,
        708,
        392,
        620,
        "Top Words",
        "Raw lexical frequency across selected leaders",
        top_word_items,
        color_cycle=SERIES_COLORS,
    )

    draw_top_profiles_panel(draw, 1370, 708, 376, 620, summary)

    return image


def save_individual_figures(summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dashboard = draw_dashboard(summary)
    dashboard.save(OUTPUT_DIR / "targetable_kol_dashboard.png", dpi=(300, 300))

    # Cropped panels for slide reuse.
    crops = {
        "targetable_kol_big5.png": (54, 318, 874, 678),
        "targetable_kol_linguistic_markers.png": (54, 708, 934, 1328),
        "targetable_kol_strategy_and_roles.png": (896, 318, 1746, 678),
        "targetable_kol_top_profiles.png": (1370, 708, 1746, 1328),
    }
    for name, box in crops.items():
        dashboard.crop(box).save(OUTPUT_DIR / name, dpi=(300, 300))


def main():
    summary = load_summary(INPUT_SUMMARY)
    save_individual_figures(summary)
    print(f"Wrote figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
