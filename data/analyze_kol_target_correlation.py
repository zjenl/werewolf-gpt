import argparse
import json
import math
import os
from collections import Counter, defaultdict
from statistics import mean, median

def safe_mean(values):
    return mean(values) if values else 0.0


def rankdata(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = safe_mean(xs)
    mean_y = safe_mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def spearman_corr(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson_corr(rankdata(xs), rankdata(ys))


def load_games(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_analysis_rows(games):
    player_rows = []
    game_rows = []

    for game in games:
        game_id = game.get("Game_ID")
        windows = game.get("windows", [])
        player_profiles = game.get("player_profiles", [])
        kol_profile = next(
            (profile for profile in player_profiles if profile.get("influence") == "kol"),
            None
        )
        kol_player = kol_profile.get("player") if kol_profile else None
        leader_target_overlap = sum(
            1
            for window in windows
            if window.get("targeted_player") == window.get("discussion_leader")
            and window.get("targeted_player") not in {None, "None", "Group", ""}
        )
        windows_targeting_kol = sum(
            1 for window in windows if kol_player and window.get("targeted_player") == kol_player
        )

        for profile in player_profiles:
            row = {
                "game_id": game_id,
                "player": profile.get("player"),
                "end_role": profile.get("endRole"),
                "influence": profile.get("influence"),
                "discussion_leader_count": profile.get("discussion_leader_count", 0),
                "werewolf_target_count": profile.get("werewolf_target_count", 0),
                "werewolf_target_rank": profile.get("werewolf_target_rank"),
                "voted_werewolf": profile.get("voted_werewolf"),
                "is_kol": profile.get("influence") == "kol",
            }
            player_rows.append(row)

        game_rows.append({
            "game_id": game_id,
            "window_count": len(windows),
            "kol_player": kol_player,
            "kol_target_count": kol_profile.get("werewolf_target_count", 0) if kol_profile else 0,
            "kol_target_rank": kol_profile.get("werewolf_target_rank") if kol_profile else None,
            "windows_targeting_kol": windows_targeting_kol,
            "leader_target_overlap": leader_target_overlap,
            "leader_target_overlap_rate": (leader_target_overlap / len(windows)) if windows else 0.0,
        })

    return player_rows, game_rows


def summarize(player_rows, game_rows):
    leader_counts = [row["discussion_leader_count"] for row in player_rows]
    target_counts = [row["werewolf_target_count"] for row in player_rows]

    by_influence = defaultdict(list)
    for row in player_rows:
        by_influence[row["influence"]].append(row["werewolf_target_count"])

    kol_games = [row for row in game_rows if row["kol_player"]]
    kol_rank_counter = Counter(row["kol_target_rank"] for row in kol_games if row["kol_target_rank"] is not None)

    return {
        "player_count": len(player_rows),
        "game_count": len(game_rows),
        "pearson_leader_target": pearson_corr(leader_counts, target_counts),
        "spearman_leader_target": spearman_corr(leader_counts, target_counts),
        "avg_target_count_by_influence": {
            influence: safe_mean(values)
            for influence, values in by_influence.items()
        },
        "median_target_count_by_influence": {
            influence: median(values) if values else 0
            for influence, values in by_influence.items()
        },
        "kol_rank_distribution": dict(sorted(kol_rank_counter.items())),
        "games_where_kol_rank_1": sum(1 for row in kol_games if row["kol_target_rank"] == 1),
        "games_with_kol": len(kol_games),
        "avg_windows_targeting_kol": safe_mean([row["windows_targeting_kol"] for row in kol_games]),
        "avg_leader_target_overlap_rate": safe_mean([row["leader_target_overlap_rate"] for row in game_rows]),
    }


def save_summary(summary, output_dir):
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


def save_text_summary(summary, output_dir):
    path = os.path.join(output_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("KOL vs Werewolf Target Analysis\n")
        f.write("=" * 32 + "\n\n")
        f.write(f"Games: {summary['game_count']}\n")
        f.write(f"Player rows: {summary['player_count']}\n")
        f.write(f"Pearson correlation (leader_count vs target_count): {summary['pearson_leader_target']}\n")
        f.write(f"Spearman correlation (leader_count vs target_count): {summary['spearman_leader_target']}\n")
        f.write(f"Games with KOL: {summary['games_with_kol']}\n")
        f.write(f"Games where KOL target rank = 1: {summary['games_where_kol_rank_1']}\n")
        f.write(f"Average windows targeting KOL: {summary['avg_windows_targeting_kol']:.3f}\n")
        f.write(f"Average leader-target overlap rate: {summary['avg_leader_target_overlap_rate']:.3f}\n\n")
        f.write("Average target count by influence:\n")
        for influence, value in summary["avg_target_count_by_influence"].items():
            f.write(f"- {influence}: {value:.3f}\n")
        f.write("\nMedian target count by influence:\n")
        for influence, value in summary["median_target_count_by_influence"].items():
            f.write(f"- {influence}: {value}\n")
        f.write("\nKOL target rank distribution:\n")
        for rank, count in summary["kol_rank_distribution"].items():
            f.write(f"- rank {rank}: {count}\n")
    return path


def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(path, width, height, elements):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
        f.write('<rect width="100%" height="100%" fill="white"/>')
        for element in elements:
            f.write(element)
        f.write("</svg>")


def svg_text(x, y, text, size=12, anchor="start", weight="normal"):
    return f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape_xml(text)}</text>'


def svg_line(x1, y1, x2, y2, stroke="#333", width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"/>'


def svg_rect(x, y, width, height, fill, stroke="#333"):
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'


def svg_circle(cx, cy, r, fill, stroke="none", opacity=0.85):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'


def plot_scatter(player_rows, output_dir):
    width, height = 800, 520
    left, right, top, bottom = 80, 30, 50, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = {"kol": "#d62728", "normal": "#1f77b4", "low_influence": "#2ca02c"}

    xs = [row["discussion_leader_count"] for row in player_rows]
    ys = [row["werewolf_target_count"] for row in player_rows]
    max_x = max(xs) if xs else 1
    max_y = max(ys) if ys else 1

    def scale_x(value):
        return left + (0 if max_x == 0 else (value / max_x) * plot_w)

    def scale_y(value):
        return top + plot_h - (0 if max_y == 0 else (value / max_y) * plot_h)

    elements = [
        svg_text(width / 2, 28, "Discussion Leadership vs Werewolf Targeting", size=18, anchor="middle", weight="bold"),
        svg_line(left, top, left, top + plot_h),
        svg_line(left, top + plot_h, left + plot_w, top + plot_h),
    ]

    for tick in range(max_x + 1):
        x = scale_x(tick)
        elements.append(svg_line(x, top + plot_h, x, top + plot_h + 6))
        elements.append(svg_text(x, top + plot_h + 22, tick, size=11, anchor="middle"))

    for tick in range(max_y + 1):
        y = scale_y(tick)
        elements.append(svg_line(left - 6, y, left, y))
        elements.append(svg_text(left - 12, y + 4, tick, size=11, anchor="end"))

    for row in player_rows:
        elements.append(
            svg_circle(
                scale_x(row["discussion_leader_count"]),
                scale_y(row["werewolf_target_count"]),
                5,
                colors.get(row["influence"], "#7f7f7f")
            )
        )

    legend_x = width - 155
    legend_y = 70
    for idx, influence in enumerate(["kol", "normal", "low_influence"]):
        elements.append(svg_circle(legend_x, legend_y + idx * 24, 6, colors[influence]))
        elements.append(svg_text(legend_x + 14, legend_y + 4 + idx * 24, influence, size=12))

    elements.append(svg_text(width / 2, height - 18, "Discussion Leader Count", size=13, anchor="middle"))
    elements.append(svg_text(18, height / 2, "Werewolf Target Count", size=13, anchor="middle"))

    path = os.path.join(output_dir, "leader_count_vs_target_count.svg")
    write_svg(path, width, height, elements)
    return path


def plot_avg_target_by_influence(summary, output_dir):
    width, height = 700, 460
    left, right, top, bottom = 80, 30, 50, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    order = ["kol", "normal", "low_influence"]
    labels = [label for label in order if label in summary["avg_target_count_by_influence"]]
    values = [summary["avg_target_count_by_influence"][label] for label in labels]
    colors = {"kol": "#d62728", "normal": "#1f77b4", "low_influence": "#2ca02c"}
    max_y = max(values) if values else 1

    elements = [
        svg_text(width / 2, 28, "Average Target Count by Influence", size=18, anchor="middle", weight="bold"),
        svg_line(left, top, left, top + plot_h),
        svg_line(left, top + plot_h, left + plot_w, top + plot_h),
    ]

    bar_w = plot_w / max(len(labels), 1) * 0.55
    gap = plot_w / max(len(labels), 1)
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * gap + (gap - bar_w) / 2
        bar_h = 0 if max_y == 0 else (value / max_y) * plot_h
        y = top + plot_h - bar_h
        elements.append(svg_rect(x, y, bar_w, bar_h, colors[label]))
        elements.append(svg_text(x + bar_w / 2, top + plot_h + 24, label, anchor="middle"))
        elements.append(svg_text(x + bar_w / 2, y - 8, f"{value:.2f}", anchor="middle", size=11))

    path = os.path.join(output_dir, "avg_target_count_by_influence.svg")
    write_svg(path, width, height, elements)
    return path


def plot_kol_rank_distribution(summary, output_dir):
    width, height = 700, 460
    left, right, top, bottom = 80, 30, 50, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    rank_dist = summary["kol_rank_distribution"]
    ranks = list(rank_dist.keys())
    counts = list(rank_dist.values())
    max_y = max(counts) if counts else 1

    elements = [
        svg_text(width / 2, 28, "How Often the KOL Is the Top Werewolf Target", size=18, anchor="middle", weight="bold"),
        svg_line(left, top, left, top + plot_h),
        svg_line(left, top + plot_h, left + plot_w, top + plot_h),
    ]

    bar_w = plot_w / max(len(ranks), 1) * 0.55
    gap = plot_w / max(len(ranks), 1)
    for idx, (rank, count) in enumerate(zip(ranks, counts)):
        x = left + idx * gap + (gap - bar_w) / 2
        bar_h = 0 if max_y == 0 else (count / max_y) * plot_h
        y = top + plot_h - bar_h
        elements.append(svg_rect(x, y, bar_w, bar_h, "#9467bd"))
        elements.append(svg_text(x + bar_w / 2, top + plot_h + 24, str(rank), anchor="middle"))
        elements.append(svg_text(x + bar_w / 2, y - 8, count, anchor="middle", size=11))

    path = os.path.join(output_dir, "kol_target_rank_distribution.svg")
    write_svg(path, width, height, elements)
    return path


def plot_overlap_histogram(game_rows, output_dir):
    width, height = 760, 460
    left, right, top, bottom = 80, 30, 50, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [row["leader_target_overlap_rate"] for row in game_rows]
    bins = [0] * 10
    for value in values:
        index = min(int(value * 10), 9)
        bins[index] += 1
    max_y = max(bins) if bins else 1

    elements = [
        svg_text(width / 2, 28, "Leader-Target Overlap by Game", size=18, anchor="middle", weight="bold"),
        svg_line(left, top, left, top + plot_h),
        svg_line(left, top + plot_h, left + plot_w, top + plot_h),
    ]

    bar_w = plot_w / len(bins) * 0.8
    gap = plot_w / len(bins)
    for idx, count in enumerate(bins):
        x = left + idx * gap + (gap - bar_w) / 2
        bar_h = 0 if max_y == 0 else (count / max_y) * plot_h
        y = top + plot_h - bar_h
        elements.append(svg_rect(x, y, bar_w, bar_h, "#ff7f0e"))
        bucket_label = f"{idx/10:.1f}-{(idx+1)/10:.1f}"
        elements.append(svg_text(x + bar_w / 2, top + plot_h + 26, bucket_label, anchor="middle", size=10))
        elements.append(svg_text(x + bar_w / 2, y - 8, count, anchor="middle", size=11))

    path = os.path.join(output_dir, "leader_target_overlap_rate.svg")
    write_svg(path, width, height, elements)
    return path


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="data/filtered_labeled_games_50.json")
    parser.add_argument("--output-dir", default="data/kol_target_analysis_50")
    return parser


def main():
    args = build_arg_parser().parse_args()
    ensure_dir(args.output_dir)

    games = load_games(args.input_file)
    player_rows, game_rows = build_analysis_rows(games)
    summary = summarize(player_rows, game_rows)

    summary_json = save_summary(summary, args.output_dir)
    summary_txt = save_text_summary(summary, args.output_dir)
    scatter_plot = plot_scatter(player_rows, args.output_dir)
    influence_plot = plot_avg_target_by_influence(summary, args.output_dir)
    rank_plot = plot_kol_rank_distribution(summary, args.output_dir)
    overlap_plot = plot_overlap_histogram(game_rows, args.output_dir)

    print(f"Saved summary JSON: {summary_json}")
    print(f"Saved summary text: {summary_txt}")
    print(f"Saved plot: {scatter_plot}")
    print(f"Saved plot: {influence_plot}")
    print(f"Saved plot: {rank_plot}")
    print(f"Saved plot: {overlap_plot}")


if __name__ == "__main__":
    main()
