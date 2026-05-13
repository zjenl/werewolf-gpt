import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


RESULTS_DIR = Path("werewolf-results")
OUTPUT_DIR = RESULTS_DIR / "analysis"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_mode(summary):
    if summary.get("personality_aware_werewolf_persuasion"):
        return "personality_aware"
    if summary.get("structured_werewolf_persuasion"):
        return "structured"
    if summary.get("targeted_werewolf_persuasion"):
        return "targeted"
    return "baseline"


def normalize_model(summary):
    model_mode = summary.get("model_mode")
    if model_mode:
        return model_mode

    model = summary.get("model", "")
    if model == "gpt-5-nano":
        return "openai-gpt5-nano"
    if "gpt-4" in model:
        return "openai-gpt4"
    return model or "unknown"


def infer_games_path(summary_path):
    stem = summary_path.stem
    if stem.endswith("-1"):
        base = stem[:-2]
        candidate = summary_path.with_name(f"{base}-games-1.json")
        if candidate.exists():
            return candidate
    candidate = summary_path.with_name(f"{stem}-games.json")
    if candidate.exists():
        return candidate
    return None


def classify_result_text(result_text):
    text = (result_text or "").lower()
    if "tie between" in text:
        return "tie"
    if "no player was voted out" in text:
        return "no_kill"
    if "was killed" in text:
        return "single_kill"
    return "other"


def avg(values):
    return mean(values) if values else 0.0


def extract_dialogue_metrics(game_dialogues):
    if not game_dialogues:
        return {}

    total_turns = []
    day_turns = []
    vote_turns = []
    night_turns = []
    word_counts = []
    question_turns = 0
    warning_count = 0

    for game in game_dialogues:
        dialogue = game.get("Dialogue", [])
        total_turns.append(len(dialogue))
        day_turns.append(sum(1 for turn in dialogue if turn.get("phase") == "DAY"))
        vote_turns.append(sum(1 for turn in dialogue if turn.get("phase") == "VOTE"))
        night_turns.append(sum(1 for turn in dialogue if turn.get("phase") == "NIGHT"))
        if game.get("warning"):
            warning_count += 1

        for turn in dialogue:
            utterance = turn.get("utterance", "")
            word_counts.append(len(utterance.split()))
            if "?" in utterance:
                question_turns += 1

    return {
        "games_with_dialogue": len(game_dialogues),
        "avg_total_turns": avg(total_turns),
        "avg_day_turns": avg(day_turns),
        "avg_vote_turns": avg(vote_turns),
        "avg_night_turns": avg(night_turns),
        "avg_words_per_turn": avg(word_counts),
        "question_turn_rate": (question_turns / sum(total_turns)) if total_turns and sum(total_turns) else 0.0,
        "warning_rate": warning_count / len(game_dialogues) if game_dialogues else 0.0,
    }


def summarize_experiment(summary_path):
    payload = read_json(summary_path)
    summary = payload["summary"]
    games = payload.get("games", [])
    games_path = infer_games_path(summary_path)
    dialogue_games = read_json(games_path) if games_path and games_path.exists() else None

    completed_games = [game for game in games if game.get("status") == "completed"]
    outcome_types = Counter(classify_result_text(game.get("result")) for game in completed_games)
    vote_margins = []
    killed_werewolf = 0

    for game in completed_games:
        votes = game.get("votes", {})
        if votes:
            sorted_votes = sorted(votes.values(), reverse=True)
            top = sorted_votes[0]
            second = sorted_votes[1] if len(sorted_votes) > 1 else 0
            vote_margins.append(top - second)

        if game.get("winner") == "villagers":
            killed_werewolf += 1

    dialogue_metrics = extract_dialogue_metrics(dialogue_games) if dialogue_games else {}

    row = {
        "file": summary_path.name,
        "experiment": summary_path.stem,
        "model_mode": normalize_model(summary),
        "mode": normalize_mode(summary),
        "model": summary.get("model"),
        "player_count": summary.get("player_count"),
        "discussion_depth": summary.get("discussion_depth"),
        "parallel_games": summary.get("parallel_games"),
        "games_requested": summary.get("games_requested"),
        "games_completed": summary.get("games_completed"),
        "games_failed": summary.get("games_failed"),
        "werewolf_wins": summary.get("werewolf_wins"),
        "werewolf_win_rate": summary.get("werewolf_win_rate"),
        "villager_win_rate": 1 - summary.get("werewolf_win_rate", 0.0) if summary.get("games_completed") else 0.0,
        "tie_rate": outcome_types["tie"] / len(completed_games) if completed_games else 0.0,
        "no_kill_rate": outcome_types["no_kill"] / len(completed_games) if completed_games else 0.0,
        "single_kill_rate": outcome_types["single_kill"] / len(completed_games) if completed_games else 0.0,
        "avg_vote_margin": avg(vote_margins),
        "median_vote_margin": median(vote_margins) if vote_margins else 0.0,
        "games_path": games_path.name if games_path else "",
    }
    row.update(dialogue_metrics)
    return row


def load_all_experiments(results_dir):
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if "-games" in path.stem:
            continue
        data = read_json(path)
        if not isinstance(data, dict) or "summary" not in data:
            continue
        rows.append(summarize_experiment(path))
    return rows


def aggregate(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)

    aggregates = []
    for value, items in grouped.items():
        aggregates.append({
            key: value,
            "runs": len(items),
            "avg_werewolf_win_rate": avg([item["werewolf_win_rate"] for item in items]),
            "avg_tie_rate": avg([item["tie_rate"] for item in items]),
            "avg_no_kill_rate": avg([item["no_kill_rate"] for item in items]),
            "avg_vote_margin": avg([item["avg_vote_margin"] for item in items]),
            "avg_total_turns": avg([item.get("avg_total_turns", 0.0) for item in items]),
            "avg_words_per_turn": avg([item.get("avg_words_per_turn", 0.0) for item in items]),
            "avg_warning_rate": avg([item.get("warning_rate", 0.0) for item in items]),
        })
    return sorted(aggregates, key=lambda item: (item[key], item["runs"]))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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


def svg_text(x, y, text, size=12, anchor="start", weight="normal", fill="#142033"):
    return f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{escape_xml(text)}</text>'


def svg_line(x1, y1, x2, y2, stroke="#333", width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"/>'


def svg_rect(x, y, width, height, fill, stroke="#333"):
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'


def bar_chart(path, title, labels, values, y_label, fill="#4c78a8"):
    width, height = 900, 520
    left, right, top, bottom = 80, 30, 55, 140
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_y = max(values) if values else 1
    elements = [
        svg_text(width / 2, 30, title, size=20, anchor="middle", weight="bold"),
        svg_line(left, top, left, top + plot_h),
        svg_line(left, top + plot_h, left + plot_w, top + plot_h),
        svg_text(20, height / 2, y_label, size=13, anchor="middle"),
    ]
    bar_w = plot_w / max(len(labels), 1) * 0.65
    gap = plot_w / max(len(labels), 1)
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * gap + (gap - bar_w) / 2
        bar_h = 0 if max_y == 0 else (value / max_y) * plot_h
        y = top + plot_h - bar_h
        elements.append(svg_rect(x, y, bar_w, bar_h, fill))
        elements.append(svg_text(x + bar_w / 2, top + plot_h + 20, label, size=10, anchor="middle"))
        elements.append(svg_text(x + bar_w / 2, y - 8, f"{value:.3f}" if isinstance(value, float) else value, size=10, anchor="middle"))
    write_svg(path, width, height, elements)


def grouped_bar_chart(path, title, group_labels, series_labels, series_values, y_label, colors):
    width, height = 980, 540
    left, right, top, bottom = 80, 30, 55, 140
    plot_w = width - left - right
    plot_h = height - top - bottom
    flat = [value for series in series_values for value in series]
    max_y = max(flat) if flat else 1
    elements = [
        svg_text(width / 2, 30, title, size=20, anchor="middle", weight="bold"),
        svg_line(left, top, left, top + plot_h),
        svg_line(left, top + plot_h, left + plot_w, top + plot_h),
        svg_text(20, height / 2, y_label, size=13, anchor="middle"),
    ]

    group_gap = plot_w / max(len(group_labels), 1)
    total_bar_width = group_gap * 0.72
    bar_w = total_bar_width / max(len(series_labels), 1)

    for g_idx, group_label in enumerate(group_labels):
        group_start = left + g_idx * group_gap + (group_gap - total_bar_width) / 2
        for s_idx, series_label in enumerate(series_labels):
            value = series_values[s_idx][g_idx]
            x = group_start + s_idx * bar_w
            bar_h = 0 if max_y == 0 else (value / max_y) * plot_h
            y = top + plot_h - bar_h
            elements.append(svg_rect(x, y, bar_w * 0.88, bar_h, colors[s_idx]))
        elements.append(svg_text(group_start + total_bar_width / 2, top + plot_h + 20, group_label, size=10, anchor="middle"))

    legend_x = width - 180
    legend_y = 70
    for idx, series_label in enumerate(series_labels):
        elements.append(svg_rect(legend_x, legend_y + idx * 22, 12, 12, colors[idx], stroke="none"))
        elements.append(svg_text(legend_x + 18, legend_y + 11 + idx * 22, series_label, size=11))

    write_svg(path, width, height, elements)


def write_html_report(path, rows, by_model, by_mode):
    top_rows = sorted(rows, key=lambda row: row["werewolf_win_rate"], reverse=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Werewolf Results Analysis</title>
<style>
body{font-family:Inter,Arial,sans-serif;margin:32px;background:#f7f9fc;color:#132033}
h1,h2{margin:0 0 14px} .panel{background:#fff;border:1px solid #d9e1ee;border-radius:10px;padding:18px 20px;margin:18px 0}
table{width:100%;border-collapse:collapse;font-size:14px} th,td{padding:10px 12px;border-bottom:1px solid #e7edf6;text-align:left}
th{font-size:12px;text-transform:uppercase;color:#607089} img{max-width:100%;border:1px solid #e2e8f2;border-radius:8px;background:#fff}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.muted{color:#607089}
</style></head><body>""")
        f.write("<h1>Werewolf Results Analysis</h1>")
        f.write(f"<p class='muted'>Analyzed {len(rows)} experiment summaries from <code>werewolf-results/</code>.</p>")
        f.write("<div class='grid'>")
        f.write("<div class='panel'><h2>Top Win Rates</h2><table><thead><tr><th>Experiment</th><th>Model</th><th>Mode</th><th>Werewolf win rate</th></tr></thead><tbody>")
        for row in top_rows[:8]:
            f.write(f"<tr><td>{escape_xml(row['experiment'])}</td><td>{escape_xml(row['model_mode'])}</td><td>{escape_xml(row['mode'])}</td><td>{row['werewolf_win_rate']:.3f}</td></tr>")
        f.write("</tbody></table></div>")
        f.write("<div class='panel'><h2>Aggregate Takeaways</h2><ul>")
        if by_model:
            best_model = max(by_model, key=lambda row: row["avg_werewolf_win_rate"])
            f.write(f"<li>Best average werewolf win rate by model mode: <strong>{escape_xml(best_model['model_mode'])}</strong> ({best_model['avg_werewolf_win_rate']:.3f}).</li>")
        if by_mode:
            best_mode = max(by_mode, key=lambda row: row["avg_werewolf_win_rate"])
            f.write(f"<li>Best average werewolf win rate by persuasion mode: <strong>{escape_xml(best_mode['mode'])}</strong> ({best_mode['avg_werewolf_win_rate']:.3f}).</li>")
        f.write("<li>See CSV files in this folder for per-experiment metrics and grouped comparisons.</li>")
        f.write("</ul></div></div>")
        f.write("<div class='grid'>")
        for name in ["win_rate_by_experiment.svg", "avg_win_rate_by_model_mode.svg", "avg_win_rate_by_mode.svg", "dialogue_stats_by_mode.svg"]:
            f.write(f"<div class='panel'><img src='{name}' alt='{name}'></div>")
        f.write("</div></body></html>")


def main():
    ensure_dir(OUTPUT_DIR)
    rows = load_all_experiments(RESULTS_DIR)
    by_model = aggregate(rows, "model_mode")
    by_mode = aggregate(rows, "mode")

    write_csv(OUTPUT_DIR / "experiments.csv", rows)
    write_csv(OUTPUT_DIR / "by_model_mode.csv", by_model)
    write_csv(OUTPUT_DIR / "by_mode.csv", by_mode)
    write_json(OUTPUT_DIR / "experiments.json", rows)

    exp_labels = [row["experiment"] for row in rows]
    exp_win_rates = [row["werewolf_win_rate"] for row in rows]
    bar_chart(
        OUTPUT_DIR / "win_rate_by_experiment.svg",
        "Werewolf Win Rate by Experiment",
        exp_labels,
        exp_win_rates,
        "Werewolf Win Rate",
        fill="#4c78a8"
    )

    model_labels = [row["model_mode"] for row in by_model]
    model_rates = [row["avg_werewolf_win_rate"] for row in by_model]
    bar_chart(
        OUTPUT_DIR / "avg_win_rate_by_model_mode.svg",
        "Average Werewolf Win Rate by Model Mode",
        model_labels,
        model_rates,
        "Average Werewolf Win Rate",
        fill="#8b1e3f"
    )

    mode_labels = [row["mode"] for row in by_mode]
    mode_rates = [row["avg_werewolf_win_rate"] for row in by_mode]
    bar_chart(
        OUTPUT_DIR / "avg_win_rate_by_mode.svg",
        "Average Werewolf Win Rate by Persuasion Mode",
        mode_labels,
        mode_rates,
        "Average Werewolf Win Rate",
        fill="#2f855a"
    )

    grouped_bar_chart(
        OUTPUT_DIR / "dialogue_stats_by_mode.svg",
        "Dialogue and Warning Patterns by Persuasion Mode",
        mode_labels,
        ["avg_total_turns", "avg_words_per_turn", "avg_warning_rate"],
        [
            [row["avg_total_turns"] for row in by_mode],
            [row["avg_words_per_turn"] for row in by_mode],
            [row["avg_warning_rate"] for row in by_mode],
        ],
        "Metric Value",
        ["#4c78a8", "#f28e2b", "#e15759"],
    )

    write_html_report(OUTPUT_DIR / "report.html", rows, by_model, by_mode)

    print(f"Saved analysis folder: {OUTPUT_DIR}")
    print(f"Experiments analyzed: {len(rows)}")
    print("Generated:")
    print("- experiments.csv")
    print("- by_model_mode.csv")
    print("- by_mode.csv")
    print("- experiments.json")
    print("- win_rate_by_experiment.svg")
    print("- avg_win_rate_by_model_mode.svg")
    print("- avg_win_rate_by_mode.svg")
    print("- dialogue_stats_by_mode.svg")
    print("- report.html")


if __name__ == "__main__":
    main()
