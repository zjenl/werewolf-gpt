import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values):
    return mean(values) if values else 0.0


def safe_median(values):
    return median(values) if values else 0.0


def safe_div(numerator, denominator):
    if not denominator:
        return 0.0
    return numerator / denominator


def percent(value):
    return round(value * 100.0, 2)


def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_games_path(summary_path):
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


def infer_log_path(summary_path):
    stem = summary_path.stem
    candidate = summary_path.with_suffix(".log")
    if candidate.exists():
        return candidate
    parts = stem.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        candidate = summary_path.with_name(f"{parts[0]}-{parts[1]}.log")
        if candidate.exists():
            return candidate
    return None


def infer_persuasion_mode(summary):
    if summary.get("personality_aware_werewolf_persuasion"):
        return "personality_aware"
    if summary.get("structured_werewolf_persuasion"):
        return "structured"
    if summary.get("targeted_werewolf_persuasion"):
        return "targeted"
    return "baseline"


def infer_model_family(summary):
    model_mode = str(summary.get("model_mode") or "").lower()
    model = str(summary.get("model") or "").lower()
    combined = f"{model_mode} {model}"
    if "grok" in combined:
        return "grok"
    if "qwen" in combined:
        return "qwen"
    if "gemini" in combined:
        return "gemini"
    if "gpt" in combined or "openai" in combined:
        return "gpt5_nano"
    return "unknown"


def normalize_experiment_label(summary_path, summary):
    family = infer_model_family(summary)
    mode = infer_persuasion_mode(summary)
    run_name = summary_path.stem
    return f"{family}:{mode}:{run_name}"


def summarize_vote_pattern(votes):
    if not votes:
        return 0, 0.0, False
    top_vote = max(votes)
    total_votes = sum(votes)
    tie = votes.count(top_vote) > 1
    concentration = safe_div(top_vote, total_votes)
    return top_vote, concentration, tie


def get_werewolf_names_from_summary_game(game_summary):
    werewolves = []
    for player in game_summary.get("players") or []:
        if player.get("card") == "Werewolf":
            werewolves.append(player.get("name"))
    return werewolves


def summarize_summary_game_votes(game_summary):
    votes = game_summary.get("votes") or {}
    werewolf_names = get_werewolf_names_from_summary_game(game_summary)
    werewolf_vote_total = sum(votes.get(name, 0) for name in werewolf_names)
    non_werewolf_vote_total = sum(
        count for name, count in votes.items() if name not in set(werewolf_names)
    )
    return {
        "werewolf_vote_total": werewolf_vote_total,
        "non_werewolf_vote_total": non_werewolf_vote_total,
        "werewolf_vote_share": safe_div(werewolf_vote_total, sum(votes.values())),
    }


def get_werewolf_players(game):
    players = game.get("playerNames") or []
    roles = game.get("startRoles") or []
    werewolves = []
    for player, role in zip(players, roles):
        if role == "Werewolf":
            werewolves.append(player)
    return set(werewolves)


def build_detailed_game_metrics(game, experiment_id, model_family, persuasion_mode):
    dialogue = game.get("Dialogue") or []
    werewolf_players = get_werewolf_players(game)
    phase_counts = Counter(turn.get("phase") or "UNKNOWN" for turn in dialogue)
    speaker_counts = Counter(turn.get("speaker") or "UNKNOWN" for turn in dialogue)
    werewolf_turns = sum(count for speaker, count in speaker_counts.items() if speaker in werewolf_players)

    utterances = [str(turn.get("utterance") or "") for turn in dialogue]
    utterance_word_counts = [len(text.split()) for text in utterances if text.strip()]
    utterance_char_counts = [len(text) for text in utterances if text.strip()]
    warning = bool(game.get("warning"))

    votes = game.get("votingOutcome") or []
    top_vote, vote_concentration, vote_tie = summarize_vote_pattern(votes)

    return {
        "experiment_id": experiment_id,
        "game_id": game.get("Game_ID") or game.get("EG_ID"),
        "model_family": model_family,
        "persuasion_mode": persuasion_mode,
        "player_count": len(game.get("playerNames") or []),
        "warning": warning,
        "total_turns": len(dialogue),
        "day_turns": phase_counts.get("DAY", 0),
        "night_turns": phase_counts.get("NIGHT", 0),
        "vote_turns": phase_counts.get("VOTE", 0),
        "distinct_speakers": len([speaker for speaker in speaker_counts if speaker != "UNKNOWN"]),
        "werewolf_turns": werewolf_turns,
        "werewolf_turn_share": safe_div(werewolf_turns, len(dialogue)),
        "avg_utterance_words": round(safe_mean(utterance_word_counts), 4),
        "avg_utterance_chars": round(safe_mean(utterance_char_counts), 4),
        "median_utterance_words": round(safe_median(utterance_word_counts), 4),
        "vote_top_count": top_vote,
        "vote_concentration": round(vote_concentration, 4),
        "vote_tie": vote_tie,
    }


def analyze_experiment(summary_path):
    data = load_json(summary_path)
    if not isinstance(data, dict) or "summary" not in data or "games" not in data:
        return None

    summary = data["summary"]
    game_summaries = data["games"] or []
    games_path = infer_games_path(summary_path)
    log_path = infer_log_path(summary_path)
    detailed_games = load_json(games_path) if games_path else []

    persuasion_mode = infer_persuasion_mode(summary)
    model_family = infer_model_family(summary)
    experiment_id = normalize_experiment_label(summary_path, summary)

    warning_count = sum(1 for game in detailed_games if game.get("warning"))
    total_turns = []
    day_turns = []
    night_turns = []
    vote_turns = []
    utterance_words = []
    utterance_chars = []
    werewolf_turn_shares = []
    vote_concentrations = []
    vote_ties = 0
    werewolf_vote_totals = []
    non_werewolf_vote_totals = []
    werewolf_vote_shares = []
    per_game_rows = []

    for game in detailed_games:
        row = build_detailed_game_metrics(game, experiment_id, model_family, persuasion_mode)
        per_game_rows.append(row)
        total_turns.append(row["total_turns"])
        day_turns.append(row["day_turns"])
        night_turns.append(row["night_turns"])
        vote_turns.append(row["vote_turns"])
        utterance_words.append(row["avg_utterance_words"])
        utterance_chars.append(row["avg_utterance_chars"])
        werewolf_turn_shares.append(row["werewolf_turn_share"])
        vote_concentrations.append(row["vote_concentration"])
        vote_ties += int(row["vote_tie"])

    wins = [bool(game.get("werewolf_win")) for game in game_summaries if game.get("status") == "completed"]
    for game_summary in game_summaries:
        if game_summary.get("status") != "completed":
            continue
        vote_summary = summarize_summary_game_votes(game_summary)
        werewolf_vote_totals.append(vote_summary["werewolf_vote_total"])
        non_werewolf_vote_totals.append(vote_summary["non_werewolf_vote_total"])
        werewolf_vote_shares.append(vote_summary["werewolf_vote_share"])

    completed = summary.get("games_completed", 0)
    failed = summary.get("games_failed", 0)
    requested = summary.get("games_requested", len(game_summaries))

    row = {
        "experiment_id": experiment_id,
        "run_name": summary_path.stem,
        "summary_file": str(summary_path),
        "games_file": str(games_path) if games_path else "",
        "log_file": str(log_path) if log_path else "",
        "model": summary.get("model", ""),
        "model_mode": summary.get("model_mode", ""),
        "model_family": model_family,
        "persuasion_mode": persuasion_mode,
        "player_count": summary.get("player_count", ""),
        "discussion_depth": summary.get("discussion_depth", ""),
        "parallel_games": summary.get("parallel_games", ""),
        "games_requested": requested,
        "games_completed": completed,
        "games_failed": failed,
        "failure_rate": round(safe_div(failed, requested), 4),
        "werewolf_wins": summary.get("werewolf_wins", sum(wins)),
        "werewolf_win_rate": round(summary.get("werewolf_win_rate", safe_div(sum(wins), len(wins))), 4),
        "completed_game_rows": len(wins),
        "paired_detailed_games": len(detailed_games),
        "warning_count": warning_count,
        "warning_rate": round(safe_div(warning_count, len(detailed_games)), 4) if detailed_games else 0.0,
        "avg_total_turns": round(safe_mean(total_turns), 4),
        "avg_day_turns": round(safe_mean(day_turns), 4),
        "avg_night_turns": round(safe_mean(night_turns), 4),
        "avg_vote_turns": round(safe_mean(vote_turns), 4),
        "avg_utterance_words": round(safe_mean(utterance_words), 4),
        "avg_utterance_chars": round(safe_mean(utterance_chars), 4),
        "avg_werewolf_turn_share": round(safe_mean(werewolf_turn_shares), 4),
        "avg_vote_concentration": round(safe_mean(vote_concentrations), 4),
        "vote_tie_rate": round(safe_div(vote_ties, len(detailed_games)), 4) if detailed_games else 0.0,
        "avg_votes_on_werewolves": round(safe_mean(werewolf_vote_totals), 4),
        "avg_votes_on_non_werewolves": round(safe_mean(non_werewolf_vote_totals), 4),
        "avg_werewolf_vote_share": round(safe_mean(werewolf_vote_shares), 4),
    }
    return row, per_game_rows


def collect_experiments(results_dir):
    experiment_rows = []
    game_rows = []
    for path in sorted(results_dir.glob("*.json")):
        if "-games" in path.stem:
            continue
        if path.name.startswith("analysis_"):
            continue
        analyzed = analyze_experiment(path)
        if not analyzed:
            continue
        experiment_row, per_game_rows = analyzed
        experiment_rows.append(experiment_row)
        game_rows.extend(per_game_rows)
    return experiment_rows, game_rows


def group_rows(rows, group_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    output = []
    for group_value, group_rows_list in sorted(grouped.items()):
        output.append({
            group_key: group_value,
            "runs": len(group_rows_list),
            "mean_win_rate": round(safe_mean([row["werewolf_win_rate"] for row in group_rows_list]), 4),
            "median_win_rate": round(safe_median([row["werewolf_win_rate"] for row in group_rows_list]), 4),
            "mean_failure_rate": round(safe_mean([row["failure_rate"] for row in group_rows_list]), 4),
            "mean_warning_rate": round(safe_mean([row["warning_rate"] for row in group_rows_list]), 4),
            "mean_total_turns": round(safe_mean([row["avg_total_turns"] for row in group_rows_list]), 4),
            "mean_day_turns": round(safe_mean([row["avg_day_turns"] for row in group_rows_list]), 4),
            "mean_utterance_words": round(safe_mean([row["avg_utterance_words"] for row in group_rows_list]), 4),
            "mean_werewolf_turn_share": round(safe_mean([row["avg_werewolf_turn_share"] for row in group_rows_list]), 4),
        })
    return output


def build_model_mode_matrix(rows):
    models = sorted({row["model_family"] for row in rows})
    modes = ["baseline", "targeted", "structured", "personality_aware"]
    matrix = []
    for model in models:
        row = {"model_family": model}
        for mode in modes:
            matches = [
                experiment_row["werewolf_win_rate"]
                for experiment_row in rows
                if experiment_row["model_family"] == model and experiment_row["persuasion_mode"] == mode
            ]
            row[mode] = round(safe_mean(matches), 4) if matches else None
        matrix.append(row)
    return matrix


def build_metric_matrix(rows, metric_key, models=None, modes=None):
    models = models or sorted({row["model_family"] for row in rows})
    modes = modes or ["baseline", "targeted", "structured", "personality_aware"]
    matrix = []
    for model in models:
        row = {"model_family": model}
        for mode in modes:
            matches = [
                experiment_row[metric_key]
                for experiment_row in rows
                if experiment_row["model_family"] == model and experiment_row["persuasion_mode"] == mode
            ]
            row[mode] = round(safe_mean(matches), 4) if matches else None
        matrix.append(row)
    return matrix


def pearson_corr(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = safe_mean(xs)
    mean_y = safe_mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs)) * math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denominator == 0:
        return None
    return numerator / denominator


def build_overview_summary(experiment_rows):
    best_run = max(experiment_rows, key=lambda row: row["werewolf_win_rate"])
    worst_run = min(experiment_rows, key=lambda row: row["werewolf_win_rate"])
    model_groups = group_rows(experiment_rows, "model_family")
    mode_groups = group_rows(experiment_rows, "persuasion_mode")
    avg_turns = [row["avg_day_turns"] for row in experiment_rows]
    win_rates = [row["werewolf_win_rate"] for row in experiment_rows]
    werewolf_talk = [row["avg_werewolf_turn_share"] for row in experiment_rows]

    return {
        "experiment_count": len(experiment_rows),
        "models": sorted({row["model_family"] for row in experiment_rows}),
        "persuasion_modes": sorted({row["persuasion_mode"] for row in experiment_rows}),
        "best_run": {
            "experiment_id": best_run["experiment_id"],
            "win_rate": best_run["werewolf_win_rate"],
            "model_family": best_run["model_family"],
            "persuasion_mode": best_run["persuasion_mode"],
        },
        "worst_run": {
            "experiment_id": worst_run["experiment_id"],
            "win_rate": worst_run["werewolf_win_rate"],
            "model_family": worst_run["model_family"],
            "persuasion_mode": worst_run["persuasion_mode"],
        },
        "mean_win_rate": round(safe_mean(win_rates), 4),
        "median_win_rate": round(safe_median(win_rates), 4),
        "turn_winrate_correlation": pearson_corr(avg_turns, win_rates),
        "werewolf_talk_winrate_correlation": pearson_corr(werewolf_talk, win_rates),
        "by_model_family": model_groups,
        "by_persuasion_mode": mode_groups,
        "model_mode_matrix": build_model_mode_matrix(experiment_rows),
    }


def write_svg(path, width, height, elements):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        )
        f.write('<rect width="100%" height="100%" fill="white"/>')
        for element in elements:
            f.write(element)
        f.write("</svg>")


def svg_text(x, y, text, size=12, anchor="start", fill="#111827", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{escape_xml(text)}</text>'
    )


def svg_line(x1, y1, x2, y2, stroke="#d1d5db", width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"/>'


def svg_rect(x, y, width, height, fill, stroke="#9ca3af", stroke_width=1, rx=0):
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx}"/>'
    )


def svg_circle(cx, cy, r, fill, opacity=0.85):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" opacity="{opacity}"/>'


MODE_COLORS = {
    "baseline": "#94a3b8",
    "targeted": "#2563eb",
    "structured": "#7c3aed",
    "personality_aware": "#059669",
}

MODEL_COLORS = {
    "gpt5_nano": "#111827",
    "grok": "#dc2626",
    "qwen": "#ea580c",
    "gemini": "#16a34a",
    "unknown": "#6b7280",
}


def plot_experiment_win_rates(rows, output_dir):
    ordered = sorted(rows, key=lambda row: row["werewolf_win_rate"], reverse=True)
    width = 1240
    row_height = 26
    top = 64
    left = 320
    right = 48
    bottom = 56
    height = top + bottom + len(ordered) * row_height
    plot_w = width - left - right

    elements = [
        svg_text(width / 2, 30, "Werewolf Win Rate by Experiment", size=22, anchor="middle", weight="bold"),
        svg_text(width / 2, 52, "Colored by persuasion mode; labels include model family and run name.", size=12, anchor="middle", fill="#4b5563"),
        svg_line(left, top, left, height - bottom, stroke="#374151"),
    ]

    for tick in range(0, 101, 10):
        x = left + plot_w * tick / 100.0
        elements.append(svg_line(x, top, x, height - bottom))
        elements.append(svg_text(x, height - bottom + 20, f"{tick}%", size=11, anchor="middle", fill="#4b5563"))

    for index, row in enumerate(ordered):
        y = top + index * row_height
        bar_y = y + 5
        bar_h = row_height - 10
        bar_w = plot_w * row["werewolf_win_rate"]
        label = f"{row['model_family']} | {row['persuasion_mode']} | {row['run_name']}"
        elements.append(svg_text(left - 12, y + 17, label, size=11, anchor="end"))
        elements.append(svg_rect(left, bar_y, bar_w, bar_h, MODE_COLORS[row["persuasion_mode"]], stroke="none", rx=2))
        elements.append(svg_text(left + bar_w + 8, y + 17, f"{percent(row['werewolf_win_rate']):.1f}%", size=11))

    path = output_dir / "experiment_win_rates.svg"
    write_svg(path, width, height, elements)
    return path


def plot_grouped_bars(rows, key, title, subtitle, output_name, output_dir):
    width = 880
    height = 460
    left, right, top, bottom = 90, 40, 70, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    elements = [
        svg_text(width / 2, 30, title, size=22, anchor="middle", weight="bold"),
        svg_text(width / 2, 52, subtitle, size=12, anchor="middle", fill="#4b5563"),
        svg_line(left, top, left, top + plot_h, stroke="#374151"),
        svg_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#374151"),
    ]

    values = [row["mean_win_rate"] for row in rows]
    max_value = max(values) if values else 1.0
    bar_gap = 24
    bar_width = (plot_w - bar_gap * (len(rows) + 1)) / max(len(rows), 1)

    for tick in range(0, 101, 10):
        y = top + plot_h - plot_h * tick / 100.0
        elements.append(svg_line(left, y, left + plot_w, y))
        elements.append(svg_text(left - 10, y + 4, f"{tick}%", size=11, anchor="end", fill="#4b5563"))

    for index, row in enumerate(rows):
        x = left + bar_gap + index * (bar_width + bar_gap)
        h = plot_h * safe_div(percent(row["mean_win_rate"]), 100.0)
        y = top + plot_h - h
        fill = MODE_COLORS.get(row.get(key), MODEL_COLORS.get(row.get(key), "#2563eb"))
        elements.append(svg_rect(x, y, bar_width, h, fill, stroke="none", rx=2))
        elements.append(svg_text(x + bar_width / 2, top + plot_h + 20, row[key], size=11, anchor="middle"))
        elements.append(svg_text(x + bar_width / 2, y - 8, f"{percent(row['mean_win_rate']):.1f}%", size=11, anchor="middle"))

    path = output_dir / output_name
    write_svg(path, width, height, elements)
    return path


def plot_scatter(rows, x_key, y_key, title, subtitle, output_name, output_dir):
    width = 880
    height = 520
    left, right, top, bottom = 90, 40, 70, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [row[x_key] for row in rows]
    ys = [row[y_key] for row in rows]
    max_x = max(xs) if xs else 1.0
    max_y = max(ys) if ys else 1.0

    def scale_x(value):
        return left + plot_w * safe_div(value, max_x if max_x else 1.0)

    def scale_y(value):
        return top + plot_h - plot_h * safe_div(value, max_y if max_y else 1.0)

    elements = [
        svg_text(width / 2, 30, title, size=22, anchor="middle", weight="bold"),
        svg_text(width / 2, 52, subtitle, size=12, anchor="middle", fill="#4b5563"),
        svg_line(left, top, left, top + plot_h, stroke="#374151"),
        svg_line(left, top + plot_h, left + plot_w, top + plot_h, stroke="#374151"),
    ]

    for tick in range(0, 6):
        x_value = max_x * tick / 5.0
        x = scale_x(x_value)
        elements.append(svg_line(x, top, x, top + plot_h))
        elements.append(svg_text(x, top + plot_h + 20, f"{x_value:.1f}", size=11, anchor="middle", fill="#4b5563"))

    for tick in range(0, 11):
        y_value = max_y * tick / 10.0
        y = scale_y(y_value)
        elements.append(svg_line(left, y, left + plot_w, y))
        elements.append(svg_text(left - 10, y + 4, f"{percent(y_value):.0f}%", size=11, anchor="end", fill="#4b5563"))

    for row in rows:
        elements.append(
            svg_circle(
                scale_x(row[x_key]),
                scale_y(row[y_key]),
                6,
                MODEL_COLORS.get(row["model_family"], "#6b7280")
            )
        )

    legend_x = width - 150
    legend_y = 90
    for idx, family in enumerate(sorted({row["model_family"] for row in rows})):
        elements.append(svg_circle(legend_x, legend_y + idx * 24, 6, MODEL_COLORS.get(family, "#6b7280")))
        elements.append(svg_text(legend_x + 12, legend_y + 4 + idx * 24, family, size=12))

    path = output_dir / output_name
    write_svg(path, width, height, elements)
    return path


def plot_heatmap(matrix_rows, output_dir):
    modes = ["baseline", "targeted", "structured", "personality_aware"]
    width = 860
    height = 420
    left, top = 180, 90
    cell_w, cell_h = 140, 56

    values = [row[mode] for row in matrix_rows for mode in modes if row[mode] is not None]
    min_value = min(values) if values else 0.0
    max_value = max(values) if values else 1.0

    def color_for(value):
        if value is None:
            return "#f3f4f6"
        ratio = safe_div(value - min_value, (max_value - min_value) or 1.0)
        red = int(245 - 180 * ratio)
        green = int(247 - 90 * ratio)
        blue = int(250 - 220 * ratio)
        return f"rgb({red},{green},{blue})"

    elements = [
        svg_text(width / 2, 30, "Win Rate Heatmap: Model Family x Persuasion Mode", size=22, anchor="middle", weight="bold"),
        svg_text(width / 2, 52, "Cells show mean Werewolf win rate for the matching experiment slice.", size=12, anchor="middle", fill="#4b5563"),
    ]

    for col, mode in enumerate(modes):
        x = left + col * cell_w
        elements.append(svg_text(x + cell_w / 2, top - 14, mode, size=12, anchor="middle", weight="bold"))

    for row_idx, row in enumerate(matrix_rows):
        y = top + row_idx * cell_h
        elements.append(svg_text(left - 12, y + cell_h / 2 + 4, row["model_family"], size=12, anchor="end", weight="bold"))
        for col, mode in enumerate(modes):
            x = left + col * cell_w
            value = row[mode]
            elements.append(svg_rect(x, y, cell_w, cell_h, color_for(value), stroke="#d1d5db"))
            label = "n/a" if value is None else f"{percent(value):.1f}%"
            elements.append(svg_text(x + cell_w / 2, y + cell_h / 2 + 4, label, size=13, anchor="middle"))

    path = output_dir / "model_mode_heatmap.svg"
    write_svg(path, width, height, elements)
    return path


def write_text_summary(path, overview, experiment_rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Werewolf Results Analysis\n")
        f.write("=========================\n\n")
        f.write(f"Experiments analyzed: {overview['experiment_count']}\n")
        f.write(f"Mean Werewolf win rate: {percent(overview['mean_win_rate']):.2f}%\n")
        f.write(f"Median Werewolf win rate: {percent(overview['median_win_rate']):.2f}%\n")
        f.write(
            f"Correlation between average day turns and win rate: "
            f"{overview['turn_winrate_correlation']}\n"
        )
        f.write(
            f"Correlation between Werewolf talk share and win rate: "
            f"{overview['werewolf_talk_winrate_correlation']}\n\n"
        )
        f.write("Best run:\n")
        f.write(
            f"- {overview['best_run']['experiment_id']} "
            f"({percent(overview['best_run']['win_rate']):.2f}% win rate)\n"
        )
        f.write("Worst run:\n")
        f.write(
            f"- {overview['worst_run']['experiment_id']} "
            f"({percent(overview['worst_run']['win_rate']):.2f}% win rate)\n\n"
        )
        f.write("By model family:\n")
        for row in overview["by_model_family"]:
            f.write(
                f"- {row['model_family']}: mean win rate {percent(row['mean_win_rate']):.2f}% "
                f"across {row['runs']} runs\n"
            )
        f.write("\nBy persuasion mode:\n")
        for row in overview["by_persuasion_mode"]:
            f.write(
                f"- {row['persuasion_mode']}: mean win rate {percent(row['mean_win_rate']):.2f}% "
                f"across {row['runs']} runs\n"
            )
        f.write("\nPer-experiment snapshot:\n")
        for row in sorted(experiment_rows, key=lambda item: item["werewolf_win_rate"], reverse=True):
            f.write(
                f"- {row['run_name']}: {percent(row['werewolf_win_rate']):.2f}% win rate, "
                f"{percent(row['failure_rate']):.2f}% failure, "
                f"{row['avg_total_turns']:.1f} avg turns, "
                f"{row['avg_votes_on_werewolves']:.2f} avg votes on werewolves\n"
            )


def write_metric_tables_markdown(path, metric_tables):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Werewolf Results Metric Tables\n\n")
        f.write("Rows are model families and columns are persuasion modes.\n\n")
        for metric_name, metric_table in metric_tables:
            columns = [key for key in metric_table[0].keys() if key != "model_family"]
            f.write(f"## {metric_name}\n\n")
            f.write("| model_family | " + " | ".join(columns) + " |\n")
            f.write("|---|" + "|".join(["---"] * len(columns)) + "|\n")
            for row in metric_table:
                values = []
                for column in columns:
                    value = row[column]
                    values.append("n/a" if value is None else str(value))
                f.write("| " + row["model_family"] + " | " + " | ".join(values) + " |\n")
            f.write("\n")


def write_html_report(path, overview, experiment_rows, chart_files):
    best = overview["best_run"]
    worst = overview["worst_run"]
    top_rows = sorted(experiment_rows, key=lambda row: row["werewolf_win_rate"], reverse=True)

    def table_rows(rows):
        rendered = []
        for row in rows:
            rendered.append(
                "<tr>"
                f"<td>{escape_xml(row['run_name'])}</td>"
                f"<td>{escape_xml(row['model_family'])}</td>"
                f"<td>{escape_xml(row['persuasion_mode'])}</td>"
                f"<td>{percent(row['werewolf_win_rate']):.1f}%</td>"
                f"<td>{percent(row['failure_rate']):.1f}%</td>"
                f"<td>{row['avg_total_turns']:.1f}</td>"
                f"<td>{row['avg_werewolf_turn_share']:.2f}</td>"
                "</tr>"
            )
        return "\n".join(rendered)

    charts_html = "\n".join(
        f'<section class="chart"><h2>{escape_xml(chart.stem.replace("_", " ").title())}</h2><img src="{escape_xml(chart.name)}" alt="{escape_xml(chart.name)}"/></section>'
        for chart in chart_files
    )

    metric_tables_path = Path(path).with_name("metric_tables.md")
    metric_tables_note = metric_tables_path.name if metric_tables_path.exists() else "metric_tables.md"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Werewolf Results Analysis</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 32px;
      color: #111827;
      background: #ffffff;
      line-height: 1.5;
    }}
    h1, h2 {{
      margin-bottom: 8px;
    }}
    .lede {{
      color: #4b5563;
      max-width: 980px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 16px;
      margin: 24px 0 32px;
    }}
    .stat {{
      border: 1px solid #d1d5db;
      padding: 16px;
      border-radius: 6px;
    }}
    .stat .value {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 8px;
    }}
    .chart {{
      margin: 28px 0 36px;
    }}
    img {{
      width: 100%;
      max-width: 1100px;
      border: 1px solid #e5e7eb;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      max-width: 1100px;
      margin-top: 16px;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 10px 12px;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #f9fafb;
    }}
  </style>
</head>
<body>
  <h1>Werewolf Results Analysis</h1>
  <p class="lede">
    This report compares every experiment summary in <code>werewolf-results</code> and, when present,
    pairs it with the matching <code>-games.json</code> file to add dialogue-level metrics such as turn
    counts, utterance length, warning rate, vote concentration, and Werewolf speaking share.
  </p>
  <p class="lede">
    Wide model-by-mode tables are also available in <code>{escape_xml(metric_tables_note)}</code> and the
    companion CSV files in this directory.
  </p>

  <div class="grid">
    <div class="stat">
      <div>Experiments analyzed</div>
      <div class="value">{overview['experiment_count']}</div>
    </div>
    <div class="stat">
      <div>Mean Werewolf win rate</div>
      <div class="value">{percent(overview['mean_win_rate']):.1f}%</div>
    </div>
    <div class="stat">
      <div>Best run</div>
      <div class="value">{percent(best['win_rate']):.1f}%</div>
      <div>{escape_xml(best['experiment_id'])}</div>
    </div>
    <div class="stat">
      <div>Worst run</div>
      <div class="value">{percent(worst['win_rate']):.1f}%</div>
      <div>{escape_xml(worst['experiment_id'])}</div>
    </div>
  </div>

  {charts_html}

  <section>
    <h2>Experiment Table</h2>
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Model</th>
          <th>Mode</th>
          <th>Win Rate</th>
          <th>Failure Rate</th>
          <th>Avg Turns</th>
          <th>Werewolf Talk Share</th>
        </tr>
      </thead>
      <tbody>
        {table_rows(top_rows)}
      </tbody>
    </table>
  </section>
</body>
</html>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def ensure_non_empty(experiment_rows):
    if not experiment_rows:
        raise SystemExit("No experiment summary JSON files were found in the results directory.")


def main():
    parser = argparse.ArgumentParser(description="Analyze all experiment outputs in werewolf-results.")
    parser.add_argument(
        "--results-dir",
        default="werewolf-results",
        help="Directory containing summary JSONs, paired -games.json files, and logs.",
    )
    parser.add_argument(
        "--output-dir",
        default="werewolf-results/analysis",
        help="Directory where analysis tables, JSON, SVGs, and HTML report will be written.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_rows, game_rows = collect_experiments(results_dir)
    ensure_non_empty(experiment_rows)

    overview = build_overview_summary(experiment_rows)
    by_model = group_rows(experiment_rows, "model_family")
    by_mode = group_rows(experiment_rows, "persuasion_mode")
    model_mode_matrix = build_model_mode_matrix(experiment_rows)
    metric_specs = [
        ("win_rate_matrix", "werewolf_win_rate"),
        ("avg_votes_on_werewolves_matrix", "avg_votes_on_werewolves"),
        ("avg_werewolf_vote_share_matrix", "avg_werewolf_vote_share"),
        ("failure_rate_matrix", "failure_rate"),
        ("warning_rate_matrix", "warning_rate"),
        ("avg_total_turns_matrix", "avg_total_turns"),
        ("avg_utterance_words_matrix", "avg_utterance_words"),
        ("avg_werewolf_turn_share_matrix", "avg_werewolf_turn_share"),
        ("avg_vote_concentration_matrix", "avg_vote_concentration"),
    ]
    metric_tables = [
        (name, build_metric_matrix(experiment_rows, metric_key))
        for name, metric_key in metric_specs
    ]

    experiment_csv_fields = [
        "experiment_id", "run_name", "summary_file", "games_file", "log_file", "model", "model_mode",
        "model_family", "persuasion_mode", "player_count", "discussion_depth", "parallel_games",
        "games_requested", "games_completed", "games_failed", "failure_rate", "werewolf_wins",
        "werewolf_win_rate", "completed_game_rows", "paired_detailed_games", "warning_count",
        "warning_rate", "avg_total_turns", "avg_day_turns", "avg_night_turns", "avg_vote_turns",
        "avg_utterance_words", "avg_utterance_chars", "avg_werewolf_turn_share",
        "avg_vote_concentration", "vote_tie_rate", "avg_votes_on_werewolves",
        "avg_votes_on_non_werewolves", "avg_werewolf_vote_share",
    ]
    game_csv_fields = [
        "experiment_id", "game_id", "model_family", "persuasion_mode", "player_count", "warning",
        "total_turns", "day_turns", "night_turns", "vote_turns", "distinct_speakers",
        "werewolf_turns", "werewolf_turn_share", "avg_utterance_words", "avg_utterance_chars",
        "median_utterance_words", "vote_top_count", "vote_concentration", "vote_tie",
    ]

    write_csv(output_dir / "experiment_metrics.csv", experiment_rows, experiment_csv_fields)
    write_csv(output_dir / "game_metrics.csv", game_rows, game_csv_fields)
    write_csv(output_dir / "model_family_summary.csv", by_model, list(by_model[0].keys()))
    write_csv(output_dir / "persuasion_mode_summary.csv", by_mode, list(by_mode[0].keys()))
    write_csv(output_dir / "model_mode_matrix.csv", model_mode_matrix, list(model_mode_matrix[0].keys()))
    for name, table in metric_tables:
        write_csv(output_dir / f"{name}.csv", table, list(table[0].keys()))

    write_json(output_dir / "experiment_metrics.json", experiment_rows)
    write_json(output_dir / "game_metrics.json", game_rows)
    write_json(output_dir / "overview_summary.json", overview)
    write_json(
        output_dir / "metric_tables.json",
        {name: table for name, table in metric_tables},
    )

    chart_files = [
        plot_experiment_win_rates(experiment_rows, output_dir),
        plot_grouped_bars(
            by_mode,
            "persuasion_mode",
            "Average Werewolf Win Rate by Persuasion Mode",
            "Means are computed across all experiment runs in the directory.",
            "mode_win_rates.svg",
            output_dir,
        ),
        plot_grouped_bars(
            by_model,
            "model_family",
            "Average Werewolf Win Rate by Model Family",
            "Comparing GPT-5 nano, Grok, Qwen, and Gemini result sets.",
            "model_win_rates.svg",
            output_dir,
        ),
        plot_scatter(
            experiment_rows,
            "avg_day_turns",
            "werewolf_win_rate",
            "Average Day Turns vs Werewolf Win Rate",
            "Each point is one experiment; color denotes model family.",
            "day_turns_vs_winrate.svg",
            output_dir,
        ),
        plot_scatter(
            experiment_rows,
            "avg_werewolf_turn_share",
            "werewolf_win_rate",
            "Werewolf Talk Share vs Werewolf Win Rate",
            "Do more verbally dominant Werewolves actually win more often?",
            "werewolf_talk_share_vs_winrate.svg",
            output_dir,
        ),
        plot_heatmap(model_mode_matrix, output_dir),
    ]

    write_text_summary(output_dir / "summary.txt", overview, experiment_rows)
    write_metric_tables_markdown(output_dir / "metric_tables.md", metric_tables)
    write_html_report(output_dir / "report.html", overview, experiment_rows, chart_files)

    print(f"Analyzed {len(experiment_rows)} experiments from {results_dir}")
    print(f"Wrote analysis bundle to {output_dir}")


if __name__ == "__main__":
    main()
