import argparse
import json
import re
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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


def extract_game_number(game):
    for key in ("game_number", "Game_ID", "EG_ID"):
        value = game.get(key)
        if value is None:
            continue
        if isinstance(value, int):
            return value
        match = re.search(r"(\d+)$", str(value))
        if match:
            return int(match.group(1))
    return None


def count_player_werewolves_in_summary_game(game):
    return sum(1 for player in (game.get("players") or []) if player.get("card") == "Werewolf")


def count_player_werewolves_in_detailed_game(game):
    return sum(1 for role in (game.get("startRoles") or []) if role == "Werewolf")


def clean_experiment(summary_path, output_dir):
    payload = load_json(summary_path)
    if not isinstance(payload, dict) or "summary" not in payload or "games" not in payload:
        return None

    games_path = infer_games_path(summary_path)
    detailed_games = load_json(games_path) if games_path else []
    detailed_by_number = {}
    for idx, game in enumerate(detailed_games, start=1):
        game_number = extract_game_number(game) or idx
        detailed_by_number[game_number] = game

    kept_summary_games = []
    kept_detailed_games = []
    skipped_completed = 0

    for game in payload.get("games", []):
        if game.get("status") != "completed":
            continue

        game_number = extract_game_number(game)
        detailed_game = detailed_by_number.get(game_number)
        summary_count = count_player_werewolves_in_summary_game(game)
        detailed_count = count_player_werewolves_in_detailed_game(detailed_game) if detailed_game else None

        keep = summary_count == 1
        if detailed_game is not None:
            keep = keep and detailed_count == 1

        if keep:
            kept_summary_games.append(game)
            if detailed_game is not None:
                kept_detailed_games.append(detailed_game)
        else:
            skipped_completed += 1

    werewolf_wins = sum(1 for game in kept_summary_games if game.get("werewolf_win"))
    kept_completed = len(kept_summary_games)

    cleaned_payload = {
        "summary": dict(payload["summary"]),
        "games": kept_summary_games,
        "cleaning": {
            "filter": "exactly_one_player_werewolf",
            "original_games_requested": payload["summary"].get("games_requested", len(payload.get("games", []))),
            "original_games_completed": payload["summary"].get("games_completed", 0),
            "original_games_failed": payload["summary"].get("games_failed", 0),
            "kept_completed_games": kept_completed,
            "removed_completed_games": skipped_completed,
        },
    }
    cleaned_payload["summary"]["games_requested"] = kept_completed
    cleaned_payload["summary"]["games_completed"] = kept_completed
    cleaned_payload["summary"]["games_failed"] = 0
    cleaned_payload["summary"]["werewolf_wins"] = werewolf_wins
    cleaned_payload["summary"]["werewolf_win_rate"] = (werewolf_wins / kept_completed) if kept_completed else 0.0

    summary_output_path = output_dir / summary_path.name
    write_json(summary_output_path, cleaned_payload)

    if games_path:
        games_output_path = output_dir / games_path.name
        write_json(games_output_path, kept_detailed_games)

    return {
        "summary_file": summary_path.name,
        "games_file": games_path.name if games_path else None,
        "kept_completed_games": kept_completed,
        "removed_completed_games": skipped_completed,
    }


def main():
    parser = argparse.ArgumentParser(description="Filter experiment result files to completed games with exactly one player Werewolf.")
    parser.add_argument(
        "--results-dir",
        default="werewolf-results",
        help="Directory containing original experiment summary JSON files and paired -games.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default="werewolf-results/one_werewolf_only",
        help="Directory where cleaned summary and -games.json files will be written.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_runs = []
    for summary_path in sorted(results_dir.glob("*.json")):
        if "-games" in summary_path.stem:
            continue
        if summary_path.parent.name == "analysis":
            continue
        cleaned = clean_experiment(summary_path, output_dir)
        if cleaned:
            cleaned_runs.append(cleaned)

    write_json(output_dir / "cleaning_summary.json", cleaned_runs)
    print(f"Cleaned {len(cleaned_runs)} experiment summaries into {output_dir}")


if __name__ == "__main__":
    main()
