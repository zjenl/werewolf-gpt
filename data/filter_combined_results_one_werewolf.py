import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def count_summary_werewolves(game):
    return sum(1 for player in (game.get("players") or []) if player.get("card") == "Werewolf")


def count_detailed_werewolves(game):
    return sum(1 for role in (game.get("startRoles") or []) if role == "Werewolf")


def filter_pair(summary_path, detailed_path):
    summary_payload = load_json(summary_path)
    detailed_games = load_json(detailed_path)

    summary_games = [game for game in (summary_payload.get("games") or []) if game.get("status") == "completed"]
    if len(summary_games) != len(detailed_games):
        raise ValueError(
            f"Mismatched lengths for {summary_path.name} ({len(summary_games)}) "
            f"and {detailed_path.name} ({len(detailed_games)})"
        )

    kept_summary_games = []
    kept_detailed_games = []

    for summary_game, detailed_game in zip(summary_games, detailed_games):
        if count_summary_werewolves(summary_game) == 1 and count_detailed_werewolves(detailed_game) == 1:
            kept_summary_games.append(summary_game)
            kept_detailed_games.append(detailed_game)

    werewolf_wins = sum(1 for game in kept_summary_games if game.get("werewolf_win"))
    filtered_count = len(kept_summary_games)

    summary_payload["games"] = kept_summary_games
    summary = summary_payload["summary"]
    summary["games_requested"] = filtered_count
    summary["games_completed"] = filtered_count
    summary["games_failed"] = 0
    summary["werewolf_wins"] = werewolf_wins
    summary["werewolf_win_rate"] = (werewolf_wins / filtered_count) if filtered_count else 0.0

    summary_payload["one_werewolf_filter"] = {
        "applied": True,
        "original_game_count": len(summary_games),
        "kept_game_count": filtered_count,
        "removed_game_count": len(summary_games) - filtered_count,
    }

    write_json(summary_path, summary_payload)
    write_json(detailed_path, kept_detailed_games)

    return {
        "summary_file": summary_path.name,
        "detailed_file": detailed_path.name,
        "original_games": len(summary_games),
        "kept_games": filtered_count,
        "removed_games": len(summary_games) - filtered_count,
        "werewolf_win_rate": summary["werewolf_win_rate"],
    }


def main():
    parser = argparse.ArgumentParser(description="Filter merged result files in place to keep only games with exactly one player Werewolf.")
    parser.add_argument(
        "--results-dir",
        default="werewolf-results",
        help="Directory containing merged summary files like gemini-targeted.json and gemini-targeted-game.json.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    target_stems = [
        "gpt5-nano-normal",
        "gpt5-nano-targeted",
        "gpt5-nano-prior",
        "gemini-normal",
        "gemini-targeted",
        "gemini-prior",
        "qwen-normal",
        "qwen-targeted",
        "qwen-prior",
    ]
    manifest = []

    for stem in target_stems:
        summary_path = results_dir / f"{stem}.json"
        detailed_path = summary_path.with_name(f"{summary_path.stem}-game.json")
        if not summary_path.exists() or not detailed_path.exists():
            continue

        result = filter_pair(summary_path, detailed_path)
        manifest.append(result)
        print(
            f"Filtered {summary_path.name}: kept {result['kept_games']} / {result['original_games']} "
            f"games (removed {result['removed_games']})"
        )

    write_json(results_dir / "combined_one_werewolf_filter_manifest.json", manifest)
    print(f"Wrote combined_one_werewolf_filter_manifest.json with {len(manifest)} entries")


if __name__ == "__main__":
    main()
