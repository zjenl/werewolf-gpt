import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path


OUTPUT_MODELS = {"gpt5_nano": "gpt5-nano", "gemini": "gemini", "qwen": "qwen"}
OUTPUT_MODES = ["normal", "targeted", "prior"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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


def paired_games_path(summary_path, summary):
    synthesized = summary.get("synthesized_from")
    if synthesized:
        candidate = summary_path.with_name(str(synthesized))
        if candidate.exists():
            return candidate

    stem = summary_path.stem
    candidates = []
    parts = stem.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        candidates.append(summary_path.with_name(f"{parts[0]}-games-{parts[1]}.json"))
    candidates.extend(
        [
            summary_path.with_name(f"{stem}-games.json"),
            summary_path.with_name(f"{stem}_game.json"),
            summary_path.with_name(f"{stem}-game.json"),
        ]
    )
    if stem.endswith("-new"):
        candidates.append(summary_path.with_name(f"{stem[:-4]}-games-new.json"))
        candidates.append(summary_path.with_name(f"{stem}.json").with_name(f"{stem}-game.json"))
    if stem.endswith("_new"):
        candidates.append(summary_path.with_name(f"{stem}_game.json"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def is_combined_output_name(path):
    stem = path.stem
    for model_slug in OUTPUT_MODELS.values():
        for mode in OUTPUT_MODES:
            if stem in {f"{model_slug}-{mode}", f"{model_slug}-{mode}-game"}:
                return True
    return False


def collect_grouped_runs(results_dir):
    grouped = {}
    for summary_path in sorted(results_dir.glob("*.json")):
        if summary_path.parent.name == "analysis":
            continue
        if "-games" in summary_path.stem or summary_path.stem.endswith("-game"):
            continue
        if summary_path.stem.endswith("_game"):
            continue
        if is_combined_output_name(summary_path):
            continue

        try:
            payload = load_json(summary_path)
        except Exception:
            continue
        if not isinstance(payload, dict) or "summary" not in payload or "games" not in payload:
            continue

        summary = payload["summary"]
        model_family = infer_model_family(summary)
        mode = infer_mode(summary)
        if model_family not in OUTPUT_MODELS or mode not in OUTPUT_MODES:
            continue

        key = (model_family, mode)
        grouped.setdefault(key, []).append(
            {
                "summary_path": summary_path,
                "games_path": paired_games_path(summary_path, summary),
                "payload": payload,
            }
        )
    return grouped


def combine_group(model_family, mode, runs):
    first_summary = deepcopy(runs[0]["payload"]["summary"])
    combined_summary_games = []
    combined_detailed_games = []

    game_number = 1
    for run in runs:
        summary_file_name = run["summary_path"].name
        for game in run["payload"].get("games", []):
            combined_game = deepcopy(game)
            combined_game["source_summary_file"] = summary_file_name
            combined_game["game_number"] = game_number
            combined_summary_games.append(combined_game)
            game_number += 1

        if run["games_path"] and run["games_path"].exists():
            detailed_games = load_json(run["games_path"])
            for detailed_game in detailed_games:
                combined_detailed_game = deepcopy(detailed_game)
                combined_detailed_game["source_summary_file"] = summary_file_name
                combined_detailed_game["source_games_file"] = run["games_path"].name
                combined_detailed_games.append(combined_detailed_game)

    completed_games = [game for game in combined_summary_games if game.get("status") == "completed"]
    failed_games = [game for game in combined_summary_games if game.get("status") != "completed"]
    werewolf_wins = sum(1 for game in completed_games if game.get("werewolf_win"))

    first_summary["games_requested"] = len(combined_summary_games)
    first_summary["games_completed"] = len(completed_games)
    first_summary["games_failed"] = len(failed_games)
    first_summary["werewolf_wins"] = werewolf_wins
    first_summary["werewolf_win_rate"] = (werewolf_wins / len(completed_games)) if completed_games else 0.0
    first_summary["generated_at"] = datetime.now().astimezone().isoformat()
    first_summary["source_runs"] = [run["summary_path"].name for run in runs]
    first_summary["combined_from_count"] = len(runs)

    return {
        "summary": first_summary,
        "games": combined_summary_games,
    }, combined_detailed_games


def main():
    parser = argparse.ArgumentParser(description="Combine result JSON files by model family and persuasion mode.")
    parser.add_argument(
        "--results-dir",
        default="werewolf-results",
        help="Directory containing experiment summary JSON files and paired game JSON files.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    grouped = collect_grouped_runs(results_dir)

    outputs = []
    for (model_family, mode), runs in sorted(grouped.items()):
        summary_payload, detailed_games = combine_group(model_family, mode, runs)
        model_slug = OUTPUT_MODELS[model_family]
        summary_output_path = results_dir / f"{model_slug}-{mode}.json"
        detailed_output_path = results_dir / f"{model_slug}-{mode}-game.json"
        write_json(summary_output_path, summary_payload)
        write_json(detailed_output_path, detailed_games)
        outputs.append(
            {
                "summary_file": summary_output_path.name,
                "detailed_file": detailed_output_path.name,
                "source_runs": summary_payload["summary"]["source_runs"],
                "games_requested": summary_payload["summary"]["games_requested"],
                "games_completed": summary_payload["summary"]["games_completed"],
                "games_failed": summary_payload["summary"]["games_failed"],
            }
        )
        print(
            f"Wrote {summary_output_path.name} and {detailed_output_path.name} "
            f"from {len(runs)} runs ({summary_payload['summary']['games_completed']} completed games)"
        )

    write_json(results_dir / "combined_results_manifest.json", outputs)
    print(f"Wrote combined_results_manifest.json with {len(outputs)} combined outputs")


if __name__ == "__main__":
    main()
