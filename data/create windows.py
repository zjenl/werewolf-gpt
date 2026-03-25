import json

INPUT_FILE = "filtered_labeled_games.json"
OUTPUT_FILE = "filtered_windowed_games.json"

NO_TARGET_LABELS = {"None", None, ""}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    games = json.load(f)

output = []

for game in games:
    dialogue = game.get("Dialogue", [])

    # 1) remove all turns with no target
    filtered_dialogue = []
    for turn in dialogue:
        target = turn.get("targeted_player")
        if target in NO_TARGET_LABELS:
            continue

        filtered_dialogue.append({
            "speaker": turn.get("speaker"),
            "utterance": turn.get("utterance"),
            "annotation": turn.get("annotation", []),
            "targeted_player": target,
            "effect": turn.get("effect")
        })

    # 2) build windows of consecutive same targeted_player
    windows = []
    current_window = None

    for turn in filtered_dialogue:
        target = turn["targeted_player"]

        clean_turn = {
            "speaker": turn["speaker"],
            "utterance": turn["utterance"],
            "annotation": turn["annotation"],
            "effect": turn["effect"]
        }

        if current_window is None:
            current_window = {
                "window_id": 0,
                "targeted_player": target,
                "Dialogue": [clean_turn]
            }
        elif target == current_window["targeted_player"]:
            current_window["Dialogue"].append(clean_turn)
        else:
            windows.append(current_window)
            current_window = {
                "window_id": len(windows),
                "targeted_player": target,
                "Dialogue": [clean_turn]
            }

    if current_window is not None:
        windows.append(current_window)

    output.append({
        "Game_ID": game.get("Game_ID"),
        "windows": windows,
        "playerNames": game.get("playerNames", []),
        "startRoles": game.get("startRoles", []),
        "endRoles": game.get("endRoles", []),
        "votingOutcome": game.get("votingOutcome", []),
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Saved {len(output)} games to {OUTPUT_FILE}")