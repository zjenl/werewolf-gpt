import json
import os
from collections import Counter

DATA_FOLDER = "."
PRE_OUTPUT = "filtered_merged_games.json"

def merge_consecutive_turns(dialogue):
    if not dialogue:
        return []

    merged = []
    current = {
        "speaker": dialogue[0].get("speaker"),
        "utterance_parts": [dialogue[0].get("utterance", "")],
        "annotation": list(dialogue[0].get("annotation", [])),
    }

    for turn in dialogue[1:]:
        speaker = turn.get("speaker")
        utterance = turn.get("utterance", "")
        annotation = turn.get("annotation", [])

        if speaker == current["speaker"]:
            current["utterance_parts"].append(utterance)
            current["annotation"].extend(annotation)
        else:
            merged.append({
                "speaker": current["speaker"],
                "utterance": " <TURN_SPLIT> ".join(current["utterance_parts"]),
                "annotation": list(dict.fromkeys(current["annotation"]))
            })
            current = {
                "speaker": speaker,
                "utterance_parts": [utterance],
                "annotation": list(annotation),
            }

    merged.append({
        "speaker": current["speaker"],
        "utterance": " <TURN_SPLIT> ".join(current["utterance_parts"]),
        "annotation": list(dict.fromkeys(current["annotation"]))
    })

    return merged


all_games = []

for filename in os.listdir(DATA_FOLDER):
    if not filename.endswith(".json"):
        continue

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    for game in data:
        start_roles = game["startRoles"]
        end_roles = game["endRoles"]
        votes = game["votingOutcome"]

        # --- stable werewolf
        stable_werewolves = [
            i for i in range(len(start_roles))
            if start_roles[i] == "Werewolf" and end_roles[i] == "Werewolf"
        ]
        if not stable_werewolves:
            continue

        # --- voting
        valid_votes = [v for v in votes if isinstance(v, int)]
        if not valid_votes:
            continue

        vote_counts = Counter(v - 1 for v in valid_votes)
        max_votes = max(vote_counts.values())
        most_voted = [p for p, c in vote_counts.items() if c == max_votes]

        if not any(w not in most_voted for w in stable_werewolves):
            continue

        merged_dialogue = merge_consecutive_turns(game["Dialogue"])
        if not merged_dialogue:
            continue

        game["Dialogue"] = merged_dialogue
        all_games.append(game)

# --- save BEFORE labeling
with open(PRE_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(all_games, f, indent=2, ensure_ascii=False)

print(f"Saved pre-label dataset: {PRE_OUTPUT} ({len(all_games)} games)")