import json
from copy import deepcopy
from openai import OpenAI

client = OpenAI()

INPUT_FILE = "filtered_merged_games.json"
OUTPUT_FILE = "filtered_labeled_games.json"

MODEL_NAME = "gpt-4.1"

SYSTEM_PROMPT = """
Label persuasion target for each merged turn in a One Night Ultimate Werewolf game.

You MUST:
- Label EVERY merged turn
- Use ONLY allowed labels
- Follow schema exactly

Target definition:
The primarily player that the speaker is trying to influence. This can be trying to gain
trust from someone, trying to convince someone to vote out someone else, etc.
A positive effect is when the targeted player is less likely to vote out the speaker and 
vice versa. For example, if James accuses Elliot, Elliot 
is the targeted player with a negative effect as Elliot is less likely to trust James. 
However, if the accusation is right, the effect is positive as the player is more likely 
to trust the speaker. If James defends himself, Group is the targeted player with 
a positive effect as Group is more likely to trust James.
If there is no target or the persuasive effect is little, label the effect as "None".

Allowed labels for targeted_player:
- any player name
- "Group"
- "None"

Allowed labels for effect:
- "Positive"
- "Negative"
- "None"
"""

# STRICT SCHEMA
LABEL_SCHEMA = {
    "name": "target_labels",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "merged_turn_index": {"type": "integer"},
                        "targeted_player": {"type": "string"},
                        "effect": {"type": "string"}
                    },
                    "required": ["merged_turn_index", "targeted_player", "effect"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["labels"],
        "additionalProperties": False
    }
}


def validate_labels(labels_by_index, num_turns, player_names):
    allowed_targeted_players = set(player_names) | {"Group", "None"}
    allowed_effects = {"Positive", "Negative", "None"}

    # check all indices present
    if set(labels_by_index.keys()) != set(range(num_turns)):
        return False

    # check label validity
    for item in labels_by_index.values():
        if item["targeted_player"] not in allowed_targeted_players:
            return False
        if item["effect"] not in allowed_effects:
            return False

    return True


def label_game_strict(game, max_retries=3):
    player_names = game["playerNames"]
    dialogue = game["Dialogue"]

    payload = {
        "playerNames": player_names,
        "dialogue": [
            {
                "merged_turn_index": i,
                "speaker": t["speaker"],
                "utterance": t["utterance"],
                "annotation": t["annotation"]
            }
            for i, t in enumerate(dialogue)
        ]
    }

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": LABEL_SCHEMA
                }
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            labels_by_index = {
                item["merged_turn_index"]: item
                for item in parsed["labels"]
            }

            if validate_labels(labels_by_index, len(dialogue), player_names):
                # attach labels
                for i, turn in enumerate(dialogue):
                    label = labels_by_index[i]
                    turn["targeted_player"] = label["targeted_player"]
                    turn["effect"] = label["effect"]
                return game

        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)

    # fallback (never crash pipeline)
    print("fallback to None")
    for turn in game["Dialogue"]:
        turn["targeted_player"] = "None"
        turn["effect"] = "none"
    return game


# --- load preprocessed dataset
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    games = json.load(f)

labeled_games = []

for g in games:
    labeled_games.append(label_game_strict(deepcopy(g)))

# --- save AFTER labeling
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(labeled_games, f, indent=2, ensure_ascii=False)

print(f"Saved labeled dataset: {OUTPUT_FILE}")