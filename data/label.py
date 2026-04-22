import json
from collections import Counter
from copy import deepcopy

from openai import OpenAI

client = OpenAI()

INPUT_FILE = "filtered_merged_games.json"
OUTPUT_FILE = "filtered_labeled_games.json"

MODEL_NAME = "gpt-5-nano"
NO_TARGET_LABELS = {"None", None, ""}

SYSTEM_PROMPT = """
Label Werewolf persuasion targets and discussion leaders in a One Night Ultimate Werewolf game.

You MUST:
- Label EVERY merged turn with targeted_player
- Label EVERY persuasion window with discussion_leader
- Label EVERY player's Big Five personality profile
- Use ONLY allowed labels
- Follow schema exactly

Target definition:
Only label the person or group that the Werewolf speaker is trying to persuade in order
to gain their trust or belief. The target can be a specific player or the whole table.

If the speaker is not a Werewolf, targeted_player MUST be "None".
If a Werewolf speaker is not clearly trying to gain trust from anyone, targeted_player
MUST be "None".
Use a specific player name when the Werewolf speaker is mainly trying to influence one
player. Use "Group" only for an explicit broad appeal, role claim, or defense aimed at
the whole table. Do not use "Group" as a catch-all for general conversation, narration,
questions, accusations, or weak persuasive signals.

Persuasion window definition:
After assigning targeted_player to every turn, create persuasion
windows by grouping all turns in dialogue order when they have the same
consecutive targeted_player (turns labeled "None" in between same targeted_player are also
considered part of the same window). Window IDs must start at 0 and increase by 1.

Discussion leader definition:
For each persuasion window, label discussion_leader as the player who is leading
everyone else in that window. This is not necessarily the first speaker, the loudest
speaker, or the targeted player. Choose the person who most clearly steers the
conversation, frames the issue, sets the agenda, asks the questions others respond to,
or influences how other players interpret the situation.
If no one clearly leads a window, use "None".

Allowed labels for targeted_player:
- any player name
- "Group"
- "None"

Allowed labels for discussion_leader:
- any player name
- "None"

Big Five personality profile:
For each player, infer their personality from their dialogue and table behavior. Label
each trait as exactly one of: "low", "moderate", "high".

Traits:
- openness
- conscientiousness
- extraversion
- agreeableness
- neuroticism
"""

LABEL_SCHEMA = {
    "name": "game_and_window_labels",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "turn_labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "merged_turn_index": {"type": "integer"},
                        "targeted_player": {"type": "string"}
                    },
                    "required": ["merged_turn_index", "targeted_player"],
                    "additionalProperties": False
                }
            },
            "window_labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "window_id": {"type": "integer"},
                        "discussion_leader": {"type": "string"}
                    },
                    "required": ["window_id", "discussion_leader"],
                    "additionalProperties": False
                }
            },
            "personality_profiles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "player": {"type": "string"},
                        "openness": {"type": "string"},
                        "conscientiousness": {"type": "string"},
                        "extraversion": {"type": "string"},
                        "agreeableness": {"type": "string"},
                        "neuroticism": {"type": "string"}
                    },
                    "required": [
                        "player",
                        "openness",
                        "conscientiousness",
                        "extraversion",
                        "agreeableness",
                        "neuroticism"
                    ],
                    "additionalProperties": False
                }
            }
        },
        "required": ["turn_labels", "window_labels", "personality_profiles"],
        "additionalProperties": False
    }
}


def validate_turn_labels(labels_by_index, game, player_names):
    allowed_targets = set(player_names) | {"Group", "None"}
    end_roles_by_player = get_role_by_player(game, "endRoles")
    start_roles_by_player = get_role_by_player(game, "startRoles")
    dialogue = game.get("Dialogue", [])

    if set(labels_by_index.keys()) != set(range(len(dialogue))):
        return False

    target_counts = Counter()
    for index, item in labels_by_index.items():
        if item["targeted_player"] not in allowed_targets:
            return False
        speaker = dialogue[index].get("speaker")
        speaker_role = end_roles_by_player.get(speaker) or start_roles_by_player.get(speaker)
        if speaker_role != "Werewolf" and item["targeted_player"] != "None":
            return False
        if item["targeted_player"] not in NO_TARGET_LABELS:
            target_counts[item["targeted_player"]] += 1

    if not target_counts:
        return True

    non_none_count = sum(target_counts.values())
    most_common_target, most_common_count = target_counts.most_common(1)[0]
    if non_none_count >= 3 and most_common_target == "Group" and most_common_count / non_none_count > 0.5:
        return False

    return True


def validate_window_labels(labels_by_window_id, windows, player_names):
    allowed_leaders = set(player_names) | {"None"}
    expected_window_ids = {window["window_id"] for window in windows}

    if set(labels_by_window_id.keys()) != expected_window_ids:
        return False

    for item in labels_by_window_id.values():
        if item["discussion_leader"] not in allowed_leaders:
            return False

    return True


def validate_personality_profiles(profiles_by_player, player_names):
    allowed_labels = {"low", "moderate", "high"}
    trait_names = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism"
    ]

    if set(profiles_by_player.keys()) != set(player_names):
        return False

    for profile in profiles_by_player.values():
        for trait_name in trait_names:
            value = profile[trait_name]
            if value not in allowed_labels:
                return False

    return True


def get_role_by_player(game, role_key):
    player_names = game["playerNames"]
    roles = game.get(role_key, [])
    return {
        player_name: roles[i] if i < len(roles) else None
        for i, player_name in enumerate(player_names)
    }


def build_game_windows(game):
    def clean_turn(turn):
        return {
            "speaker": turn.get("speaker"),
            "utterance": turn.get("utterance"),
            "annotation": turn.get("annotation", [])
        }

    windows = []
    current_window = None
    pending_none_turns = []

    for turn in game.get("Dialogue", []):
        target = turn.get("targeted_player")

        if target in NO_TARGET_LABELS:
            if current_window is not None:
                pending_none_turns.append(clean_turn(turn))
            continue

        cleaned_turn = clean_turn(turn)

        if current_window is None:
            current_window = {
                "window_id": 0,
                "targeted_player": target,
                "Dialogue": [cleaned_turn]
            }
        elif target == current_window["targeted_player"]:
            current_window["Dialogue"].extend(pending_none_turns)
            current_window["Dialogue"].append(cleaned_turn)
        else:
            windows.append(current_window)
            current_window = {
                "window_id": len(windows),
                "targeted_player": target,
                "Dialogue": [cleaned_turn]
            }

        pending_none_turns = []

    if current_window is not None:
        windows.append(current_window)

    return {
        "Game_ID": game.get("Game_ID"),
        "Dialogue": game.get("Dialogue", []),
        "windows": windows,
        "playerNames": game.get("playerNames", []),
        "startRoles": game.get("startRoles", []),
        "endRoles": game.get("endRoles", []),
        "votingOutcome": game.get("votingOutcome", []),
    }


def get_kol(windows):
    leader_counts = Counter(
        window.get("discussion_leader")
        for window in windows
        if window.get("discussion_leader") not in {None, "None", ""}
    )
    if not leader_counts:
        return "None"

    first_seen = {}
    for index, window in enumerate(windows):
        leader = window.get("discussion_leader")
        if leader not in first_seen:
            first_seen[leader] = index

    return max(
        leader_counts,
        key=lambda leader: (leader_counts[leader], -first_seen[leader])
    )


def build_prompt_payload(game):
    player_names = game["playerNames"]
    start_roles_by_player = get_role_by_player(game, "startRoles")
    end_roles_by_player = get_role_by_player(game, "endRoles")

    return {
        "playerNames": player_names,
        "playerRoles": {
            player_name: {
                "startRole": start_roles_by_player.get(player_name),
                "endRole": end_roles_by_player.get(player_name),
            }
            for player_name in player_names
        },
        "dialogue": [
            {
                "merged_turn_index": i,
                "speaker": turn["speaker"],
                "speaker_role": end_roles_by_player.get(turn["speaker"]) or start_roles_by_player.get(turn["speaker"]),
                "utterance": turn["utterance"],
                "annotation": turn["annotation"]
            }
            for i, turn in enumerate(game["Dialogue"])
        ]
    }


def attach_turn_labels(game, labels_by_index):
    for i, turn in enumerate(game["Dialogue"]):
        turn["targeted_player"] = labels_by_index[i]["targeted_player"]


def attach_window_labels(windowed_game, labels_by_window_id):
    for window in windowed_game["windows"]:
        label = labels_by_window_id[window["window_id"]]
        window["discussion_leader"] = label["discussion_leader"]
    windowed_game["kol"] = get_kol(windowed_game["windows"])


def get_leader_counts(windows, player_names):
    leader_counts = Counter({player_name: 0 for player_name in player_names})
    for window in windows:
        leader = window.get("discussion_leader")
        if leader in leader_counts:
            leader_counts[leader] += 1
    return leader_counts


def get_influence_statuses(windowed_game, player_names):
    leader_counts = get_leader_counts(windowed_game.get("windows", []), player_names)
    kol = windowed_game.get("kol")
    statuses = {player_name: "normal" for player_name in player_names}

    if kol in statuses:
        statuses[kol] = "kol"

    if not any(leader_counts.values()):
        return statuses

    non_kol_players = [player_name for player_name in player_names if player_name != kol]
    if not non_kol_players:
        return statuses

    min_leads = min(leader_counts[player_name] for player_name in non_kol_players)
    for player_name in non_kol_players:
        if leader_counts[player_name] == min_leads:
            statuses[player_name] = "low_influence"

    return statuses


def get_werewolf_target_counts(game, player_names):
    target_counts = Counter({player_name: 0 for player_name in player_names})
    for turn in game.get("Dialogue", []):
        target = turn.get("targeted_player")
        if target in target_counts:
            target_counts[target] += 1
    return target_counts


def get_werewolf_target_ranks(target_counts, player_names):
    sorted_counts = sorted(
        {target_counts[player_name] for player_name in player_names},
        reverse=True
    )
    count_to_rank = {
        count: rank
        for rank, count in enumerate(sorted_counts, start=1)
    }
    return {
        player_name: count_to_rank[target_counts[player_name]]
        for player_name in player_names
    }


def get_voted_werewolf_flags(windowed_game, player_names):
    end_roles = windowed_game.get("endRoles") or windowed_game.get("startRoles") or []
    voting_outcome = windowed_game.get("votingOutcome") or []
    flags = {}

    for index, player_name in enumerate(player_names):
        vote = voting_outcome[index] if index < len(voting_outcome) else None
        voted_player_index = None

        if isinstance(vote, int):
            voted_player_index = vote - 1
        elif isinstance(vote, str) and vote in player_names:
            voted_player_index = player_names.index(vote)

        if voted_player_index is None or voted_player_index < 0 or voted_player_index >= len(end_roles):
            flags[player_name] = None
        else:
            flags[player_name] = end_roles[voted_player_index] == "Werewolf"

    return flags


def get_most_used_strategies(game, player_names):
    strategy_counts = {
        player_name: Counter()
        for player_name in player_names
    }

    for turn in game.get("Dialogue", []):
        speaker = turn.get("speaker")
        if speaker not in strategy_counts:
            continue

        annotations = turn.get("annotation") or []
        if isinstance(annotations, str):
            annotations = [annotations]

        for annotation in annotations:
            strategy_counts[speaker][annotation] += 1

    most_used_strategies = {}
    for player_name, counts in strategy_counts.items():
        if not counts:
            most_used_strategies[player_name] = None
            continue

        strategy_pool = {
            strategy: count
            for strategy, count in counts.items()
            if strategy != "No Strategy"
        }
        if not strategy_pool:
            strategy_pool = dict(counts)

        most_used_strategies[player_name] = max(
            strategy_pool,
            key=lambda strategy: (strategy_pool[strategy], strategy)
        )

    return most_used_strategies


def build_player_profiles(labeled_game, windowed_game, profiles_by_player, player_names):
    influence_statuses = get_influence_statuses(windowed_game, player_names)
    target_counts = get_werewolf_target_counts(labeled_game, player_names)
    target_ranks = get_werewolf_target_ranks(target_counts, player_names)
    voted_werewolf_flags = get_voted_werewolf_flags(windowed_game, player_names)
    most_used_strategies = get_most_used_strategies(labeled_game, player_names)

    return [
        {
            "player": player_name,
            "personalities": {
                "openness": profiles_by_player[player_name]["openness"],
                "conscientiousness": profiles_by_player[player_name]["conscientiousness"],
                "extraversion": profiles_by_player[player_name]["extraversion"],
                "agreeableness": profiles_by_player[player_name]["agreeableness"],
                "neuroticism": profiles_by_player[player_name]["neuroticism"]
            },
            "influence": influence_statuses[player_name],
            "most_used_strategy": most_used_strategies[player_name],
            "werewolf_target_count": target_counts[player_name],
            "werewolf_target_rank": target_ranks[player_name],
            "voted_werewolf": voted_werewolf_flags[player_name]
        }
        for player_name in player_names
    ]


def attach_player_profiles(labeled_game, windowed_game, profiles_by_player, player_names):
    windowed_game["player_profiles"] = build_player_profiles(
        labeled_game,
        windowed_game,
        profiles_by_player,
        player_names
    )


def apply_fallback_labels(game):
    for turn in game["Dialogue"]:
        turn["targeted_player"] = "None"

    windowed_game = build_game_windows(game)
    for window in windowed_game["windows"]:
        window["discussion_leader"] = "None"
    windowed_game["kol"] = "None"
    windowed_game["player_profiles"] = []
    return game, windowed_game


def label_game_strict(game, max_retries=3):
    player_names = game["playerNames"]
    payload = build_prompt_payload(game)

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

            parsed = json.loads(response.choices[0].message.content)
            labels_by_index = {
                item["merged_turn_index"]: item
                for item in parsed["turn_labels"]
            }

            if not validate_turn_labels(labels_by_index, game, player_names):
                continue

            labeled_game = deepcopy(game)
            attach_turn_labels(labeled_game, labels_by_index)
            windowed_game = build_game_windows(labeled_game)

            labels_by_window_id = {
                item["window_id"]: item
                for item in parsed["window_labels"]
            }

            if not validate_window_labels(labels_by_window_id, windowed_game["windows"], player_names):
                continue

            profiles_by_player = {
                item["player"]: item
                for item in parsed["personality_profiles"]
            }

            if not validate_personality_profiles(profiles_by_player, player_names):
                continue

            attach_window_labels(windowed_game, labels_by_window_id)
            attach_player_profiles(labeled_game, windowed_game, profiles_by_player, player_names)
            return windowed_game

        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)

    print("fallback to None")
    return apply_fallback_labels(deepcopy(game))[1]


def write_json_file(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        games = json.load(f)

    labeled_games = []

    for game in games:
        labeled_games.append(label_game_strict(deepcopy(game)))

    write_json_file(OUTPUT_FILE, labeled_games)
    print(f"Saved labeled dataset: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
