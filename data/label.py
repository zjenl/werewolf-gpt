import json
from collections import Counter

from openai import OpenAI

client = OpenAI()

INPUT_FILE = "filtered_merged_games.json"
OUTPUT_FILE = "filtered_labeled_games.json"

MODEL_NAME = "gpt-5-nano"
API_SEED = 1234

SYSTEM_PROMPT = """
Section Werewolf persuasion windows, label each window's target and discussion leader,
and label player personalities in the same response for a One Night Ultimate Werewolf game.

You MUST:
- Create persuasion windows from the merged dialogue
- Label each persuasion window with one targeted_player
- Label each persuasion window with one discussion_leader
- Label EVERY player's Big Five personality profile
- Use ONLY allowed labels
- Follow schema exactly

Persuasion window definition:
A persuasion window is a contiguous range of merged turns centered on one single discussion
topic, regardless of whether the Werewolf speaker is present or not.
You MUST cover the entire dialogue from merged_turn_index 0 to the final merged_turn_index.
Every merged turn must belong to exactly one window. You should try to create more than 10 
windows per game if possible.

Windows must be contiguous:
- first window starts at merged_turn_index 0
- each next window starts at previous end_merged_turn_index + 1
- final window ends at the last merged_turn_index

Window index rules:
- window_id starts at 0 and increases by 1
- start_merged_turn_index and end_merged_turn_index are inclusive
- windows must be in dialogue order
- windows must not overlap

Discussion leader definition:
For each persuasion window, label discussion_leader as the player other than the Werewolf speaker
who is leading everyone else in that window. This is not necessarily the first speaker, the loudest
speaker, or the targeted player. Choose the person who most clearly steers the
conversation, frames the issue, sets the agenda, asks the questions others respond to,
or influences how other players interpret the situation.
If no one clearly leads a window, use "None".

Allowed labels for discussion_leader:
- any player name
- "None"

Target definition:
Label the person that the Werewolf speaker is trying to persuade or appeal to in order
to gain their trust, liking, or belief in the persuasion window. The target can be a specific player
or the whole group. Use context in the window to determine the target. For example, if a Werewolf 
agrees with the discussion topic or the discussion leader's opinion, then the target is the discussion leader. 
ALWAYS use a specific player name when possible. The target player cannot be the Werewolf speaker.

Allowed labels for targeted_player:
- any player name EXCLUDING the Werewolf speaker (always use a specific player name when possible)
- "Group" (only use when the Werewolf speaker is explicitly appealing to the whole group)
- "None" if no werewolf speaker is present in the window

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
    "name": "window_target_leader_and_personality_labels",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "target_windows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "window_id": {"type": "integer"},
                        "start_merged_turn_index": {"type": "integer"},
                        "end_merged_turn_index": {"type": "integer"},
                        "targeted_player": {"type": "string"},
                        "discussion_leader": {"type": "string"}
                    },
                    "required": [
                        "window_id",
                        "start_merged_turn_index",
                        "end_merged_turn_index",
                        "targeted_player",
                        "discussion_leader"
                    ],
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
        "required": ["target_windows", "personality_profiles"],
        "additionalProperties": False
    }
}


def get_role_by_player(game, role_key):
    player_names = game["playerNames"]
    roles = game.get(role_key, [])
    return {
        player_name: roles[i] if i < len(roles) else None
        for i, player_name in enumerate(player_names)
    }


def build_label_prompt_payload(game):
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


def get_personalities(profile):
    trait_names = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism"
    ]
    profile = profile or {}
    return {
        trait_name: profile.get(trait_name)
        for trait_name in trait_names
    }


def get_dialogue_for_window(game, window):
    dialogue = game.get("Dialogue", [])
    start = window.get("start_merged_turn_index")
    end = window.get("end_merged_turn_index")

    if not isinstance(start, int) or not isinstance(end, int):
        return []
    if start < 0 or end >= len(dialogue) or start > end:
        return []

    return [
        {
            "merged_turn_index": index,
            "speaker": turn.get("speaker"),
            "utterance": turn.get("utterance"),
            "annotation": turn.get("annotation", [])
        }
        for index, turn in enumerate(dialogue[start:end + 1], start=start)
    ]


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


def get_werewolf_target_counts(windows, player_names):
    target_counts = Counter({player_name: 0 for player_name in player_names})
    for window in windows:
        target = window.get("targeted_player")
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


def build_player_profiles(game, labeled_game, profiles_by_player, player_names):
    windows = labeled_game.get("windows", [])
    end_roles = game.get("endRoles", [])
    influence_statuses = get_influence_statuses(labeled_game, player_names)
    target_counts = get_werewolf_target_counts(windows, player_names)
    target_ranks = get_werewolf_target_ranks(target_counts, player_names)
    voted_werewolf_flags = get_voted_werewolf_flags(game, player_names)
    most_used_strategies = get_most_used_strategies(game, player_names)

    return [
        {
            "player": player_name,
            "endRole": end_roles[index] if index < len(end_roles) else None,
            "personalities": get_personalities(profiles_by_player.get(player_name)),
            "influence": influence_statuses[player_name],
            "most_used_strategy": most_used_strategies[player_name],
            "werewolf_target_count": target_counts[player_name],
            "werewolf_target_rank": target_ranks[player_name],
            "voted_werewolf": voted_werewolf_flags[player_name]
        }
        for index, player_name in enumerate(player_names)
    ]


def attach_player_profiles(game, labeled_game, profiles_by_player, player_names):
    labeled_game["player_profiles"] = build_player_profiles(
        game,
        labeled_game,
        profiles_by_player,
        player_names
    )


def parse_model_json(response):
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty content")

    content = content.strip()
    if content.startswith("```"):
        if content.startswith("```json"):
            content = content[len("```json"):].strip()
        else:
            content = content[len("```"):].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

    return json.loads(content)


def build_labeled_game(game, parsed):
    player_names = game.get("playerNames", [])
    windows = [
        {
            **dict(window),
            "Dialogue": get_dialogue_for_window(game, window)
        }
        for window in parsed.get("target_windows", [])
        if isinstance(window, dict)
    ]
    personality_profiles = [
        dict(profile)
        for profile in parsed.get("personality_profiles", [])
        if isinstance(profile, dict)
    ]
    profiles_by_player = {
        profile.get("player"): profile
        for profile in personality_profiles
        if isinstance(profile, dict)
    }

    labeled_game = {
        "Game_ID": game.get("Game_ID"),
        "playerNames": player_names,
        "startRoles": game.get("startRoles", []),
        "endRoles": game.get("endRoles", []),
        "votingOutcome": game.get("votingOutcome", []),
        "windows": windows,
        "kol": get_kol(windows),
    }
    attach_player_profiles(game, labeled_game, profiles_by_player, player_names)
    labeled_game.pop("kol", None)
    return labeled_game


def label_game_targets(game, max_retries=3):
    payload = build_label_prompt_payload(game)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                seed=API_SEED,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": LABEL_SCHEMA
                }
            )

            parsed = parse_model_json(response)
            return build_labeled_game(game, parsed)

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed:", e)

    print("fallback to None")
    return build_labeled_game(game, {
        "target_windows": [],
        "personality_profiles": []
    })


def write_json_file(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        games = json.load(f)

    labeled_games = [
        label_game_targets(game)
        for game in games
    ]

    write_json_file(OUTPUT_FILE, labeled_games)
    print(f"Saved labeled dataset: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
