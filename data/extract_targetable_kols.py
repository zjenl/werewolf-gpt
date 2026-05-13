import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
from statistics import mean, median


DEFAULT_INPUT = "data/filtered_labeled_games.json"
DEFAULT_OUTPUT_PLAYERS = "data/targetable_kol_players.json"
DEFAULT_OUTPUT_SUMMARY = "data/targetable_kol_attribute_summary.json"
DEFAULT_OUTPUT_TEXT = "data/targetable_kol_attribute_summary.txt"

TRAIT_NAMES = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]

TOKEN_RE = re.compile(r"[A-Za-z']+")
QUESTION_WORDS = {"who", "what", "when", "where", "why", "how"}
HEDGE_WORDS = {
    "maybe", "perhaps", "probably", "possibly", "guess", "might", "could",
    "seems", "seem", "think", "unsure", "maybe", "likely",
}
CERTAINTY_WORDS = {
    "definitely", "clearly", "obviously", "sure", "certain", "know",
    "exactly", "absolutely", "must",
}
AGREEMENT_WORDS = {
    "yeah", "yes", "agree", "right", "true", "exactly", "fair", "sense",
    "ok", "okay",
}
DISAGREEMENT_WORDS = {
    "no", "wrong", "disagree", "dont", "don't", "isnt", "isn't",
    "not", "false",
}
ACCUSATION_WORDS = {
    "wolf", "werewolf", "liar", "lying", "suspicious", "sus", "guilty",
    "evil", "accuse", "accusing",
}
EVIDENCE_WORDS = {
    "because", "saw", "seen", "proof", "evidence", "role", "swapped",
    "switch", "middle", "claim", "claimed",
}
ACTION_WORDS = {
    "vote", "kill", "pick", "choose", "lynch", "go", "trust", "push",
}
INCLUSIVE_WORDS = {"we", "us", "our", "ours", "everyone", "all", "together"}
SELF_WORDS = {"i", "me", "my", "mine", "myself"}
YOU_WORDS = {"you", "your", "yours", "yourself"}
NEGATION_WORDS = {"no", "not", "never", "none", "nothing", "neither", "dont", "don't", "cant", "can't"}


def load_games(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_mean(values):
    return mean(values) if values else 0.0


def safe_median(values):
    return median(values) if values else 0.0


def safe_div(numerator, denominator):
    if not denominator:
        return 0.0
    return numerator / denominator


def role_by_player(game, key):
    names = game.get("playerNames", [])
    roles = game.get(key, [])
    return {name: roles[i] if i < len(roles) else None for i, name in enumerate(names)}


def is_selected_profile(profile):
    return (
        profile.get("influence") == "kol"
        and profile.get("werewolf_target_rank") == 1
        and profile.get("voted_werewolf") is False
    )


def collect_player_utterances(game, player_name):
    utterances = []
    for window in game.get("windows", []):
        for turn in window.get("Dialogue", []):
            if turn.get("speaker") != player_name:
                continue
            utterances.append({
                "window_id": window.get("window_id"),
                "merged_turn_index": turn.get("merged_turn_index"),
                "speaker": turn.get("speaker"),
                "utterance": turn.get("utterance"),
                "annotation": turn.get("annotation"),
                "window_targeted_player": window.get("targeted_player"),
                "window_discussion_leader": window.get("discussion_leader"),
            })
    utterances.sort(key=lambda item: (item.get("merged_turn_index", -1), item.get("window_id", -1)))
    return utterances


def collect_targeted_windows(game, player_name):
    windows = []
    for window in game.get("windows", []):
        if window.get("targeted_player") != player_name:
            continue
        windows.append({
            "window_id": window.get("window_id"),
            "start_merged_turn_index": window.get("start_merged_turn_index"),
            "end_merged_turn_index": window.get("end_merged_turn_index"),
            "targeted_player": window.get("targeted_player"),
            "discussion_leader": window.get("discussion_leader"),
        })
    return windows


def tokenize(text):
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def analyze_utterances(utterances, player_names, speaker_name):
    counts = Counter()
    total_words = 0
    total_utterances = len(utterances)
    unique_word_counter = Counter()
    other_names = {name.lower() for name in player_names if name != speaker_name}

    for utterance in utterances:
        text = utterance.get("utterance") or ""
        tokens = tokenize(text)
        total_words += len(tokens)
        unique_word_counter.update(tokens)

        if "?" in text:
            counts["question_utterances"] += 1
        if "!" in text:
            counts["exclamation_utterances"] += 1

        token_set = set(tokens)
        if token_set & QUESTION_WORDS:
            counts["question_word_utterances"] += 1
        if token_set & HEDGE_WORDS:
            counts["hedge_utterances"] += 1
        if token_set & CERTAINTY_WORDS:
            counts["certainty_utterances"] += 1
        if token_set & AGREEMENT_WORDS:
            counts["agreement_utterances"] += 1
        if token_set & DISAGREEMENT_WORDS:
            counts["disagreement_utterances"] += 1
        if token_set & ACCUSATION_WORDS:
            counts["accusation_utterances"] += 1
        if token_set & EVIDENCE_WORDS:
            counts["evidence_utterances"] += 1
        if token_set & ACTION_WORDS:
            counts["action_utterances"] += 1
        if token_set & INCLUSIVE_WORDS:
            counts["inclusive_utterances"] += 1
        if token_set & SELF_WORDS:
            counts["self_reference_utterances"] += 1
        if token_set & YOU_WORDS:
            counts["you_reference_utterances"] += 1
        if token_set & NEGATION_WORDS:
            counts["negation_utterances"] += 1
        if token_set & other_names:
            counts["name_mention_utterances"] += 1

        counts["question_word_tokens"] += sum(1 for token in tokens if token in QUESTION_WORDS)
        counts["hedge_tokens"] += sum(1 for token in tokens if token in HEDGE_WORDS)
        counts["certainty_tokens"] += sum(1 for token in tokens if token in CERTAINTY_WORDS)
        counts["agreement_tokens"] += sum(1 for token in tokens if token in AGREEMENT_WORDS)
        counts["disagreement_tokens"] += sum(1 for token in tokens if token in DISAGREEMENT_WORDS)
        counts["accusation_tokens"] += sum(1 for token in tokens if token in ACCUSATION_WORDS)
        counts["evidence_tokens"] += sum(1 for token in tokens if token in EVIDENCE_WORDS)
        counts["action_tokens"] += sum(1 for token in tokens if token in ACTION_WORDS)
        counts["inclusive_tokens"] += sum(1 for token in tokens if token in INCLUSIVE_WORDS)
        counts["self_reference_tokens"] += sum(1 for token in tokens if token in SELF_WORDS)
        counts["you_reference_tokens"] += sum(1 for token in tokens if token in YOU_WORDS)
        counts["negation_tokens"] += sum(1 for token in tokens if token in NEGATION_WORDS)
        counts["name_mention_tokens"] += sum(1 for token in tokens if token in other_names)

    top_words = [
        {"word": word, "count": count}
        for word, count in unique_word_counter.most_common(20)
    ]

    return {
        "total_utterances": total_utterances,
        "total_words": total_words,
        "avg_words_per_utterance": round(safe_div(total_words, total_utterances), 4),
        "unique_word_count": len(unique_word_counter),
        "question_utterance_rate": round(safe_div(counts["question_utterances"], total_utterances), 4),
        "exclamation_utterance_rate": round(safe_div(counts["exclamation_utterances"], total_utterances), 4),
        "question_word_utterance_rate": round(safe_div(counts["question_word_utterances"], total_utterances), 4),
        "hedge_utterance_rate": round(safe_div(counts["hedge_utterances"], total_utterances), 4),
        "certainty_utterance_rate": round(safe_div(counts["certainty_utterances"], total_utterances), 4),
        "agreement_utterance_rate": round(safe_div(counts["agreement_utterances"], total_utterances), 4),
        "disagreement_utterance_rate": round(safe_div(counts["disagreement_utterances"], total_utterances), 4),
        "accusation_utterance_rate": round(safe_div(counts["accusation_utterances"], total_utterances), 4),
        "evidence_utterance_rate": round(safe_div(counts["evidence_utterances"], total_utterances), 4),
        "action_utterance_rate": round(safe_div(counts["action_utterances"], total_utterances), 4),
        "inclusive_utterance_rate": round(safe_div(counts["inclusive_utterances"], total_utterances), 4),
        "self_reference_utterance_rate": round(safe_div(counts["self_reference_utterances"], total_utterances), 4),
        "you_reference_utterance_rate": round(safe_div(counts["you_reference_utterances"], total_utterances), 4),
        "negation_utterance_rate": round(safe_div(counts["negation_utterances"], total_utterances), 4),
        "name_mention_utterance_rate": round(safe_div(counts["name_mention_utterances"], total_utterances), 4),
        "question_word_token_rate": round(safe_div(counts["question_word_tokens"], total_words), 4),
        "hedge_token_rate": round(safe_div(counts["hedge_tokens"], total_words), 4),
        "certainty_token_rate": round(safe_div(counts["certainty_tokens"], total_words), 4),
        "agreement_token_rate": round(safe_div(counts["agreement_tokens"], total_words), 4),
        "disagreement_token_rate": round(safe_div(counts["disagreement_tokens"], total_words), 4),
        "accusation_token_rate": round(safe_div(counts["accusation_tokens"], total_words), 4),
        "evidence_token_rate": round(safe_div(counts["evidence_tokens"], total_words), 4),
        "action_token_rate": round(safe_div(counts["action_tokens"], total_words), 4),
        "inclusive_token_rate": round(safe_div(counts["inclusive_tokens"], total_words), 4),
        "self_reference_token_rate": round(safe_div(counts["self_reference_tokens"], total_words), 4),
        "you_reference_token_rate": round(safe_div(counts["you_reference_tokens"], total_words), 4),
        "negation_token_rate": round(safe_div(counts["negation_tokens"], total_words), 4),
        "name_mention_token_rate": round(safe_div(counts["name_mention_tokens"], total_words), 4),
        "top_words": top_words,
    }


def build_selected_player_record(game, profile):
    player_name = profile.get("player")
    start_roles = role_by_player(game, "startRoles")
    end_roles = role_by_player(game, "endRoles")
    utterances = collect_player_utterances(game, player_name)
    linguistic_markers = analyze_utterances(utterances, game.get("playerNames", []), player_name)

    return {
        "game_id": game.get("Game_ID"),
        "player": player_name,
        "utterances": utterances,
        "profile": {
            "personalities": profile.get("personalities", {}),
            "discussion_leader_count": profile.get("discussion_leader_count"),
            "most_used_strategy": profile.get("most_used_strategy"),
            "startRole": start_roles.get(player_name),
            "endRole": end_roles.get(player_name) or profile.get("endRole"),
            "werewolf_target_count": profile.get("werewolf_target_count"),
            "werewolf_target_rank": profile.get("werewolf_target_rank"),
            "voted_werewolf": profile.get("voted_werewolf"),
            "influence": profile.get("influence"),
        },
        "linguistic_markers": linguistic_markers,
        "targeted_windows": collect_targeted_windows(game, player_name),
    }


def summarize_selected_players(selected_players):
    strategy_counts = Counter()
    start_role_counts = Counter()
    end_role_counts = Counter()
    discussion_leader_counts = []
    werewolf_target_counts = []
    utterance_counts = []
    utterance_word_counts = []
    targeted_window_counts = []
    trait_value_counts = {trait: Counter() for trait in TRAIT_NAMES}
    joint_trait_profiles = Counter()
    marker_values = defaultdict(list)
    global_top_words = Counter()

    for player in selected_players:
        profile = player.get("profile", {})
        personalities = profile.get("personalities", {})
        linguistic_markers = player.get("linguistic_markers", {})

        strategy_counts[profile.get("most_used_strategy") or "None"] += 1
        start_role_counts[profile.get("startRole") or "None"] += 1
        end_role_counts[profile.get("endRole") or "None"] += 1
        discussion_leader_counts.append(profile.get("discussion_leader_count") or 0)
        werewolf_target_counts.append(profile.get("werewolf_target_count") or 0)
        utterance_counts.append(len(player.get("utterances", [])))
        targeted_window_counts.append(len(player.get("targeted_windows", [])))

        for utterance in player.get("utterances", []):
            text = (utterance.get("utterance") or "").strip()
            if text:
                utterance_word_counts.append(len(text.split()))

        for key, value in linguistic_markers.items():
            if key == "top_words":
                for row in value:
                    global_top_words[row["word"]] += row["count"]
            elif isinstance(value, (int, float)):
                marker_values[key].append(value)

        trait_signature = []
        for trait in TRAIT_NAMES:
            value = personalities.get(trait) or "None"
            trait_value_counts[trait][value] += 1
            trait_signature.append(f"{trait}={value}")
        joint_trait_profiles["; ".join(trait_signature)] += 1

    linguistic_marker_summary = {
        key: {
            "mean": round(safe_mean(values), 4),
            "median": round(safe_median(values), 4),
            "min": round(min(values), 4) if values else 0,
            "max": round(max(values), 4) if values else 0,
        }
        for key, values in sorted(marker_values.items())
    }

    return {
        "selected_player_count": len(selected_players),
        "games_represented": len({player.get("game_id") for player in selected_players}),
        "most_used_strategy_counts": dict(strategy_counts.most_common()),
        "start_role_counts": dict(start_role_counts.most_common()),
        "end_role_counts": dict(end_role_counts.most_common()),
        "trait_value_counts": {
            trait: dict(counter.most_common())
            for trait, counter in trait_value_counts.items()
        },
        "top_joint_trait_profiles": [
            {"trait_profile": trait_profile, "count": count}
            for trait_profile, count in joint_trait_profiles.most_common(10)
        ],
        "discussion_leader_count_stats": {
            "mean": round(safe_mean(discussion_leader_counts), 4),
            "median": round(safe_median(discussion_leader_counts), 4),
            "min": min(discussion_leader_counts) if discussion_leader_counts else 0,
            "max": max(discussion_leader_counts) if discussion_leader_counts else 0,
        },
        "werewolf_target_count_stats": {
            "mean": round(safe_mean(werewolf_target_counts), 4),
            "median": round(safe_median(werewolf_target_counts), 4),
            "min": min(werewolf_target_counts) if werewolf_target_counts else 0,
            "max": max(werewolf_target_counts) if werewolf_target_counts else 0,
        },
        "targeted_window_count_stats": {
            "mean": round(safe_mean(targeted_window_counts), 4),
            "median": round(safe_median(targeted_window_counts), 4),
            "min": min(targeted_window_counts) if targeted_window_counts else 0,
            "max": max(targeted_window_counts) if targeted_window_counts else 0,
        },
        "utterance_count_stats": {
            "mean": round(safe_mean(utterance_counts), 4),
            "median": round(safe_median(utterance_counts), 4),
            "min": min(utterance_counts) if utterance_counts else 0,
            "max": max(utterance_counts) if utterance_counts else 0,
        },
        "utterance_word_count_stats": {
            "mean": round(safe_mean(utterance_word_counts), 4),
            "median": round(safe_median(utterance_word_counts), 4),
            "min": min(utterance_word_counts) if utterance_word_counts else 0,
            "max": max(utterance_word_counts) if utterance_word_counts else 0,
        },
        "linguistic_marker_summary": linguistic_marker_summary,
        "top_words_across_selected_players": [
            {"word": word, "count": count}
            for word, count in global_top_words.most_common(30)
        ],
    }


def write_text_summary(path, summary):
    lines = [
        "Targetable KOL Attribute Summary",
        "===============================",
        "",
        f"Selected players: {summary['selected_player_count']}",
        f"Games represented: {summary['games_represented']}",
        "",
        "Most used strategies:",
    ]
    for label, count in summary["most_used_strategy_counts"].items():
        lines.append(f"- {label}: {count}")

    lines.extend([
        "",
        "Start roles:",
    ])
    for label, count in summary["start_role_counts"].items():
        lines.append(f"- {label}: {count}")

    lines.extend([
        "",
        "End roles:",
    ])
    for label, count in summary["end_role_counts"].items():
        lines.append(f"- {label}: {count}")

    lines.extend([
        "",
        "Trait value counts:",
    ])
    for trait, counts in summary["trait_value_counts"].items():
        lines.append(f"- {trait}:")
        for label, count in counts.items():
            lines.append(f"  - {label}: {count}")

    lines.extend([
        "",
        "Top joint trait profiles:",
    ])
    for row in summary["top_joint_trait_profiles"]:
        lines.append(f"- {row['trait_profile']}: {row['count']}")

    for stat_name in [
        "discussion_leader_count_stats",
        "werewolf_target_count_stats",
        "targeted_window_count_stats",
        "utterance_count_stats",
        "utterance_word_count_stats",
    ]:
        stats = summary[stat_name]
        lines.extend([
            "",
            f"{stat_name}:",
            f"- mean: {stats['mean']}",
            f"- median: {stats['median']}",
            f"- min: {stats['min']}",
            f"- max: {stats['max']}",
        ])

    lines.extend([
        "",
        "Linguistic marker summary:",
    ])
    for marker_name, stats in summary["linguistic_marker_summary"].items():
        lines.append(f"- {marker_name}:")
        lines.append(f"  - mean: {stats['mean']}")
        lines.append(f"  - median: {stats['median']}")
        lines.append(f"  - min: {stats['min']}")
        lines.append(f"  - max: {stats['max']}")

    lines.extend([
        "",
        "Top words across selected players:",
    ])
    for row in summary["top_words_across_selected_players"]:
        lines.append(f"- {row['word']}: {row['count']}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def extract_selected_players(games):
    selected = []
    for game in games:
        for profile in game.get("player_profiles", []):
            if is_selected_profile(profile):
                selected.append(build_selected_player_record(game, profile))
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Extract KOL players most targeted by Werewolves who did not vote Werewolf."
    )
    parser.add_argument("--input-file", default=DEFAULT_INPUT)
    parser.add_argument("--players-output", default=DEFAULT_OUTPUT_PLAYERS)
    parser.add_argument("--summary-output", default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument("--text-output", default=DEFAULT_OUTPUT_TEXT)
    args = parser.parse_args()

    games = load_games(args.input_file)
    selected_players = extract_selected_players(games)
    summary = summarize_selected_players(selected_players)

    write_json(args.players_output, selected_players)
    write_json(args.summary_output, summary)
    write_text_summary(args.text_output, summary)

    print(f"Selected players: {len(selected_players)}")
    print(f"Wrote player records to {args.players_output}")
    print(f"Wrote attribute summary to {args.summary_output}")
    print(f"Wrote text summary to {args.text_output}")


if __name__ == "__main__":
    main()
