import json
from collections import Counter, defaultdict

FILE_PATH = "filtered_labeled_games.json"
TARGET_SPEAKER = "Ashley"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count who Ashley is targeting in persuasive turns
target_counts = Counter()

# Optional: keep examples for each target
target_examples = defaultdict(list)

for game in data:
    game_id = game.get("Game_ID", "UnknownGame")
    for turn in game.get("Dialogue", []):
        if turn.get("speaker") != TARGET_SPEAKER:
            continue

        target = turn.get("targeted_player", "None")
        annotations = turn.get("annotation", [])

        # Only keep actual persuasive / strategy turns
        if not annotations or annotations == ["No Strategy"]:
            continue
        if target in [None, "None"]:
            continue

        target_counts[target] += 1

        if len(target_examples[target]) < 5:
            target_examples[target].append({
                "game_id": game_id,
                "utterance": turn.get("utterance", ""),
                "annotation": annotations
            })

print(f"\nWho Ashley mainly focuses on persuading\n{'='*45}")

if not target_counts:
    print("No persuasive Ashley turns with a specific target were found.")
else:
    total = sum(target_counts.values())
    for target, count in target_counts.most_common():
        pct = count / total * 100
        print(f"{target}: {count} times ({pct:.1f}%)")

    main_target, main_count = target_counts.most_common(1)[0]
    print(f"\nAshley mainly focuses on persuading: {main_target} ({main_count} turns)")

    print(f"\nExamples of Ashley targeting each player\n{'='*45}")
    for target, examples in target_examples.items():
        print(f"\nTARGET: {target}")
        for ex in examples:
            print(f"- [{ex['game_id']}] {ex['utterance']}")
            print(f"  annotation={ex['annotation']}")

# Optional: save results to a text file
with open("ashley_target_analysis.txt", "w", encoding="utf-8") as out:
    out.write("Who Ashley mainly focuses on persuading\n")
    out.write("=" * 45 + "\n")
    if not target_counts:
        out.write("No persuasive Ashley turns with a specific target were found.\n")
    else:
        total = sum(target_counts.values())
        for target, count in target_counts.most_common():
            pct = count / total * 100
            out.write(f"{target}: {count} times ({pct:.1f}%)\n")

        main_target, main_count = target_counts.most_common(1)[0]
        out.write(f"\nAshley mainly focuses on persuading: {main_target} ({main_count} turns)\n")

print("\nDone. Results also saved to ashley_target_analysis.txt")
