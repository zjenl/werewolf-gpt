# Werewolf GPT

`werewolf-gpt` is a research-oriented implementation of **One Night Ultimate Werewolf** built around two connected workflows:

1. **LLM simulation** of multi-player Werewolf games under different prompting strategies and model backends
2. **Human-game annotation and analysis** for studying persuasion targets, discussion leaders, KOLs, and influenceable opinion leaders

The repo now supports end-to-end work from human data preprocessing, AI-assisted labeling, and descriptive analysis through to prompt engineering and comparative simulation experiments.

This project was originally deployed from the upstream repository:

- <https://github.com/surfkansas/werewolf-gpt/tree/main?tab=readme-ov-file>

## Project Overview

The project asks a simple question: **who does the Werewolf try to persuade, and can that insight improve AI Werewolf play?**

The current workflow is:

1. preprocess human Werewolf games
2. label persuasion windows, targets, discussion leaders, and player personalities
3. build player profiles and identify KOLs / influence categories
4. analyze how often Werewolves target influential players
5. derive a human-data prior about influenceable opinion leaders
6. translate that prior into a new Werewolf prompt
7. compare prompt conditions in large simulation batches across multiple models

## Repository Structure

### Core simulation

- [werewolf.py](werewolf.py)  
  Main game simulator and batch runner.

- [prompts/rules.txt](prompts/rules.txt)  
  Shared role/game instructions.

- [prompts/werewolf_targeted_day.txt](prompts/werewolf_targeted_day.txt)  
  Targeted persuasion prompt.

- [prompts/werewolf_personality_leader_prior_day.txt](prompts/werewolf_personality_leader_prior_day.txt)  
  Personality-aware prompt with a human-derived prior about influenceable opinion leaders.

### Human data labeling and analysis

- [data/filter.py](data/filter.py)  
  Filters and merges the human game dataset into a cleaner format for annotation.

- [data/label.py](data/label.py)  
  AI-assisted labeling pipeline for persuasion windows, targets, discussion leaders, and Big Five personality traits.

- [data/analyze_kol_target_correlation.py](data/analyze_kol_target_correlation.py)  
  KOL vs Werewolf-target analysis and SVG figure generation.

- [data/extract_targetable_kols.py](data/extract_targetable_kols.py)  
  Extracts the subgroup of KOLs who were targeted most and did not vote Werewolf, then summarizes their traits and linguistic markers.

- [data/analyze_werewolf_results.py](data/analyze_werewolf_results.py)  
  Sweeps the `werewolf-results/` directory and generates experiment-level summaries, CSVs, JSON, SVGs, and HTML reports.

### Figure/table rendering

- [data/render_three_mode_tables.py](data/render_three_mode_tables.py)  
  Renders normal / targeted / prior comparison tables from live result files.

- [data/render_metric_tables_figure.py](data/render_metric_tables_figure.py)  
  Renders polished comparison-table figures from aggregated metrics.

- [data/render_targetable_kol_figures.py](data/render_targetable_kol_figures.py)  
  Renders summary figures for the influenceable-KOL subgroup.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API Keys

### Simulation

`werewolf.py` supports direct OpenAI and OpenRouter-compatible backends.

- For OpenAI mode, set:

```bash
export OPENAI_API_KEY="your_key_here"
```

- For OpenRouter modes, set:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

Optional OpenRouter metadata:

```bash
export OPENROUTER_APP_TITLE="Werewolf GPT"
export OPENROUTER_HTTP_REFERER="https://your-site.example"
```

### Labeling

[data/label.py](data/label.py) is currently **OpenRouter-only** and expects:

```bash
export OPENROUTER_API_KEY="your_key_here"
```

## Running the Game

Basic interactive run:

```bash
python3 werewolf.py
```

Markdown-rendered run:

```bash
python3 werewolf.py --render-markdown
```

### Model Modes

Current model presets:

- `openai-gpt5-nano`
- `openrouter-grok`
- `openrouter-qwen`
- `openrouter-gemini`

Example:

```bash
python3 werewolf.py --model-mode openrouter-qwen
```

### Batch Runs

Run 100 games and save both compact results and generated game dialogue:

```bash
python3 werewolf.py \
  --model-mode openai-gpt5-nano \
  --games 100 \
  --parallel-games 5 \
  --player-count 5 \
  --discussion-depth 20 \
  --results-file results/baseline-100.json \
  --games-json-file results/baseline-100-games.json
```

Batch outputs:

- compact experiment results (`results/*.json`)
- generated game dialogues (`results/*-games.json`)

These files are updated incrementally during long runs.

## Werewolf Prompt Conditions

The simulator currently supports two special Werewolf prompting modes:

- `--targeted-werewolf-persuasion`  
  The Werewolf explicitly identifies the most useful player to persuade and aims its public statement at that player.

- `--leader-prior-personality-werewolf-persuasion`  
  The Werewolf uses a human-data prior about influenceable opinion leaders on top of targeted persuasion logic.

- `--staged-werewolf-persuasion`  
  The Werewolf uses one prompt mode for an initial discussion block, triggers an interim vote, then switches to a second mode for continued discussion before the final vote.

Example targeted run:

```bash
python3 werewolf.py \
  --model-mode openrouter-grok \
  --games 100 \
  --parallel-games 5 \
  --player-count 5 \
  --discussion-depth 20 \
  --targeted-werewolf-persuasion \
  --results-file werewolf-results/target-100.json \
  --games-json-file werewolf-results/target-100-games.json
```

Example prior run:

```bash
python3 werewolf.py \
  --model-mode openrouter-qwen \
  --games 100 \
  --parallel-games 5 \
  --player-count 5 \
  --discussion-depth 20 \
  --leader-prior-personality-werewolf-persuasion \
  --results-file werewolf-results/qwen-prior-100.json \
  --games-json-file werewolf-results/qwen-prior-100-games.json
```

## Human Data Pipeline

### 1. Filter and merge the dataset

Start from the human game data in [data/train.json](data/train.json), then run the filter/merge step:

```bash
.venv/bin/python data/filter.py
```

The filtered merged game files in `data/` are used as inputs to the labeling stage.

### 2. Label persuasion windows and player personalities

The labeling pipeline:

- sections the game into persuasion windows
- labels each window with:
  - `targeted_player`
  - `discussion_leader`
- labels each player's Big Five personality traits
- builds window-level outputs and `player_profiles`

Run:

```bash
.venv/bin/python data/label.py
```

Current defaults in [data/label.py](data/label.py):

- input: `filtered_merged_games_50.json`
- output: `filtered_labeled_games_50.json`
- model: `x-ai/grok-4.1-fast`
- parallel workers: `5`

### 3. Extract influenceable opinion leaders

This script extracts the subgroup:

- `influence == "kol"`
- `werewolf_target_rank == 1`
- `voted_werewolf == false`

Run:

```bash
.venv/bin/python data/extract_targetable_kols.py \
  --input-file data/filtered_labeled_games_50.json \
  --players-output data/targetable_kol_players_50.json \
  --summary-output data/targetable_kol_attribute_summary_50.json \
  --text-output data/targetable_kol_attribute_summary_50.txt
```

Outputs include:

- selected player utterances and profiles
- attribute summaries
- rule-based linguistic marker summaries

## Analysis Workflows

### KOL vs target correlation

Run:

```bash
.venv/bin/python data/analyze_kol_target_correlation.py \
  --input-file data/filtered_labeled_games_50.json \
  --output-dir data/kol_target_analysis_50
```

Outputs include:

- [leader_count_vs_target_count.svg](data/kol_target_analysis_50/leader_count_vs_target_count.svg)
- [avg_target_count_by_influence.svg](data/kol_target_analysis_50/avg_target_count_by_influence.svg)
- [kol_target_rank_distribution.svg](data/kol_target_analysis_50/kol_target_rank_distribution.svg)
- [leader_target_overlap_rate.svg](data/kol_target_analysis_50/leader_target_overlap_rate.svg)
- [target_count_boxplot_by_influence.svg](data/kol_target_analysis_50/target_count_boxplot_by_influence.svg)

### Experiment sweep across simulation results

Run:

```bash
.venv/bin/python data/analyze_werewolf_results.py
```

This scans `werewolf-results/` and writes:

- experiment metrics CSV / JSON
- game-level metrics CSV / JSON
- model/mode comparison summaries
- SVG plots
- HTML report

Main output folder:

- [werewolf-results/analysis](werewolf-results/analysis)

### Three-mode comparison tables

These tables aggregate:

- `normal`
- `targeted`
- `prior`

across:

- `gpt5_nano`
- `gemini`
- `qwen`

Run:

```bash
.venv/bin/python data/render_three_mode_tables.py
```

Outputs include:

- [win_rate_normal_targeted_prior.png](werewolf-results/analysis/win_rate_normal_targeted_prior.png)
- [avg_votes_on_werewolves_normal_targeted_prior.png](werewolf-results/analysis/avg_votes_on_werewolves_normal_targeted_prior.png)
- [avg_werewolf_vote_share_normal_targeted_prior.png](werewolf-results/analysis/avg_werewolf_vote_share_normal_targeted_prior.png)
- [three_mode_tables_combined.png](werewolf-results/analysis/three_mode_tables_combined.png)
- [three_mode_tables_current.md](werewolf-results/analysis/three_mode_tables_current.md)

### Influenceable-KOL figures

Run:

```bash
.venv/bin/python data/render_targetable_kol_figures.py
```

Outputs include:

- [targetable_kol_dashboard.png](data/targetable_kol_figures_50/targetable_kol_dashboard.png)
- [targetable_kol_big5.png](data/targetable_kol_figures_50/targetable_kol_big5.png)
- [targetable_kol_linguistic_markers.png](data/targetable_kol_figures_50/targetable_kol_linguistic_markers.png)

## Current Method Summary

The current project methodology is:

1. preprocess human games
2. label persuasion windows, discussion leaders, targets, and personalities
3. build player profiles and KOL labels
4. analyze whether Werewolves target influential players
5. isolate influenceable opinion leaders
6. summarize their personality, strategy, and linguistic traits
7. encode those findings as a new Werewolf prompt prior
8. run comparative LLM simulation batches
9. evaluate win rate, votes on Werewolves, and vote share across models and prompt conditions

## Notes

- The repo now contains both legacy and current outputs; some folders include earlier experiment artifacts kept for comparison.
- Some older result filenames do not perfectly match the final prompt taxonomy, so analysis scripts rely on the stored JSON flags rather than filenames alone.
- The simulation currently keeps `startRoles` and `endRoles` aligned because no live role-swapping night actions are implemented.

## License

See [LICENSE](LICENSE).
