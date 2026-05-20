import json
import os
import random
import threading
import time
import uuid
import hashlib
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import click
import colorama
import dotenv
import openai
from colorama import Fore, Style
colorama.init()

from openai import OpenAI

OPENROUTER_API_BASE_URL = 'https://openrouter.ai/api/v1'
MODEL_PRESETS = {
    'openai-gpt5-nano': {
        'model': 'gpt-5-nano',
        'api_base_url': None,
    },
    'openrouter-grok': {
        'model': 'x-ai/grok-4.1-fast',
        'api_base_url': OPENROUTER_API_BASE_URL,
    },
    'openrouter-qwen': {
        'model': 'qwen/qwen3.5-flash-02-23',
        'api_base_url': OPENROUTER_API_BASE_URL,
    },
    'openrouter-gemini': {
        'model': 'google/gemini-2.5-flash-lite',
        'api_base_url': OPENROUTER_API_BASE_URL,
    },
}
DEFAULT_MODEL_MODE = 'openai-gpt5-nano'
DEFAULT_MODEL = MODEL_PRESETS[DEFAULT_MODEL_MODE]['model']
DEFAULT_API_BASE_URL = MODEL_PRESETS[DEFAULT_MODEL_MODE]['api_base_url']
DEFAULT_RESULTS_FILE = 'results/baseline-results.json'
DEFAULT_GAMES_JSON_FILE = 'results/baseline-games.json'
MODEL_MAX_RETRIES = 3
MODEL_RETRY_SLEEP_SECONDS = 5
MODEL_SEED = 12345
WEREWOLF_PROMPT_MODE_CHOICES = ('normal', 'targeted', 'prior')

if os.path.isfile('.env'):
    dotenv.load_dotenv()

client_state = threading.local()
configured_api_base_url = DEFAULT_API_BASE_URL


def get_model_preset(mode):
    preset = MODEL_PRESETS.get(mode)
    if preset is None:
        raise click.ClickException(f'Unknown model mode: {mode}')
    return preset


def resolve_model_config(model_mode, model, api_base_url):
    preset = get_model_preset(model_mode)
    resolved_model = model or preset['model']
    if api_base_url is None:
        resolved_api_base_url = preset['api_base_url']
    else:
        resolved_api_base_url = api_base_url

    return {
        'model_mode': model_mode,
        'model': resolved_model,
        'api_base_url': resolved_api_base_url,
    }


def configure_client(api_base_url=None):
    global configured_api_base_url

    if api_base_url is not None:
        configured_api_base_url = api_base_url
    else:
        api_base_url = configured_api_base_url

    thread_client = getattr(client_state, 'client', None)
    thread_base_url = getattr(client_state, 'base_url', None)
    if thread_client is not None and thread_base_url == api_base_url:
        return thread_client

    uses_openrouter = bool(api_base_url and 'openrouter.ai' in api_base_url)
    if uses_openrouter:
        api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise click.ClickException('Set OPENROUTER_API_KEY in your environment or .env file before running OpenRouter model games.')
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise click.ClickException('Set OPENAI_API_KEY in your environment or .env file before running OpenAI model games.')

    client_args = {'api_key': api_key}
    if api_base_url:
        client_args['base_url'] = api_base_url

    if uses_openrouter:
        headers = {
            'X-OpenRouter-Title': os.getenv('OPENROUTER_APP_TITLE', 'Werewolf GPT')
        }
        referer = os.getenv('OPENROUTER_HTTP_REFERER')
        if referer:
            headers['HTTP-Referer'] = referer
        client_args['default_headers'] = headers

    thread_client = OpenAI(**client_args)
    client_state.client = thread_client
    client_state.base_url = api_base_url
    openai.api_key = api_key
    return thread_client


def run_model_prompt(prompt, model, json_mode=False, prompt_cache_key=None):
    messages = prompt
    if isinstance(prompt, str):
        messages = [
            {
                'role': 'user',
                'content': prompt
            }
        ]

    request_args = {
        'model': model,
        'seed': MODEL_SEED,
        'messages': messages
    }

    client = configure_client()
    api_base_url = getattr(client_state, 'base_url', None)
    uses_openrouter = bool(api_base_url and 'openrouter.ai' in api_base_url)

    if not uses_openrouter:
        if prompt_cache_key:
            request_args['prompt_cache_key'] = prompt_cache_key
        request_args['prompt_cache_retention'] = 'in_memory'

    if json_mode:
        request_args['response_format'] = {'type': 'json_object'}

    last_error = None
    for attempt in range(MODEL_MAX_RETRIES):
        try:
            response = client.chat.completions.create(**request_args)
            message = response.choices[0].message
            return message.content or ''
        except Exception as e:
            last_error = e
            if attempt == MODEL_MAX_RETRIES - 1:
                break
            time.sleep(MODEL_RETRY_SLEEP_SECONDS * (attempt + 1))

    raise last_error


def parse_json_response(message_json):
    try:
        return json.loads(message_json)
    except ValueError:
        pass

    start = message_json.find('{')
    end = message_json.rfind('}')
    if start != -1 and end != -1 and start < end:
        try:
            return json.loads(message_json[start:end + 1])
        except ValueError:
            pass

    raise ValueError('Response did not contain valid JSON.')


def return_dict_from_json_or_fix(message_json, model):

    try:
        message_dict = parse_json_response(message_json)

    except ValueError:
        fixed_json = run_model_prompt(
            "I have a JSON string, but it is not valid JSON. Possibly, the message contains other text besides just the JSON. "
            "Could you make it valid? Or, if there is valid JSON in the response, please just extract the JSON and do NOT update it. "
            "Please respond ONLY in valid JSON. Do not comment on your response. Do not start or end with backticks. "
            "You must ONLY respond in JSON.\n\n"
            f"Bad JSON:\n{message_json}",
            model,
            json_mode=True
        )

        try:
            message_dict = parse_json_response(fixed_json)

        except ValueError:
            raise ValueError(f'Unable to get valid JSON response. Original Response: {message_json} Attempted Fix: {fixed_json}')

    return message_dict


def stringify_action_value(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value)


def get_first_action_value(action, keys, default=''):
    for key in keys:
        value = stringify_action_value(action.get(key))
        if value:
            return value
    return default


class Player:

    def __init__(self, player_name, player_number, other_players, card, card_list, model):
        self.player_number = player_number
        self.player_name = player_name
        self.other_players = other_players
        self.card = card
        self.card_thought = card
        self.display_card = card
        self.rules_prompt_prefix = open('prompts/rules.txt').read().format(player_name = player_name, other_players = '; '.join(other_players), card = card, card_list = card_list)
        self.memory = []
        self.model = model
        self.prompt_cache_key = hashlib.sha256(self.rules_prompt_prefix.encode('utf-8')).hexdigest()

    def append_memory(self, memory_item):
        self.memory.append(memory_item)

    def run_prompt(self, prompt):
        memory_block = ''
        if len(self.memory) > 0:
            memory_block = '\n\nYou have the following memory of interactions in this game:\n\n' + '\n\n'.join(self.memory)

        messages = [
            {
                'role': 'system',
                'content': self.rules_prompt_prefix
            },
            {
                'role': 'user',
                'content': 'Use your role instructions and the evolving game state to make the best move.'
            },
            {
                'role': 'user',
                'content': memory_block + '\n\n' + prompt
            }
        ]

        response_text = run_model_prompt(
            messages,
            self.model,
            json_mode=True,
            prompt_cache_key=f'werewolf-player-{self.prompt_cache_key}'
        )
        if not response_text:
            print("No text returned from model.")
            return ""

        return response_text

class ConsoleRenderingEngine:

    player_colors = [Fore.YELLOW, Fore.GREEN, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]

    def __init__(self):
        pass

    def get_player_colored_name(self, player):
        return f'{self.player_colors[player.player_number - 1]}{Style.BRIGHT}{player.player_name}{Style.RESET_ALL}'

    def type_line(self, text):
        for char in text:
            print(char, end='', flush=True)
            time.sleep(random.uniform(0.005, 0.015))
        print()

    def render_system_message(self, statement, ref_players=[], ref_cards=[], no_wait=False):
        print()
        ref_players_formatted = []
        for player in ref_players:
            ref_players_formatted.append(self.get_player_colored_name(player))
        ref_cards_formatted = []
        for card in ref_cards:
            ref_cards_formatted.append(f'{Fore.RED}{Style.BRIGHT}{card}{Style.RESET_ALL}')
        print(statement.format(ref_players = ref_players_formatted, ref_cards = ref_cards_formatted));
        if not no_wait:
            time.sleep(random.uniform(1, 3))

    def render_phase(self, phase):
        print()
        print(f'=== The {Fore.RED}{Style.BRIGHT}{phase}{Style.RESET_ALL} phase will now commence. ===')

    def render_game_statement(self, statement):
        print()
        print(f'{Fore.WHITE}{Style.BRIGHT}GAME{Style.RESET_ALL}: ', end='')
        self.type_line(statement)
        time.sleep(random.uniform(1, 3))
        
    def render_player_turn_init(self, player):
        print()
        player_colored_name = self.get_player_colored_name(player)
        print(f'{player_colored_name} (thoughts as {player.card_thought}): ', end='', flush=True)

    def render_player_turn(self, player, statement, reasoning):
        player_colored_name = self.get_player_colored_name(player)
        self.type_line(reasoning)
        time.sleep(random.uniform(1, 3))
        if statement is not None:
            print(f'{player_colored_name}: ', end='')
            self.type_line(statement)

    def render_player_vote(self, player, voted_player, reasoning):
        player_colored_name = self.get_player_colored_name(player)
        self.type_line(reasoning)
        time.sleep(random.uniform(1, 3))
        print(f'{player_colored_name} [{player.display_card}]: ', end='')
        self.type_line(f'I am voting for {voted_player}.')

    def render_vote_results(self, votes, players):
        print()
        print('The votes were:')
        print()
        for player in players:
            if votes[player.player_name] > 0:
                print(f'{player.player_name} : {player.card} : {votes[player.player_name]}')

    def render_game_details(self, player_count, discussion_depth, model):
        print()
        print('## Run Details')
        print()
        print(f'* Model: {model}')
        print(f'* Player Count: {player_count}')
        print(f'* Discussion Depth: {discussion_depth}')
        print()

class MarkdownRenderingEngine:

    def __init__(self):
        print('# Werewolf GPT - Recorded Play')

    def render_system_message(self, statement, ref_players=[], ref_cards=[], no_wait=False):
        print()
        ref_players_formatted = []
        for player in ref_players:
            ref_players_formatted.append(f'**{player.player_name}**')
        ref_cards_formatted = []
        for card in ref_cards:
            ref_cards_formatted.append(f'***{card}***')
        print(statement.format(ref_players = ref_players_formatted, ref_cards = ref_cards_formatted));

    def render_phase(self, phase):
        print()
        print('---')
        print()
        print(f'## The ***{phase}*** phase will now commence.')

    def render_game_statement(self, statement, ref_players=[], ref_cards=[]):
        print()
        print(f'>***GAME:*** {statement}')

    def render_player_turn_init(self, player):
        # Markdown rendering doesn't need to do anything here. This method is called when
        # an AI begins to think of it's actions.
        pass

    def render_player_turn(self, player, statement, reasoning):
        print()
        print(f'***{player.player_name} (thoughts as {player.card_thought}):*** {reasoning}')
        if statement is not None:
            print(f'> **{player.player_name}:** {statement}')

    def render_player_vote(self, player, voted_player, reasoning):
        print()
        print(f'***{player.player_name} (thoughts as {player.card_thought}):*** {reasoning}')
        print(f'> **{player.player_name} [{player.display_card}]:** I am voting for {voted_player}.')

    def render_vote_results(self, votes, players):
        print()
        print('The votes were:')
        print()
        for player in players:
            if votes[player.player_name] > 0:
                print(f'* {player.player_name} : {player.card} : {votes[player.player_name]}')

    def render_game_details(self, player_count, discussion_depth, model):
        print()
        print('## Run Details')
        print()
        print(f'* Model: {model}')
        print(f'* Player Count: {player_count}')
        print(f'* Discussion Depth: {discussion_depth}')


class SilentRenderingEngine:

    def render_system_message(self, statement, ref_players=[], ref_cards=[], no_wait=False):
        pass

    def render_phase(self, phase):
        pass

    def render_game_statement(self, statement, ref_players=[], ref_cards=[]):
        pass

    def render_player_turn_init(self, player):
        pass

    def render_player_turn(self, player, statement, reasoning):
        pass

    def render_player_vote(self, player, voted_player, reasoning):
        pass

    def render_vote_results(self, votes, players):
        pass

    def render_game_details(self, player_count, discussion_depth, model):
        pass


class Game:

    def __init__(self, player_count, discussion_depth, model, render_markdown=False, silent=False, targeted_werewolf_persuasion=False, leader_prior_personality_werewolf_persuasion=False, staged_werewolf_persuasion=False, stage_one_mode='targeted', stage_two_mode='prior', stage_one_rounds=10, stage_two_rounds=5):
        self.player_count = player_count
        self.discussion_depth = discussion_depth
        self.card_list = None
        self.player_names = []
        self.players = []
        self.middle_cards = []
        self.model = model
        self.result = None
        self.dialogue = []
        self.voting_outcome = []
        self.warning = ''
        self.targeted_werewolf_persuasion = targeted_werewolf_persuasion
        self.leader_prior_personality_werewolf_persuasion = leader_prior_personality_werewolf_persuasion
        self.staged_werewolf_persuasion = staged_werewolf_persuasion
        self.stage_one_mode = stage_one_mode
        self.stage_two_mode = stage_two_mode
        self.stage_one_rounds = stage_one_rounds
        self.stage_two_rounds = stage_two_rounds
        self.stage_vote_history = []
        self.vote_change_summary = None
        self.branch_results = {}

        if silent:
            self.rendering_engine = SilentRenderingEngine()
        elif render_markdown:
            self.rendering_engine = MarkdownRenderingEngine()
        else:
            self.rendering_engine = ConsoleRenderingEngine()

    def record_dialogue_turn(self, speaker, phase, utterance, thoughts, stage=None):
        rec_id = len(self.dialogue) + 1
        record = {
            'Rec_Id': rec_id,
            'speaker': speaker,
            'timestamp': f'00:{rec_id:02d}',
            'phase': phase,
            'utterance': utterance or '',
            'thoughts': thoughts or ''
        }
        if stage is not None:
            record['stage'] = stage
        self.dialogue.append(record)

    def append_warning(self, message):
        self.warning = (self.warning + '; ' if self.warning else '') + message

    def get_reasoning(self, action, raw_response, player_name, phase):
        reasoning = get_first_action_value(action, ['reasoning', 'thoughts', 'rationale', 'explanation', 'justification'])
        if reasoning:
            return reasoning

        self.append_warning(f'{player_name} did not provide reasoning during {phase}')
        raw_response = stringify_action_value(raw_response)
        if raw_response:
            return f'No reasoning supplied. Raw response: {raw_response}'
        return 'No reasoning supplied.'

    def get_statement(self, action, player_name):
        statement = get_first_action_value(action, ['statement', 'utterance', 'message', 'response', 'question', 'claim'])
        if statement:
            return statement

        self.append_warning(f'{player_name} did not provide a day statement')
        return 'I need more information before making a claim.'

    def get_targeted_werewolf_reasoning(self, action, raw_response, player_name):
        reasoning = self.get_reasoning(action, raw_response, player_name, 'DAY')
        analysis_fields = [
            ('Influence analysis', action.get('influence_analysis')),
            ('Personality analysis', action.get('personality_analysis')),
            ('Strategy plan', action.get('strategy_plan')),
            ('Target analysis', action.get('target_analysis') or action.get('analysis') or action.get('player_analysis'))
        ]
        for label, analysis in analysis_fields:
            if analysis:
                reasoning += f'\n{label}: {stringify_action_value(analysis)}'
        return reasoning

    def get_player_name_from_action(self, action, keys):
        candidate = get_first_action_value(action, keys)
        if candidate in self.player_names:
            return candidate

        for value in action.values():
            value = stringify_action_value(value)
            if value in self.player_names:
                return value

        return candidate

    def resolve_active_werewolf_prompt_mode(self):
        if self.staged_werewolf_persuasion:
            return None
        if self.leader_prior_personality_werewolf_persuasion:
            return 'prior'
        if self.targeted_werewolf_persuasion:
            return 'targeted'
        return 'normal'

    def uses_targeted_reasoning_mode(self, prompt_mode):
        return prompt_mode in {'targeted', 'prior'}

    def get_werewolf_prompt_file(self, prompt_mode):
        prompt_files = {
            'targeted': 'prompts/werewolf_targeted_day.txt',
            'prior': 'prompts/werewolf_personality_leader_prior_day.txt',
        }
        return prompt_files.get(prompt_mode)

    def get_day_prompt(self, player, default_day_prompt, werewolf_prompt_mode=None):
        active_prompt_mode = werewolf_prompt_mode or self.resolve_active_werewolf_prompt_mode()
        if active_prompt_mode == 'normal' or active_prompt_mode is None or player.card != 'Werewolf':
            return default_day_prompt

        known_werewolves = [
            other_player.player_name
            for other_player in self.players
            if other_player.card == 'Werewolf' and other_player.player_name != player.player_name
        ]
        persuasion_candidates = [
            other_player.player_name
            for other_player in self.players
            if other_player.player_name != player.player_name and other_player.player_name not in known_werewolves
        ]

        known_werewolves_text = '; '.join(known_werewolves) if known_werewolves else 'None'
        persuasion_candidates_text = '; '.join(persuasion_candidates)

        prompt_file = self.get_werewolf_prompt_file(active_prompt_mode)
        if prompt_file is None:
            return default_day_prompt

        return open(prompt_file).read().format(
            known_werewolves=known_werewolves_text,
            persuasion_candidates=persuasion_candidates_text
        )

    def to_ego4d_like_game(self, game_number):
        game_json = {
            'EG_ID': str(uuid.uuid4()),
            'Game_ID': f'GeneratedGame{game_number}',
            'Dialogue': self.dialogue,
            'playerNames': self.player_names,
            'votingOutcome': self.voting_outcome,
            'startRoles': [player.card for player in self.players],
            'endRoles': [player.card for player in self.players],
            'warning': self.warning
        }
        if self.staged_werewolf_persuasion and self.stage_vote_history:
            game_json['stageVotingOutcomes'] = [stage_vote['voting_outcome'] for stage_vote in self.stage_vote_history]
            game_json['stageVoteHistory'] = self.stage_vote_history
            game_json['voteChangeSummary'] = self.vote_change_summary
            game_json['branchResults'] = self.branch_results
        return game_json

    def summarize_werewolf_votes(self, votes):
        werewolf_names = {player.player_name for player in self.players if player.card == 'Werewolf'}
        total_votes = sum(votes.values())
        werewolf_vote_total = sum(votes.get(name, 0) for name in werewolf_names)
        return {
            'werewolf_vote_total': werewolf_vote_total,
            'werewolf_vote_share': werewolf_vote_total / total_votes if total_votes else 0.0,
        }

    def format_vote_results_for_memory(self, stage_label, votes):
        ordered = ', '.join(f'{player.player_name}: {votes[player.player_name]}' for player in self.players)
        return f'{stage_label} VOTE RESULTS: {ordered}'

    def compute_vote_change_summary(self):
        if len(self.stage_vote_history) < 1 or not self.branch_results:
            return None

        interim_stage = self.stage_vote_history[0]
        summary = {
            'intermediate_mode': interim_stage.get('mode'),
            'intermediate_votes_on_werewolves': interim_stage['werewolf_vote_total'],
            'intermediate_werewolf_vote_share': interim_stage['werewolf_vote_share'],
        }

        for branch_key, branch_label in [('same_mode', 'same_mode'), ('switched_mode', 'switched_mode')]:
            branch = self.branch_results.get(branch_key)
            if not branch:
                continue
            changed_voters = [
                player_name
                for player_name in self.player_names
                if interim_stage['vote_by_voter'].get(player_name) != branch['vote_by_voter'].get(player_name)
            ]
            summary[f'{branch_label}_mode'] = branch.get('mode')
            summary[f'{branch_label}_votes_on_werewolves'] = branch['werewolf_vote_total']
            summary[f'{branch_label}_delta_votes_on_werewolves'] = branch['werewolf_vote_total'] - interim_stage['werewolf_vote_total']
            summary[f'{branch_label}_werewolf_vote_share'] = branch['werewolf_vote_share']
            summary[f'{branch_label}_delta_werewolf_vote_share'] = branch['werewolf_vote_share'] - interim_stage['werewolf_vote_share']
            summary[f'{branch_label}_players_changed_votes'] = len(changed_voters)
            summary[f'{branch_label}_changed_voters'] = changed_voters
            summary[f'{branch_label}_werewolf_win'] = branch['werewolf_win']

        if self.branch_results.get('same_mode') and self.branch_results.get('switched_mode'):
            summary['switch_vs_same_delta_votes_on_werewolves'] = (
                self.branch_results['switched_mode']['werewolf_vote_total']
                - self.branch_results['same_mode']['werewolf_vote_total']
            )
            summary['switch_vs_same_delta_werewolf_vote_share'] = (
                self.branch_results['switched_mode']['werewolf_vote_share']
                - self.branch_results['same_mode']['werewolf_vote_share']
            )
            summary['switch_changed_outcome'] = (
                self.branch_results['switched_mode']['werewolf_win']
                != self.branch_results['same_mode']['werewolf_win']
            )
        return summary

    def snapshot_branch_state(self):
        return {
            'dialogue': deepcopy(self.dialogue),
            'warning': self.warning,
            'players_memory': {player.player_name: list(player.memory) for player in self.players},
        }

    def restore_branch_state(self, snapshot):
        self.dialogue = deepcopy(snapshot['dialogue'])
        self.warning = snapshot['warning']
        for player in self.players:
            player.memory = list(snapshot['players_memory'][player.player_name])

    def run_branch(self, snapshot, mode_label, stage_label, discussion_rounds):
        self.restore_branch_state(snapshot)
        self.rendering_engine.render_phase(f'{stage_label} DAY')
        self.day(discussion_depth=discussion_rounds, werewolf_prompt_mode=mode_label, stage_label=stage_label)
        self.rendering_engine.render_phase(f'{stage_label} FINAL VOTE')
        branch_vote = self.vote(stage_label=stage_label, mode_label=mode_label, final_vote=True, record_as_game_result=False)
        branch_vote['dialogue'] = deepcopy(self.dialogue)
        branch_vote['warning'] = self.warning
        return branch_vote

    def play(self):

        self.initialize_game()
        
        self.rendering_engine.render_system_message(open('intro.txt').read().strip(), no_wait=True)

        self.rendering_engine.render_system_message(self.card_list, no_wait=True)
    
        self.introduce_players()

        self.show_middle_cards()

        self.rendering_engine.render_phase('NIGHT')

        self.rendering_engine.render_game_statement('Everyone, close your eyes.')

        self.night_werewolf()

        self.night_minion()

        self.night_mason()

        self.night_seer()

        self.rendering_engine.render_phase('DAY')

        if self.staged_werewolf_persuasion:
            self.day(discussion_depth=self.stage_one_rounds, werewolf_prompt_mode=self.stage_one_mode, stage_label='STAGE_1')
            self.rendering_engine.render_phase('INTERIM VOTE')
            interim_vote = self.vote(stage_label='INTERMEDIATE', mode_label=self.stage_one_mode, final_vote=False, reveal_results=False)
            branch_snapshot = self.snapshot_branch_state()
            same_mode_branch = self.run_branch(branch_snapshot, self.stage_one_mode, 'SAME_MODE', self.stage_two_rounds)
            switched_mode_branch = self.run_branch(branch_snapshot, self.stage_two_mode, 'SWITCH_MODE', self.stage_two_rounds)
            self.branch_results = {
                'same_mode': same_mode_branch,
                'switched_mode': switched_mode_branch,
            }
            self.stage_vote_history = [interim_vote]
            self.vote_change_summary = self.compute_vote_change_summary()

            self.dialogue = deepcopy(switched_mode_branch['dialogue'])
            self.warning = switched_mode_branch['warning']
            self.voting_outcome = switched_mode_branch['voting_outcome']
            self.result = {
                'winner': switched_mode_branch['winner'],
                'werewolf_win': switched_mode_branch['werewolf_win'],
                'result': switched_mode_branch['result'],
                'targeted_werewolf_persuasion': self.targeted_werewolf_persuasion,
                'leader_prior_personality_werewolf_persuasion': self.leader_prior_personality_werewolf_persuasion,
                'staged_werewolf_persuasion': True,
                'votes': switched_mode_branch['votes'],
                'players': [
                    {
                        'name': player.player_name,
                        'card': player.card
                    }
                    for player in self.players
                ],
                'middle_cards': self.middle_cards,
                'stage_vote_history': self.stage_vote_history,
                'branch_results': self.branch_results,
                'vote_change_summary': self.vote_change_summary,
            }
        else:
            self.day()

            self.rendering_engine.render_phase('VOTE')

            self.vote()

        self.rendering_engine.render_game_details(self.player_count, self.discussion_depth, self.model)

        return self.result

    def initialize_game(self):
        if self.player_count < 3 or self.player_count > 5:
            raise ValueError('Number of players must be between 3 and 5 inclusive.')

        alloted_cards = ['Werewolf', 'Werewolf', 'Seer', 'Mason', 'Mason']

        while len(alloted_cards) < self.player_count + 3:
            if 'Minion' not in alloted_cards:
                alloted_cards.append('Minion')
            else:
                alloted_cards.append('Villager')

        non_werewolf_cards = [card for card in alloted_cards if card != 'Werewolf']
        random.shuffle(non_werewolf_cards)

        player_cards = ['Werewolf'] + non_werewolf_cards[:self.player_count - 1]
        middle_cards = ['Werewolf'] + non_werewolf_cards[self.player_count - 1:]

        random.shuffle(player_cards)
        random.shuffle(middle_cards)

        dealt_cards = player_cards + middle_cards
        card_list = '* ' + '\n* '.join(dealt_cards)
        self.card_list = card_list

        self.player_names = self.get_player_names(self.player_count)
        self.players = [Player(name, i, self.get_other_players(i, self.player_names), player_cards[i - 1], card_list, self.model) for i, name in enumerate(self.player_names, 1)]
        self.middle_cards = middle_cards

    def introduce_players(self):
        for player in self.players:
            self.rendering_engine.render_system_message(f'Player number {player.player_number} is named {{ref_players[0]}}, and they have the {{ref_cards[0]}} card.',
                ref_players=[player], ref_cards=[player.card], no_wait=True)

    def show_middle_cards(self):
        self.rendering_engine.render_system_message('The cards face-down in the middle of the board are {ref_cards[0]}, {ref_cards[1]}, and {ref_cards[2]}.',
            ref_cards=self.middle_cards)    

    def night_werewolf(self):
        self.rendering_engine.render_game_statement('Werewolves, wake up and look for other Werewolves.')

        werewolf_players = [player for player in self.players if player.card == 'Werewolf']

        if len(werewolf_players) == 0:
            self.rendering_engine.render_system_message('There are no werewolves in play.')
        elif len(werewolf_players) == 1:
            middle_card = random.choice(self.middle_cards)

            message = f'GAME: You are the only werewolf. You can deduce that the other werewolf card is in the middle cards. ' \
                + 'You randomly picked one of the center cards and were able to see that it was: {middle_card}'
            werewolf_players[0].append_memory(message)

            self.rendering_engine.render_system_message('There is one werewolf in play, {ref_players[0]}. The werewolf randomly viewed the middle card: {ref_cards[0]}.', 
                ref_players = werewolf_players, ref_cards = [middle_card])
        else:
            message_one = f'GAME (NIGHT PHASE): You are have seen that the other werewolf is {werewolf_players[1].player_name}.'
            werewolf_players[0].append_memory(message_one)
            message_two = f'GAME (NIGHT PHASE): You are have seen that the other werewolf is {werewolf_players[0].player_name}.'
            werewolf_players[1].append_memory(message_two)

            self.rendering_engine.render_system_message('There are two werewolves in play, {ref_players[0]} and {ref_players[1]}. They are both now aware of each other.',
                ref_players = werewolf_players)

        self.rendering_engine.render_game_statement('Werewolves, close your eyes.')

    def night_minion(self):
        self.rendering_engine.render_game_statement('Minion, wake up. Werewolves, stick out your thumb so the Minion can see who you are.')

        minion_players = [player for player in self.players if player.card == 'Minion']
        werewolf_players = [player for player in self.players if player.card == 'Werewolf']

        if len(minion_players) == 0:
            self.rendering_engine.render_system_message('There are no minions in play.')
        else:
            if len(werewolf_players) == 0:
                message = 'GAME (NIGHT PHASE): There are no werewolves in play. Both werewolves are currently in the middle cards.'
                minion_players[0].append_memory(message)

                self.rendering_engine.render_system_message('{ref_players[0]} is a minion and is aware that no one is a werewolf.',
                    ref_players = minion_players)
            elif len(werewolf_players) == 1:
                message = f'GAME (NIGHT PHASE): There are is one werewolf in play. {werewolf_players[0].player_name} is a werewolf. They do not know that you are the minion. ' \
                    + 'The other werewolf is in the middle cards.'
                minion_players[0].append_memory(message)

                self.rendering_engine.render_system_message('{ref_players[0]} is a minion and is aware that {ref_players[1]} is a werewolf.',
                    ref_players = minion_players + werewolf_players)
            else:
                message = f'GAME (NIGHT PHASE): There are two werewolves in play. {werewolf_players[0].player_name} and {werewolf_players[1].player_name} are a werewolves. ' \
                    + 'They do not know that you are the minion.'
                minion_players[0].append_memory(message)

                self.rendering_engine.render_system_message('{ref_players[0]} is a minion and is aware that both {ref_players[1]} and {ref_players[2]} are werewolves.',
                    ref_players = minion_players + werewolf_players)

        self.rendering_engine.render_game_statement('Werewolves, put your thumbs away. Minion, close your eyes.')

    def night_mason(self):
        self.rendering_engine.render_game_statement('Masons, wake up and look for other Masons.')

        mason_players = [player for player in self.players if player.card == 'Mason']

        if len(mason_players) == 0:
            self.rendering_engine.render_system_message('There are no masons in play.')
        elif len(mason_players) == 1:
            message = f'GAME: You are the only mason. You can deduce that the other mason card is in the middle cards.'
            mason_players[0].append_memory(message)

            self.rendering_engine.render_system_message('There is one mason in play, {ref_players[0]}. They are aware they are the only mason in play.',
                ref_players = mason_players)
        else:
            message_one = f'GAME (NIGHT PHASE): You are have seen that the other mason is {mason_players[1].player_name}.'
            mason_players[0].append_memory(message_one)
            message_two = f'GAME (NIGHT PHASE): You are have seen that the other mason is {mason_players[0].player_name}.'
            mason_players[1].append_memory(message_two)

            self.rendering_engine.render_system_message('There are two masons in play, {ref_players[0]} and {ref_players[1]}. ' \
                + 'They are both now aware of each other.', ref_players = mason_players)

        self.rendering_engine.render_game_statement('Masons, close your eyes.')

    def night_seer(self):
        self.rendering_engine.render_game_statement('Seer, wake up. You may look at another player’s card or two of the center cards.')

        seer_players = [player for player in self.players if player.card == 'Seer']

        if len(seer_players) == 0:
            self.rendering_engine.render_system_message('There are no seers in play.')
        else:
            self.rendering_engine.render_system_message('There is one seer in play, {ref_players[0]}. They are thinking about their action.'
                , ref_players = seer_players)
            
            self.rendering_engine.render_player_turn_init(seer_players[0])

            prompt = open('prompts/seer.txt').read()
            response = seer_players[0].run_prompt(prompt)

            action = return_dict_from_json_or_fix(response, self.model)
            reasoning = self.get_reasoning(action, response, seer_players[0].player_name, 'NIGHT')
            choice = get_first_action_value(action, ['choice', 'action', 'selection'], 'center').lower()
            if choice not in ['player', 'center']:
                self.append_warning(f'{seer_players[0].player_name} supplied invalid seer choice {choice}')
                choice = 'center'
            
            thoughts_message = f'NIGHT ROUND THOUGHTS: {reasoning}'
            seer_players[0].append_memory(thoughts_message)
            self.record_dialogue_turn(seer_players[0].player_name, 'NIGHT', f'I choose to look at {choice}.', reasoning)

            self.rendering_engine.render_player_turn(seer_players[0], None, reasoning)

            if choice == 'player':
                player_name = self.get_player_name_from_action(action, ['player', 'target_player', 'selected_player'])
                player = next((p for p in self.players if p.player_name == player_name), None)
                if player is None or player.player_name == seer_players[0].player_name:
                    self.append_warning(f'{seer_players[0].player_name} supplied invalid seer target {player_name} and viewed center cards instead')
                    viewed_cards = random.sample(self.middle_cards, k=2)

                    message = f'GAME (NIGHT PHASE): You have seen two cards in the center of the table: {viewed_cards[0]} and {viewed_cards[1]}'
                    seer_players[0].append_memory(message)

                    self.rendering_engine.render_system_message('The seer looked at two cards from the center of the table and saw the cards {ref_cards[0]} and {ref_cards[1]}',
                        ref_cards = viewed_cards)
                    self.rendering_engine.render_game_statement('Seer, close your eyes.')
                    return
                
                message = f'GAME (NIGHT PHASE): You are have seen that {player.player_name} has the card: {player.card}.'
                seer_players[0].append_memory(message)

                self.rendering_engine.render_system_message('The seer looked at a card from {ref_players[0]} and saw the card {ref_cards[0]}',
                    ref_players = [player], ref_cards = [player.card])
            else:
                viewed_cards = random.sample(self.middle_cards, k=2)
                
                message = f'GAME (NIGHT PHASE): You have seen two cards in the center of the table: {viewed_cards[0]} and {viewed_cards[1]}'
                seer_players[0].append_memory(message)
                
                self.rendering_engine.render_system_message('The seer looked at two cards from the center of the table and saw the cards {ref_cards[0]} and {ref_cards[1]}',
                    ref_cards = viewed_cards)

        self.rendering_engine.render_game_statement('Seer, close your eyes.')

    def day(self, discussion_depth=None, werewolf_prompt_mode=None, stage_label=None):
        if stage_label == 'STAGE_2':
            self.rendering_engine.render_game_statement('Everyone, continue the discussion with the interim vote in mind.')
        else:
            self.rendering_engine.render_game_statement('Everyone, Wake up!')

        day_prompt = open('prompts/day.txt').read()

        pointer = -1

        discussion_count = 0
        discussion_depth = discussion_depth if discussion_depth is not None else self.discussion_depth

        target_player = None

        while discussion_count < discussion_depth:
            if target_player is None:
                pointer += 1
                if pointer > len(self.players) - 1:
                    pointer = 0
                player = self.players[pointer]
            else:
                try:
                    player = [player for player in self.players if player.player_name == target_player][0]
                    target_player = None
                except:
                    print()
                    print(f'SYSTEM NOTE: The AI supplied {target_player} as the target player. To avoid a crash, we will skip this directed discussion.')
                    print()

            self.rendering_engine.render_player_turn_init(player)

            prompt = self.get_day_prompt(player, day_prompt, werewolf_prompt_mode=werewolf_prompt_mode)
            response = player.run_prompt(prompt)

            action = return_dict_from_json_or_fix(response, self.model)
            if (
                self.targeted_werewolf_persuasion
                or self.leader_prior_personality_werewolf_persuasion
                or self.uses_targeted_reasoning_mode(werewolf_prompt_mode)
            ) and player.card == 'Werewolf':
                reasoning = self.get_targeted_werewolf_reasoning(action, response, player.player_name)
            else:
                reasoning = self.get_reasoning(action, response, player.player_name, 'DAY')
            statement = self.get_statement(action, player.player_name)
            if any(key in action for key in ['target_player', 'target', 'player']):
                target_player = self.get_player_name_from_action(action, ['target_player', 'target', 'player'])
                if target_player not in self.player_names or target_player == player.player_name:
                    self.append_warning(f'{player.player_name} supplied invalid day target {target_player}')
                    target_player = None

            thoughts_message = f'DAY ROUND THOUGHTS: {reasoning}'
            player.append_memory(thoughts_message)

            message = f'{player.player_name}: {statement}'
            for i_player in self.players:
                i_player.append_memory(message)

            self.record_dialogue_turn(player.player_name, 'DAY', statement, reasoning, stage=stage_label)

            self.rendering_engine.render_player_turn(
                player,
                statement,
                reasoning
            )

            discussion_count += 1

    def vote(self, stage_label='FINAL', mode_label=None, final_vote=True, record_as_game_result=True, reveal_results=True):
        vote_intro = 'It\'s time to vote!' if final_vote else 'It\'s time for an interim vote before the mode switch!'
        self.rendering_engine.render_game_statement(vote_intro)

        vote_prompt = open('prompts/vote.txt').read()

        votes = {}
        vote_by_player_number = {}
        vote_by_voter = {}

        for player in self.players:
            votes[player.player_name] = 0

        for player in self.players:
            self.rendering_engine.render_player_turn_init(player)

            response = player.run_prompt(vote_prompt)

            action = return_dict_from_json_or_fix(response, self.model)
            reasoning = self.get_reasoning(action, response, player.player_name, 'VOTE')
            voted_player = self.get_player_name_from_action(action, ['voted_player', 'vote', 'player', 'target_player', 'target'])

            if voted_player not in votes or voted_player == player.player_name:
                voted_player = random.choice([p.player_name for p in self.players if p.player_name != player.player_name])
                self.append_warning(f'{player.player_name} supplied an invalid vote and was assigned {voted_player}')

            self.rendering_engine.render_player_vote(player, voted_player, reasoning)

            votes[voted_player] += 1
            vote_by_player_number[player.player_number] = next(p.player_number for p in self.players if p.player_name == voted_player)
            vote_by_voter[player.player_name] = voted_player
            self.record_dialogue_turn(player.player_name, 'VOTE', f'I am voting for {voted_player}.', reasoning, stage=stage_label)

        voting_outcome = [
            vote_by_player_number.get(player.player_number, 'NA')
            for player in self.players
        ]

        if reveal_results:
            self.rendering_engine.render_vote_results(votes, self.players)

        werewolf_vote_summary = self.summarize_werewolf_votes(votes)
        stage_vote_summary = {
            'stage': stage_label,
            'mode': mode_label,
            'votes': votes,
            'vote_by_voter': vote_by_voter,
            'voting_outcome': voting_outcome,
            'werewolf_vote_total': werewolf_vote_summary['werewolf_vote_total'],
            'werewolf_vote_share': werewolf_vote_summary['werewolf_vote_share'],
        }
        if self.staged_werewolf_persuasion:
            self.stage_vote_history.append(stage_vote_summary)

        if reveal_results:
            vote_results_memory = self.format_vote_results_for_memory(stage_label, votes)
            for player in self.players:
                player.append_memory(vote_results_memory)

        if not final_vote:
            if reveal_results:
                self.rendering_engine.render_game_statement(vote_results_memory)
            else:
                self.rendering_engine.render_game_statement('The interim votes have been recorded privately. Continue the discussion without seeing the tally.')
            return stage_vote_summary

        self.voting_outcome = voting_outcome

        max_votes = max(votes.values())

        if max_votes == 1:
            werewolf_players = [player for player in self.players if player.card == 'Werewolf']
            minion_players = [player for player in self.players if player.card == 'Minion']

            if len(werewolf_players) + len(minion_players) == 0:
                game_result = 'No player was voted out. The villagers win.'
            else:
                game_result = 'No player was voted out. The werewolves win.'
        else:
            players_with_max_votes = [player for player in self.players if votes[player.player_name] == max_votes]

            if len(players_with_max_votes) > 1:
                game_result = f'There was a tie between {", ".join([player.player_name for player in players_with_max_votes])}.'
                if all(player.card != 'Werewolf' for player in players_with_max_votes):
                    game_result += ' The werewolves win.'
                else:
                    game_result += ' The villagers win.'
            else:
                killed_player = players_with_max_votes[0]
                game_result = f'{killed_player.player_name} was killed.'

                if killed_player.card == 'Werewolf':
                    game_result += ' The villagers win.'
                else:
                    game_result += ' The werewolves win.'

        self.rendering_engine.render_game_statement(game_result)

        winner = 'werewolves' if game_result.endswith('The werewolves win.') else 'villagers'
        final_result = {
            'winner': winner,
            'werewolf_win': winner == 'werewolves',
            'result': game_result,
            'targeted_werewolf_persuasion': self.targeted_werewolf_persuasion,
            'leader_prior_personality_werewolf_persuasion': self.leader_prior_personality_werewolf_persuasion,
            'staged_werewolf_persuasion': self.staged_werewolf_persuasion,
            'votes': votes,
            'players': [
                {
                    'name': player.player_name,
                    'card': player.card
                }
                for player in self.players
            ],
            'middle_cards': self.middle_cards
        }
        final_result.update(stage_vote_summary)
        if record_as_game_result:
            self.vote_change_summary = self.compute_vote_change_summary()
            self.result = final_result
            if self.staged_werewolf_persuasion:
                self.result['stage_vote_history'] = self.stage_vote_history
                self.result['vote_change_summary'] = self.vote_change_summary
            return self.result
        return final_result

    def get_player_names(self, player_count):
        name_options = ['Alexandra', 'Alexia', 'Andrei', 'Cristina', 'Dragos', 'Dracula', 'Emil', 'Ileana', 'Kraven', 'Larisa', 'Lucian', 'Marius', 'Michael', 'Mircea', 'Radu', 'Semira', 'Selene', 'Stefan', 'Viktor', 'Vladimir']
        return random.sample(name_options, player_count)

    def get_other_players(self, player_number, player_names):
        return [name for i, name in enumerate(player_names, 1) if i != player_number]

#game = Game(player_count = 5, discussion_depth = 20, use_gpt4 = True    , render_markdown = False)
#game.play()


def write_json_file(json_file, payload):
    results_dir = os.path.dirname(json_file)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def run_batch_game(game_number, player_count, discussion_depth, model, targeted_werewolf_persuasion, leader_prior_personality_werewolf_persuasion, staged_werewolf_persuasion, stage_one_mode, stage_two_mode, stage_one_rounds, stage_two_rounds):
    game = Game(
        player_count=player_count,
        discussion_depth=discussion_depth,
        model=model,
        silent=True,
        targeted_werewolf_persuasion=targeted_werewolf_persuasion,
        leader_prior_personality_werewolf_persuasion=leader_prior_personality_werewolf_persuasion,
        staged_werewolf_persuasion=staged_werewolf_persuasion,
        stage_one_mode=stage_one_mode,
        stage_two_mode=stage_two_mode,
        stage_one_rounds=stage_one_rounds,
        stage_two_rounds=stage_two_rounds
    )
    result = game.play()
    result['game_number'] = game_number
    result['status'] = 'completed'
    return result, game.to_ego4d_like_game(game_number)


def record_batch_result(payload, games_json, result, game_json, results_file, games_json_file):
    if result['status'] == 'completed':
        payload['summary']['games_completed'] += 1
        if result['werewolf_win']:
            payload['summary']['werewolf_wins'] += 1
        games_json.append(game_json)
    else:
        payload['summary']['games_failed'] += 1

    completed = payload['summary']['games_completed']
    if completed > 0:
        payload['summary']['werewolf_win_rate'] = payload['summary']['werewolf_wins'] / completed

    payload['games'].append(result)
    payload['games'].sort(key=lambda item: item['game_number'])
    games_json.sort(key=lambda item: int(item['Game_ID'].replace('GeneratedGame', '')))
    write_json_file(results_file, payload)
    write_json_file(games_json_file, games_json)


def finalize_batch_summary(payload):
    completed_games = [game for game in payload['games'] if game.get('status') == 'completed']
    summary = payload['summary']
    if not summary.get('staged_werewolf_persuasion'):
        return

    intermediate_vote_totals = []
    same_mode_vote_totals = []
    switched_mode_vote_totals = []
    intermediate_vote_shares = []
    same_mode_vote_shares = []
    switched_mode_vote_shares = []
    same_mode_changed_vote_counts = []
    switched_mode_changed_vote_counts = []
    games_with_switch_vote_change = 0
    same_mode_wins = 0
    switched_mode_wins = 0
    games_with_outcome_change = 0

    for game in completed_games:
        vote_change_summary = game.get('vote_change_summary')
        if not vote_change_summary:
            continue
        intermediate_vote_totals.append(vote_change_summary['intermediate_votes_on_werewolves'])
        same_mode_vote_totals.append(vote_change_summary['same_mode_votes_on_werewolves'])
        switched_mode_vote_totals.append(vote_change_summary['switched_mode_votes_on_werewolves'])
        intermediate_vote_shares.append(vote_change_summary['intermediate_werewolf_vote_share'])
        same_mode_vote_shares.append(vote_change_summary['same_mode_werewolf_vote_share'])
        switched_mode_vote_shares.append(vote_change_summary['switched_mode_werewolf_vote_share'])
        same_mode_changed_vote_counts.append(vote_change_summary['same_mode_players_changed_votes'])
        switched_mode_changed_vote_counts.append(vote_change_summary['switched_mode_players_changed_votes'])
        if vote_change_summary['switched_mode_players_changed_votes'] > 0:
            games_with_switch_vote_change += 1
        same_mode_wins += int(vote_change_summary['same_mode_werewolf_win'])
        switched_mode_wins += int(vote_change_summary['switched_mode_werewolf_win'])
        games_with_outcome_change += int(vote_change_summary['switch_changed_outcome'])

    if not intermediate_vote_totals:
        return

    game_count = len(intermediate_vote_totals)
    summary['avg_intermediate_votes_on_werewolves'] = sum(intermediate_vote_totals) / game_count
    summary['avg_same_mode_final_votes_on_werewolves'] = sum(same_mode_vote_totals) / game_count
    summary['avg_switched_mode_final_votes_on_werewolves'] = sum(switched_mode_vote_totals) / game_count
    summary['avg_delta_same_mode_votes_on_werewolves'] = (
        summary['avg_same_mode_final_votes_on_werewolves'] - summary['avg_intermediate_votes_on_werewolves']
    )
    summary['avg_delta_switched_mode_votes_on_werewolves'] = (
        summary['avg_switched_mode_final_votes_on_werewolves'] - summary['avg_intermediate_votes_on_werewolves']
    )
    summary['avg_intermediate_werewolf_vote_share'] = sum(intermediate_vote_shares) / game_count
    summary['avg_same_mode_final_werewolf_vote_share'] = sum(same_mode_vote_shares) / game_count
    summary['avg_switched_mode_final_werewolf_vote_share'] = sum(switched_mode_vote_shares) / game_count
    summary['avg_delta_same_mode_werewolf_vote_share'] = (
        summary['avg_same_mode_final_werewolf_vote_share'] - summary['avg_intermediate_werewolf_vote_share']
    )
    summary['avg_delta_switched_mode_werewolf_vote_share'] = (
        summary['avg_switched_mode_final_werewolf_vote_share'] - summary['avg_intermediate_werewolf_vote_share']
    )
    summary['avg_same_mode_changed_votes'] = sum(same_mode_changed_vote_counts) / game_count
    summary['avg_switched_mode_changed_votes'] = sum(switched_mode_changed_vote_counts) / game_count
    summary['same_mode_werewolf_wins'] = same_mode_wins
    summary['switched_mode_werewolf_wins'] = switched_mode_wins
    summary['same_mode_werewolf_win_rate'] = same_mode_wins / game_count
    summary['switched_mode_werewolf_win_rate'] = switched_mode_wins / game_count
    summary['games_with_any_switched_vote_change'] = games_with_switch_vote_change
    summary['games_with_any_switched_vote_change_rate'] = games_with_switch_vote_change / game_count
    summary['games_with_branch_outcome_change'] = games_with_outcome_change
    summary['games_with_branch_outcome_change_rate'] = games_with_outcome_change / game_count


def run_batch(player_count, discussion_depth, model_mode, model, game_count, results_file, games_json_file, targeted_werewolf_persuasion, leader_prior_personality_werewolf_persuasion, staged_werewolf_persuasion, stage_one_mode, stage_two_mode, stage_one_rounds, stage_two_rounds, parallel_games):
    payload = {
        'summary': {
            'model_mode': model_mode,
            'model': model,
            'player_count': player_count,
            'discussion_depth': discussion_depth,
            'targeted_werewolf_persuasion': targeted_werewolf_persuasion,
            'leader_prior_personality_werewolf_persuasion': leader_prior_personality_werewolf_persuasion,
            'staged_werewolf_persuasion': staged_werewolf_persuasion,
            'stage_one_mode': stage_one_mode if staged_werewolf_persuasion else None,
            'stage_two_mode': stage_two_mode if staged_werewolf_persuasion else None,
            'stage_one_rounds': stage_one_rounds if staged_werewolf_persuasion else None,
            'stage_two_rounds': stage_two_rounds if staged_werewolf_persuasion else None,
            'parallel_games': parallel_games,
            'games_requested': game_count,
            'games_completed': 0,
            'games_failed': 0,
            'werewolf_wins': 0,
            'werewolf_win_rate': 0.0,
            'avg_intermediate_votes_on_werewolves': 0.0,
            'avg_same_mode_final_votes_on_werewolves': 0.0,
            'avg_switched_mode_final_votes_on_werewolves': 0.0,
            'avg_delta_same_mode_votes_on_werewolves': 0.0,
            'avg_delta_switched_mode_votes_on_werewolves': 0.0,
            'avg_intermediate_werewolf_vote_share': 0.0,
            'avg_same_mode_final_werewolf_vote_share': 0.0,
            'avg_switched_mode_final_werewolf_vote_share': 0.0,
            'avg_delta_same_mode_werewolf_vote_share': 0.0,
            'avg_delta_switched_mode_werewolf_vote_share': 0.0,
            'avg_same_mode_changed_votes': 0.0,
            'avg_switched_mode_changed_votes': 0.0,
            'same_mode_werewolf_wins': 0,
            'switched_mode_werewolf_wins': 0,
            'same_mode_werewolf_win_rate': 0.0,
            'switched_mode_werewolf_win_rate': 0.0,
            'games_with_any_switched_vote_change': 0,
            'games_with_any_switched_vote_change_rate': 0.0,
            'games_with_branch_outcome_change': 0,
            'games_with_branch_outcome_change_rate': 0.0,
            'generated_at': datetime.now(timezone.utc).isoformat()
        },
        'games': []
    }
    games_json = []

    if parallel_games <= 1:
        for game_number in range(1, game_count + 1):
            try:
                result, game_json = run_batch_game(game_number, player_count, discussion_depth, model, targeted_werewolf_persuasion, leader_prior_personality_werewolf_persuasion, staged_werewolf_persuasion, stage_one_mode, stage_two_mode, stage_one_rounds, stage_two_rounds)
            except Exception as e:
                result = {
                    'game_number': game_number,
                    'status': 'failed',
                    'error': str(e),
                    'winner': None,
                    'werewolf_win': None,
                    'result': None
                }
                game_json = None
                click.echo(f'Game {game_number} error: {e}', err=True)

            record_batch_result(payload, games_json, result, game_json, results_file, games_json_file)
            click.echo(f'Game {game_number}/{game_count}: {result["status"]}', err=True)
    else:
        finished = 0
        with ThreadPoolExecutor(max_workers=parallel_games) as executor:
            futures = {
                executor.submit(run_batch_game, game_number, player_count, discussion_depth, model, targeted_werewolf_persuasion, leader_prior_personality_werewolf_persuasion, staged_werewolf_persuasion, stage_one_mode, stage_two_mode, stage_one_rounds, stage_two_rounds): game_number
                for game_number in range(1, game_count + 1)
            }

            for future in as_completed(futures):
                game_number = futures[future]
                try:
                    result, game_json = future.result()
                except Exception as e:
                    result = {
                        'game_number': game_number,
                        'status': 'failed',
                        'error': str(e),
                        'winner': None,
                        'werewolf_win': None,
                        'result': None
                    }
                    game_json = None
                    click.echo(f'Game {game_number} error: {e}', err=True)

                record_batch_result(payload, games_json, result, game_json, results_file, games_json_file)
                finished += 1
                click.echo(f'Game {game_number}/{game_count}: {result["status"]} ({finished}/{game_count} finished)', err=True)

    finalize_batch_summary(payload)
    write_json_file(results_file, payload)
    write_json_file(games_json_file, games_json)
    return payload['summary']

@click.command()
@click.option('--player-count', type=int, default=5, help='Number of players')
@click.option('--discussion-depth', type=int, default=20, help='Number of discussion rounds')
@click.option('--model-mode', type=click.Choice(list(MODEL_PRESETS.keys())), default=DEFAULT_MODEL_MODE, show_default=True, help='Preset model/API mode')
@click.option('--model', default=None, help='Optional model override used for every player and JSON repair call')
@click.option('--api-base-url', default=None, help='Optional OpenAI-compatible API base URL override')
@click.option('--games', type=int, default=5, show_default=True, help='Number of games to run. Use 1000 for a baseline batch.')
@click.option('--parallel-games', type=int, default=1, show_default=True, help='Number of independent games to run concurrently during batch mode')
@click.option('--results-file', default=DEFAULT_RESULTS_FILE, show_default=True, help='JSON file for batch results')
@click.option('--games-json-file', default=DEFAULT_GAMES_JSON_FILE, show_default=True, help='Ego4D-like JSON file containing generated game dialogue')
@click.option('--targeted-werewolf-persuasion', is_flag=True, default=False, help='Make Werewolf players analyze candidates and target the highest-utility player during day discussion')
@click.option('--leader-prior-personality-werewolf-persuasion', is_flag=True, default=False, help='Make Werewolf players use human-game priors about influenceable discussion leaders on top of targeted persuasion')
@click.option('--staged-werewolf-persuasion', is_flag=True, default=False, help='Run a two-stage day discussion with an interim vote after stage one and a mode switch before stage two')
@click.option('--stage-one-mode', type=click.Choice(WEREWOLF_PROMPT_MODE_CHOICES), default='targeted', show_default=True, help='Werewolf prompt mode used during stage one of staged persuasion')
@click.option('--stage-two-mode', type=click.Choice(WEREWOLF_PROMPT_MODE_CHOICES), default='prior', show_default=True, help='Werewolf prompt mode used during stage two of staged persuasion')
@click.option('--stage-one-rounds', type=int, default=10, show_default=True, help='Number of day discussion turns before the interim vote in staged persuasion mode')
@click.option('--stage-two-rounds', type=int, default=10, show_default=True, help='Number of day discussion turns after the mode switch in staged persuasion mode')
@click.option('--use-gpt4', is_flag=True, default=False, help='Legacy alias: use openai/gpt-4 instead of the default model')
@click.option('--render-markdown', is_flag=True, default=False, help='Render output as markdown')
def play_game(player_count, discussion_depth, model_mode, model, api_base_url, games, parallel_games, results_file, games_json_file, targeted_werewolf_persuasion, leader_prior_personality_werewolf_persuasion, staged_werewolf_persuasion, stage_one_mode, stage_two_mode, stage_one_rounds, stage_two_rounds, use_gpt4, render_markdown):
    model_config = resolve_model_config(model_mode, model, api_base_url)
    model = model_config['model']
    api_base_url = model_config['api_base_url']

    if use_gpt4:
        model = 'openai/gpt-4'
        api_base_url = OPENROUTER_API_BASE_URL

    configure_client(api_base_url)

    if games < 1:
        raise click.ClickException('--games must be at least 1.')
    if parallel_games < 1:
        raise click.ClickException('--parallel-games must be at least 1.')
    if stage_one_rounds < 1 or stage_two_rounds < 1:
        raise click.ClickException('--stage-one-rounds and --stage-two-rounds must both be at least 1.')
    if staged_werewolf_persuasion:
        discussion_depth = stage_one_rounds + stage_two_rounds

    if games > 1:
        parallel_games = min(parallel_games, games)
        summary = run_batch(player_count, discussion_depth, model_mode, model, games, results_file, games_json_file, targeted_werewolf_persuasion, leader_prior_personality_werewolf_persuasion, staged_werewolf_persuasion, stage_one_mode, stage_two_mode, stage_one_rounds, stage_two_rounds, parallel_games)
        click.echo()
        click.echo(f'Batch complete. Results saved to {results_file}')
        click.echo(f'Generated game dialogue saved to {games_json_file}')
        click.echo(f'Parallel games: {summary["parallel_games"]}')
        click.echo(f'Games completed: {summary["games_completed"]}')
        click.echo(f'Games failed: {summary["games_failed"]}')
        click.echo(f'Werewolf wins: {summary["werewolf_wins"]}')
        click.echo(f'Werewolf win rate: {summary["werewolf_win_rate"]:.2%}')
        if summary.get('staged_werewolf_persuasion'):
            click.echo(f'Stage one mode: {summary["stage_one_mode"]} ({summary["stage_one_rounds"]} rounds)')
            click.echo(f'Stage two mode: {summary["stage_two_mode"]} ({summary["stage_two_rounds"]} rounds)')
            click.echo(f'Avg intermediate votes on Werewolves: {summary["avg_intermediate_votes_on_werewolves"]:.2f}')
            click.echo(f'Avg same-mode final votes on Werewolves: {summary["avg_same_mode_final_votes_on_werewolves"]:.2f}')
            click.echo(f'Avg switched-mode final votes on Werewolves: {summary["avg_switched_mode_final_votes_on_werewolves"]:.2f}')
            click.echo(f'Avg switched delta in votes on Werewolves: {summary["avg_delta_switched_mode_votes_on_werewolves"]:+.2f}')
            click.echo(f'Same-mode Werewolf win rate: {summary["same_mode_werewolf_win_rate"]:.2%}')
            click.echo(f'Switched-mode Werewolf win rate: {summary["switched_mode_werewolf_win_rate"]:.2%}')
            click.echo(f'Games with any switched-branch vote change: {summary["games_with_any_switched_vote_change"]} ({summary["games_with_any_switched_vote_change_rate"]:.2%})')
            click.echo(f'Games where switching changed the final winner: {summary["games_with_branch_outcome_change"]} ({summary["games_with_branch_outcome_change_rate"]:.2%})')
    else:
        game = Game(
            player_count=player_count,
            discussion_depth=discussion_depth,
            model=model,
            render_markdown=render_markdown,
            targeted_werewolf_persuasion=targeted_werewolf_persuasion,
            leader_prior_personality_werewolf_persuasion=leader_prior_personality_werewolf_persuasion,
            staged_werewolf_persuasion=staged_werewolf_persuasion,
            stage_one_mode=stage_one_mode,
            stage_two_mode=stage_two_mode,
            stage_one_rounds=stage_one_rounds,
            stage_two_rounds=stage_two_rounds
        )
        game.play()

if __name__ == '__main__':
    play_game()
