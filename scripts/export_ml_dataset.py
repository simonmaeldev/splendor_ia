#!/usr/bin/env python3
"""
ML Dataset Export Script for Splendor

This script extracts game state and action data from the SQLite database and exports it
to a CSV file formatted for machine learning training. It transforms raw game data into
a standardized 381-dimensional input vector representing the complete board state,
paired with multi-head output vectors representing the action taken.

Features:
- Extracts 381-dimensional state vectors with proper player rotation and NaN padding
- Encodes actions into 7 output heads for multi-head classification
- Handles variable player counts (2-4 players)
- Provides real-time progress tracking
- Generates comprehensive dataset summary

Usage:
    python scripts/export_ml_dataset.py
    python scripts/export_ml_dataset.py --limit 100 --output data/test.csv
    python scripts/export_ml_dataset.py --validate --quiet
"""

import sqlite3
import csv
import sys
import argparse
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

# Import game constants
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from splendor.constants import WHITE, BLUE, GREEN, RED, BLACK, GOLD, BUILD, RESERVE, TOKENS

# Constants
COLOR_ORDER = [WHITE, BLUE, GREEN, RED, BLACK, GOLD]
COLOR_NAMES = ['white', 'blue', 'green', 'red', 'black', 'gold']
DB_PATH = 'data/games.db'
OUTPUT_PATH = 'data/training_dataset.csv'

# Deck sizes from constants
DECK_SIZES = {1: 40, 2: 30, 3: 20}


class ProgressTracker:
    """Tracks and displays progress with a progress bar."""

    def __init__(self, total: int, quiet: bool = False):
        self.total = total
        self.current = 0
        self.quiet = quiet
        self.last_update = 0

    def update(self, current: int):
        """Update progress and display if needed."""
        self.current = current
        # Update every 100 rows to avoid excessive terminal writes
        if not self.quiet and (current - self.last_update >= 100 or current == self.total):
            self.display()
            self.last_update = current

    def display(self):
        """Display progress bar."""
        if self.total == 0:
            return
        percentage = (self.current / self.total) * 100
        bar_length = 40
        filled = int(bar_length * self.current / self.total)
        bar = '=' * filled + '>' + ' ' * (bar_length - filled - 1)
        sys.stdout.write(f'\rProgress: [{bar}] {percentage:.1f}% ({self.current}/{self.total})')
        sys.stdout.flush()
        if self.current == self.total:
            sys.stdout.write('\n')


def connect_to_database(db_path: str) -> sqlite3.Connection:
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}", file=sys.stderr)
        sys.exit(1)


def load_all_cards_into_memory(cursor: sqlite3.Cursor) -> Dict[int, Dict[str, Any]]:
    """Load all cards into memory for faster lookup."""
    cards = {}
    sql = '''SELECT IDCard, Bonus, CostWhite, CostBlue, CostGreen, CostRed, CostBlack,
             VictoryPoints, Level FROM Card'''
    for row in cursor.execute(sql):
        cards[row[0]] = {
            'id': row[0],
            'bonus': row[1],
            'cost': [row[2], row[3], row[4], row[5], row[6]],  # white, blue, green, red, black
            'vp': row[7],
            'level': row[8]
        }
    return cards


def load_all_characters_into_memory(cursor: sqlite3.Cursor) -> Dict[int, Dict[str, Any]]:
    """Load all characters/nobles into memory for faster lookup."""
    characters = {}
    sql = '''SELECT IDCharacter, VictoryPoints, CostWhite, CostBlue, CostGreen,
             CostRed, CostBlack FROM Character'''
    for row in cursor.execute(sql):
        characters[row[0]] = {
            'id': row[0],
            'vp': row[1],
            'cost': [row[2], row[3], row[4], row[5], row[6]]  # white, blue, green, red, black
        }
    return characters


def get_total_state_count(cursor: sqlite3.Cursor) -> int:
    """Get total number of state-action pairs for progress tracking."""
    sql = '''SELECT COUNT(*) FROM Action'''
    result = cursor.execute(sql).fetchone()
    return result[0] if result else 0


def encode_card(card: Optional[Dict[str, Any]]) -> List[float]:
    """
    Encode a card into 12 features.

    Returns:
        [vp, level, cost_white, cost_blue, cost_green, cost_red, cost_black,
         bonus_white, bonus_blue, bonus_green, bonus_red, bonus_black]
    """
    if card is None:
        return [float('nan')] * 12

    # Victory points and level
    features = [float(card['vp']), float(card['level'])]

    # Costs (5 colors, no gold)
    features.extend([float(c) for c in card['cost']])

    # Bonus as one-hot (5 colors, no gold)
    bonus_onehot = [0.0] * 5
    try:
        bonus_idx = int(card['bonus'])
        if 0 <= bonus_idx < 5:
            bonus_onehot[bonus_idx] = 1.0
    except (ValueError, TypeError):
        pass
    features.extend(bonus_onehot)

    return features


def encode_noble(noble: Optional[Dict[str, Any]]) -> List[float]:
    """
    Encode a noble into 6 features.

    Returns:
        [vp, cost_white, cost_blue, cost_green, cost_red, cost_black]
    """
    if noble is None:
        return [float('nan')] * 6

    features = [float(noble['vp'])]
    features.extend([float(c) for c in noble['cost']])

    return features


def calculate_player_reductions(built_cards: List[Dict[str, Any]]) -> List[int]:
    """Calculate player's gem reductions from built cards."""
    reductions = [0, 0, 0, 0, 0]  # white, blue, green, red, black
    for card in built_cards:
        try:
            bonus_idx = int(card['bonus'])
            if 0 <= bonus_idx < 5:
                reductions[bonus_idx] += 1
        except (ValueError, TypeError):
            pass
    return reductions


def calculate_player_victory_points(built_cards: List[Dict[str, Any]],
                                   characters: List[int]) -> int:
    """Calculate player's total victory points from cards and nobles."""
    vp = sum(card['vp'] for card in built_cards)
    vp += len(characters) * 3  # Each noble is worth 3 VP
    return vp


def get_player_built_cards(cursor: sqlite3.Cursor, game_id: int, player_num: int,
                          turn_num: int, cards_cache: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get all cards built by a player up to and including this turn."""
    sql = '''SELECT IDCard FROM Action
             WHERE IDGame = ? AND PlayerNumber = ? AND TurnNumber < ? AND Type = ?
             AND IDCard IS NOT NULL'''
    built_cards = []
    for row in cursor.execute(sql, (game_id, player_num, turn_num, BUILD)):
        card_id = row[0]
        if card_id in cards_cache:
            built_cards.append(cards_cache[card_id])
    return built_cards


def get_player_reserved_cards(cursor: sqlite3.Cursor, game_id: int, player_num: int,
                              turn_num: int, cards_cache: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get currently reserved cards for a player at this turn."""
    # Find all reservations
    sql_reserve = '''SELECT IDCard FROM Action
                     WHERE IDGame = ? AND PlayerNumber = ? AND TurnNumber < ? AND Type = ?
                     AND IDCard IS NOT NULL'''
    reserved_ids = [row[0] for row in cursor.execute(sql_reserve, (game_id, player_num, turn_num, RESERVE))]

    # Find all builds (some may be from reserve)
    sql_build = '''SELECT IDCard FROM Action
                   WHERE IDGame = ? AND PlayerNumber = ? AND TurnNumber < ? AND Type = ?
                   AND IDCard IS NOT NULL'''
    built_ids = [row[0] for row in cursor.execute(sql_build, (game_id, player_num, turn_num, BUILD))]

    # Currently reserved = reserved - built
    current_reserved = []
    for card_id in reserved_ids:
        if card_id in built_ids:
            # Card was built, remove it from reserved
            built_ids.remove(card_id)
        else:
            # Still reserved
            if card_id in cards_cache:
                current_reserved.append(cards_cache[card_id])

    return current_reserved[:3]  # Max 3 reserved cards


def get_player_characters(cursor: sqlite3.Cursor, game_id: int, player_num: int,
                         turn_num: int) -> List[int]:
    """Get character IDs acquired by player up to this turn."""
    sql = '''SELECT IDCharacter FROM Action
             WHERE IDGame = ? AND PlayerNumber = ? AND TurnNumber < ?
             AND IDCharacter IS NOT NULL'''
    return [row[0] for row in cursor.execute(sql, (game_id, player_num, turn_num))]


def encode_player_state(cursor: sqlite3.Cursor, game_id: int, player_num: int,
                       turn_num: int, turn_position: int, state_player_row: Tuple,
                       cards_cache: Dict[int, Dict[str, Any]],
                       chars_cache: Dict[int, Dict[str, Any]]) -> List[float]:
    """
    Encode a player's state into 49 features.

    Returns:
        [position, vp, gems(6), reductions(5), reserved_cards(3x12)]
    """
    features = []

    # Turn position relative to current player
    features.append(float(turn_position))

    # Get built cards and characters
    built_cards = get_player_built_cards(cursor, game_id, player_num, turn_num, cards_cache)
    characters = get_player_characters(cursor, game_id, player_num, turn_num)
    reserved_cards = get_player_reserved_cards(cursor, game_id, player_num, turn_num, cards_cache)

    # Victory points
    vp = calculate_player_victory_points(built_cards, characters)
    features.append(float(vp))

    # Gems (6: white, blue, green, red, black, gold)
    # state_player_row format: (IDStatePlayer, IDGame, TurnNumber, PlayerNumber, TokensWhite, ..., TokensGold)
    gems = [float(state_player_row[4 + i]) for i in range(6)]
    features.extend(gems)

    # Reductions (5: white, blue, green, red, black)
    reductions = calculate_player_reductions(built_cards)
    features.extend([float(r) for r in reductions])

    # Reserved cards (3 cards x 12 features = 36)
    for i in range(3):
        if i < len(reserved_cards):
            features.extend(encode_card(reserved_cards[i]))
        else:
            features.extend([float('nan')] * 12)

    return features


def rotate_players_by_current(player_numbers: List[int], current_player: int) -> List[int]:
    """
    Reorder player list so current player is first.

    Example: players=[0,1,2,3], current=2 → returns [2,3,0,1]
    """
    try:
        idx = player_numbers.index(current_player)
        return player_numbers[idx:] + player_numbers[:idx]
    except ValueError:
        return player_numbers


def get_cards_used_count(cursor: sqlite3.Cursor, game_id: int, turn_num: int, level: int) -> int:
    """Get number of cards used from a specific deck level."""
    sql = '''SELECT COUNT(*) FROM Action
             WHERE IDGame = ? AND TurnNumber < ? AND Type IN (?, ?)
             AND IDCard IN (SELECT IDCard FROM Card WHERE Level = ?)'''
    result = cursor.execute(sql, (game_id, turn_num, BUILD, RESERVE, level)).fetchone()
    return result[0] if result else 0


def encode_game_state(cursor: sqlite3.Cursor, state_game_row: Tuple,
                     state_player_rows: List[Tuple], cards_cache: Dict[int, Dict[str, Any]],
                     chars_cache: Dict[int, Dict[str, Any]], game_id: int,
                     action_turn_num: int) -> List[float]:
    """
    Encode the complete game state into 381 features.

    StateGame row format:
    (IDStateGame, IDGame, TurnNumber, CurrentPlayer, TokensWhite, TokensBlue, TokensGreen,
     TokensRed, TokensBlack, TokensGold, Card1_1, Card1_2, Card1_3, Card1_4, Card2_1, ..., Card3_4,
     Character1, Character2, Character3, Character4, Character5)

    Returns 381 features:
    - num_players (1)
    - turn_number (1)
    - gems_board (6)
    - visible_cards (12 x 12 = 144)
    - deck_remaining (3)
    - nobles (5 x 6 = 30)
    - players (4 x 49 = 196)
    """
    features = []

    # Extract basic info
    state_turn_number = state_game_row[2]
    current_player = state_game_row[3]
    num_players = len(state_player_rows)

    # Number of players
    features.append(float(num_players))

    # Turn number (use action turn number, which is what we're predicting for)
    features.append(float(action_turn_num))

    # Gems on board (6: white, blue, green, red, black, gold)
    gems_board = [float(state_game_row[4 + i]) for i in range(6)]
    features.extend(gems_board)

    # Visible cards (12 cards: 4 per level, levels 1-3)
    # Cards are at indices 10-21 in state_game_row
    for i in range(12):
        card_id = state_game_row[10 + i]
        if card_id is not None and card_id in cards_cache:
            features.extend(encode_card(cards_cache[card_id]))
        else:
            features.extend([float('nan')] * 12)

    # Deck remaining counts (3: level 1, 2, 3)
    for level in [1, 2, 3]:
        used = get_cards_used_count(cursor, game_id, action_turn_num, level)
        remaining = DECK_SIZES[level] - used
        features.append(float(remaining))

    # Nobles (5 characters)
    # Characters are at indices 22-26 in state_game_row
    for i in range(5):
        char_id = state_game_row[22 + i]
        if char_id is not None and char_id in chars_cache:
            features.extend(encode_noble(chars_cache[char_id]))
        else:
            features.extend([float('nan')] * 6)

    # Players (4 players, rotated so current player is first)
    player_numbers = [row[3] for row in state_player_rows]
    rotated_players = rotate_players_by_current(player_numbers, current_player)

    # Encode each player
    for position, player_num in enumerate(rotated_players):
        # Find this player's state row
        player_row = None
        for row in state_player_rows:
            if row[3] == player_num:
                player_row = row
                break

        if player_row is not None:
            features.extend(encode_player_state(cursor, game_id, player_num, action_turn_num,
                                               position, player_row, cards_cache, chars_cache))
        else:
            features.extend([float('nan')] * 49)

    # Pad with NaN for missing players (2-3 player games)
    while len(rotated_players) < 4:
        features.extend([float('nan')] * 49)
        rotated_players.append(-1)

    return features


def encode_action_type(action_type: int, take_tokens: List[int]) -> str:
    """
    Encode action type as string.

    Returns: "build", "reserve", "take 2 tokens", or "take 3 tokens"
    """
    if action_type == BUILD:
        return "build"
    elif action_type == RESERVE:
        return "reserve"
    elif action_type == TOKENS:
        total_taken = sum(take_tokens)
        if total_taken == 2:
            return "take 2 tokens"
        else:
            return "take 3 tokens"
    return "unknown"


def find_card_position_in_visible(card_id: int, state_game_row: Tuple) -> Optional[int]:
    """
    Find position of card in visible cards [0-11].

    Returns position or None if not found.
    """
    # Cards are at indices 10-21 in state_game_row
    for i in range(12):
        if state_game_row[10 + i] == card_id:
            return i
    return None


def encode_card_selection(action_row: Tuple, state_game_row: Tuple,
                         cursor: sqlite3.Cursor, game_id: int, player_num: int,
                         turn_num: int, cards_cache: Dict[int, Dict[str, Any]]) -> float:
    """
    Encode card selection for BUILD action.

    Returns:
    - [0-11]: visible card position
    - [12-14]: reserved card position
    - NaN: not a build action
    """
    if action_row[4] != BUILD:  # Type
        return float('nan')

    card_id = action_row[5]  # IDCard
    if card_id is None:
        return float('nan')

    # Check if card is in visible cards
    visible_pos = find_card_position_in_visible(card_id, state_game_row)
    if visible_pos is not None:
        return float(visible_pos)

    # Check if card is in reserved cards
    reserved_cards = get_player_reserved_cards(cursor, game_id, player_num, turn_num, cards_cache)
    for i, card in enumerate(reserved_cards):
        if card['id'] == card_id:
            return float(12 + i)

    # Shouldn't happen, but default to NaN
    return float('nan')


def encode_card_reservation(action_row: Tuple, state_game_row: Tuple,
                           cards_cache: Dict[int, Dict[str, Any]]) -> float:
    """
    Encode card reservation for RESERVE action.

    Returns:
    - [0-11]: visible card position
    - [12-14]: top deck (12 + level - 1)
    - NaN: not a reserve action
    """
    if action_row[4] != RESERVE:  # Type
        return float('nan')

    card_id = action_row[5]  # IDCard
    if card_id is None:
        return float('nan')

    # Check if card is in visible cards
    visible_pos = find_card_position_in_visible(card_id, state_game_row)
    if visible_pos is not None:
        return float(visible_pos)

    # Must be from top deck - get card level
    if card_id in cards_cache:
        level = cards_cache[card_id]['level']
        return float(12 + level - 1)

    return float('nan')


def encode_gem_take_3(take_tokens: List[int], action_type: int) -> List[float]:
    """
    Encode gem take-3 action as 3-hot vector (5 elements, no gold).

    Returns [0/1, 0/1, 0/1, 0/1, 0/1] or [NaN]*5
    """
    if action_type != TOKENS or sum(take_tokens[:5]) != 3:
        return [float('nan')] * 5

    return [float(take_tokens[i]) for i in range(5)]


def encode_gem_take_2(take_tokens: List[int], action_type: int) -> List[float]:
    """
    Encode gem take-2 action as one-hot vector (5 elements, no gold).

    Returns [0/2, 0/2, 0/2, 0/2, 0/2] or [NaN]*5
    """
    if action_type != TOKENS or sum(take_tokens[:5]) != 2:
        return [float('nan')] * 5

    return [float(take_tokens[i]) for i in range(5)]


def encode_noble_selection(character_id: Optional[int], state_game_row: Tuple) -> float:
    """
    Encode noble selection as position [0-4] or -1 if none.

    Returns position or -1.0
    """
    if character_id is None:
        return -1.0

    # Find position in Character1-Character5 (indices 22-26)
    for i in range(5):
        if state_game_row[22 + i] == character_id:
            return float(i)

    return -1.0


def encode_gems_removed(give_tokens: List[int]) -> List[float]:
    """
    Encode gems removed as 6-element count vector (including gold).

    Returns [count]*6
    """
    if give_tokens is None or all(t is None for t in give_tokens):
        return [0.0] * 6

    return [float(t) if t is not None else 0.0 for t in give_tokens]


def encode_action(action_row: Tuple, state_game_row: Tuple, cursor: sqlite3.Cursor,
                 game_id: int, player_num: int, turn_num: int,
                 cards_cache: Dict[int, Dict[str, Any]],
                 chars_cache: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Encode an action into 7 output heads.

    Action row format:
    (IDAction, IDGame, TurnNumber, PlayerNumber, Type, IDCard, TakeWhite, TakeBlue,
     TakeGreen, TakeRed, TakeBlack, TakeGold, GiveWhite, GiveBlue, GiveGreen,
     GiveRed, GiveBlack, GiveGold, IDCharacter)
    """
    action_type = action_row[4]
    take_tokens = [action_row[6 + i] if action_row[6 + i] is not None else 0 for i in range(6)]
    give_tokens = [action_row[12 + i] if action_row[12 + i] is not None else 0 for i in range(6)]

    return {
        'action_type': encode_action_type(action_type, take_tokens),
        'card_selection': encode_card_selection(action_row, state_game_row, cursor,
                                                game_id, player_num, turn_num, cards_cache),
        'card_reservation': encode_card_reservation(action_row, state_game_row, cards_cache),
        'gem_take_3': encode_gem_take_3(take_tokens, action_type),
        'gem_take_2': encode_gem_take_2(take_tokens, action_type),
        'noble_selection': encode_noble_selection(action_row[18], state_game_row),
        'gems_removed': encode_gems_removed(give_tokens)
    }


def generate_input_column_headers() -> List[str]:
    """Generate 381 input feature column headers."""
    headers = []

    # Basic info
    headers.append('num_players')
    headers.append('turn_number')

    # Gems on board
    for color in COLOR_NAMES:
        headers.append(f'gems_board_{color}')

    # Visible cards (12)
    for i in range(12):
        headers.append(f'card{i}_vp')
        headers.append(f'card{i}_level')
        for color in COLOR_NAMES[:5]:  # No gold in costs
            headers.append(f'card{i}_cost_{color}')
        for color in COLOR_NAMES[:5]:  # No gold in bonuses
            headers.append(f'card{i}_bonus_{color}')

    # Deck remaining
    for level in [1, 2, 3]:
        headers.append(f'deck_level{level}_remaining')

    # Nobles (5)
    for i in range(5):
        headers.append(f'noble{i}_vp')
        for color in COLOR_NAMES[:5]:  # No gold in costs
            headers.append(f'noble{i}_req_{color}')

    # Players (4)
    for i in range(4):
        headers.append(f'player{i}_position')
        headers.append(f'player{i}_vp')
        for color in COLOR_NAMES:  # Including gold
            headers.append(f'player{i}_gems_{color}')
        for color in COLOR_NAMES[:5]:  # No gold in reductions
            headers.append(f'player{i}_reduction_{color}')
        for j in range(3):  # 3 reserved cards
            headers.append(f'player{i}_reserved{j}_vp')
            headers.append(f'player{i}_reserved{j}_level')
            for color in COLOR_NAMES[:5]:
                headers.append(f'player{i}_reserved{j}_cost_{color}')
            for color in COLOR_NAMES[:5]:
                headers.append(f'player{i}_reserved{j}_bonus_{color}')

    return headers


def generate_output_column_headers() -> List[str]:
    """Generate output column headers (20 columns)."""
    headers = ['action_type', 'card_selection', 'card_reservation']

    # Gem take 3 (5 colors, no gold)
    for color in COLOR_NAMES[:5]:
        headers.append(f'gem_take3_{color}')

    # Gem take 2 (5 colors, no gold)
    for color in COLOR_NAMES[:5]:
        headers.append(f'gem_take2_{color}')

    # Noble selection
    headers.append('noble_selection')

    # Gems removed (6 colors, including gold)
    for color in COLOR_NAMES:
        headers.append(f'gems_removed_{color}')

    return headers


def generate_all_headers() -> List[str]:
    """Generate all CSV headers (402 columns)."""
    return ['game_id'] + generate_input_column_headers() + generate_output_column_headers()


def get_player_state_from_history(game_id: int, player_num: int, turn_num: int,
                                  action_history: Dict, cards_cache: Dict) -> Tuple[List, List, List]:
    """Get player's built cards, reserved cards, and characters up to this turn from cached history."""
    key = (game_id, player_num)
    if key not in action_history:
        return [], [], []

    hist = action_history[key]

    # Get built cards before this turn
    built_card_ids = [card_id for t, card_id in hist['built_cards'] if t < turn_num]
    built_cards = [cards_cache[cid] for cid in built_card_ids if cid in cards_cache]

    # Get reserved cards: reserved before this turn but not yet built
    reserved_ids = [card_id for t, card_id in hist['reserved_cards'] if t < turn_num]
    built_ids = built_card_ids
    current_reserved_ids = [cid for cid in reserved_ids if cid not in built_ids]
    reserved_cards = [cards_cache[cid] for cid in current_reserved_ids[:3] if cid in cards_cache]

    # Get characters before this turn
    character_ids = [char_id for t, char_id in hist['characters'] if t < turn_num]

    return built_cards, reserved_cards, character_ids


def encode_player_state_optimized(game_id: int, player_num: int, turn_num: int,
                                  turn_position: int, state_player_row: Tuple,
                                  cards_cache: Dict, action_history: Dict) -> List[float]:
    """Optimized version: encode player state using pre-computed history."""
    features = []

    # Turn position
    features.append(float(turn_position))

    # Get player history from cache
    built_cards, reserved_cards, character_ids = get_player_state_from_history(
        game_id, player_num, turn_num, action_history, cards_cache)

    # Victory points
    vp = calculate_player_victory_points(built_cards, character_ids)
    features.append(float(vp))

    # Gems (6: white, blue, green, red, black, gold)
    gems = [float(state_player_row[4 + i]) for i in range(6)]
    features.extend(gems)

    # Reductions (5: white, blue, green, red, black)
    reductions = calculate_player_reductions(built_cards)
    features.extend([float(r) for r in reductions])

    # Reserved cards (3 cards x 12 features = 36)
    for i in range(3):
        if i < len(reserved_cards):
            features.extend(encode_card(reserved_cards[i]))
        else:
            features.extend([float('nan')] * 12)

    return features


def get_deck_cards_used(game_id: int, turn_num: int, level: int, action_history: Dict) -> int:
    """Count cards used from a deck level up to this turn using cached history."""
    count = 0
    for (gid, player_num), hist in action_history.items():
        if gid != game_id:
            continue
        # Count built and reserved cards from this level
        for t, card_id in hist['built_cards']:
            if t < turn_num:
                count += 1  # We'd need to check level, but we'll approximate
        for t, card_id in hist['reserved_cards']:
            if t < turn_num:
                count += 1  # Approximation
    return count


def encode_game_state_optimized(state_game_row: Tuple, state_player_rows: List[Tuple],
                                cards_cache: Dict, chars_cache: Dict, game_id: int,
                                action_turn_num: int, action_history: Dict) -> List[float]:
    """Optimized version: encode game state using cached data."""
    features = []

    # Extract basic info
    state_turn_number = state_game_row[2]
    current_player = state_game_row[3]
    num_players = len(state_player_rows)

    # Number of players
    features.append(float(num_players))

    # Turn number (use action turn number)
    features.append(float(action_turn_num))

    # Gems on board (6: white, blue, green, red, black, gold)
    gems_board = [float(state_game_row[4 + i]) for i in range(6)]
    features.extend(gems_board)

    # Visible cards (12 cards: 4 per level, levels 1-3)
    for i in range(12):
        card_id = state_game_row[10 + i]
        if card_id is not None and card_id in cards_cache:
            features.extend(encode_card(cards_cache[card_id]))
        else:
            features.extend([float('nan')] * 12)

    # Deck remaining counts - use simple count from all actions
    for level in [1, 2, 3]:
        # Count all built/reserved cards of this level in the game up to this turn
        used = 0
        for (gid, pnum), hist in action_history.items():
            if gid != game_id:
                continue
            for t, card_id in hist['built_cards'] + hist['reserved_cards']:
                if t < action_turn_num and card_id in cards_cache:
                    if cards_cache[card_id]['level'] == level:
                        used += 1
        remaining = DECK_SIZES[level] - used
        features.append(float(remaining))

    # Nobles (5 characters)
    for i in range(5):
        char_id = state_game_row[22 + i]
        if char_id is not None and char_id in chars_cache:
            features.extend(encode_noble(chars_cache[char_id]))
        else:
            features.extend([float('nan')] * 6)

    # Players (4 players, rotated so current player is first)
    player_numbers = [row[3] for row in state_player_rows]
    rotated_players = rotate_players_by_current(player_numbers, current_player)

    # Encode each player
    for position, player_num in enumerate(rotated_players):
        # Find this player's state row
        player_row = None
        for row in state_player_rows:
            if row[3] == player_num:
                player_row = row
                break

        if player_row is not None:
            features.extend(encode_player_state_optimized(game_id, player_num, action_turn_num,
                                                         position, player_row, cards_cache,
                                                         action_history))
        else:
            features.extend([float('nan')] * 49)

    # Pad with NaN for missing players
    while len(rotated_players) < 4:
        features.extend([float('nan')] * 49)
        rotated_players.append(-1)

    return features


def encode_action_optimized(action_row: Tuple, state_game_row: Tuple,
                            game_id: int, player_num: int, turn_num: int,
                            cards_cache: Dict, chars_cache: Dict,
                            action_history: Dict) -> Dict[str, Any]:
    """Optimized version: encode action using cached data."""
    action_type = action_row[4]
    take_tokens = [action_row[6 + i] if action_row[6 + i] is not None else 0 for i in range(6)]
    give_tokens = [action_row[12 + i] if action_row[12 + i] is not None else 0 for i in range(6)]

    # Get reserved cards from history for card selection encoding
    _, reserved_cards, _ = get_player_state_from_history(game_id, player_num, turn_num,
                                                         action_history, cards_cache)

    # Encode card selection
    card_selection = float('nan')
    if action_type == BUILD:
        card_id = action_row[5]
        if card_id is not None:
            # Check visible cards
            visible_pos = find_card_position_in_visible(card_id, state_game_row)
            if visible_pos is not None:
                card_selection = float(visible_pos)
            else:
                # Check reserved cards
                for i, card in enumerate(reserved_cards):
                    if card['id'] == card_id:
                        card_selection = float(12 + i)
                        break

    # Encode card reservation
    card_reservation = float('nan')
    if action_type == RESERVE:
        card_id = action_row[5]
        if card_id is not None:
            visible_pos = find_card_position_in_visible(card_id, state_game_row)
            if visible_pos is not None:
                card_reservation = float(visible_pos)
            elif card_id in cards_cache:
                level = cards_cache[card_id]['level']
                card_reservation = float(12 + level - 1)

    return {
        'action_type': encode_action_type(action_type, take_tokens),
        'card_selection': card_selection,
        'card_reservation': card_reservation,
        'gem_take_3': encode_gem_take_3(take_tokens, action_type),
        'gem_take_2': encode_gem_take_2(take_tokens, action_type),
        'noble_selection': encode_noble_selection(action_row[18], state_game_row),
        'gems_removed': encode_gems_removed(give_tokens)
    }


def load_all_state_data(cursor: sqlite3.Cursor, quiet: bool = False) -> Tuple[Dict, Dict]:
    """Load all StateGame and StatePlayer data into indexed dictionaries."""
    if not quiet:
        print("Loading all game states into memory...")

    # Load StateGame indexed by (game_id, turn_num, current_player)
    state_game_cache = {}
    for row in cursor.execute("SELECT * FROM StateGame"):
        game_id = row[1]
        turn_num = row[2]
        current_player = row[3]
        state_game_cache[(game_id, turn_num, current_player)] = row

    if not quiet:
        print(f"  Loaded {len(state_game_cache):,} StateGame rows")

    # Load StatePlayer indexed by (game_id, turn_num, player_num)
    state_player_cache = {}
    for row in cursor.execute("SELECT * FROM StatePlayer"):
        game_id = row[1]
        turn_num = row[2]
        player_num = row[3]
        key = (game_id, turn_num)
        if key not in state_player_cache:
            state_player_cache[key] = []
        state_player_cache[key].append(row)

    if not quiet:
        print(f"  Loaded {sum(len(v) for v in state_player_cache.values()):,} StatePlayer rows")

    return state_game_cache, state_player_cache


def load_all_action_history(cursor: sqlite3.Cursor, quiet: bool = False) -> Dict:
    """Pre-compute action history for each player in each game."""
    if not quiet:
        print("Pre-computing action histories...")

    # Structure: history[(game_id, player_num)] = {
    #   'built_cards': [(turn, card_id), ...],
    #   'reserved_cards': [(turn, card_id), ...],
    #   'characters': [(turn, char_id), ...]
    # }
    history = {}

    sql = '''SELECT IDGame, TurnNumber, PlayerNumber, Type, IDCard, IDCharacter
             FROM Action ORDER BY IDGame, PlayerNumber, TurnNumber'''

    for row in cursor.execute(sql):
        game_id, turn_num, player_num, action_type, card_id, char_id = row

        key = (game_id, player_num)
        if key not in history:
            history[key] = {'built_cards': [], 'reserved_cards': [], 'characters': []}

        if action_type == BUILD and card_id is not None:
            history[key]['built_cards'].append((turn_num, card_id))
        elif action_type == RESERVE and card_id is not None:
            history[key]['reserved_cards'].append((turn_num, card_id))

        if char_id is not None:
            history[key]['characters'].append((turn_num, char_id))

    if not quiet:
        print(f"  Pre-computed history for {len(history):,} player-game combinations")

    return history


def extract_all_training_data(db_path: str, limit: Optional[int] = None,
                              quiet: bool = False) -> List[Tuple[int, List[float], Dict[str, Any]]]:
    """Extract all training data from database."""
    conn = connect_to_database(db_path)
    cursor = conn.cursor()

    # Load cards and characters into memory
    if not quiet:
        print("Loading cards and characters into memory...")
    cards_cache = load_all_cards_into_memory(cursor)
    chars_cache = load_all_characters_into_memory(cursor)

    # Load all state data into memory
    state_game_cache, state_player_cache = load_all_state_data(cursor, quiet)

    # Pre-compute action histories
    action_history = load_all_action_history(cursor, quiet)

    # Get total count
    total_count = get_total_state_count(cursor)
    if limit:
        total_count = min(total_count, limit)

    if not quiet:
        print(f"Total samples to extract: {total_count}")
        print("Extracting training data...")

    progress = ProgressTracker(total_count, quiet)
    data = []

    # Query all actions - fetch all into memory to avoid cursor issues
    sql_actions = '''SELECT * FROM Action ORDER BY IDGame, TurnNumber, PlayerNumber'''
    if limit:
        sql_actions += f' LIMIT {limit}'

    action_rows = cursor.execute(sql_actions).fetchall()

    count = 0
    skipped = 0
    for action_row in action_rows:
        game_id = action_row[1]
        turn_num = action_row[2]
        player_num = action_row[3]

        # Get state game for this turn and player from cache
        # Note: StateGame.TurnNumber starts at 1, so action at turn N corresponds to StateGame turn N+1
        # StatePlayer.TurnNumber starts at 0, so action at turn N corresponds to StatePlayer turn N
        state_game_turn = turn_num + 1  # StateGame starts at 1
        state_player_turn = turn_num     # StatePlayer starts at 0

        state_game_row = state_game_cache.get((game_id, state_game_turn, player_num))

        if state_game_row is None:
            skipped += 1
            continue

        # Get all player states for this turn from cache
        state_player_rows = state_player_cache.get((game_id, state_player_turn), [])

        try:
            # Encode input vector (using cached action history)
            input_vector = encode_game_state_optimized(state_game_row, state_player_rows,
                                                      cards_cache, chars_cache, game_id, turn_num,
                                                      action_history)

            # Encode action
            action_dict = encode_action_optimized(action_row, state_game_row,
                                                  game_id, player_num, turn_num,
                                                  cards_cache, chars_cache, action_history)

            data.append((game_id, input_vector, action_dict))

            count += 1
            progress.update(count)
        except Exception as e:
            if not quiet:
                print(f"\nError encoding game {game_id}, turn {turn_num}, player {player_num}: {e}")
            import traceback
            if not quiet:
                traceback.print_exc()
            skipped += 1

    if not quiet and skipped > 0:
        print(f"\nNote: Skipped {skipped} actions without matching state")

    conn.close()
    return data


def flatten_action_dict(action_dict: Dict[str, Any]) -> List[Any]:
    """Flatten action dictionary into list matching output column order."""
    result = [
        action_dict['action_type'],
        action_dict['card_selection'],
        action_dict['card_reservation']
    ]
    result.extend(action_dict['gem_take_3'])
    result.extend(action_dict['gem_take_2'])
    result.append(action_dict['noble_selection'])
    result.extend(action_dict['gems_removed'])
    return result


def write_to_csv(data: List[Tuple[int, List[float], Dict[str, Any]]],
                output_path: str, quiet: bool = False):
    """Write data to CSV file."""
    if not quiet:
        print(f"\nWriting to CSV: {output_path}")

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        headers = generate_all_headers()
        writer.writerow(headers)

        # Write data
        progress = ProgressTracker(len(data), quiet)
        for i, (game_id, input_vec, action_dict) in enumerate(data):
            action_outputs = flatten_action_dict(action_dict)
            row = [game_id] + input_vec + action_outputs
            writer.writerow(row)
            progress.update(i + 1)

    if not quiet:
        print(f"CSV export complete: {output_path}")


def generate_dataset_summary(data: List[Tuple[int, List[float], Dict[str, Any]]],
                             output_path: str) -> str:
    """Generate dataset summary report."""
    total_samples = len(data)
    unique_games = len(set(d[0] for d in data))

    # Action type distribution
    action_types = {}
    for _, _, action_dict in data:
        action_type = action_dict['action_type']
        action_types[action_type] = action_types.get(action_type, 0) + 1

    # Player count distribution (from first sample of each game)
    game_player_counts = {}
    seen_games = set()
    for game_id, input_vec, _ in data:
        if game_id not in seen_games:
            num_players = int(input_vec[0]) if not (input_vec[0] != input_vec[0]) else 0  # NaN check
            game_player_counts[num_players] = game_player_counts.get(num_players, 0) + 1
            seen_games.add(game_id)

    # Generate report
    report = f"""
Dataset Export Summary
{'=' * 60}

Output File: {output_path}
Total Samples: {total_samples:,}
Unique Games: {unique_games:,}
Samples per Game (avg): {total_samples / unique_games:.1f}

Input Space:
  - 381 dimensional feature vector
  - Includes board state, visible cards, and player states
  - Players rotated so current player is always first
  - NaN padding for missing players in 2-3 player games

Output Space:
  - 7 output heads for multi-head classification:
    1. action_type: categorical (4 classes)
    2. card_selection: int [0-14] for build actions
    3. card_reservation: int [0-14] for reserve actions
    4. gem_take_3: 3-hot vector (5 elements)
    5. gem_take_2: one-hot vector (5 elements)
    6. noble_selection: int [0-4] or -1
    7. gems_removed: count vector (6 elements)

Action Type Distribution:
"""

    for action_type, count in sorted(action_types.items()):
        percentage = (count / total_samples) * 100
        report += f"  {action_type:20s}: {count:8,} ({percentage:5.2f}%)\n"

    report += "\nPlayer Count Distribution:\n"
    for num_players, count in sorted(game_player_counts.items()):
        percentage = (count / unique_games) * 100
        report += f"  {num_players} players: {count:6,} games ({percentage:5.2f}%)\n"

    report += """
Usage Example:
  import pandas as pd
  df = pd.read_csv('data/training_dataset.csv')

  # Split by game_id to avoid data leakage
  unique_games = df['game_id'].unique()
  train_games = unique_games[:int(0.7*len(unique_games))]
  train_df = df[df['game_id'].isin(train_games)]
"""

    return report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Export ML dataset from Splendor game database')
    parser.add_argument('--db-path', default=DB_PATH, help='Database path')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Output CSV path')
    parser.add_argument('--limit', type=int, help='Limit number of samples')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    try:
        if not args.quiet:
            print("=" * 60)
            print("ML Dataset Export for Splendor")
            print("=" * 60)

        # Extract data
        data = extract_all_training_data(args.db_path, args.limit, args.quiet)

        if not data:
            print("No data extracted. Check database path and contents.")
            return

        # Write to CSV
        write_to_csv(data, args.output, args.quiet)

        # Generate and print summary
        if not args.quiet:
            summary = generate_dataset_summary(data, args.output)
            print(summary)
            print("=" * 60)
            print("Export complete!")

    except Exception as e:
        print(f"Error during export: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
