"""
CSV Exporter Module for Real-Time Game Data Export

This module provides functionality to export game data to CSV files during gameplay.
It reuses encoding logic from export_ml_dataset.py to ensure consistency and eliminates
the need for expensive post-processing.

Features:
- Exports each game to both game-specific CSV and aggregated all_games.csv
- Handles variable player counts with appropriate NaN padding
- Encodes state and actions using consistent 381 input + 20 output format
- Ensures data consistency with deep copies

Usage:
    from splendor.csv_exporter import export_game_to_csv

    # At end of game
    export_game_to_csv(game_id, num_players, states_and_actions, 'data/games')
"""

import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from copy import deepcopy
import math

# Import game constants
from splendor.constants import WHITE, BLUE, GREEN, RED, BLACK, GOLD, BUILD, RESERVE, TOKENS

# Constants matching export_ml_dataset.py
COLOR_ORDER = [WHITE, BLUE, GREEN, RED, BLACK, GOLD]
COLOR_NAMES = ['white', 'blue', 'green', 'red', 'black', 'gold']
DECK_SIZES = {1: 40, 2: 30, 3: 20}


def encode_card(card_obj) -> List:
    """
    Encode a card into 12 features.

    Args:
        card_obj: Card object with attributes: vp, level, cost, bonus

    Returns:
        [vp, level, cost_white, cost_blue, cost_green, cost_red, cost_black,
         bonus_white, bonus_blue, bonus_green, bonus_red, bonus_black]
    """
    if card_obj is None:
        return [float('nan')] * 12

    # Victory points and level (integers)
    features = [card_obj.vp, card_obj.lvl]

    # Costs (5 colors, no gold) - integers
    features.extend([card_obj.cost[i] for i in range(5)])

    # Bonus as one-hot (5 colors, no gold) - integers
    bonus_onehot = [0] * 5
    if 0 <= card_obj.bonus < 5:
        bonus_onehot[card_obj.bonus] = 1
    features.extend(bonus_onehot)

    return features


def encode_noble(noble_obj) -> List:
    """
    Encode a noble into 6 features.

    Args:
        noble_obj: Character object with attributes: vp, cost

    Returns:
        [vp, cost_white, cost_blue, cost_green, cost_red, cost_black]
    """
    if noble_obj is None:
        return [float('nan')] * 6

    # Victory points and costs - integers
    features = [noble_obj.vp]
    features.extend([noble_obj.cost[i] for i in range(5)])

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


def generate_input_column_headers() -> List[str]:
    """Generate 382 input feature column headers."""
    headers = []

    # Basic info
    headers.append('num_players')
    headers.append('turn_number')
    headers.append('current_player')

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
    """Generate all CSV headers (402 columns total)."""
    return ['game_id'] + generate_input_column_headers() + generate_output_column_headers()


def encode_player_state_from_board(board, player_num: int, turn_position: int) -> List:
    """
    Encode a player's state into 49 features from board state.

    Args:
        board: Board object containing full game state
        player_num: Absolute player number (0-3)
        turn_position: Position relative to current player (0-3, unused)

    Returns:
        [position, vp, gems(6), reductions(5), reserved_cards(3x12)]
    """
    features = []

    # Absolute player position - integer (0-3)
    features.append(player_num)

    # Get player object
    player = board.players[player_num]

    # Victory points - integer
    vp = player.getVictoryPoints()
    features.append(vp)

    # Gems (6: white, blue, green, red, black, gold) - integers
    # Use deep copy to avoid reference issues
    gems = deepcopy(player.tokens)
    features.extend(gems)

    # Reductions (5: white, blue, green, red, black) - integers from built cards bonuses
    reductions = [0, 0, 0, 0, 0]
    for card in player.built:
        if 0 <= card.bonus < 5:
            reductions[card.bonus] += 1
    features.extend(reductions)

    # Reserved cards (3 cards x 12 features = 36)
    reserved_cards = deepcopy(player.reserved)
    for i in range(3):
        if i < len(reserved_cards):
            features.extend(encode_card(reserved_cards[i]))
        else:
            features.extend([float('nan')] * 12)

    return features


def encode_game_state_from_board(board, action_turn_num: int) -> List:
    """
    Encode the complete game state into 382 features from board state.

    Args:
        board: Board object containing full game state
        action_turn_num: Turn number for this action

    Returns:
        382 features:
        - num_players (1)
        - turn_number (1)
        - current_player (1)
        - gems_board (6)
        - visible_cards (12 x 12 = 144, padded with NaN when decks depleted)
        - deck_remaining (3)
        - nobles (5 x 6 = 30)
        - players (4 x 49 = 196)
    """
    features = []

    # Number of players - integer
    num_players = len(board.players)
    features.append(num_players)

    # Turn number - integer
    features.append(action_turn_num)

    # Current player - integer (0-based)
    features.append(board.currentPlayer)

    # Gems on board (6: white, blue, green, red, black, gold) - integers
    gems_board = deepcopy(board.tokens)
    features.extend(gems_board)

    # Visible cards (12 cards: 4 per level, levels 1-3)
    # displayedCards is [level1[0-4], level2[0-4], level3[0-4]]
    # Always encode exactly 4 slots per level, padding with NaN if deck is depleted
    for level_cards in board.displayedCards:
        for i in range(4):
            if i < len(level_cards):
                features.extend(encode_card(level_cards[i]))
            else:
                # Deck depleted, no card in this slot - encode as NaN
                features.extend(encode_card(None))

    # Deck remaining counts (3: level 1, 2, 3) - integers
    features.append(len(board.deckLVL1))
    features.append(len(board.deckLVL2))
    features.append(len(board.deckLVL3))

    # Nobles (5 characters)
    nobles = deepcopy(board.characters)
    for i in range(5):
        if i < len(nobles):
            features.extend(encode_noble(nobles[i]))
        else:
            features.extend([float('nan')] * 6)

    # Players (4 players, rotated so current player is first, with absolute positions preserved)
    player_numbers = list(range(num_players))
    current_player = board.currentPlayer
    rotated_players = rotate_players_by_current(player_numbers, current_player)

    # Encode each player (player_num is absolute position, stored in playerX_position field)
    for position, player_num in enumerate(rotated_players):
        features.extend(encode_player_state_from_board(board, player_num, position))

    # Pad with NaN for missing players (2-3 player games)
    while len(rotated_players) < 4:
        features.extend([float('nan')] * 49)
        rotated_players.append(-1)

    # Validation
    if len(features) != 382:
        raise ValueError(f"Expected 382 features, got {len(features)}")

    return features


def find_card_position_in_visible(card, board) -> Optional[int]:
    """
    Find position of card in visible cards [0-11].

    Returns position or None if not found.
    """
    # Search through displayed cards
    position = 0
    for level_cards in board.displayedCards:
        for displayed_card in level_cards:
            if displayed_card is not None and card is not None:
                # Compare card IDs or attributes
                if (displayed_card.vp == card.vp and
                    displayed_card.lvl == card.lvl and
                    displayed_card.bonus == card.bonus and
                    displayed_card.cost == card.cost):
                    return position
            position += 1
    return None


def find_card_in_reserved(card, player) -> Optional[int]:
    """
    Find position of card in player's reserved cards [12-14].

    Returns position or None if not found.
    """
    for i, reserved_card in enumerate(player.reserved):
        if (reserved_card.vp == card.vp and
            reserved_card.lvl == card.lvl and
            reserved_card.bonus == card.bonus and
            reserved_card.cost == card.cost):
            return 12 + i
    return None


def encode_action_from_move(move, board) -> Dict[str, Any]:
    """
    Encode an action from a Move object into 7 output heads.

    Args:
        move: Move object with actionType, action (card or token list), tokensToRemove, character
        board: Board object for context

    Returns:
        Dict with 7 output heads:
        - action_type: str ("build", "reserve", "take 2 tokens", "take 3 tokens")
        - card_selection: float (0-14 or NaN)
        - card_reservation: float (0-14 or NaN)
        - gem_take_3: List[float] (5 elements)
        - gem_take_2: List[float] (5 elements)
        - noble_selection: float (0-4 or -1)
        - gems_removed: List[float] (6 elements)
    """
    action_type = move.actionType

    # For TOKENS action, move.action is the list of tokens
    # For BUILD/RESERVE, move.action is the Card
    if action_type == TOKENS:
        take_tokens = move.action if isinstance(move.action, list) else [0] * 6
    else:
        take_tokens = [0] * 6

    give_tokens = move.tokensToRemove if move.tokensToRemove else [0] * 6

    # Encode action type
    if action_type == BUILD:
        action_type_str = "build"
    elif action_type == RESERVE:
        action_type_str = "reserve"
    elif action_type == TOKENS:
        # Check pattern: take 2 = contains a 2, take 3 = contains only 1s
        if any(t == 2 for t in take_tokens[:5]):
            action_type_str = "take 2 tokens"  # Has a 2: [2,0,0,0,0], [0,2,0,0,0], etc.
        else:
            action_type_str = "take 3 tokens"  # Only 1s: [1,1,1,0,0], [1,1,0,0,0], [1,0,0,0,0]
    else:
        action_type_str = "unknown"

    # Card selection (for BUILD) - integer (0-14) or NaN
    card_selection = float('nan')
    if action_type == BUILD and move.action:
        # move.action is the Card for BUILD actions
        card = move.action
        # Check visible cards
        pos = find_card_position_in_visible(card, board)
        if pos is not None:
            card_selection = pos  # integer
        else:
            # Check reserved cards
            player = board.players[board.currentPlayer]
            pos = find_card_in_reserved(card, player)
            if pos is not None:
                card_selection = pos  # integer

    # Card reservation (for RESERVE) - integer (0-14) or NaN
    card_reservation = float('nan')
    if action_type == RESERVE:
        if isinstance(move.action, int):
            # Reserved from top deck - move.action is the deck level
            card_reservation = 12 + move.action - 1  # integer
        else:
            # Reserved from visible - move.action is the Card
            card = move.action
            pos = find_card_position_in_visible(card, board)
            if pos is not None:
                card_reservation = pos  # integer

    # Gem take 3 - integers (0 or 1)
    # Use same logic as action_type determination: if no 2s, it's a "take 3 tokens" intent
    gem_take_3 = [float('nan')] * 5
    if action_type == TOKENS and not any(t == 2 for t in take_tokens[:5]):
        gem_take_3 = [take_tokens[i] for i in range(5)]

    # Gem take 2 - integers (0 or 2)
    # Use same logic as action_type determination: if contains a 2, it's "take 2 tokens"
    gem_take_2 = [float('nan')] * 5
    if action_type == TOKENS and any(t == 2 for t in take_tokens[:5]):
        gem_take_2 = [take_tokens[i] for i in range(5)]

    # Noble selection - integer (-1 or 0-4)
    noble_selection = -1
    if hasattr(move, 'character') and move.character is not None:
        # Find position in board.characters
        for i, char in enumerate(board.characters):
            if char is not None and move.character is not None:
                if (char.vp == move.character.vp and char.cost == move.character.cost):
                    noble_selection = i
                    break

    # Gems removed - integers
    gems_removed = [g if g is not None else 0 for g in give_tokens]

    return {
        'action_type': action_type_str,
        'card_selection': card_selection,
        'card_reservation': card_reservation,
        'gem_take_3': gem_take_3,
        'gem_take_2': gem_take_2,
        'noble_selection': noble_selection,
        'gems_removed': gems_removed
    }


def flatten_action_dict(action_dict: Dict[str, Any]) -> List[Any]:
    """
    Convert action dict to flat list matching output column order.

    Returns 20 values.
    """
    result = []
    result.append(action_dict['action_type'])
    result.append(action_dict['card_selection'])
    result.append(action_dict['card_reservation'])
    result.extend(action_dict['gem_take_3'])
    result.extend(action_dict['gem_take_2'])
    result.append(action_dict['noble_selection'])
    result.extend(action_dict['gems_removed'])
    return result


def write_game_specific_csv(game_id: int, nb_players: int, data_rows: List[List[Any]],
                            output_dir: str):
    """
    Write game-specific CSV file without NaN padding.

    Args:
        game_id: Game ID
        nb_players: Number of players (2-4)
        data_rows: List of data rows [game_id, input_features, output_features]
        output_dir: Base output directory (e.g., 'data/games')
    """
    output_path = Path(output_dir) / f"{nb_players}_games" / f"{game_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate headers for this player count
    # Note: We still use full headers with NaN padding for consistency
    headers = generate_all_headers()

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data_rows)


def append_to_all_games_csv(game_id: int, nb_players: int, data_rows: List[List[Any]],
                            output_file: str):
    """
    Append game data to aggregated all_games.csv with 4-player padding.

    Args:
        game_id: Game ID
        nb_players: Number of players (2-4)
        data_rows: List of data rows (already padded to 4 players)
        output_file: Path to all_games.csv
    """
    output_path = Path(output_file)

    # Check if file exists
    file_exists = output_path.exists()

    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write headers if new file
        if not file_exists:
            headers = generate_all_headers()
            writer.writerow(headers)

        # Write data rows (already padded)
        writer.writerows(data_rows)


def export_game_to_csv(game_id: int, nb_players: int, states_and_actions: List[Tuple],
                       output_dir: str = 'data/games') -> bool:
    """
    Export a complete game to CSV files.

    Args:
        game_id: Game ID
        nb_players: Number of players (2-4)
        states_and_actions: List of (board_state, move, turn_num) tuples
        output_dir: Base output directory

    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert states and actions to CSV rows
        csv_data = []

        for board_state, move, turn_num in states_and_actions:
            # Encode game state (381 features)
            input_features = encode_game_state_from_board(board_state, turn_num)

            # Encode action (20 features)
            action_dict = encode_action_from_move(move, board_state)
            output_features = flatten_action_dict(action_dict)

            # Combine into row: game_id + input + output
            row = [game_id] + input_features + output_features
            csv_data.append(row)

        # Write game-specific CSV
        write_game_specific_csv(game_id, nb_players, csv_data, output_dir)

        # Write to all_games.csv
        all_games_path = Path(output_dir) / 'all_games.csv'
        append_to_all_games_csv(game_id, nb_players, csv_data, str(all_games_path))

        return True

    except Exception as e:
        print(f"Error exporting game {game_id} to CSV: {e}")
        return False
