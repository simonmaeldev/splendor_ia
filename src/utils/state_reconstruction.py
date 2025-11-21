"""
State Reconstruction Module for Splendor AI

This module reconstructs internal game state (Board object) from CSV input features.
The reconstructed board state is functionally equivalent to the original game state,
allowing us to call board.getMoves() to get valid moves for action masking.

This reverses the encoding performed by csv_exporter.py:
    CSV features (382) → Board object with Card, Character, Player objects

Key features:
- Parses 382 CSV features into structured Python objects
- Handles NaN padding for variable player counts (2-4 players)
- Creates dummy cards for deck contents (only counts matter for getMoves())
- Handles player rotation (CSV has current player first, needs un-rotation)
- Never modifies splendor/* files (validation confirms reconstruction correctness)

Usage:
    from src.utils.state_reconstruction import reconstruct_board_from_csv_row
    import csv

    with open('game.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            board = reconstruct_board_from_csv_row(row)
            valid_moves = board.getMoves()
"""

from typing import List, Dict, Any, Optional
import math

# Import game objects
from splendor.cards import Card
from splendor.characters import Character
from splendor.player import Player
from splendor.board import Board
from splendor.constants import WHITE, BLUE, GREEN, RED, BLACK, GOLD


def parse_card_features(features: List[float]) -> Optional[Card]:
    """
    Parse 12 features into a Card object.

    Features: [vp, level, cost_white, cost_blue, cost_green, cost_red, cost_black,
               bonus_white, bonus_blue, bonus_green, bonus_red, bonus_black]

    Args:
        features: List of 12 float values representing a card

    Returns:
        Card object or None if features are all NaN (empty slot)

    Example:
        >>> features = [1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 2.0, 0, 1, 0, 0, 0]
        >>> card = parse_card_features(features)
        >>> card.vp == 1 and card.lvl == 2 and card.bonus == BLUE
        True
    """
    # Check if all features are NaN (empty slot)
    if all(math.isnan(f) for f in features):
        return None

    # Extract vp and level (convert floats to ints)
    vp = int(features[0])
    level = int(features[1])

    # Extract costs (5 colors: white, blue, green, red, black)
    cost = [int(features[2 + i]) for i in range(5)]

    # Extract bonus from one-hot encoding (5 colors)
    bonus_onehot = [int(features[7 + i]) for i in range(5)]

    # Find which bonus color (index of the 1 in one-hot)
    bonus = 0
    for i, val in enumerate(bonus_onehot):
        if val == 1:
            bonus = i
            break

    # Create Card object
    card = Card(vp, bonus, cost, level)

    # Set visible to True (per spec requirement)
    card.visible = True

    return card


def parse_noble_features(features: List[float]) -> Optional[Character]:
    """
    Parse 6 features into a Character (noble) object.

    Features: [vp, req_white, req_blue, req_green, req_red, req_black]

    Args:
        features: List of 6 float values representing a noble

    Returns:
        Character object or None if features are all NaN (empty slot)

    Example:
        >>> features = [3.0, 4.0, 4.0, 0.0, 0.0, 0.0]
        >>> noble = parse_noble_features(features)
        >>> noble.vp == 3 and noble.cost == [4, 4, 0, 0, 0]
        True
    """
    # Check if all features are NaN (empty slot)
    if all(math.isnan(f) for f in features):
        return None

    # Extract vp
    vp = int(features[0])

    # Extract costs (5 colors: white, blue, green, red, black)
    cost = [int(features[1 + i]) for i in range(5)]

    # Create Character object
    return Character(vp, cost)


def create_dummy_deck(level: int, count: int) -> List[Card]:
    """
    Create dummy cards for a deck.

    Dummy cards are used to represent unknown deck contents. The getMoves() function
    only checks if decks are non-empty to determine if top-deck reservation is possible,
    so the actual card properties don't matter.

    Args:
        level: Deck level (1, 2, or 3)
        count: Number of cards in the deck

    Returns:
        List of dummy Card objects

    Example:
        >>> deck = create_dummy_deck(1, 5)
        >>> len(deck) == 5 and all(c.lvl == 1 for c in deck)
        True
    """
    dummy_cards = []
    for _ in range(count):
        # Create dummy card: vp=0, bonus=0, cost=[0,0,0,0,0], lvl=level
        card = Card(0, 0, [0, 0, 0, 0, 0], level)
        # Cards in deck are not visible
        card.visible = False
        dummy_cards.append(card)

    return dummy_cards


def parse_player_features(features: List[float], player_num: int) -> Optional[Player]:
    """
    Parse 49 features into a Player object.

    Features: [position, vp, gems(6), reductions(5), reserved_cards(3x12)]

    Args:
        features: List of 49 float values representing a player
        player_num: Player number (0-3) for identification

    Returns:
        Player object or None if features are all NaN (player doesn't exist)

    Example:
        >>> features = [0.0, 5.0, 2, 1, 0, 3, 1, 0, 1, 0, 2, 0, 1] + [float('nan')] * 36
        >>> player = parse_player_features(features, 0)
        >>> player.vp == 5 and player.tokens[WHITE] == 2
        True
    """
    # Check if all features are NaN (player doesn't exist)
    if all(math.isnan(f) for f in features):
        return None

    # Extract position (should match player_num)
    position = int(features[0])

    # Create Player object
    player = Player(str(player_num), "ISMCTS_PARA1000")

    # Extract vp
    player.vp = int(features[1])

    # Extract gems/tokens (6 colors including gold)
    player.tokens = [int(features[2 + i]) for i in range(6)]

    # Extract reductions (5 colors, no gold)
    player.reductions = [int(features[8 + i]) for i in range(5)]

    # Extract reserved cards (3 cards x 12 features = 36)
    for i in range(3):
        start_idx = 13 + (i * 12)
        card_features = features[start_idx:start_idx + 12]
        card = parse_card_features(card_features)
        if card is not None:
            player.reserved.append(card)

    return player


def reconstruct_board_from_csv_row(row: Dict[str, Any]) -> Board:
    """
    Reconstruct Board object from CSV row dictionary.

    This is the main reconstruction function that converts 382 CSV features
    into a complete Board object that can call getMoves().

    Args:
        row: Dictionary with column names as keys (from CSV reader)

    Returns:
        Board object representing the game state

    Example:
        >>> import csv
        >>> with open('game.csv') as f:
        ...     reader = csv.DictReader(f)
        ...     row = next(reader)
        ...     board = reconstruct_board_from_csv_row(row)
        ...     moves = board.getMoves()
    """
    def safe_float(val):
        """Convert value to float, handling empty strings as NaN."""
        if val == '' or val is None:
            return float('nan')
        return float(val)

    # Extract basic info
    num_players = int(safe_float(row['num_players']))
    turn_number = int(safe_float(row['turn_number']))
    current_player = int(safe_float(row['current_player']))

    # Create Board object
    # Note: Board.__init__ initializes with random decks/cards, we'll overwrite them
    board = Board(num_players, ["ISMCTS_PARA1000"] * num_players)
    board.nbTurn = turn_number
    board.currentPlayer = current_player

    # Reconstruct board tokens (6 colors)
    board.tokens = []
    for color in ['white', 'blue', 'green', 'red', 'black', 'gold']:
        board.tokens.append(int(safe_float(row[f'gems_board_{color}'])))

    # Reconstruct visible cards (12 cards: 4 per level)
    board.displayedCards = [[], [], []]
    for card_idx in range(12):
        # Extract 12 features for this card
        features = []
        features.append(safe_float(row[f'card{card_idx}_vp']))
        features.append(safe_float(row[f'card{card_idx}_level']))
        for color in ['white', 'blue', 'green', 'red', 'black']:
            features.append(safe_float(row[f'card{card_idx}_cost_{color}']))
        for color in ['white', 'blue', 'green', 'red', 'black']:
            features.append(safe_float(row[f'card{card_idx}_bonus_{color}']))

        card = parse_card_features(features)

        # Determine which level this card belongs to
        level = card_idx // 4  # 0-3 -> level 0, 4-7 -> level 1, 8-11 -> level 2

        if card is not None:
            board.displayedCards[level].append(card)

    # Reconstruct deck counts with dummy cards
    deck1_count = int(safe_float(row['deck_level1_remaining']))
    deck2_count = int(safe_float(row['deck_level2_remaining']))
    deck3_count = int(safe_float(row['deck_level3_remaining']))

    board.deckLVL1 = create_dummy_deck(1, deck1_count)
    board.deckLVL2 = create_dummy_deck(2, deck2_count)
    board.deckLVL3 = create_dummy_deck(3, deck3_count)
    board.decks = [board.deckLVL1, board.deckLVL2, board.deckLVL3]

    # Reconstruct nobles (5 nobles)
    board.characters = []
    for noble_idx in range(5):
        features = []
        features.append(safe_float(row[f'noble{noble_idx}_vp']))
        for color in ['white', 'blue', 'green', 'red', 'black']:
            features.append(safe_float(row[f'noble{noble_idx}_req_{color}']))

        noble = parse_noble_features(features)
        if noble is not None:
            board.characters.append(noble)

    # Reconstruct players with rotation handling
    # CSV has players rotated so current player is first (player0_*)
    # We need to un-rotate to get absolute player positions
    board.players = [None] * num_players

    for csv_player_idx in range(4):
        # Extract 49 features for this player
        features = []

        # Check if this player exists (check if position is NaN)
        position_key = f'player{csv_player_idx}_position'
        if position_key not in row or row[position_key] == '':
            # Player doesn't exist, rest are NaN
            break

        position_val = safe_float(row[position_key])
        if math.isnan(position_val):
            # Player doesn't exist, rest are NaN
            break

        features.append(position_val)
        features.append(safe_float(row[f'player{csv_player_idx}_vp']))

        for color in ['white', 'blue', 'green', 'red', 'black', 'gold']:
            features.append(safe_float(row[f'player{csv_player_idx}_gems_{color}']))

        for color in ['white', 'blue', 'green', 'red', 'black']:
            features.append(safe_float(row[f'player{csv_player_idx}_reduction_{color}']))

        # Reserved cards (3 cards x 12 features)
        for reserved_idx in range(3):
            features.append(safe_float(row[f'player{csv_player_idx}_reserved{reserved_idx}_vp']))
            features.append(safe_float(row[f'player{csv_player_idx}_reserved{reserved_idx}_level']))
            for color in ['white', 'blue', 'green', 'red', 'black']:
                features.append(safe_float(row[f'player{csv_player_idx}_reserved{reserved_idx}_cost_{color}']))
            for color in ['white', 'blue', 'green', 'red', 'black']:
                features.append(safe_float(row[f'player{csv_player_idx}_reserved{reserved_idx}_bonus_{color}']))

        player = parse_player_features(features, csv_player_idx)

        if player is not None:
            # Get absolute position from player features
            absolute_position = int(features[0])
            # Place player at absolute position
            board.players[absolute_position] = player

    return board


def reconstruct_board_from_features(features: List[float]) -> Board:
    """
    Reconstruct Board object from flat list of 382 features.

    This function is useful for programmatic reconstruction without CSV files.

    Args:
        features: List of 382 float values

    Returns:
        Board object representing the game state

    Example:
        >>> features = [3.0, 1.0, 0.0, ...] # 382 features
        >>> board = reconstruct_board_from_features(features)
        >>> moves = board.getMoves()
    """
    # Import here to avoid circular dependency
    from splendor.csv_exporter import generate_input_column_headers

    # Create dictionary mapping feature names to values
    headers = generate_input_column_headers()

    if len(features) != len(headers):
        raise ValueError(f"Expected {len(headers)} features, got {len(features)}")

    row_dict = {header: features[i] for i, header in enumerate(headers)}

    # Call the main reconstruction function
    return reconstruct_board_from_csv_row(row_dict)
