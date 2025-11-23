"""
Validation Module for State Reconstruction

This module provides validation utilities to ensure reconstructed board states
are correct and functionally equivalent to original states.

Validation strategies:
1. getMoves() execution: Verify board.getMoves() runs without errors
2. Action validation: Verify CSV action is in valid moves list
3. Round-trip encoding: Verify reconstruct → encode produces same features

Usage:
    from src.utils.validate_reconstruction import validate_action_in_moves
    from src.utils.state_reconstruction import reconstruct_board_from_csv_row

    board = reconstruct_board_from_csv_row(row)
    action = parse_action_from_csv(row, board)
    is_valid, message = validate_action_in_moves(board, action)
"""

from typing import List, Dict, Any, Optional, Tuple
import math

# Import game objects
from src.splendor.cards import Card
from src.splendor.characters import Character
from src.splendor.board import Board
from src.splendor.move import Move
from src.splendor.constants import WHITE, BLUE, GREEN, RED, BLACK, GOLD, BUILD, RESERVE, TOKENS


def parse_action_from_csv(row: Dict[str, Any], board: Board) -> Move:
    """
    Parse action from CSV output columns into a Move object.

    Args:
        row: CSV row dictionary containing output features
        board: Reconstructed board (needed for Card/Character references)

    Returns:
        Move object representing the action

    Example:
        >>> # BUILD action from visible card
        >>> action = parse_action_from_csv(row, board)
        >>> action.actionType == BUILD
        True
    """
    def safe_float(val):
        """Convert value to float, handling empty strings as NaN."""
        if val == '' or val is None:
            return float('nan')
        return float(val)

    # Get action type string
    action_type_str = row['action_type']

    # Map action type string to constant
    if action_type_str == 'build':
        action_type = BUILD
    elif action_type_str == 'reserve':
        action_type = RESERVE
    elif action_type_str in ['take 2 tokens', 'take 3 tokens']:
        action_type = TOKENS
    else:
        raise ValueError(f"Unknown action type: {action_type_str}")

    # Initialize action components
    action = None
    tokens_to_remove = [0, 0, 0, 0, 0, 0]
    character = None

    # Parse action based on type
    if action_type == BUILD:
        # Get card selection
        card_selection = safe_float(row['card_selection'])
        if not math.isnan(card_selection):
            card_idx = int(card_selection)

            if card_idx < 12:
                # Build from visible card - use relative indexing across all visible cards
                visible_cards = []
                for level_cards in board.displayedCards:
                    visible_cards.extend(level_cards)

                if card_idx < len(visible_cards):
                    action = visible_cards[card_idx]
            else:
                # Build from reserved card (12-14)
                reserved_idx = card_idx - 12
                current_player = board.players[board.currentPlayer]
                if reserved_idx < len(current_player.reserved):
                    action = current_player.reserved[reserved_idx]

    elif action_type == RESERVE:
        # Get card reservation
        card_reservation = safe_float(row['card_reservation'])
        if not math.isnan(card_reservation):
            card_idx = int(card_reservation)

            if card_idx < 12:
                # Reserve from visible card - use relative indexing across all visible cards
                visible_cards = []
                for level_cards in board.displayedCards:
                    visible_cards.extend(level_cards)

                if card_idx < len(visible_cards):
                    action = visible_cards[card_idx]
            else:
                # Reserve from top deck (12-14 maps to deck levels 1-3)
                deck_level = (card_idx - 12) + 1  # 12->1, 13->2, 14->3
                action = deck_level  # For top deck reservation, action is the deck level (int)

    elif action_type == TOKENS:
        # Get token list based on action type
        if action_type_str == 'take 3 tokens':
            # Get gem_take3 values
            tokens = []
            for color in ['white', 'blue', 'green', 'red', 'black']:
                val = safe_float(row[f'gem_take3_{color}'])
                tokens.append(0 if math.isnan(val) else int(val))
            tokens.append(0)  # No gold in take actions
        else:  # 'take 2 tokens'
            # Get gem_take2 values
            tokens = []
            for color in ['white', 'blue', 'green', 'red', 'black']:
                val = safe_float(row[f'gem_take2_{color}'])
                tokens.append(0 if math.isnan(val) else int(val))
            tokens.append(0)  # No gold in take actions

        action = tokens

    # Parse tokens to remove
    for i, color in enumerate(['white', 'blue', 'green', 'red', 'black', 'gold']):
        val = safe_float(row[f'gems_removed_{color}'])
        tokens_to_remove[i] = 0 if math.isnan(val) else int(val)

    # Parse noble selection
    noble_selection = int(safe_float(row['noble_selection']))
    if noble_selection >= 0 and noble_selection < len(board.characters):
        character = board.characters[noble_selection]

    # Create Move object
    return Move(action_type, action, tokens_to_remove, character)


def validate_action_in_moves(board: Board, expected_action: Move) -> Tuple[bool, str]:
    """
    Validate that expected action is in board's valid moves.

    Args:
        board: Reconstructed board
        expected_action: Expected action from CSV

    Returns:
        Tuple of (success: bool, message: str)

    Example:
        >>> is_valid, msg = validate_action_in_moves(board, action)
        >>> if not is_valid:
        ...     print(f"Validation failed: {msg}")
    """
    try:
        # Call getMoves() - this is the main validation
        valid_moves = board.getMoves()

        # Check if expected action is in valid moves
        for move in valid_moves:
            if move == expected_action:
                return (True, "Action is valid")

        # Action not found in valid moves
        return (False, f"Action not in {len(valid_moves)} valid moves")

    except Exception as e:
        # getMoves() raised an exception - reconstruction is incorrect
        return (False, f"getMoves() failed: {str(e)}")


def encode_board_state(board: Board, turn_num: int) -> List[float]:
    """
    Encode board state back to 382 features for round-trip testing.

    Args:
        board: Board object to encode
        turn_num: Turn number for encoding

    Returns:
        List of 382 features

    Example:
        >>> features = encode_board_state(board, 1)
        >>> len(features) == 382
        True
    """
    from src.splendor.csv_exporter import encode_game_state_from_board
    return encode_game_state_from_board(board, turn_num)


def compare_features(original: List[float], reconstructed: List[float],
                     tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Compare two feature lists element-by-element with NaN-aware logic.

    Args:
        original: Original feature list
        reconstructed: Reconstructed feature list
        tolerance: Tolerance for floating point comparison

    Returns:
        Dictionary with comparison results:
        - 'match': bool (all features match)
        - 'num_diffs': int (count of differing features)
        - 'diff_indices': List[int] (indices where features differ)
        - 'diff_details': List[Tuple[int, float, float]] (index, original, reconstructed)

    Example:
        >>> result = compare_features(original, reconstructed)
        >>> if not result['match']:
        ...     print(f"Found {result['num_diffs']} differences")
    """
    if len(original) != len(reconstructed):
        return {
            'match': False,
            'num_diffs': abs(len(original) - len(reconstructed)),
            'diff_indices': [],
            'diff_details': [],
            'error': f"Length mismatch: {len(original)} vs {len(reconstructed)}"
        }

    diff_indices = []
    diff_details = []

    for i in range(len(original)):
        orig_val = original[i]
        recon_val = reconstructed[i]

        # Handle NaN comparisons
        orig_is_nan = isinstance(orig_val, float) and math.isnan(orig_val)
        recon_is_nan = isinstance(recon_val, float) and math.isnan(recon_val)

        if orig_is_nan and recon_is_nan:
            # Both NaN - match
            continue
        elif orig_is_nan or recon_is_nan:
            # One NaN, one not - mismatch
            diff_indices.append(i)
            diff_details.append((i, orig_val, recon_val))
        else:
            # Neither NaN - compare values with tolerance
            if abs(float(orig_val) - float(recon_val)) > tolerance:
                diff_indices.append(i)
                diff_details.append((i, orig_val, recon_val))

    return {
        'match': len(diff_indices) == 0,
        'num_diffs': len(diff_indices),
        'diff_indices': diff_indices,
        'diff_details': diff_details
    }


def validate_round_trip(original_features: List[float], board: Board,
                        turn_num: int) -> Dict[str, Any]:
    """
    Perform round-trip validation: original → reconstruct → encode → compare.

    Args:
        original_features: Original 382 features from CSV
        board: Reconstructed board
        turn_num: Turn number

    Returns:
        Dictionary with validation results

    Example:
        >>> result = validate_round_trip(original_features, board, turn_num)
        >>> if result['comparison']['match']:
        ...     print("Round-trip successful!")
    """
    # Encode reconstructed board
    reconstructed_features = encode_board_state(board, turn_num)

    # Compare features
    comparison = compare_features(original_features, reconstructed_features)

    return {
        'success': comparison['match'],
        'comparison': comparison,
        'original_count': len(original_features),
        'reconstructed_count': len(reconstructed_features)
    }
