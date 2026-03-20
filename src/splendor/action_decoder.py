"""Action decoder for translating model predictions to Move objects.

This module converts multi-head neural network predictions into executable Move objects
that the Splendor game engine understands.

The model has 7 prediction heads:
- action_type: Which type of action (BUILD/RESERVE/TAKE2/TAKE3) - 4 classes
- card_selection: Which card to build (if BUILD) - 15 classes
- card_reservation: Which card to reserve (if RESERVE) - 15 classes
- gem_take3: Which gems to take (if TAKE3) - 26 classes
- gem_take2: Which gem color to take 2 of (if TAKE2) - 5 classes
- noble: Which noble to select (if available) - 5 classes
- gems_removed: Which gems to discard (if overflow) - 84 classes

Example usage:
    >>> predictions = {
    ...     'action_type': 0,  # BUILD
    ...     'card_selection': 5,
    ...     'noble': 2,
    ...     'gems_removed': 0
    ... }
    >>> move = decode_predictions_to_move(predictions, board)
    >>> move.actionType == BUILD
    True
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from imitation_learning.constants import CLASS_TO_COMBO_TAKE3, CLASS_TO_REMOVAL

if TYPE_CHECKING:
    from .board import Board
    from .cards import Card
    from .characters import Character

from .constants import BUILD, RESERVE, TOKENS, WHITE, BLUE, GREEN, RED, BLACK, NO_TOKENS, MAX_NB_TOKENS, GOLD, TAKEONEGOLD
from .custom_operators import add, substract
from .move import Move


def decode_gem_take3(class_idx: int) -> List[int]:
    """Decode gem_take3 class index to token list.

    Args:
        class_idx: Class index (0-25) from model prediction
            - 0: No gems (empty)
            - 1-5: Single gems
            - 6-15: Two gems (combinations)
            - 16-25: Three gems (combinations)

    Returns:
        List of 6 integers representing tokens to take [w, b, g, r, bl, gold]

    Example:
        >>> decode_gem_take3(0)  # No gems
        [0, 0, 0, 0, 0, 0]
        >>> decode_gem_take3(1)  # White
        [1, 0, 0, 0, 0, 0]
        >>> decode_gem_take3(6)  # White + Blue
        [1, 1, 0, 0, 0, 0]
    """
    # Get tuple of color names from class index
    color_tuple = CLASS_TO_COMBO_TAKE3.get(class_idx, tuple())

    # Convert color names to indices
    color_map = {
        'white': WHITE,
        'blue': BLUE,
        'green': GREEN,
        'red': RED,
        'black': BLACK
    }

    # Initialize token list (6 elements: 5 colors + gold)
    tokens = [0] * 6

    # Set 1 for each selected color
    for color_name in color_tuple:
        color_idx = color_map.get(color_name)
        if color_idx is not None:
            tokens[color_idx] = 1

    return tokens


def decode_gem_take2(class_idx: int) -> List[int]:
    """Decode gem_take2 class index to token list.

    Args:
        class_idx: Class index (0-4) representing color
            - 0: White
            - 1: Blue
            - 2: Green
            - 3: Red
            - 4: Black

    Returns:
        List of 6 integers with 2 tokens of selected color [w, b, g, r, bl, gold]

    Example:
        >>> decode_gem_take2(0)  # Take 2 white
        [2, 0, 0, 0, 0, 0]
        >>> decode_gem_take2(2)  # Take 2 green
        [0, 0, 2, 0, 0, 0]
    """
    tokens = [0] * 6

    # Validate class index
    if 0 <= class_idx < 5:
        tokens[class_idx] = 2

    return tokens


def decode_gems_removed(class_idx: int) -> List[int]:
    """Decode gems_removed class index to token removal list.

    Args:
        class_idx: Class index (0-83) representing removal pattern
            - 0: No removal (0, 0, 0, 0, 0, 0)
            - Other: Various combinations of removing 0-3 gems per color, total ≤ 3

    Returns:
        List of 6 integers representing gems to remove [w, b, g, r, bl, gold]

    Example:
        >>> decode_gems_removed(0)  # No removal
        [0, 0, 0, 0, 0, 0]
        >>> decode_gems_removed(1)  # Remove 1 white
        [1, 0, 0, 0, 0, 0]
    """
    # Get removal tuple from class index
    removal_tuple = CLASS_TO_REMOVAL.get(class_idx, (0, 0, 0, 0, 0, 0))

    # Convert tuple to list
    return list(removal_tuple)


def decode_card_selection(class_idx: int, board: 'Board') -> Optional['Card']:
    """Map card_selection index to Card object from board.

    Args:
        class_idx: Class index (0-14)
            - 0-11: Visible cards on board (level 1: 0-3, level 2: 4-7, level 3: 8-11)
            - 12-14: Reserved cards of current player
        board: Board object with displayedCards and current player

    Returns:
        Card object or None if index is invalid

    Example:
        >>> card = decode_card_selection(0, board)  # First level 1 card
        >>> card = decode_card_selection(12, board)  # First reserved card
    """
    current_player = board.players[board.currentPlayer]

    if 0 <= class_idx < 12:
        flat_displayed_cards = [card for lvl in board.displayedCards for card in lvl]
        return flat_displayed_cards[class_idx]

    elif 12 <= class_idx < 15:
        # Reserved cards: indices 12-14
        reserved_idx = class_idx - 12

        if reserved_idx < len(current_player.reserved):
            return current_player.reserved[reserved_idx]

    return None


def decode_card_reservation(class_idx: int, board: 'Board') -> Optional[object]:
    """Map card_reservation index to Card object or deck level.

    Args:
        class_idx: Class index (0-14)
            - 0-11: Visible cards on board (level 1: 0-3, level 2: 4-7, level 3: 8-11)
            - 12: Top deck level 1
            - 13: Top deck level 2
            - 14: Top deck level 3
        board: Board object with displayedCards and decks

    Returns:
        Card object (for visible card) or int 1-3 (for top deck level) or None

    Example:
        >>> result = decode_card_reservation(0, board)  # Visible card
        >>> isinstance(result, Card)
        True
        >>> result = decode_card_reservation(12, board)  # Top deck level 1
        >>> result == 1
        True
    """
    if 0 <= class_idx < 12:
        # Visible cards: same mapping as card_selection
        level = class_idx // 4
        card_in_level = class_idx % 4

        if level < len(board.displayedCards) and card_in_level < len(board.displayedCards[level]):
            return board.displayedCards[level][card_in_level]

    elif 12 <= class_idx < 15:
        # Top deck: indices 12-14 map to deck levels 1-3
        deck_level = class_idx - 11  # 12→1, 13→2, 14→3

        # Check if deck has cards
        if deck_level >= 1 and deck_level <= 3:
            deck_idx = deck_level - 1
            if deck_idx < len(board.decks) and len(board.decks[deck_idx]) > 0:
                return deck_level

    return None


def decode_noble_selection(class_idx: int, board: 'Board') -> Optional['Character']:
    """Map noble index to Character object from board.

    Args:
        class_idx: Class index (0-4) corresponding to board.characters[0-4]
        board: Board object with characters list

    Returns:
        Character object or None if index is invalid or no noble at that position

    Example:
        >>> noble = decode_noble_selection(0, board)  # First noble
        >>> noble = decode_noble_selection(5, board)  # Invalid → None
    """
    if 0 <= class_idx < len(board.characters):
        return board.characters[class_idx]

    return None


def validate_token_removal(tokens_after_taking: List[int], tokens_to_remove: List[int]) -> List[int]:
    """Validate and fix token removal to ensure no negative tokens.

    Args:
        tokens_after_taking: Player's token count after taking new tokens (6 elements)
        tokens_to_remove: Predicted token removal (6 elements)

    Returns:
        Validated token removal list that won't cause negative tokens

    Example:
        >>> tokens_after = [4, 4, 2, 1, 2, 0]  # Player has 13 tokens after taking
        >>> tokens_remove = [0, 0, 0, 2, 1, 0]  # Would make red negative
        >>> validate_token_removal(tokens_after, tokens_remove)
        [0, 1, 0, 1, 1, 0]  # Safe removal (3 tokens removed)
    """
    # Check if predicted removal is valid (no negative tokens)
    result_tokens = substract(tokens_after_taking, tokens_to_remove)

    # If all tokens are non-negative, the removal is valid
    if all(t >= 0 for t in result_tokens):
        return tokens_to_remove

    # Invalid removal - generate a safe removal pattern
    # Calculate how many tokens need to be removed
    total_tokens = sum(tokens_after_taking)
    tokens_to_discard = max(0, total_tokens - MAX_NB_TOKENS)

    # If no removal needed, return empty removal
    if tokens_to_discard == 0:
        return [0] * 6

    # Remove tokens in order of abundance (most plentiful first)
    # Create list of (count, index) pairs
    token_counts = [(count, idx) for idx, count in enumerate(tokens_after_taking)]
    # Sort by count descending, then by index for consistency
    token_counts.sort(key=lambda x: (-x[0], x[1]))

    # Build safe removal
    safe_removal = [0] * 6
    remaining_to_remove = tokens_to_discard

    for count, idx in token_counts:
        if remaining_to_remove == 0:
            break

        # Remove as many as possible from this color (up to what we have)
        can_remove = min(count, remaining_to_remove)
        safe_removal[idx] = can_remove
        remaining_to_remove -= can_remove

    return safe_removal


def decode_predictions_to_move(predictions: Dict[str, int], board: 'Board') -> Move:
    """Decode all model predictions into a Move object.

    This is the main entry point for converting neural network outputs into
    executable game actions.

    Args:
        predictions: Dict mapping head name to predicted class index
            Required keys: 'action_type', 'card_selection', 'card_reservation',
                          'gem_take3', 'gem_take2', 'noble', 'gems_removed'
        board: Board object needed for card/noble lookups

    Returns:
        Move object ready to be executed by the game engine

    Example:
        >>> predictions = {
        ...     'action_type': 0,  # BUILD
        ...     'card_selection': 5,
        ...     'card_reservation': 0,
        ...     'gem_take3': 0,
        ...     'gem_take2': 0,
        ...     'noble': 2,
        ...     'gems_removed': 0
        ... }
        >>> move = decode_predictions_to_move(predictions, board)
        >>> move.actionType == BUILD
        True
        >>> isinstance(move.action, Card)
        True

    Notes:
        - Action types: 0=BUILD, 1=RESERVE, 2=TAKE2, 3=TAKE3
        - For BUILD: uses card_selection, noble, gems_removed
        - For RESERVE: uses card_reservation, gems_removed
        - For TAKE2/TAKE3: uses gem_take2/gem_take3, gems_removed
        - If predictions are invalid, returns a safe fallback move
        - Logic follows board.py:doMove structure: main action → token removal → noble selection
    """
    action_type = predictions.get('action_type', 0)

    # Create a deep copy for simulation without side effects
    board_copy = board.clone()

    # Decode action based on type and simulate on board copy
    if action_type == BUILD:  # BUILD
        card = decode_card_selection(predictions.get('card_selection', 0), board)
        if card is None:
            raise ValueError("trying to build a None card")

        # Simulate BUILD action on copy
        board_copy.build(Move(BUILD, card, NO_TOKENS, None))
        action = card
        final_action_type = BUILD

    elif action_type == RESERVE:  # RESERVE
        card_or_level = decode_card_reservation(predictions.get('card_reservation', 0), board)

        # Fallback if invalid
        if card_or_level is None:
            if len(board.displayedCards[0]) > 0:
                card_or_level = board.displayedCards[0][0]

        # Simulate RESERVE action on copy
        board_copy.reserve(Move(RESERVE, card_or_level, NO_TOKENS, None))
        action = card_or_level
        final_action_type = RESERVE

    elif action_type == 2:  # TAKE2
        tokens = decode_gem_take2(predictions.get('gem_take2', 0))

        # Simulate TAKE2 action on copy
        board_copy.takeTokens(Move(TOKENS, tokens, NO_TOKENS, None))
        action = tokens
        final_action_type = TOKENS

    else:  # TAKE3 (action_type == 3)
        tokens = decode_gem_take3(predictions.get('gem_take3', 0))

        # Simulate TAKE3 action on copy
        board_copy.takeTokens(Move(TOKENS, tokens, NO_TOKENS, None))
        action = tokens
        final_action_type = TOKENS

    # Common logic: Get player state after the action
    current_player_copy = board_copy.getCurrentPlayer()
    tokens_after_action = current_player_copy.tokens
    bonus_after_action = current_player_copy.getTotalBonus()

    # Common logic: Check if token removal is needed (sum > 10)
    tokens_to_remove = decode_gems_removed(predictions.get('gems_removed', 0))
    if sum(tokens_after_action) > MAX_NB_TOKENS:
        tokens_to_remove = validate_token_removal(tokens_after_action, tokens_to_remove)
    else:
        tokens_to_remove = NO_TOKENS

    # Common logic: Check for available nobles based on post-action bonus
    possible_nobles = list(filter(
        lambda c: all(color >= 0 for color in substract(bonus_after_action, c.cost)),
        board_copy.characters
    ))
    noble = None
    if possible_nobles:
        noble = decode_noble_selection(predictions.get('noble', 0), board)

    return Move(final_action_type, action, tokens_to_remove, noble)


if __name__ == "__main__":
    # Test decoding functions
    print("Testing action decoder...")

    # Test gem_take3 decoding
    print("\nTesting gem_take3 decoding:")
    print(f"Class 0 (empty): {decode_gem_take3(0)}")
    print(f"Class 1 (white): {decode_gem_take3(1)}")
    print(f"Class 6 (white+blue): {decode_gem_take3(6)}")
    print(f"Class 16 (white+blue+green): {decode_gem_take3(16)}")

    # Test gem_take2 decoding
    print("\nTesting gem_take2 decoding:")
    for i in range(5):
        print(f"Class {i}: {decode_gem_take2(i)}")

    # Test gems_removed decoding
    print("\nTesting gems_removed decoding:")
    print(f"Class 0 (no removal): {decode_gems_removed(0)}")
    print(f"Class 1 (remove 1 white): {decode_gems_removed(1)}")

    print("\n✓ Action decoder test passed!")
