"""Advanced feature engineering for Splendor imitation learning.

This module extracts strategic game-state features that encode domain knowledge:
- Token availability and taking opportunities
- Card purchasing feasibility and distances (per player, per card)
- Noble acquisition opportunities and distances (per player, per noble)
- Player competitive positioning (leaderboard, VP gaps, buying power)
- Game progression metrics (distance to end, deck depletion)

Total features: ~893 new features
- Token features: 26
- Card features: 540 (4 players × 15 cards × 9)
- Noble features: 148 (4 players × 37)
- Card-noble synergy: 120 (4 players × 15 cards × 2)
- Player comparison: 50
- Game progression: 9

Example usage:
    >>> import pandas as pd
    >>> from feature_engineering import extract_all_features
    >>> row = df.iloc[0]
    >>> features = extract_all_features(row)
    >>> print(f"Extracted {len(features)} features")
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from splendor.board import Board
from splendor.cards import Card
from splendor.characters import Character
from splendor.constants import (
    GOLD,
    MAX_NB_TOKENS,
    NB_TOKEN_2,
    NB_TOKEN_3,
    NB_TOKEN_4,
    VP_GOAL,
)
from splendor.player import Player
from utils.state_reconstruction import reconstruct_board_from_csv_row

# Type aliases
FeatureDict = Dict[str, float]


def extract_all_features(row: pd.Series, board: 'Board' = None) -> FeatureDict:
    """Main entry point: extract all strategic features from a game state.

    Args:
        row: Pandas Series containing raw game state from CSV
        board: Optional pre-reconstructed Board object (optimization to avoid redundant reconstruction)
               If None, will reconstruct from row (backward compatible)

    Returns:
        Dictionary mapping feature names to float values

    Example:
        >>> features = extract_all_features(df.iloc[0])
        >>> len(features)  # ~893 features

    Note:
        OPTIMIZATION: When processing many rows, reconstruct the board once and
        pass it to both mask generation and feature engineering to avoid
        redundant board reconstructions (2x speedup).
    """
    features: FeatureDict = {}

    try:
        # Reconstruct board state from CSV if not provided
        if board is None:
            board = reconstruct_board_from_csv_row(row.to_dict())

        # Extract all feature categories
        features.update(extract_token_features(row, board))
        features.update(extract_card_features(row, board))
        features.update(extract_noble_features(row, board))
        features.update(extract_card_noble_synergy(row, board))
        features.update(extract_player_comparison_features(row, board))
        features.update(extract_game_progression_features(row, board))

    except Exception as e:
        # If extraction fails, return empty dict (will be filled with zeros)
        game_id = row.get("game_id", "unknown")
        turn_num = row.get("turn_number", "unknown")
        print(
            f"WARNING: Feature extraction failed for game_id={game_id}, "
            f"turn={turn_num}: {e}"
        )

    return features


def extract_token_features(row: pd.Series, board: Board) -> FeatureDict:
    """Extract token-related features (26 features).

    Features per color (white, blue, green, red, black):
    - can_take2_{color}: Binary (1 if pile >= 4)
    - tokens_left_if_take2_{color}: Remaining tokens after taking 2
    - tokens_left_if_take1_{color}: Remaining tokens after taking 1
    - max_tokens_pile_{color}: Max based on num_players
    - maximum_takeable_this_turn_{color}: min(2, available)

    Args:
        row: CSV row
        board: Reconstructed Board object

    Returns:
        Dict with 26 token features
    """
    features: FeatureDict = {}
    colors = ["white", "blue", "green", "red", "black"]
    num_players = int(row.get("num_players", 4))

    # Determine max tokens based on player count
    if num_players == 2:
        max_regular = NB_TOKEN_2
    elif num_players == 3:
        max_regular = NB_TOKEN_3
    else:
        max_regular = NB_TOKEN_4

    for idx, color in enumerate(colors):
        col_name = f"gems_board_{color}"
        tokens_available = int(row.get(col_name, 0))

        # Can take 2 of this color?
        can_take2 = 1.0 if tokens_available >= 4 else 0.0
        features[f"can_take2_{color}"] = can_take2

        # Tokens remaining after taking 2
        if can_take2:
            features[f"tokens_left_if_take2_{color}"] = float(tokens_available - 2)
        else:
            features[f"tokens_left_if_take2_{color}"] = float(tokens_available)

        # Tokens remaining after taking 1
        features[f"tokens_left_if_take1_{color}"] = float(max(0, tokens_available - 1))

        # Max pile size for this game
        features[f"max_tokens_pile_{color}"] = float(max_regular)

        # Maximum takeable this turn
        features[f"maximum_takeable_this_turn_{color}"] = float(
            min(2, tokens_available)
        )

    # Gold pile max (always 5 regardless of player count)
    features["max_tokens_pile_gold"] = 5.0

    return features


def extract_card_features(row: pd.Series, board: Board) -> FeatureDict:
    """Extract card-related features for all players (540 features).

    Per player, per card (15 cards: 12 visible + 3 reserved):
    - player{i}_card{j}_can_build: Binary
    - player{i}_card{j}_must_use_gold: Binary
    - player{i}_card{j}_distance_{color}: Distance per color (5)
    - player{i}_card{j}_distance_total: Sum of distances
    - player{i}_card{j}_vp_if_buy: Projected VP after purchase

    Args:
        row: CSV row
        board: Reconstructed Board object

    Returns:
        Dict with 540 card features (4 players × 15 cards × 9 features)
    """
    features: FeatureDict = {}
    colors = ["white", "blue", "green", "red", "black"]

    num_players = len(board.players)

    # Extract features for actual players
    for player_idx in range(num_players):
        player = board.players[player_idx]

        # Collect all cards this player could interact with
        all_cards = []

        # Add 12 visible cards from board
        for level in range(3):
            for card in board.displayedCards[level]:
                all_cards.append(card)

        # Add player's reserved cards (up to 3)
        for card in player.reserved:
            all_cards.append(card)

        # Pad to exactly 15 cards
        while len(all_cards) < 15:
            all_cards.append(None)

        # Extract features for each card
        for card_idx, card in enumerate(all_cards):
            prefix = f"player{player_idx}_card{card_idx}_"

            if card is None:
                # No card in this slot
                features[f"{prefix}can_build"] = 0.0
                features[f"{prefix}must_use_gold"] = 0.0
                for color in colors:
                    features[f"{prefix}distance_{color}"] = 0.0
                features[f"{prefix}distance_total"] = 0.0
                features[f"{prefix}vp_if_buy"] = 0.0
            else:
                # Can the player build this card?
                can_build = player.canBuild(card)
                features[f"{prefix}can_build"] = 1.0 if can_build else 0.0

                # Calculate real cost (after bonuses)
                real_cost = player.realCost(card)

                # Check if must use gold
                must_use_gold = 0.0
                if can_build:
                    # Need gold if any real cost > player tokens for that color
                    for i in range(5):
                        if real_cost[i] > player.tokens[i]:
                            must_use_gold = 1.0
                            break
                features[f"{prefix}must_use_gold"] = must_use_gold

                # Distance per color (how many more gems needed)
                total_distance = 0.0
                for i, color in enumerate(colors):
                    distance = max(0, real_cost[i] - player.tokens[i])
                    features[f"{prefix}distance_{color}"] = float(distance)
                    total_distance += distance

                features[f"{prefix}distance_total"] = total_distance

                # Projected VP if we buy this card
                projected_vp = player.vp + card.vp

                # Check if buying this card would enable acquiring a noble
                new_reductions = player.reductions.copy()
                new_reductions[card.bonus] += 1

                nobles_after = 0
                for noble in board.characters:
                    if noble is None:
                        continue
                    # Check if player can acquire this noble after buying card
                    can_acquire = all(
                        new_reductions[i] >= noble.cost[i] for i in range(5)
                    )
                    if can_acquire:
                        nobles_after += 1

                # Add noble bonus VP (assuming 3 VP per noble)
                if nobles_after > len(player.characters):
                    projected_vp += 3.0

                features[f"{prefix}vp_if_buy"] = float(projected_vp)

    # Pad features for missing players with zeros
    for player_idx in range(num_players, 4):
        for card_idx in range(15):
            prefix = f"player{player_idx}_card{card_idx}_"
            features[f"{prefix}can_build"] = 0.0
            features[f"{prefix}must_use_gold"] = 0.0
            for color in colors:
                features[f"{prefix}distance_{color}"] = 0.0
            features[f"{prefix}distance_total"] = 0.0
            features[f"{prefix}vp_if_buy"] = 0.0

    return features


def extract_noble_features(row: pd.Series, board: Board) -> FeatureDict:
    """Extract noble-related features for all players (148 features).

    Per player, per noble (5 nobles):
    - player{i}_noble{j}_distance_{color}: Distance per color (5)
    - player{i}_noble{j}_distance_total: Sum of distances
    - player{i}_noble{j}_acquirable: Binary

    Aggregate per player:
    - player{i}_closest_noble_distance: Min distance to any noble
    - player{i}_nobles_acquirable_count: Count of immediately acquirable nobles

    Args:
        row: CSV row
        board: Reconstructed Board object

    Returns:
        Dict with 148 noble features (4 players × 37 features)
    """
    features: FeatureDict = {}
    colors = ["white", "blue", "green", "red", "black"]

    num_players = len(board.players)

    # Extract features for actual players
    for player_idx in range(num_players):
        player = board.players[player_idx]
        reductions = player.getTotalBonus()

        # Pad nobles to exactly 5
        nobles = board.characters[:]
        while len(nobles) < 5:
            nobles.append(None)

        min_distance = float("inf")
        acquirable_count = 0

        for noble_idx, noble in enumerate(nobles[:5]):
            prefix = f"player{player_idx}_noble{noble_idx}_"

            if noble is None:
                # No noble in this slot
                for color in colors:
                    features[f"{prefix}distance_{color}"] = 0.0
                features[f"{prefix}distance_total"] = 0.0
                features[f"{prefix}acquirable"] = 0.0
            else:
                # Calculate distance for each color
                total_distance = 0.0
                for i, color in enumerate(colors):
                    distance = max(0, noble.cost[i] - reductions[i])
                    features[f"{prefix}distance_{color}"] = float(distance)
                    total_distance += distance

                features[f"{prefix}distance_total"] = total_distance

                # Is this noble immediately acquirable?
                acquirable = 1.0 if total_distance == 0.0 else 0.0
                features[f"{prefix}acquirable"] = acquirable

                # Update aggregates
                if total_distance < min_distance:
                    min_distance = total_distance
                acquirable_count += int(acquirable)

        # Aggregate features
        if min_distance == float("inf"):
            min_distance = 0.0
        features[f"player{player_idx}_closest_noble_distance"] = min_distance
        features[f"player{player_idx}_nobles_acquirable_count"] = float(
            acquirable_count
        )

    # Pad features for missing players with zeros
    for player_idx in range(num_players, 4):
        for noble_idx in range(5):
            prefix = f"player{player_idx}_noble{noble_idx}_"
            for color in colors:
                features[f"{prefix}distance_{color}"] = 0.0
            features[f"{prefix}distance_total"] = 0.0
            features[f"{prefix}acquirable"] = 0.0
        features[f"player{player_idx}_closest_noble_distance"] = 0.0
        features[f"player{player_idx}_nobles_acquirable_count"] = 0.0

    return features


def extract_card_noble_synergy(row: pd.Series, board: Board) -> FeatureDict:
    """Extract card-noble synergy features (120 features).

    Per player, per card (15 cards):
    - player{i}_card{j}_nobles_after_buy: Count of nobles acquirable after
    - player{i}_card{j}_closest_noble_distance_after_buy: Min distance after

    Args:
        row: CSV row
        board: Reconstructed Board object

    Returns:
        Dict with 120 synergy features (4 players × 15 cards × 2)
    """
    features: FeatureDict = {}

    num_players = len(board.players)

    # Extract features for actual players
    for player_idx in range(num_players):
        player = board.players[player_idx]

        # Collect all cards
        all_cards = []
        for level in range(3):
            for card in board.displayedCards[level]:
                all_cards.append(card)
        for card in player.reserved:
            all_cards.append(card)
        while len(all_cards) < 15:
            all_cards.append(None)

        for card_idx, card in enumerate(all_cards):
            prefix = f"player{player_idx}_card{card_idx}_"

            if card is None:
                features[f"{prefix}nobles_after_buy"] = 0.0
                features[f"{prefix}closest_noble_distance_after_buy"] = 0.0
            else:
                # Simulate buying this card
                new_reductions = player.reductions.copy()
                new_reductions[card.bonus] += 1

                # Count nobles and find closest
                nobles_acquirable = 0
                min_distance = float("inf")

                for noble in board.characters:
                    if noble is None:
                        continue

                    # Calculate distance with new reductions
                    distance = sum(
                        max(0, noble.cost[i] - new_reductions[i]) for i in range(5)
                    )

                    if distance == 0:
                        nobles_acquirable += 1
                    if distance < min_distance:
                        min_distance = distance

                if min_distance == float("inf"):
                    min_distance = 0.0

                features[f"{prefix}nobles_after_buy"] = float(nobles_acquirable)
                features[f"{prefix}closest_noble_distance_after_buy"] = float(
                    min_distance
                )

    # Pad features for missing players with zeros
    for player_idx in range(num_players, 4):
        for card_idx in range(15):
            prefix = f"player{player_idx}_card{card_idx}_"
            features[f"{prefix}nobles_after_buy"] = 0.0
            features[f"{prefix}closest_noble_distance_after_buy"] = 0.0

    return features


def extract_player_comparison_features(row: pd.Series, board: Board) -> FeatureDict:
    """Extract player comparison and ranking features (50 features).

    Per player (8 base features):
    - player{i}_vp: Victory points
    - player{i}_total_gems_reduction: Sum of bonuses
    - player{i}_buying_capacity: Tokens + bonuses
    - player{i}_total_gems_possessed: Sum of all tokens
    - player{i}_total_gem_colors_possessed: Count of non-zero colors
    - player{i}_num_reserved_cards: Reserved count
    - player{i}_num_nobles_acquired: Noble count
    - player{i}_num_cards_bought: Built cards count

    Per player (4 relative features):
    - player{i}_distance_to_max_vp: Gap from leader
    - player{i}_leaderboard_position: Rank by VP (0-3)
    - player{i}_vp_gap_to_leader: VP difference from 1st
    - player{i}_gems_reduction_leaderboard_position: Rank by bonuses

    Global (2 features):
    - max_vp_among_players
    - max_gems_reduction_among_players

    Args:
        row: CSV row
        board: Reconstructed Board object

    Returns:
        Dict with 50 comparison features
    """
    features: FeatureDict = {}

    num_players = len(board.players)

    # Collect player stats (pad with zeros for missing players)
    player_vps = [0.0] * 4
    player_reductions = [0.0] * 4

    # Extract features for actual players
    for player_idx in range(num_players):
        player = board.players[player_idx]
        prefix = f"player{player_idx}_"

        # Base features
        vp = float(player.vp)
        total_reduction = float(sum(player.reductions))
        total_tokens = float(sum(player.tokens[:5]))  # Exclude gold
        buying_capacity = total_tokens + total_reduction

        player_vps[player_idx] = vp
        player_reductions[player_idx] = total_reduction

        features[f"{prefix}vp"] = vp
        features[f"{prefix}total_gems_reduction"] = total_reduction
        features[f"{prefix}buying_capacity"] = buying_capacity
        features[f"{prefix}total_gems_possessed"] = float(sum(player.tokens))
        features[f"{prefix}total_gem_colors_possessed"] = float(
            sum(1 for t in player.tokens[:5] if t > 0)
        )
        features[f"{prefix}num_reserved_cards"] = float(len(player.reserved))
        features[f"{prefix}num_nobles_acquired"] = float(len(player.characters))
        features[f"{prefix}num_cards_bought"] = float(len(player.built))

    # Pad features for missing players with zeros
    for player_idx in range(num_players, 4):
        prefix = f"player{player_idx}_"
        features[f"{prefix}vp"] = 0.0
        features[f"{prefix}total_gems_reduction"] = 0.0
        features[f"{prefix}buying_capacity"] = 0.0
        features[f"{prefix}total_gems_possessed"] = 0.0
        features[f"{prefix}total_gem_colors_possessed"] = 0.0
        features[f"{prefix}num_reserved_cards"] = 0.0
        features[f"{prefix}num_nobles_acquired"] = 0.0
        features[f"{prefix}num_cards_bought"] = 0.0

    # Global features
    max_vp = max(player_vps)
    max_reduction = max(player_reductions)

    features["max_vp_among_players"] = max_vp
    features["max_gems_reduction_among_players"] = max_reduction

    # Relative features
    # VP leaderboard (descending)
    vp_ranking = sorted(
        range(4), key=lambda i: player_vps[i], reverse=True
    )  # [best_idx, 2nd_idx, ...]
    vp_positions = [0] * 4
    for pos, player_idx in enumerate(vp_ranking):
        vp_positions[player_idx] = pos

    # Reduction leaderboard (descending)
    reduction_ranking = sorted(range(4), key=lambda i: player_reductions[i], reverse=True)
    reduction_positions = [0] * 4
    for pos, player_idx in enumerate(reduction_ranking):
        reduction_positions[player_idx] = pos

    for player_idx in range(4):
        prefix = f"player{player_idx}_"

        features[f"{prefix}distance_to_max_vp"] = max_vp - player_vps[player_idx]
        features[f"{prefix}leaderboard_position"] = float(vp_positions[player_idx])
        features[f"{prefix}vp_gap_to_leader"] = player_vps[vp_ranking[0]] - player_vps[
            player_idx
        ]
        features[f"{prefix}gems_reduction_leaderboard_position"] = float(
            reduction_positions[player_idx]
        )

    return features


def extract_game_progression_features(row: pd.Series, board: Board) -> FeatureDict:
    """Extract game progression features (9 features).

    Global features (5):
    - distance_to_end_game: 15 - max VP
    - deck_level1_remaining: Cards left in deck 1
    - deck_level2_remaining: Cards left in deck 2
    - deck_level3_remaining: Cards left in deck 3
    - total_cards_bought: Sum across all players

    Per player (4):
    - player{i}_vp_to_win: 15 - player VP

    Args:
        row: CSV row
        board: Reconstructed Board object

    Returns:
        Dict with 9 progression features
    """
    features: FeatureDict = {}

    num_players = len(board.players)

    # Global features
    max_vp = max(player.vp for player in board.players)
    features["distance_to_end_game"] = float(VP_GOAL - max_vp)

    features["deck_level1_remaining"] = float(row.get("deck_level1_remaining", 0))
    features["deck_level2_remaining"] = float(row.get("deck_level2_remaining", 0))
    features["deck_level3_remaining"] = float(row.get("deck_level3_remaining", 0))

    total_cards_bought = sum(len(player.built) for player in board.players)
    features["total_cards_bought"] = float(total_cards_bought)

    # Per player - actual players
    for player_idx in range(num_players):
        player = board.players[player_idx]
        features[f"player{player_idx}_vp_to_win"] = float(VP_GOAL - player.vp)

    # Pad features for missing players with zeros
    for player_idx in range(num_players, 4):
        features[f"player{player_idx}_vp_to_win"] = 0.0

    return features


def get_all_feature_names() -> List[str]:
    """Get ordered list of all feature names that will be generated.

    This is useful for validation, debugging, and ensuring consistency.

    Returns:
        List of feature names in the order they're generated
    """
    feature_names = []
    colors = ["white", "blue", "green", "red", "black"]

    # Token features (26)
    for color in colors:
        feature_names.append(f"can_take2_{color}")
        feature_names.append(f"tokens_left_if_take2_{color}")
        feature_names.append(f"tokens_left_if_take1_{color}")
        feature_names.append(f"max_tokens_pile_{color}")
        feature_names.append(f"maximum_takeable_this_turn_{color}")
    feature_names.append("max_tokens_pile_gold")

    # Card features (540: 4 players × 15 cards × 9)
    for player_idx in range(4):
        for card_idx in range(15):
            prefix = f"player{player_idx}_card{card_idx}_"
            feature_names.append(f"{prefix}can_build")
            feature_names.append(f"{prefix}must_use_gold")
            for color in colors:
                feature_names.append(f"{prefix}distance_{color}")
            feature_names.append(f"{prefix}distance_total")
            feature_names.append(f"{prefix}vp_if_buy")

    # Noble features (148: 4 players × 37)
    for player_idx in range(4):
        for noble_idx in range(5):
            prefix = f"player{player_idx}_noble{noble_idx}_"
            for color in colors:
                feature_names.append(f"{prefix}distance_{color}")
            feature_names.append(f"{prefix}distance_total")
            feature_names.append(f"{prefix}acquirable")
        feature_names.append(f"player{player_idx}_closest_noble_distance")
        feature_names.append(f"player{player_idx}_nobles_acquirable_count")

    # Card-noble synergy (120: 4 players × 15 cards × 2)
    for player_idx in range(4):
        for card_idx in range(15):
            prefix = f"player{player_idx}_card{card_idx}_"
            feature_names.append(f"{prefix}nobles_after_buy")
            feature_names.append(f"{prefix}closest_noble_distance_after_buy")

    # Player comparison (50)
    for player_idx in range(4):
        prefix = f"player{player_idx}_"
        feature_names.append(f"{prefix}vp")
        feature_names.append(f"{prefix}total_gems_reduction")
        feature_names.append(f"{prefix}buying_capacity")
        feature_names.append(f"{prefix}total_gems_possessed")
        feature_names.append(f"{prefix}total_gem_colors_possessed")
        feature_names.append(f"{prefix}num_reserved_cards")
        feature_names.append(f"{prefix}num_nobles_acquired")
        feature_names.append(f"{prefix}num_cards_bought")

    feature_names.append("max_vp_among_players")
    feature_names.append("max_gems_reduction_among_players")

    for player_idx in range(4):
        prefix = f"player{player_idx}_"
        feature_names.append(f"{prefix}distance_to_max_vp")
        feature_names.append(f"{prefix}leaderboard_position")
        feature_names.append(f"{prefix}vp_gap_to_leader")
        feature_names.append(f"{prefix}gems_reduction_leaderboard_position")

    # Game progression (9)
    feature_names.append("distance_to_end_game")
    feature_names.append("deck_level1_remaining")
    feature_names.append("deck_level2_remaining")
    feature_names.append("deck_level3_remaining")
    feature_names.append("total_cards_bought")

    for player_idx in range(4):
        feature_names.append(f"player{player_idx}_vp_to_win")

    return feature_names


if __name__ == "__main__":
    # Validate feature count
    feature_names = get_all_feature_names()
    print(f"Total strategic features: {len(feature_names)}")
    print(f"\nExpected breakdown:")
    print(f"  Token features: 26")
    print(f"  Card features: 540")
    print(f"  Noble features: 148")
    print(f"  Card-noble synergy: 120")
    print(f"  Player comparison: 50")
    print(f"  Game progression: 9")
    print(f"  Total: 893")
    print(f"\nActual: {len(feature_names)}")

    # Show sample feature names
    print(f"\nSample feature names:")
    for i in [0, 25, 100, 200, 500, 700, 850, -1]:
        if abs(i) < len(feature_names):
            print(f"  [{i}] {feature_names[i]}")
