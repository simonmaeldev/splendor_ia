#!/usr/bin/env python3
"""
Game Replay Script for Splendor AI

This script loads game data from CSV files and displays the complete game progression
turn-by-turn with color-coded terminal output. It reverses the encoding logic from
csv_exporter.py to reconstruct human-readable game states and actions.

Usage:
    python scripts/replay_game.py --file data/games/2_games/1.csv
    python scripts/replay_game.py --game-id 1 --players 2
    python scripts/replay_game.py --help

Features:
    - Loads and decodes 402-column CSV files (game_id + 382 inputs + 20 outputs)
    - Displays board state, player states, and actions for each turn
    - Uses ANSI colors for gem types (white, blue, green, red, black, gold)
    - Handles 2, 3, and 4 player games with proper NaN handling
    - Shows all action types: build, reserve, take tokens, noble attractions
"""

import csv
import argparse
import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from splendor.constants import (
    WHITE, BLUE, GREEN, RED, BLACK, GOLD,
    bcolors, getColor, fullStrColor,
    DECK1, DECK2, DECK3, CHARACTERS
)

# Color order matching csv_exporter.py
COLOR_ORDER = [WHITE, BLUE, GREEN, RED, BLACK, GOLD]
COLOR_NAMES = ['white', 'blue', 'green', 'red', 'black', 'gold']


def is_nan(value: Any) -> bool:
    """Check if a value is NaN."""
    try:
        return math.isnan(float(value))
    except (ValueError, TypeError):
        return False


def safe_float(value: Any) -> float:
    """Convert to float, handling empty strings."""
    if value == '' or value is None:
        return float('nan')
    return float(value)


def safe_int(value: Any) -> int:
    """Convert to int, handling NaN."""
    if is_nan(value):
        return 0
    return int(float(value))


class GameCard:
    """Represents a Splendor card."""
    def __init__(self, vp: int, level: int, cost: List[int], bonus: int):
        self.vp = vp
        self.level = level
        self.cost = cost  # [white, blue, green, red, black]
        self.bonus = bonus  # 0-4 for WHITE-BLACK

    def __str__(self):
        return f"L{self.level} {self.vp}VP {fullStrColor(self.bonus)}"


class GameNoble:
    """Represents a noble/character."""
    def __init__(self, vp: int, requirements: List[int]):
        self.vp = vp
        self.requirements = requirements  # [white, blue, green, red, black]

    def __str__(self):
        return f"{self.vp}VP Noble"


class PlayerState:
    """Represents a player's state."""
    def __init__(self, position: int, vp: int, gems: List[int],
                 reductions: List[int], reserved_cards: List[Optional[GameCard]]):
        self.position = position
        self.vp = vp
        self.gems = gems  # [white, blue, green, red, black, gold]
        self.reductions = reductions  # [white, blue, green, red, black]
        self.reserved_cards = reserved_cards  # Up to 3 cards


class GameState:
    """Represents complete game state for one turn."""
    def __init__(self):
        self.num_players: int = 0
        self.turn_number: int = 0
        self.current_player: int = 0
        self.board_gems: List[int] = [0] * 6
        self.visible_cards: List[Optional[GameCard]] = []
        self.deck_remaining: List[int] = [0, 0, 0]
        self.nobles: List[Optional[GameNoble]] = []
        self.players: List[PlayerState] = []


class GameAction:
    """Represents an action taken."""
    def __init__(self):
        self.action_type: str = ""
        self.card_selection: Optional[int] = None
        self.card_reservation: Optional[int] = None
        self.gem_take_3: List[int] = []
        self.gem_take_2: List[int] = []
        self.noble_selection: int = -1
        self.gems_removed: List[int] = [0] * 6


def decode_card(features: List[float]) -> Optional[GameCard]:
    """
    Decode 12 features into a GameCard object.

    Features: [vp, level, cost_white, cost_blue, cost_green, cost_red, cost_black,
               bonus_white, bonus_blue, bonus_green, bonus_red, bonus_black]
    """
    if is_nan(features[0]):
        return None

    vp = safe_int(features[0])
    level = safe_int(features[1])
    cost = [safe_int(features[i]) for i in range(2, 7)]

    # Decode bonus from one-hot encoding
    bonus_onehot = [safe_int(features[i]) for i in range(7, 12)]
    bonus = -1
    for i, val in enumerate(bonus_onehot):
        if val == 1:
            bonus = i
            break

    return GameCard(vp, level, cost, bonus)


def decode_noble(features: List[float]) -> Optional[GameNoble]:
    """
    Decode 6 features into a GameNoble object.

    Features: [vp, req_white, req_blue, req_green, req_red, req_black]
    """
    if is_nan(features[0]):
        return None

    vp = safe_int(features[0])
    requirements = [safe_int(features[i]) for i in range(1, 6)]

    return GameNoble(vp, requirements)


def decode_player_state(features: List[float]) -> Optional[PlayerState]:
    """
    Decode 49 features into a PlayerState object.

    Features: [position, vp, gems(6), reductions(5), reserved_cards(3x12)]
    """
    if is_nan(features[0]):
        return None

    position = safe_int(features[0])
    vp = safe_int(features[1])
    gems = [safe_int(features[i]) for i in range(2, 8)]
    reductions = [safe_int(features[i]) for i in range(8, 13)]

    # Decode 3 reserved cards
    reserved_cards = []
    for i in range(3):
        start_idx = 13 + i * 12
        card_features = features[start_idx:start_idx + 12]
        card = decode_card(card_features)
        reserved_cards.append(card)

    return PlayerState(position, vp, gems, reductions, reserved_cards)


def decode_game_state(row: List[str]) -> GameState:
    """
    Decode 382 input features into a GameState object.

    CSV columns: game_id + 382 inputs + 20 outputs
    Input features start at index 1.
    """
    state = GameState()

    # Convert row to floats
    features = [safe_float(val) for val in row[1:383]]  # Skip game_id, take 382 features

    idx = 0

    # Basic info (3 features)
    state.num_players = safe_int(features[idx])
    idx += 1
    state.turn_number = safe_int(features[idx])
    idx += 1
    state.current_player = safe_int(features[idx])
    idx += 1

    # Board gems (6 features)
    state.board_gems = [safe_int(features[idx + i]) for i in range(6)]
    idx += 6

    # Visible cards (12 cards × 12 features = 144)
    for i in range(12):
        card_features = features[idx:idx + 12]
        card = decode_card(card_features)
        state.visible_cards.append(card)
        idx += 12

    # Deck remaining (3 features)
    state.deck_remaining = [safe_int(features[idx + i]) for i in range(3)]
    idx += 3

    # Nobles (5 nobles × 6 features = 30)
    for i in range(5):
        noble_features = features[idx:idx + 6]
        noble = decode_noble(noble_features)
        state.nobles.append(noble)
        idx += 6

    # Players (4 players × 49 features = 196)
    for i in range(4):
        player_features = features[idx:idx + 49]
        player = decode_player_state(player_features)
        if player:
            state.players.append(player)
        idx += 49

    return state


def decode_action(row: List[str]) -> GameAction:
    """
    Decode 20 output features into a GameAction object.

    Output features start at index 383 (after game_id + 382 inputs).
    """
    action = GameAction()

    # Start at output features
    idx = 383

    # Action type (string)
    action.action_type = row[idx]
    idx += 1

    # Card selection (float or NaN)
    val = safe_float(row[idx])
    action.card_selection = None if is_nan(val) else safe_int(val)
    idx += 1

    # Card reservation (float or NaN)
    val = safe_float(row[idx])
    action.card_reservation = None if is_nan(val) else safe_int(val)
    idx += 1

    # Gem take 3 (5 features)
    action.gem_take_3 = []
    for i in range(5):
        val = safe_float(row[idx + i])
        action.gem_take_3.append(0 if is_nan(val) else safe_int(val))
    idx += 5

    # Gem take 2 (5 features)
    action.gem_take_2 = []
    for i in range(5):
        val = safe_float(row[idx + i])
        action.gem_take_2.append(0 if is_nan(val) else safe_int(val))
    idx += 5

    # Noble selection (int)
    action.noble_selection = safe_int(row[idx])
    idx += 1

    # Gems removed (6 features)
    action.gems_removed = [safe_int(row[idx + i]) for i in range(6)]
    idx += 6

    return action


def render_tokens(tokens: List[int], label: str = "") -> str:
    """Render token counts with color coding."""
    parts = []
    if label:
        parts.append(label)

    color_labels = ['W', 'U', 'G', 'R', 'K', 'Y']
    for i, count in enumerate(tokens):
        if count > 0:
            color = getColor(i)
            parts.append(f"{color}{color_labels[i]}:{count}{bcolors.RESET}")

    return " ".join(parts) if parts else (label + "none" if label else "none")


def render_cost(cost: List[int]) -> str:
    """Render card cost with colors."""
    parts = []
    for i, c in enumerate(cost):
        if c > 0:
            color = getColor(i)
            parts.append(f"{color}{c}{bcolors.RESET}")
    return " ".join(parts) if parts else "free"


def render_card(card: Optional[GameCard], position: int) -> str:
    """Render a card with colors."""
    if card is None:
        return f"  [{position:2d}] (empty)"

    bonus_color = getColor(card.bonus)
    bonus_str = f"{bonus_color}{fullStrColor(card.bonus)}{bcolors.RESET}"
    cost_str = render_cost(card.cost)

    return f"  [{position:2d}] L{card.level} {card.vp}VP {bonus_str} Cost:[{cost_str}]"


def render_noble(noble: Optional[GameNoble], position: int) -> str:
    """Render a noble with requirements."""
    if noble is None:
        return f"  [{position}] (none)"

    req_str = render_cost(noble.requirements)
    return f"  [{position}] {noble.vp}VP Req:[{req_str}]"


def render_board_state(state: GameState) -> str:
    """Render complete board state."""
    lines = []

    lines.append(f"{bcolors.BOLD}╔═══ BOARD STATE ═══╗{bcolors.RESET}")
    lines.append(f"Tokens: {render_tokens(state.board_gems)}")
    lines.append("")

    # Visible cards by level
    lines.append(f"{bcolors.BOLD}Visible Cards:{bcolors.RESET}")
    for level in range(3):
        start_idx = level * 4
        level_cards = state.visible_cards[start_idx:start_idx + 4]
        lines.append(f"  Level {level + 1} (Deck: {state.deck_remaining[level]} remaining):")
        for i, card in enumerate(level_cards):
            lines.append(render_card(card, start_idx + i))

    lines.append("")
    lines.append(f"{bcolors.BOLD}Nobles:{bcolors.RESET}")
    for i, noble in enumerate(state.nobles):
        if noble:
            lines.append(render_noble(noble, i))

    return "\n".join(lines)


def render_player_state(player: PlayerState, is_current: bool = False) -> str:
    """Render a player's state."""
    lines = []

    marker = " ← CURRENT" if is_current else ""
    lines.append(f"{bcolors.BOLD}Player {player.position} ({player.vp} VP){marker}{bcolors.RESET}")
    lines.append(f"  Tokens: {render_tokens(player.gems)}")
    lines.append(f"  Bonuses: {render_tokens(player.reductions)}")

    # Reserved cards
    reserved_str = []
    for i, card in enumerate(player.reserved_cards):
        if card:
            bonus_color = getColor(card.bonus)
            reserved_str.append(f"L{card.level}·{card.vp}VP·{bonus_color}{fullStrColor(card.bonus)[0].upper()}{bcolors.RESET}")

    if reserved_str:
        lines.append(f"  Reserved: {' | '.join(reserved_str)}")
    else:
        lines.append(f"  Reserved: none")

    return "\n".join(lines)


def render_action(action: GameAction, state: GameState) -> str:
    """Render action details."""
    lines = []

    lines.append(f"{bcolors.BOLD}╔═══ ACTION ═══╗{bcolors.RESET}")
    lines.append(f"Type: {bcolors.BOLD}{action.action_type}{bcolors.RESET}")

    if action.action_type == "build":
        if action.card_selection is not None:
            if action.card_selection < 12:
                card = state.visible_cards[action.card_selection]
                lines.append(f"Card: {render_card(card, action.card_selection)}")
            else:
                # Built from reserved
                lines.append(f"Card: From reserved slot {action.card_selection - 12}")

    elif action.action_type == "reserve":
        if action.card_reservation is not None:
            if action.card_reservation < 12:
                card = state.visible_cards[action.card_reservation]
                lines.append(f"Card: {render_card(card, action.card_reservation)}")
            else:
                # Reserved from deck top
                deck_level = action.card_reservation - 12 + 1
                lines.append(f"Card: From top of Level {deck_level} deck")

    elif action.action_type == "take 3 tokens":
        tokens_taken = []
        for i, count in enumerate(action.gem_take_3):
            if count > 0:
                color = getColor(i)
                tokens_taken.append(f"{color}{fullStrColor(i)}{bcolors.RESET}")
        lines.append(f"Tokens taken: {', '.join(tokens_taken)}")

    elif action.action_type == "take 2 tokens":
        for i, count in enumerate(action.gem_take_2):
            if count == 2:
                color = getColor(i)
                lines.append(f"Tokens taken: 2× {color}{fullStrColor(i)}{bcolors.RESET}")
                break

    # Tokens removed
    if sum(action.gems_removed) > 0:
        lines.append(f"Tokens returned: {render_tokens(action.gems_removed)}")

    # Noble acquired
    if action.noble_selection >= 0:
        noble = state.nobles[action.noble_selection]
        lines.append(f"{bcolors.BOLD}✓ Noble acquired: {render_noble(noble, action.noble_selection)}{bcolors.RESET}")

    return "\n".join(lines)


def render_turn(turn_num: int, state: GameState, action: GameAction) -> str:
    """Render complete turn display."""
    lines = []

    # Turn header
    sep = "═" * 80
    lines.append(f"\n{bcolors.BOLD}{sep}{bcolors.RESET}")
    lines.append(f"{bcolors.BOLD}  TURN {state.turn_number} - Player {state.current_player}'s Turn{bcolors.RESET}")
    lines.append(f"{bcolors.BOLD}{sep}{bcolors.RESET}\n")

    # Board state
    lines.append(render_board_state(state))
    lines.append("")

    # Player states (rotated so current player is first)
    lines.append(f"{bcolors.BOLD}╔═══ PLAYERS ═══╗{bcolors.RESET}")
    # Find current player in list and rotate
    current_idx = -1
    for i, player in enumerate(state.players):
        if player.position == state.current_player:
            current_idx = i
            break

    if current_idx >= 0:
        rotated = state.players[current_idx:] + state.players[:current_idx]
        for i, player in enumerate(rotated):
            lines.append(render_player_state(player, i == 0))
            if i < len(rotated) - 1:
                lines.append("")
    else:
        for player in state.players:
            lines.append(render_player_state(player, player.position == state.current_player))
            lines.append("")

    lines.append("")

    # Action
    lines.append(render_action(action, state))

    return "\n".join(lines)


def load_and_replay_game(csv_path: Path) -> bool:
    """
    Load a CSV file and replay the complete game.

    Args:
        csv_path: Path to CSV file

    Returns:
        True if successful, False otherwise
    """
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return False

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

            # Validate column count
            if len(headers) != 403:  # game_id + 382 + 20
                print(f"Error: Expected 403 columns, got {len(headers)}")
                return False

            # Display game info
            rows = list(reader)
            if not rows:
                print("Error: No data rows in CSV")
                return False

            game_id = rows[0][0]
            num_players = safe_int(rows[0][1])

            print(f"\n{bcolors.BOLD}{'═' * 80}{bcolors.RESET}")
            print(f"{bcolors.BOLD}  REPLAYING GAME {game_id} ({num_players} Players){bcolors.RESET}")
            print(f"{bcolors.BOLD}{'═' * 80}{bcolors.RESET}")
            print(f"\nColor Legend:")
            print(f"  {getColor(WHITE)}White{bcolors.RESET} {getColor(BLUE)}Blue{bcolors.RESET} {getColor(GREEN)}Green{bcolors.RESET} {getColor(RED)}Red{bcolors.RESET} {getColor(BLACK)}Black{bcolors.RESET} {getColor(GOLD)}Gold{bcolors.RESET}")
            print(f"\nTotal turns: {len(rows)}")
            print(f"Scroll up/down to review game progression\n")

            # Replay each turn
            for turn_idx, row in enumerate(rows):
                state = decode_game_state(row)
                action = decode_action(row)
                print(render_turn(turn_idx + 1, state, action))

            # Game summary
            final_state = decode_game_state(rows[-1])
            print(f"\n{bcolors.BOLD}{'═' * 80}{bcolors.RESET}")
            print(f"{bcolors.BOLD}  GAME OVER{bcolors.RESET}")
            print(f"{bcolors.BOLD}{'═' * 80}{bcolors.RESET}\n")

            print("Final Scores:")
            sorted_players = sorted(final_state.players, key=lambda p: p.vp, reverse=True)
            for i, player in enumerate(sorted_players):
                marker = "★ WINNER" if i == 0 else ""
                print(f"  Player {player.position}: {player.vp} VP {marker}")

            print(f"\nTotal turns: {len(rows)}")

            return True

    except Exception as e:
        print(f"Error replaying game: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(
        description="Replay Splendor games from CSV files with color-coded terminal output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/replay_game.py --file data/games/2_games/1.csv
  python scripts/replay_game.py --game-id 5 --players 3
  python scripts/replay_game.py -f data/games/4_games/10.csv
        """
    )

    parser.add_argument('--file', '-f', type=str, help='Path to CSV file')
    parser.add_argument('--game-id', type=int, help='Game ID to load')
    parser.add_argument('--players', type=int, choices=[2, 3, 4],
                       help='Number of players (required with --game-id)')

    args = parser.parse_args()

    # Determine CSV path
    csv_path = None

    if args.file:
        csv_path = Path(args.file)
    elif args.game_id is not None:
        if args.players is None:
            print("Error: --players required when using --game-id")
            return 1
        csv_path = Path(f"data/games/{args.players}_games/{args.game_id}.csv")
    else:
        parser.print_help()
        return 1

    # Replay game
    success = load_and_replay_game(csv_path)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
