"""Play Splendor games with MODEL players.

This script demonstrates how to use trained neural network models to play Splendor.
It supports various game configurations:
- MODEL vs MODEL (compare different models)
- MODEL vs MCTS (evaluate against strong baseline)
- MODEL vs Random (sanity check)
- Multi-player games (2-4 players)

Usage:
    python scripts/play_with_model.py --model1 <path> --model2 <path> --games 10
    python scripts/play_with_model.py --model <path> --opponent mcts --games 5
    python scripts/play_with_model.py --model <path> --players 4 --games 3

Examples:
    # MODEL vs MODEL (different models)
    python scripts/play_with_model.py \\
        --model1 data/models/baseline/best_model.pth \\
        --model2 data/models/advanced/best_model.pth \\
        --games 10

    # MODEL vs MCTS (1000 iterations)
    python scripts/play_with_model.py \\
        --model data/models/baseline/best_model.pth \\
        --opponent mcts \\
        --mcts-iterations 1000 \\
        --games 5

    # 4-player game with mixed players
    python scripts/play_with_model.py \\
        --model data/models/baseline/best_model.pth \\
        --players 4 \\
        --games 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from typing import List, Dict, Optional
import random
from datetime import datetime

from splendor.board import Board
from splendor.ai_player import get_model_action
from splendor.ISMCTS import ISMCTS_para
from splendor.move import Move


def play_game(
    player_types: List[str],
    model_paths: List[Optional[str]],
    mcts_iterations: List[int],
    verbose: bool = False,
    display: bool = False
) -> Dict:
    """Play a single game with specified players.

    Args:
        player_types: List of player types ("model", "mcts", or "random")
        model_paths: List of model paths (None for non-model players)
        mcts_iterations: List of MCTS iteration counts (0 for non-MCTS players)
        verbose: If True, print game progress
        display: If True, display the game board state after each move

    Returns:
        Dict with game results
    """
    num_players = len(player_types)
    board = Board(num_players, [None] * num_players)

    start_time = datetime.now()
    # Track time spent by each player
    player_times = [0.0] * num_players

    if verbose:
        print(f"\n{'='*70}")
        print(f"Starting {num_players}-player game")
        print(f"Players: {', '.join(f'P{i}={t}' for i, t in enumerate(player_types))}")
        print(f"{'='*70}\n")

    while board.getMoves() != []:
        current_player = board.currentPlayer
        player_type = player_types[current_player]


        # Display board state before move
        if display:
            board.show()
            print(f"\nP{current_player} ({player_type}) is thinking...")

        # Time the player's move
        move_start = datetime.now()

        # Get move based on player type
        if player_type == "model":
            move = get_model_action(board, model_paths[current_player])
        elif player_type == "mcts":
            move = ISMCTS_para(board, mcts_iterations[current_player], verbose=False)
        else:  # random
            legal_moves = board.getMoves()
            move = random.choice(legal_moves)

        # Record time spent
        move_end = datetime.now()
        player_times[current_player] += (move_end - move_start).total_seconds()

        if display:
            print(f"P{current_player} plays: {move}\n")

        # Execute move
        board.doMove(move)

    # Game finished
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    winner = board.getVictorious(verbose=False)
    scores = [p.getVictoryPoints() for p in board.players]

    if verbose:
        print(f"\n{'='*70}")
        print(f"Game finished!")
        print(f"Winner: Player {winner} ({player_types[winner]})")
        print(f"Scores: {' | '.join(f'P{i}={s}' for i, s in enumerate(scores))}")
        print(f"Total turns: {board.nbTurn}")
        print(f"Duration: {duration:.1f}s")
        print(f"{'='*70}\n")

    # Calculate number of turns per player
    turns_per_player = [0] * num_players
    # Each player gets roughly nbTurn / num_players turns
    # The exact distribution depends on who starts and wins
    for i in range(board.nbTurn):
        player_idx = i % num_players
        turns_per_player[player_idx] += 1

    return {
        'winner': winner,
        'winner_type': player_types[winner],
        'scores': scores,
        'turns': board.nbTurn,
        'duration': duration,
        'player_times': player_times,
        'turns_per_player': turns_per_player
    }


def run_tournament(
    player_types: List[str],
    model_paths: List[Optional[str]],
    mcts_iterations: List[int],
    num_games: int,
    verbose: bool = False,
    display: bool = False
) -> None:
    """Run a tournament with multiple games.

    Args:
        player_types: List of player types
        model_paths: List of model paths
        mcts_iterations: List of MCTS iterations
        num_games: Number of games to play
        verbose: If True, print detailed output
        display: If True, display the game board state after each move
    """
    print(f"\n{'='*70}")
    print(f"TOURNAMENT: {num_games} GAMES")
    print(f"{'='*70}")
    print(f"Base configuration:")
    for i, pt in enumerate(player_types):
        desc = f"  Player type: {pt}"
        if pt == "model":
            desc += f" (model: {model_paths[i]})"
        elif pt == "mcts":
            desc += f" (iterations: {mcts_iterations[i]})"
        print(desc)
    print(f"Note: Player positions are randomized for each game")
    print(f"{'='*70}\n")

    results = []
    failures = 0
    # Track original player indices to accumulate statistics
    original_player_indices = list(range(len(player_types)))

    for game_num in range(1, num_games + 1):
        print(f"Game {game_num}/{num_games}...", end=" ", flush=True)

        try:
            # Randomize player order for this game
            shuffled_indices = original_player_indices.copy()
            random.shuffle(shuffled_indices)

            shuffled_player_types = [player_types[i] for i in shuffled_indices]
            shuffled_model_paths = [model_paths[i] for i in shuffled_indices]
            shuffled_mcts_iterations = [mcts_iterations[i] for i in shuffled_indices]

            result = play_game(
                shuffled_player_types,
                shuffled_model_paths,
                shuffled_mcts_iterations,
                verbose=verbose and game_num == 1,  # Only verbose for first game
                display=display and game_num == 1   # Only display for first game
            )

            # Map winner back to original player index
            result['original_winner'] = shuffled_indices[result['winner']]
            result['position_mapping'] = shuffled_indices
            results.append(result)

            print(f"✓ Winner: P{result['winner']} ({result['winner_type']}) | "
                  f"Turns: {result['turns']} | "
                  f"Scores: {result['scores']}")

        except Exception as e:
            failures += 1
            print(f"❌ FAILED: {e}")

    # Statistics
    if results:
        print(f"\n{'='*70}")
        print(f"TOURNAMENT RESULTS")
        print(f"{'='*70}")

        # Win rates by original player configuration
        num_players = len(player_types)
        win_counts = [0] * num_players
        for r in results:
            win_counts[r['original_winner']] += 1

        print(f"\nWin Rates (by player type):")
        for i in range(num_players):
            win_rate = win_counts[i] / len(results) if results else 0
            desc = player_types[i]
            if player_types[i] == "model":
                desc += f" ({Path(model_paths[i]).name})"
            elif player_types[i] == "mcts":
                desc += f" ({mcts_iterations[i]} iters)"
            print(f"  {desc}: {win_rate:.1%} ({win_counts[i]}/{len(results)})")

        # Average statistics
        avg_turns = sum(r['turns'] for r in results) / len(results)
        avg_duration = sum(r['duration'] for r in results) / len(results)
        avg_winning_score = sum(max(r['scores']) for r in results) / len(results)

        print(f"\nGame Statistics:")
        print(f"  Average turns: {avg_turns:.1f}")
        print(f"  Average duration: {avg_duration:.1f}s")
        print(f"  Average winning score: {avg_winning_score:.1f}")

        # Average play time per player type (per turn)
        print(f"\nAverage Play Time per Turn (per Player Type):")
        for i in range(num_players):
            # Collect play times and turn counts for this original player across all games
            total_time = 0.0
            total_turns = 0
            for r in results:
                # Find which position this original player was in for this game
                pos = r['position_mapping'].index(i)
                total_time += r['player_times'][pos]
                total_turns += r['turns_per_player'][pos]

            avg_time_per_turn_ms = (total_time / total_turns) * 1000 if total_turns > 0 else 0
            desc = player_types[i]
            if player_types[i] == "model":
                desc += f" ({Path(model_paths[i]).name})"
            elif player_types[i] == "mcts":
                desc += f" ({mcts_iterations[i]} iters)"
            print(f"  {desc}: {avg_time_per_turn_ms:.2f}ms/turn")

        # Per-player statistics (mapped back to original players)
        print(f"\nPer-Player Average Scores (by player type):")
        for i in range(num_players):
            # Collect scores for this original player across all games
            scores = []
            for r in results:
                # Find which position this original player was in for this game
                pos = r['position_mapping'].index(i)
                scores.append(r['scores'][pos])
            avg_score = sum(scores) / len(scores)
            desc = player_types[i]
            if player_types[i] == "model":
                desc += f" ({Path(model_paths[i]).name})"
            elif player_types[i] == "mcts":
                desc += f" ({mcts_iterations[i]} iters)"
            print(f"  {desc}: {avg_score:.1f}")

        print(f"\nCompletion Rate: {len(results)}/{num_games} ({len(results)/num_games:.1%})")
        print(f"{'='*70}\n")

    else:
        print(f"\n❌ No games completed successfully\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Play Splendor games with MODEL players",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Player configuration
    parser.add_argument(
        "--model",
        help="Path to MODEL player checkpoint (default: data/models/baseline/best_model.pth)"
    )
    parser.add_argument(
        "--model1",
        help="Path to MODEL player 1 checkpoint (for MODEL vs MODEL)"
    )
    parser.add_argument(
        "--model2",
        help="Path to MODEL player 2 checkpoint (for MODEL vs MODEL)"
    )
    parser.add_argument(
        "--opponent",
        choices=["model", "mcts", "random"],
        default="random",
        help="Opponent type (default: random)"
    )
    parser.add_argument(
        "--players",
        type=int,
        choices=[2, 3, 4],
        default=2,
        help="Number of players (default: 2)"
    )
    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=1000,
        help="MCTS iterations (default: 1000)"
    )

    # Game configuration
    parser.add_argument(
        "--games",
        type=int,
        default=5,
        help="Number of games to play (default: 5)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed game output"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the game board state after each move (only for first game)"
    )

    args = parser.parse_args()

    # Determine player configuration
    if args.model1 and args.model2:
        # MODEL vs MODEL
        player_types = ["model", "model"]
        model_paths = [args.model1, args.model2]
        mcts_iterations = [0, 0]

    elif args.model:
        # MODEL vs opponent
        if args.opponent == "model":
            # Need two models
            print("Error: --opponent model requires --model2 to be specified")
            print("Or use --model1 and --model2 instead")
            return

        player_types = ["model", args.opponent]
        model_paths = [args.model, None]
        mcts_iterations = [0, args.mcts_iterations if args.opponent == "mcts" else 0]

        # Extend for multi-player games
        if args.players > 2:
            for _ in range(args.players - 2):
                player_types.append(args.opponent)
                model_paths.append(None)
                mcts_iterations.append(args.mcts_iterations if args.opponent == "mcts" else 0)

    else:
        # Default: use default model path
        default_model = "data/models/baseline/best_model.pth"
        if not Path(default_model).exists():
            print(f"Error: Default model not found at {default_model}")
            print("Please specify a model with --model or --model1/--model2")
            return

        player_types = ["model", args.opponent]
        model_paths = [default_model, None]
        mcts_iterations = [0, args.mcts_iterations if args.opponent == "mcts" else 0]

    # Validate model paths
    for i, (ptype, mpath) in enumerate(zip(player_types, model_paths)):
        if ptype == "model":
            if not mpath:
                print(f"Error: Model path not specified for player {i}")
                return
            if not Path(mpath).exists():
                print(f"Error: Model not found at {mpath}")
                return

    # Run tournament
    run_tournament(
        player_types,
        model_paths,
        mcts_iterations,
        args.games,
        verbose=args.verbose,
        display=args.display
    )


if __name__ == "__main__":
    main()
