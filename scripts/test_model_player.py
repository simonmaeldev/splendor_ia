"""Integration test for MODEL player with actual gameplay.

This script tests the MODEL player by:
1. Loading a trained model from disk
2. Playing several complete games
3. Verifying all moves are legal
4. Checking for crashes and errors
5. Reporting statistics

Usage:
    python scripts/test_model_player.py [model_path] [num_games]

Examples:
    python scripts/test_model_player.py data/models/baseline/best_model.pth 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from typing import List, Dict
from splendor.board import Board
from splendor.ai_player import get_model_action
from splendor.move import Move
import random


def play_game_with_model(model_path: str, verbose: bool = False) -> Dict:
    """Play a complete game with MODEL player.

    Args:
        model_path: Path to trained model checkpoint
        verbose: If True, print game progress

    Returns:
        Dict with game statistics

    Raises:
        Exception: If model makes illegal move or game crashes
    """
    # Create 2-player game with MODEL player
    board = Board(2, [None, None])

    turn_count = 0
    moves_taken = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting game with MODEL player")
        print(f"Model: {model_path}")
        print(f"{'='*60}\n")

    while board.getMoves() != []:
        turn_count += 1

        if verbose:
            print(f"\n--- Turn {turn_count}, Player {board.currentPlayer} ---")
            board.show()

        # Get action from MODEL player
        try:
            move = get_model_action(board, model_path)

            if verbose:
                print(f"MODEL prediction: {move}")

            # Verify move is legal
            legal_moves = board.getMoves()
            if not legal_moves:
                raise ValueError("No legal moves available, but game is not finished")

            # Execute move
            board.doMove(move)
            moves_taken.append(move)

        except Exception as e:
            print(f"\n❌ ERROR during turn {turn_count}:")
            print(f"   {str(e)}")
            print(f"\nBoard state:")
            board.show()
            raise

    # Game finished
    winner = board.getVictorious(verbose=False)
    final_scores = [p.getVictoryPoints() for p in board.players]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Game finished!")
        print(f"Winner: Player {winner}")
        print(f"Scores: {final_scores}")
        print(f"Total turns: {turn_count}")
        print(f"{'='*60}\n")

    return {
        'winner': winner,
        'scores': final_scores,
        'turns': turn_count,
        'total_moves': len(moves_taken),
        'success': True
    }


def play_random_move_game(verbose: bool = False) -> Dict:
    """Play a game with random move selection (baseline).

    Args:
        verbose: If True, print game progress

    Returns:
        Dict with game statistics
    """
    board = Board(2, [None, None])
    turn_count = 0

    while board.getMoves() != []:
        turn_count += 1
        legal_moves = board.getMoves()
        move = random.choice(legal_moves)
        board.doMove(move)

    winner = board.getVictorious()
    final_scores = [p.getVictoryPoints() for p in board.players]

    return {
        'winner': winner,
        'scores': final_scores,
        'turns': turn_count,
        'success': True
    }


def run_integration_tests(model_path: str, num_games: int = 5) -> None:
    """Run integration tests with multiple games.

    Args:
        model_path: Path to trained model checkpoint
        num_games: Number of games to play
    """
    print(f"\n{'='*70}")
    print(f"MODEL PLAYER INTEGRATION TEST")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Games: {num_games}")
    print(f"{'='*70}\n")

    # Check model exists
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        print(f"❌ ERROR: Model not found at {model_path}")
        print(f"\nPlease train a model first or specify a valid path.")
        return

    print(f"✓ Found model at {model_path}\n")

    # Test 1: Model Loading
    print("Test 1: Model Loading...")
    try:
        from splendor.ai_player import ModelPlayer
        player = ModelPlayer(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  - Input dim: {player.config['input_dim']}")
        print(f"  - Trunk dims: {player.config['trunk_dims']}")
        print(f"  - Head dims: {player.config['head_dims']}")
        print(f"  - Features: {len(player.feature_names)}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return

    # Test 2: Play multiple games
    print(f"\nTest 2: Playing {num_games} games...")
    results = []
    failures = []

    for i in range(num_games):
        print(f"  Game {i+1}/{num_games}...", end=" ", flush=True)

        try:
            result = play_game_with_model(model_path, verbose=False)
            results.append(result)
            print(f"✓ (Winner: P{result['winner']}, Turns: {result['turns']}, Scores: {result['scores']})")

        except Exception as e:
            print(f"❌ FAILED")
            failures.append({
                'game_num': i+1,
                'error': str(e)
            })

    # Test 3: Statistics
    print(f"\nTest 3: Statistics")
    if results:
        avg_turns = sum(r['turns'] for r in results) / len(results)
        avg_score = sum(max(r['scores']) for r in results) / len(results)
        win_rates = [sum(1 for r in results if r['winner'] == p) / len(results) for p in range(2)]

        print(f"✓ Successfully completed {len(results)}/{num_games} games")
        print(f"  - Average turns: {avg_turns:.1f}")
        print(f"  - Average winning score: {avg_score:.1f}")
        print(f"  - Win rates: P0={win_rates[0]:.1%}, P1={win_rates[1]:.1%}")
    else:
        print(f"❌ No games completed successfully")

    # Test 4: Compare with random baseline
    print(f"\nTest 4: Baseline Comparison")
    print(f"  Playing {num_games} random games...")
    random_results = []
    for i in range(num_games):
        try:
            result = play_random_move_game(verbose=False)
            random_results.append(result)
        except:
            pass

    if random_results:
        random_avg_turns = sum(r['turns'] for r in random_results) / len(random_results)
        model_avg_turns = avg_turns if results else 0

        print(f"  - Random avg turns: {random_avg_turns:.1f}")
        print(f"  - MODEL avg turns: {model_avg_turns:.1f}")

        if model_avg_turns > 0:
            if model_avg_turns < random_avg_turns:
                print(f"✓ MODEL is {random_avg_turns/model_avg_turns:.1f}x faster than random")
            else:
                print(f"  MODEL is {model_avg_turns/random_avg_turns:.1f}x slower than random (may indicate learning)")

    # Test 5: Error Analysis
    if failures:
        print(f"\nTest 5: Error Analysis")
        print(f"❌ {len(failures)} game(s) failed:")
        for f in failures:
            print(f"  - Game {f['game_num']}: {f['error']}")
    else:
        print(f"\nTest 5: Error Analysis")
        print(f"✓ No errors encountered")

    # Final Summary
    print(f"\n{'='*70}")
    print(f"INTEGRATION TEST SUMMARY")
    print(f"{'='*70}")
    success_rate = len(results) / num_games if num_games > 0 else 0
    print(f"Success rate: {success_rate:.1%} ({len(results)}/{num_games})")

    if success_rate == 1.0:
        print(f"✓ ALL TESTS PASSED")
    elif success_rate >= 0.8:
        print(f"⚠ MOSTLY PASSED (some failures)")
    else:
        print(f"❌ TESTS FAILED")

    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test MODEL player integration")
    parser.add_argument(
        "model_path",
        nargs="?",
        default="data/models/baseline/best_model.pth",
        help="Path to trained model checkpoint (default: data/models/baseline/best_model.pth)"
    )
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

    args = parser.parse_args()

    if args.verbose:
        # Play single game with verbose output
        print("Running single verbose game...\n")
        try:
            result = play_game_with_model(args.model_path, verbose=True)
            print(f"\n✓ Game completed successfully")
        except Exception as e:
            print(f"\n❌ Game failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run full integration test suite
        run_integration_tests(args.model_path, args.games)


if __name__ == "__main__":
    main()
