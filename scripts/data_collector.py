"""
Safe Data Collection with Progress Tracking for Splendor Game Simulations

This module provides a robust data collection system that:
- Reads target game counts from configuration files
- Tracks progress through detailed logging
- Displays real-time progress with time estimates
- Ensures database safety through atomic commits
- Handles interruptions gracefully
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

# Import existing game functions
from main import PlayGame, saveIntoBdd


# ===== Configuration Management =====

def parse_config(filepath: str) -> Dict[int, int]:
    """
    Parse configuration file to get target game counts per player count.

    Format: "nb_players: nb_games" per line
    Example:
        2: 500
        3: 1000
        4: 500

    Args:
        filepath: Path to configuration file

    Returns:
        Dictionary mapping player count to target game count

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If format is invalid or values out of range
    """
    config: Dict[int, int] = {}

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: {filepath}\n"
            f"Please create {filepath} with format:\n"
            f"2: 500\n"
            f"3: 1000\n"
            f"4: 500"
        )

    if not lines:
        raise ValueError("Configuration file is empty")

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):  # Skip empty lines and comments
            continue

        # Parse format "nb_players: nb_games"
        match = re.match(r'^(\d+)\s*:\s*(\d+)$', line)
        if not match:
            raise ValueError(
                f"Invalid format at line {line_num}: '{line}'\n"
                f"Expected format: 'nb_players: nb_games' (e.g., '3: 1000')"
            )

        nb_players = int(match.group(1))
        nb_games = int(match.group(2))

        # Validate ranges
        if nb_players < 2 or nb_players > 4:
            raise ValueError(
                f"Invalid player count at line {line_num}: {nb_players}\n"
                f"Player count must be between 2 and 4"
            )

        if nb_games <= 0:
            raise ValueError(
                f"Invalid game count at line {line_num}: {nb_games}\n"
                f"Game count must be positive"
            )

        config[nb_players] = nb_games

    return config


# ===== Log Management =====

@dataclass
class LogEntry:
    """Represents a single game log entry"""
    nb_players: int
    nb_turns: int
    time_start: datetime
    time_end: datetime
    duration: float  # seconds
    sim_num: int
    total_sims: int


def parse_log_file(filepath: str) -> List[LogEntry]:
    """
    Parse log file to extract all completed game entries.

    Format: "nb_players players, nb_turns turns, started: time_start, ended: time_end, duration: duration_seconds s, sim_num/total_sims"

    Args:
        filepath: Path to log file

    Returns:
        List of parsed log entries
    """
    entries: List[LogEntry] = []

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        # Log file doesn't exist yet - this is fine for first run
        return entries

    malformed_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            # Parse format: "2 players, 15 turns, started: 2025-10-31 14:30:45, ended: 2025-10-31 14:31:30, duration: 45.3s, 1/500"
            pattern = r'^(\d+) players, (\d+) turns, started: ([^,]+), ended: ([^,]+), duration: ([\d.]+)s, (\d+)/(\d+)$'
            match = re.match(pattern, line)

            if not match:
                malformed_count += 1
                continue

            nb_players = int(match.group(1))
            nb_turns = int(match.group(2))
            time_start = datetime.strptime(match.group(3), '%Y-%m-%d %H:%M:%S')
            time_end = datetime.strptime(match.group(4), '%Y-%m-%d %H:%M:%S')
            duration = float(match.group(5))
            sim_num = int(match.group(6))
            total_sims = int(match.group(7))

            entries.append(LogEntry(
                nb_players=nb_players,
                nb_turns=nb_turns,
                time_start=time_start,
                time_end=time_end,
                duration=duration,
                sim_num=sim_num,
                total_sims=total_sims
            ))
        except Exception:
            malformed_count += 1
            continue

    if malformed_count > 0:
        print(f"Warning: Skipped {malformed_count} malformed log entries")

    return entries


def write_log_entry(filepath: str, entry: LogEntry) -> None:
    """
    Append a log entry to the log file.

    Args:
        filepath: Path to log file
        entry: Log entry to write
    """
    try:
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'a') as f:
            line = (
                f"{entry.nb_players} players, "
                f"{entry.nb_turns} turns, "
                f"started: {entry.time_start.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"ended: {entry.time_end.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"duration: {entry.duration:.1f}s, "
                f"{entry.sim_num}/{entry.total_sims}\n"
            )
            f.write(line)
    except Exception as e:
        print(f"Warning: Failed to write log entry: {e}")


# ===== Database Queries =====

def get_game_count_by_players(db_path: str, nb_players: int) -> int:
    """
    Query database to count games for a specific player count.

    Args:
        db_path: Path to SQLite database
        nb_players: Number of players to filter by

    Returns:
        Count of games with specified player count
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM Game WHERE NbPlayers = ?', (nb_players,))
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"Error querying database: {e}")
        return 0


# ===== Statistics Computation =====

def compute_avg_duration(log_entries: List[LogEntry], nb_players: int, limit: Optional[int] = 50) -> Optional[float]:
    """
    Compute average game duration for a specific player count.

    Args:
        log_entries: List of all log entries
        nb_players: Player count to filter by
        limit: Maximum number of entries to use for calculation (default 50).
               Set to None to use all entries.

    Returns:
        Average duration in seconds, or None if no entries found
    """
    filtered = [e for e in log_entries if e.nb_players == nb_players]

    # Apply limit if specified
    if limit is not None and len(filtered) > limit:
        filtered = filtered[:limit]

    if not filtered:
        return None

    total_duration = sum(e.duration for e in filtered)
    return total_duration / len(filtered)


# ===== Progress Display =====

def format_progress_bar(completed: int, total: int, width: int = 50) -> str:
    """
    Create ASCII progress bar.

    Args:
        completed: Number of completed items
        total: Total number of items
        width: Width of progress bar in characters

    Returns:
        Formatted progress bar string
    """
    if total == 0:
        percentage = 100
    else:
        percentage = int((completed / total) * 100)

    filled = int((completed / total) * width) if total > 0 else width
    bar = '=' * filled + '>' if filled < width else '=' * width
    bar = bar.ljust(width)

    return f"[{bar}] {percentage}%"


def format_time_estimate(seconds: float) -> str:
    """
    Format time estimate as "Xd, HH:MM".

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted time string
    """
    delta = timedelta(seconds=int(seconds))
    days = delta.days
    hours = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60

    return f"{days}d, {hours:02d}:{minutes:02d}"


def format_datetime_display(dt: datetime) -> str:
    """
    Format datetime as "Day Month, HH:MM".

    Args:
        dt: Datetime to format

    Returns:
        Formatted datetime string
    """
    return dt.strftime("%d %b, %H:%M")


def format_progress_display(
    nb_players: int,
    completed: int,
    total: int,
    session_start: datetime,
    avg_duration: Optional[float],
    log_entries: List[LogEntry]
) -> str:
    """
    Format complete progress display with all information.

    Args:
        nb_players: Current player count
        completed: Games completed for this player count
        total: Target games for this player count
        session_start: When collection started
        avg_duration: Average game duration in seconds (or None)
        log_entries: All log entries for statistics

    Returns:
        Formatted multi-line progress display
    """
    lines = []

    # Header
    lines.append(f"\n[{nb_players} players] {completed}/{total} games ({int(completed/total*100) if total > 0 else 100}%)")

    # Progress bar
    lines.append(format_progress_bar(completed, total))

    # Start time
    lines.append(f"Started on: {format_datetime_display(session_start)}")

    # Average duration
    if avg_duration is not None:
        lines.append(f"Average duration: {avg_duration:.1f}s")
    else:
        lines.append(f"Average duration: NaN")

    # Time estimates
    if avg_duration is not None and completed < total:
        remaining_games = total - completed
        estimated_seconds = remaining_games * avg_duration
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)

        lines.append(f"Estimated remaining: {format_time_estimate(estimated_seconds)}")
        lines.append(f"Estimated completion: {format_datetime_display(estimated_completion)}")
    elif completed >= total:
        lines.append(f"Status: Complete!")

    return '\n'.join(lines)


# ===== Safe Game Execution =====

def run_single_game_safe(
    nb_players: int,
    iterations: List[int],
    algorithm: str,
    db_path: str,
    log_path: str,
    sim_num: int,
    total_sims: int,
    session_start: datetime,
    log_entries: List[LogEntry]
) -> Tuple[bool, Optional[int]]:
    """
    Run a single game with error handling and logging.

    Args:
        nb_players: Number of players
        iterations: MCTS iterations per player
        algorithm: Algorithm to use (e.g., "ISMCTS_PARA")
        db_path: Path to database
        log_path: Path to log file
        sim_num: Current simulation number
        total_sims: Total simulations for this player count
        session_start: Session start time
        log_entries: Current log entries for statistics

    Returns:
        Tuple of (success: bool, nb_turns: Optional[int])
    """
    time_start = datetime.now()

    try:
        # Import Board to count turns
        from splendor.board import Board

        # Play the game
        players = [algorithm] * nb_players
        state = Board(nb_players, players, debug=False)

        # Execute game (uses existing PlayGame logic but we need to capture state)
        from splendor.ISMCTS import ISMCTS, ISMCTS_para
        from splendor.move import Move
        from typing import Any

        historyPlayers: List[List[List[int]]] = []
        historyState: List[Any] = []
        historyActionPlayers: List[List[Move]] = [[]] * nb_players

        for p in range(nb_players):
            historyPlayers += [[state.getPlayerState(p)]]

        while (state.getMoves() != []):
            historyState.append(state.getState())
            currentPlayer = state.currentPlayer

            if players[currentPlayer] == "ISMCTS_PARA":
                m = ISMCTS_para(rootstate=state, itermax=iterations[currentPlayer], verbose=False)
            elif players[currentPlayer] == "ISMCTS":
                m = ISMCTS(rootstate=state, itermax=iterations[currentPlayer], verbose=False)

            state.doMove(m)
            historyActionPlayers[currentPlayer].append(m)
            historyPlayers[currentPlayer].append(state.getPlayerState(currentPlayer))

        winner = state.getVictorious(False)
        nb_turns = state.nbTurn

        # Save to database (atomic operation)
        saveIntoBdd(state, winner, historyState, historyPlayers, historyActionPlayers, iterations, players)

        time_end = datetime.now()
        duration = (time_end - time_start).total_seconds()

        # Write log entry after successful commit
        entry = LogEntry(
            nb_players=nb_players,
            nb_turns=nb_turns,
            time_start=time_start,
            time_end=time_end,
            duration=duration,
            sim_num=sim_num,
            total_sims=total_sims
        )
        write_log_entry(log_path, entry)

        return True, nb_turns

    except Exception as e:
        print(f"\nError during game execution: {e}")
        print("Game not saved. Continuing...")
        return False, None


# ===== Main Orchestration =====

def run_data_collection(
    config_path: str = 'data/simulation_config.txt',
    db_path: str = 'data/games.db',
    log_path: str = 'data/simulation_log.txt'
) -> None:
    """
    Main orchestration loop for data collection.

    Reads configuration, determines remaining work, runs simulations,
    and tracks progress until all targets are met.

    Args:
        config_path: Path to configuration file
        db_path: Path to SQLite database
        log_path: Path to log file
    """
    print("\n=== Splendor Data Collection ===\n")

    # Parse configuration
    try:
        config = parse_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Display configuration
    config_display = " | ".join([f"{nb_players} players: {nb_games} games"
                                 for nb_players, nb_games in sorted(config.items())])
    print(f"Config: {config_display}\n")

    # Session start time
    session_start = datetime.now()

    # Default settings
    ALGORITHM = "ISMCTS_PARA"
    ITERATIONS_PER_PLAYER = 1000

    # Process each player count in order
    try:
        for nb_players in sorted(config.keys()):
            target_games = config[nb_players]

            # Check current count in database
            current_count = get_game_count_by_players(db_path, nb_players)

            if current_count >= target_games:
                print(f"[{nb_players} players] Target already met ({current_count}/{target_games}). Skipping.\n")
                continue

            remaining_games = target_games - current_count
            print(f"[{nb_players} players] Need {remaining_games} more games ({current_count}/{target_games} complete)\n")

            # Initialize cache for average duration and log entries
            cached_avg_duration = None
            cached_log_entries = None
            last_parse_count = -1  # Track when we last parsed

            # Run remaining games
            completed_this_session = 0
            while completed_this_session < remaining_games:
                sim_num = current_count + completed_this_session + 1
                current_game_count = current_count + completed_this_session

                # Determine if we need to recompute average duration
                should_recompute = False
                if current_game_count < 50:
                    # Recompute every 10 games or on first game
                    if current_game_count % 10 == 0 or current_game_count == 0:
                        should_recompute = True
                elif current_game_count == 50:
                    # Compute one final time at exactly 50 games
                    should_recompute = True
                # else: current_game_count > 50, use cached value

                # Parse log and compute average if needed
                if should_recompute or cached_log_entries is None:
                    log_entries = parse_log_file(log_path)
                    cached_log_entries = log_entries
                    cached_avg_duration = compute_avg_duration(log_entries, nb_players, limit=50)
                    last_parse_count = current_game_count
                else:
                    # Use cached values
                    log_entries = cached_log_entries

                avg_duration = cached_avg_duration

                # Display progress before game
                progress = format_progress_display(
                    nb_players=nb_players,
                    completed=current_game_count,
                    total=target_games,
                    session_start=session_start,
                    avg_duration=avg_duration,
                    log_entries=log_entries
                )
                print(progress)
                print(f"\nRunning game {sim_num}/{target_games}...")

                # Run game
                iterations = [ITERATIONS_PER_PLAYER] * nb_players
                success, nb_turns = run_single_game_safe(
                    nb_players=nb_players,
                    iterations=iterations,
                    algorithm=ALGORITHM,
                    db_path=db_path,
                    log_path=log_path,
                    sim_num=sim_num,
                    total_sims=target_games,
                    session_start=session_start,
                    log_entries=log_entries
                )

                if success:
                    print(f"Game {sim_num} completed ({nb_turns} turns)")
                    completed_this_session += 1
                else:
                    print(f"Game {sim_num} failed. Continuing to next game...")
                    # Continue without incrementing - will be retried in next run

            # Final progress for this player count
            # Parse one last time to get final stats
            log_entries = parse_log_file(log_path)
            avg_duration = compute_avg_duration(log_entries, nb_players, limit=50)
            progress = format_progress_display(
                nb_players=nb_players,
                completed=target_games,
                total=target_games,
                session_start=session_start,
                avg_duration=avg_duration,
                log_entries=log_entries
            )
            print(progress)
            print(f"\n[{nb_players} players] Target complete!\n")

        print("\n=== All targets complete! ===\n")

    except KeyboardInterrupt:
        print("\n\n=== Interrupted by user ===")
        total_saved = 0
        for nb_players in config.keys():
            count = get_game_count_by_players(db_path, nb_players)
            total_saved += count
        print(f"Total games saved: {total_saved}")
        print("Database is safe. Run again to continue.")
        print()


# ===== Validation Function =====

def validate_all_systems(
    config_path: str = 'data/simulation_config.txt',
    db_path: str = 'data/games.db',
    log_path: str = 'data/simulation_log.txt'
) -> None:
    """
    Run comprehensive validation of all system components.

    Args:
        config_path: Path to configuration file
        db_path: Path to database
        log_path: Path to log file
    """
    print("=== System Validation ===\n")

    # Test 1: Config parsing
    print("1. Testing config parsing...")
    try:
        config = parse_config(config_path)
        print(f"   ✓ Config parsed: {config}")
    except Exception as e:
        print(f"   ✗ Config parsing failed: {e}")
        return

    # Test 2: Log parsing
    print("2. Testing log parsing...")
    try:
        entries = parse_log_file(log_path)
        print(f"   ✓ Parsed {len(entries)} log entries")
    except Exception as e:
        print(f"   ✗ Log parsing failed: {e}")

    # Test 3: Database queries
    print("3. Testing database queries...")
    try:
        for nb_players in [2, 3, 4]:
            count = get_game_count_by_players(db_path, nb_players)
            print(f"   ✓ {nb_players} players: {count} games")
    except Exception as e:
        print(f"   ✗ Database queries failed: {e}")

    # Test 4: Statistics
    print("4. Testing statistics computation...")
    try:
        entries = parse_log_file(log_path)
        for nb_players in [2, 3, 4]:
            avg = compute_avg_duration(entries, nb_players)
            if avg is not None:
                print(f"   ✓ {nb_players} players: avg {avg:.1f}s")
            else:
                print(f"   - {nb_players} players: no data")
    except Exception as e:
        print(f"   ✗ Statistics failed: {e}")

    print("\n=== Validation Complete ===\n")


if __name__ == "__main__":
    # Run data collection
    run_data_collection()
