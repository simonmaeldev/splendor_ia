# Splendor AI

A machine learning project to develop an AI capable of playing Splendor at a high level, using Information Set Monte Carlo Tree Search (ISMCTS) as a baseline and progressing through supervised learning and reinforcement learning.

## Table of Contents
- [Project Description](#project-description)
- [MCTS Configuration](#mcts-configuration)
- [Project Structure](#project-structure)
- [Database Schema](#database-schema)
- [Installation](#installation)
- [Usage](#usage)

## Project Description

This project has two primary goals:
1. **Develop a strong AI for Splendor** - Build an agent capable of playing the board game Splendor at a high level
2. **Showcase AI/ML expertise** - Demonstrate proficiency in modern machine learning techniques including tree search, deep learning, and reinforcement learning

### Methodology

The project follows a multi-phase approach inspired by AlphaGo/AlphaZero methodology:

#### Phase 1: MCTS Baseline & Data Collection
- Implement Information Set Monte Carlo Tree Search (ISMCTS) to handle imperfect information
- Generate high-quality game data through self-play using parallelized MCTS
- Store complete game trajectories (states, actions, outcomes) in a relational database for analysis

#### Phase 2: Behavior Cloning (Supervised Learning)
- Train a neural network to imitate MCTS decisions using supervised learning
- Validates the deep learning architecture and feature representation
- Provides a faster inference model than full MCTS tree search

#### Phase 3: Deep Reinforcement Learning
- Use self-play reinforcement learning to surpass the MCTS baseline
- Combines policy and value networks for efficient decision-making
- Iteratively improves through competition against previous versions

#### Phase 4: Explainability & Strategy Extraction
- Analyze learned representations to understand emergent strategies
- Extract human-interpretable patterns from successful gameplay
- Identify key decision points and strategic insights

### Database-Driven Analysis

All games are recorded in a SQLite database (`data/games.db`) capturing:
- Complete game state history (board positions, token counts, visible cards)
- Player states (resources held, cards owned)
- All actions taken with full context

This enables pattern analysis, strategy mining, and comprehensive evaluation of agent performance.

## MCTS Configuration

The ISMCTS implementation uses the following hyperparameters:

### Core Parameters

- **`itermax`**: Number of MCTS simulations per move (default: 1000)
  - Configured in `scripts/main.py:135`
  - Each iteration performs: determinization → selection → expansion → simulation → backpropagation

- **`exploration`**: UCB1 exploration constant (default: 0.7 ≈ √2/2)
  - Defined in `src/splendor/node.py:35` in the `UCBSelectChild()` method
  - Balances exploitation of known good moves vs. exploration of uncertain moves
  - UCB1 formula: `wins/visits + exploration * sqrt(log(avails)/visits)` (see `src/splendor/node.py:44`)

### Parallel Processing

- **ISMCTS_para**: Parallelized version using multiprocessing
  - Uses `(cpu_count() - 2)` processes (see `src/splendor/ISMCTS.py:69`)
  - Divides iterations across processes, then merges resulting trees
  - Significantly faster for large iteration counts

### Imperfect Information Handling

- **Determinization**: One per iteration
  - Each MCTS iteration samples a possible world state consistent with current information
  - Handles hidden information (opponent cards, deck composition)
  - See `board.cloneAndRandomize()` called in `src/splendor/ISMCTS.py:26`

### Selection Policy

- **UCB1**: Upper Confidence Bound formula balances exploration/exploitation
  - Implementation: `src/splendor/node.py:44`
  - Considers: win rate, exploration bonus, availability counts

## Project Structure

```
.
├── src/
│   └── splendor/               # Main game engine package
│       ├── __init__.py         # Package initialization with exports
│       ├── board.py            # Board state representation and game logic
│       ├── cards.py            # Card class definition
│       ├── characters.py       # Noble character class definition
│       ├── constants.py        # Game constants (all 40 cards, 10 nobles)
│       ├── custom_operators.py # Custom list manipulation operators
│       ├── ISMCTS.py           # ISMCTS algorithm implementation (sequential and parallel)
│       ├── move.py             # Move and action definitions
│       ├── node.py             # MCTS tree node implementation
│       └── player.py           # Player state representation
├── scripts/
│   ├── __init__.py             # Scripts package initialization
│   ├── main.py                 # Main execution script (runs MCTS self-play games)
│   ├── data_collector.py       # Safe data collection with progress tracking
│   ├── create_database.py      # Database schema creation (DESTRUCTIVE - drops all tables)
│   └── load_database.py        # Populate database with cards and nobles from constants
├── data/
│   ├── .gitkeep                # Ensures data directory is tracked
│   ├── games.db                # SQLite database storing game history
│   ├── simulation_config.txt   # Configuration file for target game counts
│   └── simulation_log.txt      # Log file tracking completed games (auto-generated)
├── specs/                      # AI agent task specifications directory
└── README.md                   # Project documentation (this file)
```

## Database Schema

The project uses SQLite for persistent storage of game data. All tables are defined in `scripts/create_database.py`.

### Tables Overview

#### Player
Stores player identities for game tracking.
- `IDPlayer` (PRIMARY KEY): Unique player identifier
- `Name` (UNIQUE): Player name (format: "{algorithm}{iteration_count}", e.g., "ISMCTS_PARA1000")

#### Card
Defines all 40 cards in the game across 3 levels.
- `IDCard` (PRIMARY KEY): Card identifier
- `Bonus`: Permanent resource provided (White/Blue/Green/Red/Black)
- `Cost[White/Blue/Green/Red/Black]`: Resource costs to purchase
- `VictoryPoints`: Points awarded
- `Level`: Card tier (1, 2, or 3)

#### Character
Defines the 10 noble characters.
- `IDCharacter` (PRIMARY KEY): Noble identifier
- `VictoryPoints`: Points awarded (always 3)
- `Cost[White/Blue/Green/Red/Black]`: Card bonus requirements to attract noble

#### Game
High-level game metadata.
- `IDGame` (PRIMARY KEY): Game identifier
- `NbPlayers`: Number of players (2-4)
- `P1, P2, P3, P4`: Player IDs (foreign keys to Player table)
- `VictoryPoints`: Winning player's final score
- `NbTurns`: Total turns taken
- `Winner`: Winning player position (0-3)

#### StateGame
Complete board state for each turn.
- `IDStateGame` (PRIMARY KEY): State identifier
- `IDGame`: Associated game (foreign key)
- `TurnNumber`: Turn index
- `CurrentPlayer`: Active player position
- `Tokens[White/Blue/Green/Red/Black/Gold]`: Tokens remaining on board
- `Card[1-3]_[1-4]`: Visible cards at each level (12 slots total)
- `Character[1-5]`: Available nobles (up to 5)

#### StatePlayer
Individual player state for each turn.
- `IDStatePlayer` (PRIMARY KEY): State identifier
- `IDGame`: Associated game (foreign key)
- `TurnNumber`: Turn index
- `PlayerNumber`: Player position (0-3)
- `Tokens[White/Blue/Green/Red/Black/Gold]`: Tokens held by player

#### Action
Player actions taken each turn.
- `IDAction` (PRIMARY KEY): Action identifier
- `IDGame`: Associated game (foreign key)
- `TurnNumber`: Turn index
- `PlayerNumber`: Player position
- `Type`: Action type (BUILD/RESERVE/TAKE_TOKENS)
- `IDCard`: Card involved (for BUILD/RESERVE actions)
- `Take[White/Blue/Green/Red/Black/Gold]`: Tokens taken from board
- `Give[White/Blue/Green/Red/Black/Gold]`: Tokens returned to board
- `IDCharacter`: Noble attracted (if any)

### Entity Relationships

- Games reference 2-4 Players
- StateGame records link to Game (one per turn)
- StatePlayer records link to Game (one per player per turn)
- Action records link to Game (one per player per turn)
- Cards and Characters are referenced by both game states and actions

Full schema details available in `scripts/create_database.py:13-159`.

## Installation

### Prerequisites
- Python 3.x
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Steps

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository**:
```bash
git clone <repository-url>
cd splendor_ia
```

3. **Create virtual environment with uv**:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. **Install dependencies**:
The project uses only Python standard library modules (no external dependencies required):
- `sqlite3` - Database operations
- `multiprocessing` - Parallel MCTS
- `typing` - Type hints
- `random` - Randomization
- `timeit` - Performance measurement
- `math` - Mathematical operations

5. **Initialize the database**:
```bash
python scripts/create_database.py
python scripts/load_database.py
```

**Note**: Run scripts from the project root directory. Python automatically adds the current directory to `sys.path`, allowing the scripts to import from the `splendor` package.

## Usage

### ⚠️ IMPORTANT: Database Impact Warnings

**Running these scripts WILL modify `games.db`:**

#### Database Initialization

```bash
# DESTRUCTIVE: Drops ALL existing tables and recreates schema
python scripts/create_database.py

# Populates cards and nobles from constants.py
python scripts/load_database.py
```

**WARNING**: `scripts/create_database.py` will **DELETE ALL EXISTING DATA** in the database. Only run this for a fresh setup or when you want to completely reset the database.

#### Running Game Simulations

The project includes a safe, configuration-driven data collection system that tracks progress, displays real-time statistics, and handles interruptions gracefully.

##### Quick Start

1. **Configure target game counts** in `data/simulation_config.txt`:
```
# Format: nb_players: nb_games
2: 500
3: 1000
4: 500
```

2. **Run data collection**:
```bash
python scripts/main.py
```

3. **Monitor progress** - The system displays:
   - Progress bars for each player count
   - Average game duration
   - Estimated time remaining
   - Estimated completion time
   - Games completed/total

4. **Interrupt safely** - Press Ctrl+C at any time:
   - Database remains consistent (only complete games are saved)
   - Progress is automatically saved to `data/simulation_log.txt`
   - Run again to continue from where you left off

##### Configuration File Format

Edit `data/simulation_config.txt` to specify target game counts:

```
# Splendor Game Simulation Configuration
# Format: nb_players: nb_games
# Player count must be 2-4, game count must be positive

2: 500    # Target: 500 games with 2 players
3: 1000   # Target: 1000 games with 3 players
4: 500    # Target: 500 games with 4 players
```

**Features**:
- Comments supported (lines starting with `#`)
- System automatically checks current database counts
- Only runs remaining games needed to meet targets
- Processes player counts sequentially (2 → 3 → 4)

##### Progress Tracking

The system maintains detailed logs in `data/simulation_log.txt`:

```
2 players, 15 turns, started: 2025-10-31 14:30:45, ended: 2025-10-31 14:31:30, duration: 45.3s, 1/500
2 players, 18 turns, started: 2025-10-31 14:31:30, ended: 2025-10-31 14:32:20, duration: 50.1s, 2/500
...
```

**Log format**:
- `nb_players`: Number of players in game
- `nb_turns`: Total turns taken
- `started/ended`: Timestamps (YYYY-MM-DD HH:MM:SS)
- `duration`: Game duration in seconds
- `sim_num/total_sims`: Progress (current/target)

##### Example Output

```
=== Splendor Data Collection ===
Config: 2 players: 500 games | 3 players: 1000 games | 4 players: 500 games

[2 players] 245/500 games (49%)
[===========================>                         ] 49%
Started on: 31 Oct, 14:30
Average duration: 45.3s
Estimated remaining: 0d, 03:12
Estimated completion: 31 Oct, 17:42

Running game 246/500...
```

##### Safety Features

**Atomic Database Commits**:
- Each game is committed as a single transaction
- Interruptions (Ctrl+C, crashes) never leave partial games in database
- Database integrity is always maintained

**Auto-Resume**:
- System queries database on startup to determine current progress
- Automatically continues from where it left off
- Works across multiple sessions and interruptions

**Error Handling**:
- Failed games are logged but skipped
- System continues to next game without stopping
- No data corruption from game execution errors

##### Algorithm Configuration

Default settings (configured in `scripts/data_collector.py`):
- **Algorithm**: `ISMCTS_PARA` (parallel MCTS)
- **Iterations**: 1000 per player
- **Player count**: Automatically uses 2, 3, or 4 players based on config

Available algorithms:
- `"ISMCTS"`: Sequential MCTS (slower, single-threaded)
- `"ISMCTS_PARA"`: Parallel MCTS (faster, uses multiprocessing)

To modify defaults, edit `scripts/data_collector.py:490-491`.

##### Validation Commands

Check system health:

```bash
# Validate configuration parsing
python -c "import sys; sys.path.insert(0, 'scripts'); import data_collector as dc; config = dc.parse_config('data/simulation_config.txt'); print('Config valid' if config else 'Config invalid')"

# Check log file parsing
python -c "import sys; sys.path.insert(0, 'scripts'); import data_collector as dc; entries = dc.parse_log_file('data/simulation_log.txt'); print(f'Parsed {len(entries)} log entries')"

# Query database counts
python -c "import sys; sys.path.insert(0, 'scripts'); import data_collector as dc; count = dc.get_game_count_by_players('data/games.db', 3); print(f'Found {count} 3-player games')"

# Run comprehensive validation
python -c "import sys; sys.path.insert(0, 'scripts'); import data_collector as dc; dc.validate_all_systems('data/simulation_config.txt', 'data/games.db', 'data/simulation_log.txt')"

# Check database integrity
python -c "import sqlite3; conn = sqlite3.connect('data/games.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM Game'); print(f'Total games in database: {cursor.fetchone()[0]}'); conn.close()"
```

#### Database Preservation

To preserve existing game data:
```bash
# Backup before running create_database.py
cp data/games.db data/games.db.backup

# Restore if needed
cp data/games.db.backup data/games.db
```

### Implementation Details

The MCTS implementation is based on work by Peter Cowling, Edward Powley, and Daniel Whitehouse (University of York, 2012-2013). See `src/splendor/ISMCTS.py:1-5` and `src/splendor/node.py:1-5` for attribution.
