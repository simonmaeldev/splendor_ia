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
│   ├── create_database.py      # Database schema creation (DESTRUCTIVE - drops all tables)
│   └── load_database.py        # Populate database with cards and nobles from constants
├── data/
│   ├── .gitkeep                # Ensures data directory is tracked
│   └── games.db                # SQLite database storing game history
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

```bash
# ADDITIVE: Runs games and appends results to database
python scripts/main.py
```

**Note**:
- `scripts/main.py` runs an **infinite loop** (see `scripts/main.py:133`) and must be manually interrupted (Ctrl+C)
- Each game appends new records to the database (safe for existing data)
- Default configuration: 3 players, 1000 MCTS iterations per move, ISMCTS_PARA algorithm (see `scripts/main.py:135`)

#### Database Preservation

To preserve existing game data:
```bash
# Backup before running create_database.py
cp data/games.db data/games.db.backup

# Restore if needed
cp data/games.db.backup data/games.db
```

#### Configuration

Modify game parameters in `scripts/main.py:135`:
```python
PlayGame(
    [1000, 1000, 1000],  # Iterations per player
    ["ISMCTS_PARA", "ISMCTS_PARA", "ISMCTS_PARA"]  # Algorithm per player
)
```

Available algorithms:
- `"ISMCTS"`: Sequential MCTS (slower, single-threaded)
- `"ISMCTS_PARA"`: Parallel MCTS (faster, uses multiprocessing)

### Implementation Details

The MCTS implementation is based on work by Peter Cowling, Edward Powley, and Daniel Whitehouse (University of York, 2012-2013). See `src/splendor/ISMCTS.py:1-5` and `src/splendor/node.py:1-5` for attribution.
