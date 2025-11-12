# Chore: Complete README Documentation

## Chore Description
Enhance the README.md file with comprehensive documentation including:
1. A refined project description explaining the AI/ML goals and methodology
2. MCTS hyperparameters documentation
3. Updated project structure reflecting the current state
4. Database schema documentation for all tables
5. Installation instructions with uv virtual environment setup
6. Execution instructions with database impact warnings

## Relevant Files
Use these files to resolve the chore:

- **README.md** (line 1-24) - Main documentation file that needs to be updated with all sections
  - Currently has minimal project structure section
  - Needs project description, installation, usage, MCTS parameters, and database schema sections

- **ISMCTS.py** (line 13-59, 63-84) - Contains MCTS implementation with hyperparameters
  - Line 13: `ISMCTS()` function signature shows `itermax` parameter
  - Line 35: UCB exploration parameter in `UCBSelectChild()` (exploration = 0.7)
  - Line 63-84: Parallel MCTS implementation `ISMCTS_para()`
  - Line 69: CPU count calculation for parallel processing (cpu_count() - 2)

- **node.py** (line 35-44) - Contains UCB selection logic with exploration parameter
  - Line 35: `UCBSelectChild()` method with exploration constant (default 0.7)
  - Line 44: UCB1 formula implementation

- **create_database.py** (line 12-159) - Database schema definition
  - Line 13-17: Player table
  - Line 19-31: Card table
  - Line 33-44: Character table
  - Line 46-63: Game table
  - Line 65-114: StateGame table
  - Line 116-131: StatePlayer table
  - Line 133-159: Action table

- **load_database.py** (line 1-38) - Database initialization script
  - Populates cards and characters from constants
  - Shows database setup workflow

- **main.py** (line 131-139) - Main execution script
  - Line 135: Shows default MCTS configuration (1000 iterations per player, 3 players, ISMCTS_PARA)
  - Demonstrates how games are played and saved to database

### New Files
No new files need to be created.

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### 1. Add Project Description Section
- Read the current README.md to understand existing content
- Add a comprehensive "Project Description" section that:
  - Explains the dual goals: improving at Splendor and showcasing AI/ML skills
  - Describes the MCTS baseline opponent (inspired by AlphaGo/AlphaZero methodology)
  - Outlines the multi-phase ML pipeline:
    - Phase 1: MCTS game generation and data collection
    - Phase 2: Behavior cloning via supervised learning (validates DL architecture)
    - Phase 3: Deep reinforcement learning to surpass MCTS
    - Phase 4: Explainability analysis to extract learned strategies
  - Mentions the database-driven approach for pattern analysis

### 2. Document MCTS Hyperparameters
- Create an "MCTS Configuration" section documenting:
  - `itermax`: Number of MCTS iterations/simulations (default: 1000 per move in main.py:135)
  - `exploration`: UCB1 exploration constant (default: 0.7, approximately sqrt(2)/2) in node.py:35
  - Parallel processing: Uses (cpu_count() - 2) processes for ISMCTS_para in ISMCTS.py:69
  - Determinization: One per iteration for handling imperfect information
  - Selection policy: UCB1 formula from node.py:44
- Reference specific file locations for each parameter

### 3. Update Project Structure Section
- Replace the existing project structure with current state:
  - Update file descriptions to be more accurate
  - Remove "TODO" markers from files that are implemented
  - Add the specs/ directory description
  - Ensure all files in the root directory are documented
  - Use tree format for clarity

### 4. Document Database Schema
- Add a "Database Schema" section with:
  - Overview explaining the relational database structure (SQLite)
  - Table-by-table documentation:
    - **Player**: Stores player identities (IDPlayer, Name)
    - **Card**: Card definitions with costs and bonuses (40 cards across 3 levels)
    - **Character**: Noble definitions with victory points and requirements
    - **Game**: Game metadata (players, winner, victory points, turns)
    - **StateGame**: Board state per turn (tokens, visible cards, nobles)
    - **StatePlayer**: Player state per turn (tokens held)
    - **Action**: Player actions per turn (type, cards, tokens taken/given)
  - Entity-relationship overview
  - Reference create_database.py for schema details

### 5. Add Installation Instructions
- Create an "Installation" section with:
  - Prerequisites: Python 3.x, uv package manager
  - Step 1: Install uv if not already installed
  - Step 2: Clone the repository
  - Step 3: Create virtual environment with uv
  - Step 4: Install dependencies (note: no requirements file currently exists)
  - Step 5: Initialize database (run create_database.py, then load_database.py)
  - Note about Python dependencies being standard library only (sqlite3, multiprocessing, typing, random, timeit, math)

### 6. Add Usage Instructions with Database Warnings
- Create a "Usage" section with:
  - **WARNING**: Running main.py WILL modify the database (games.db)
  - Database initialization steps:
    - `python create_database.py` - DROPS all existing tables and recreates schema
    - `python load_database.py` - Populates cards and characters
  - Running simulations:
    - `python main.py` - Runs games and APPENDS to database
    - Configuration in main.py:135 (number of iterations, players, algorithm)
  - Database preservation note:
    - To preserve existing data, backup games.db before running
    - create_database.py is DESTRUCTIVE - only run for fresh setup
    - main.py is ADDITIVE - safe for existing databases with proper schema
  - Mention that main.py runs indefinitely (while True loop) and should be interrupted manually

### 7. Format and Polish README
- Ensure consistent markdown formatting
- Add table of contents if README is lengthy
- Verify all file references are accurate (file:line format where helpful)
- Check that all sections flow logically
- Add any necessary code blocks with proper syntax highlighting

## Validation Commands
Execute every command to validate the chore is complete with zero regressions.

- `cat /home/apprentyr/projects/splendor_ia/README.md` - Verify README content is complete and well-formatted
- `python /home/apprentyr/projects/splendor_ia/create_database.py` - Verify database creation script still works
- `python /home/apprentyr/projects/splendor_ia/load_database.py` - Verify database initialization script still works

## Notes
- The project uses only Python standard library dependencies (no requirements.txt needed)
- The MCTS implementation is based on Peter Cowling et al.'s work (University of York, 2012-2013)
- main.py contains a deliberate infinite loop and exception handling for continuous game generation
- The database (games.db) is already 16MB, indicating significant data has been collected
- No app/ directory or scripts/ directory exists in the current project structure - the chore description may be outdated regarding "Relevant Files" section
- The project is a flat structure with all Python files in the root directory
