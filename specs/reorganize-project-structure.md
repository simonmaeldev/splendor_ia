# Chore: Reorganize Project Structure

## Chore Description
The project files are currently all in the root directory, which makes the codebase difficult to navigate and doesn't follow conventional Python project structure. This chore will reorganize files into a conventional structure with separate directories for source code, scripts, tests, and documentation. All import statements will be updated to reflect the new structure.

The proposed structure follows Python best practices with:
- `src/splendor/` for the main game engine and logic
- `scripts/` for executable scripts (database initialization, game execution)
- `data/` for database files
- `specs/` for specifications (already exists)
- Configuration files at root level

## Relevant Files
Use these files to resolve the chore:

### Core Game Logic Files (move to `src/splendor/`)
- `board.py` - Board state representation and game logic - Core game engine component
- `cards.py` - Card class definition - Game entity
- `characters.py` - Noble character class definition - Game entity
- `constants.py` - Game constants (cards, nobles) - Configuration data
- `custom_operators.py` - Custom list manipulation operators - Utility functions
- `move.py` - Move and action definitions - Game logic
- `node.py` - MCTS tree node implementation - Algorithm component
- `player.py` - Player state representation - Game entity
- `ISMCTS.py` - ISMCTS algorithm implementation - Main algorithm

### Script Files (move to `scripts/`)
- `main.py` - Main execution script for running games - Should be in scripts/
- `create_database.py` - Database schema creation script - Database utility
- `load_database.py` - Database population script - Database utility

### Data Files (move to `data/`)
- `games.db` - SQLite database storing game history
- `games_save_1000_iter_3_players_1000_games.db` - Backup database

### Documentation Files (keep at root)
- `README.md` - Project documentation - Needs structure and commands updated

### New Files
#### `src/splendor/__init__.py`
- Package initialization file to make src/splendor a Python package
- Will export main classes for easier imports

#### `scripts/__init__.py`
- Empty file to make scripts a Python package (optional but good practice)

#### `data/.gitkeep`
- Placeholder to ensure data directory is tracked by git

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

### Step 1: Create New Directory Structure
- Create `src/splendor/` directory for main game code
- Create `scripts/` directory for executable scripts
- Create `data/` directory for database files
- Create empty `__init__.py` files in `src/splendor/` and `scripts/`
- Create `.gitkeep` in `data/` directory

### Step 2: Move Core Game Logic Files
- Move `board.py` to `src/splendor/board.py`
- Move `cards.py` to `src/splendor/cards.py`
- Move `characters.py` to `src/splendor/characters.py`
- Move `constants.py` to `src/splendor/constants.py`
- Move `custom_operators.py` to `src/splendor/custom_operators.py`
- Move `move.py` to `src/splendor/move.py`
- Move `node.py` to `src/splendor/node.py`
- Move `player.py` to `src/splendor/player.py`
- Move `ISMCTS.py` to `src/splendor/ISMCTS.py`

### Step 3: Move Script Files
- Move `main.py` to `scripts/main.py`
- Move `create_database.py` to `scripts/create_database.py`
- Move `load_database.py` to `scripts/load_database.py`

### Step 4: Move Database Files
- Move `games.db` to `data/games.db`
- Move `games_save_1000_iter_3_players_1000_games.db` to `data/games_save_1000_iter_3_players_1000_games.db`

### Step 5: Update Import Statements in Core Game Logic Files
Update all files in `src/splendor/` to use relative imports:
- Update `src/splendor/board.py` imports:
  - Change `from custom_operators import *` to `from .custom_operators import *`
  - Change `from constants import *` to `from .constants import *`
  - Change `from player import *` to `from .player import *`
  - Change `from characters import *` to `from .characters import *`
  - Change `from move import *` to `from .move import *`
  - Change `from cards import *` to `from .cards import *`
- Update `src/splendor/player.py` imports:
  - Change `from cards import *` to `from .cards import *`
  - Change `from characters import *` to `from .characters import *`
  - Change `from move import *` to `from .move import *`
  - Change `from constants import *` to `from .constants import *`
  - Change `from custom_operators import *` to `from .custom_operators import *`
- Update `src/splendor/ISMCTS.py` imports:
  - Change `from node import *` to `from .node import *`
  - Change `from board import *` to `from .board import *`
- Update `src/splendor/node.py` imports:
  - Change `from move import *` to `from .move import *`

### Step 6: Update Import Statements in Script Files
Update all files in `scripts/` to use absolute imports from the src package:
- Update `scripts/main.py` imports:
  - Change `from board import *` to `from splendor.board import *`
  - Change `from ISMCTS import *` to `from splendor.ISMCTS import *`
- Update `scripts/create_database.py` imports (if any from game code):
  - Check for any imports and update to `from splendor.X import Y` format
- Update `scripts/load_database.py` imports:
  - Change `from constants import *` to `from splendor.constants import *`

### Step 7: Update Database Paths in Script Files
Update all database connection paths to point to `data/` directory:
- Update `scripts/main.py`:
  - Change `sqlite3.connect('games.db')` to `sqlite3.connect('data/games.db')`
- Update `scripts/create_database.py`:
  - Change `sqlite3.connect('games.db')` to `sqlite3.connect('data/games.db')`
- Update `scripts/load_database.py`:
  - Change `sqlite3.connect('games.db')` to `sqlite3.connect('data/games.db')`

### Step 8: Create src/splendor/__init__.py
Create package initialization file that exports main classes for convenient imports:
- Export main classes: Board, Card, Character, Player, Move, Node
- Export main functions: ISMCTS, ISMCTS_para
- Add docstring describing the package

### Step 9: Update README.md Project Structure Section
Update the project structure diagram to reflect the new organization:
- Update directory tree to show `src/splendor/`, `scripts/`, and `data/` directories
- Update file descriptions to include directory paths
- Keep descriptions of what each file does

### Step 10: Update README.md Installation Section
Update installation instructions if needed:
- Ensure PYTHONPATH instructions are included if necessary
- Add note about running scripts from project root

### Step 11: Update README.md Usage Section
Update all command examples to use new file locations:
- Change `python create_database.py` to `python scripts/create_database.py`
- Change `python load_database.py` to `python scripts/load_database.py`
- Change `python main.py` to `python scripts/main.py`
- Update all references to file paths in commands and examples

### Step 12: Update .gitignore
Update .gitignore to handle new directory structure:
- Add `data/*.db` to ignore database files in data directory
- Remove or update old patterns that referenced root-level database files
- Ensure `__pycache__` patterns cover new directories

### Step 13: Run Validation Commands
Execute validation commands to ensure everything works correctly with zero regressions

## Validation Commands
Execute every command to validate the chore is complete with zero regressions.

- `cd /home/apprentyr/projects/splendor_ia && python scripts/create_database.py` - Test database creation script works with new paths
- `cd /home/apprentyr/projects/splendor_ia && python scripts/load_database.py` - Test database loading script works with new imports and paths
- `cd /home/apprentyr/projects/splendor_ia && python -c "from splendor.board import Board; from splendor.ISMCTS import ISMCTS; print('Imports successful')"` - Test that imports work correctly from the new package structure
- `cd /home/apprentyr/projects/splendor_ia && ls -la src/splendor/ scripts/ data/` - Verify all files are in correct locations
- `cd /home/apprentyr/projects/splendor_ia && python -m py_compile src/splendor/*.py scripts/*.py` - Verify all Python files compile without syntax errors

## Notes
- The reorganization follows conventional Python project structure with a `src/` directory approach
- Using relative imports (`.module`) within the `src/splendor/` package keeps the code modular
- Scripts use absolute imports from the `splendor` package
- All database paths now point to `data/` directory for better organization
- The project will be easier to package and distribute after this reorganization
- Running scripts from the project root with `python scripts/script_name.py` will work because Python adds the current directory to sys.path
- Consider adding a `setup.py` or `pyproject.toml` in the future to make the package installable
