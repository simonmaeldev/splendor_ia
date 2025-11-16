# Bug: Error during game execution: 'int' object has no attribute 'cost'

## Bug Description

During game simulation with data collection, the system intermittently crashes with the error:
```
Error during game execution: 'int' object has no attribute 'cost'
```

This error occurs when saving game data to the database, specifically when a player reserves a card from the top of a deck (blind reserve). The code incorrectly assumes that `move.action` is always a Card object, but for top-deck reserves, it's an integer (0, 1, or 2) representing the deck level.

**Symptoms**:
- Games fail to save to database
- Error occurs intermittently (only when players perform blind reserves)
- CSV export never happens because database save fails first
- Progress is lost for the failed game

**Expected behavior**: Top-deck reserves should be handled correctly, with the actual Card object being retrieved and its ID saved to the database.

**Actual behavior**: Code tries to access `.cost` attribute on an integer, causing an AttributeError.

## Problem Statement

The `savePlayerActions` function in `scripts/main.py` needs to handle two types of RESERVE actions:
1. **Visible card reserve**: `move.action` is a Card object
2. **Top-deck reserve**: `move.action` is an integer (0, 1, or 2)

Currently, the code only handles case 1, causing crashes for case 2.

## Solution Statement

**Chosen Approach: Option 2 - Retrieve actual card from board state**

Modify `savePlayerActions` to:
1. Accept `board_states` parameter (list of Board objects before each action)
2. Detect when `move.action` is an integer (top-deck reserve)
3. Retrieve the actual Card object from the deck before it was removed
4. Use that Card to look up the card ID for database save

**Important**: When a player reserves from the top deck, the card becomes visible only to that player (it's added to their reserved list but marked as not visible to others). We retrieve the card from the board state at the time of the action, before it was removed from the deck.

## Steps to Reproduce

1. Run game simulations with data collection:
   ```bash
   cd /home/apprentyr/projects/splendor_ia
   python3 scripts/data_collector.py
   ```

2. Wait for a game where a player performs a blind reserve (reserves from top deck)

3. Observe the error:
   ```
   Error during game execution: 'int' object has no attribute 'cost'
   Game not saved. Continuing...
   ```

4. Check that the game was not saved to the database

**Note**: This error is intermittent because not all games involve top-deck reserves. The error rate depends on the MCTS strategy and game state.

## Root Cause Analysis

### Primary Issue: `scripts/main.py:111`

```python
def savePlayerActions(cursor: sqlite3.Cursor, gameID: int, playerPos: int, history: List[Move],
                     cards: Dict[Tuple[int, int, int, int, int], int],
                     characters: Dict[Tuple[int, int, int, int, int], int]) -> None:
    # ...
    for turn, a in enumerate(history):
        if a.actionType in [BUILD, RESERVE]:
            take = [None] * 6 if a.actionType == BUILD else TAKEONEGOLD
            give = [None] * 6
            cardID = cards[tuple(a.action.cost)]  # ← BUG HERE: a.action might be int!
            characterID = characters[tuple(a.character.cost)] if a.character else None
        # ...
```

**Root cause**: Line 111 assumes `a.action` is always a Card object, but for RESERVE actions from top deck, `a.action` is an integer.

### Why `move.action` can be an integer

Looking at `src/splendor/board.py:209-219` (getPossibleReserve):
```python
def getPossibleReserve(self) -> List[Union[Card, int]]:
    reserve: List[Union[Card, int]] = []
    if self.getCurrentPlayer().canReserve():
        for i in range(0,len(self.displayedCards)):
            # add visible cards
            for j in range(0, len(self.displayedCards[i])):
                reserve.append(self.displayedCards[i][j])  # Card objects
            # if there's still a deck of this lvl
            if self.decks[i]:
                reserve.append(i)  # Integer (0, 1, or 2) ← HERE!
    return reserve
```

When a move is created for a top-deck reserve, `move.action` is set to an integer (the deck level index).

### How the Card is handled during gameplay

In `src/splendor/board.py:262-266` (getCard method):
```python
def getCard(self, move: Move) -> Card:
    if move.actionType == RESERVE and move.action in TOP_DECK:  # reserve topdeck
        return self.decks[move.action][0]  # Returns the actual Card object
    else:
        return move.action  # Already a Card object
```

And in `src/splendor/board.py:284-290` (reserve method):
```python
def reserve(self, move: Move) -> None:
    card = self.getCard(move)  # Gets actual Card object
    self.removeCard(move)       # Removes from deck (using move.action as index)
    player = self.getCurrentPlayer()
    player.reserved.append(card)  # Stores actual Card object
    # ...
```

**Key insight**: During gameplay, `getCard(move)` correctly retrieves the Card object. But when saving to database, the code uses the original `move.action` which is still an integer for top-deck reserves.

### Secondary Issue: Comment acknowledges the bug!

Line 111 has a comment:
```python
cardID = cards[tuple(a.action.cost)] #will fail if someone reserve a top deck
```

This indicates the developer was aware of the issue but didn't fix it!

### Why the error is intermittent

- Not all games involve top-deck reserves
- MCTS strategy might prefer visible cards
- Players need available reserve slots (< 3 reserved cards)
- Decks need to have cards remaining

This explains why some games succeed while others fail.

## Relevant Files

### Files to modify

- **`scripts/main.py`** (line 104-119: `savePlayerActions` function)
  - Fix line 111 to handle both Card objects and integers
  - Retrieve actual Card object for top-deck reserves
  - Use the Card to get its ID for database save

### Files for reference

- **`src/splendor/board.py`** (lines 209-219, 262-266, 284-290)
  - Shows how `move.action` can be an integer
  - Shows how `getCard()` retrieves actual Card for top-deck reserves

- **`src/splendor/move.py`**
  - Move class definition to understand structure

- **`src/splendor/constants.py`** (line 24)
  - Definition of `TOP_DECK = [0, 1, 2]`

## Step by Step Tasks

### Step 1: Understand the Move structure and card visibility
- Review Move class to confirm action field can be `Union[Card, int]`
- Verify TOP_DECK constant definition: `[0, 1, 2]`
- Understand when move.action is int vs Card:
  - **int**: Top-deck reserve (blind reserve) - card becomes visible **only** to the reserving player
  - **Card**: Visible card reserve or build action
- Confirm that `board_states[turn]` contains the board state **BEFORE** action at turn `turn`
  - This is critical: we need the deck state before the card was removed

### Step 2: Update savePlayerActions signature
- Open `scripts/main.py`
- Locate `savePlayerActions` function (line 104)
- Add `board_states: List[Board]` parameter to function signature:
  ```python
  def savePlayerActions(cursor: sqlite3.Cursor, gameID: int, playerPos: int,
                       history: List[Move], cards: Dict[Tuple[int, int, int, int, int], int],
                       characters: Dict[Tuple[int, int, int, int, int], int],
                       board_states: List[Board]) -> None:
  ```

### Step 3: Fix top-deck reserve handling
- Within `savePlayerActions`, locate line 111 (card ID retrieval)
- Replace the problematic line:
  ```python
  cardID = cards[tuple(a.action.cost)] #will fail if someone reserve a top deck
  ```
- With robust handling for both visible and top-deck reserves:
  ```python
  # Get actual card object for both visible and top-deck reserves
  if a.actionType == RESERVE and isinstance(a.action, int):
      # Top-deck reserve: a.action is deck level (0, 1, or 2)
      # The card is drawn from the top of the deck and becomes visible only to this player
      # Get card from the deck BEFORE it was removed (board_states[turn] = state before action)
      board = board_states[turn]
      if a.action < len(board.decks) and len(board.decks[a.action]) > 0:
          card = board.decks[a.action][0]  # Top card of the specified deck level
          cardID = cards[tuple(card.cost)]
      else:
          # Deck empty - shouldn't happen, but handle gracefully
          cardID = None
  else:
      # Visible card reserve or build: a.action is already a Card object
      cardID = cards[tuple(a.action.cost)]
  ```

### Step 4: Update function call in saveIntoBdd
- Locate the call to `savePlayerActions` in `saveIntoBdd` function (around line 144-145)
- Update to pass the `board_states` parameter:
  ```python
  # insert actions of players
  for i, ha in enumerate(historyActionPlayers):
      savePlayerActions(cursor, gameID, i, ha, cards, characters, board_states)
  ```

### Step 5: Test the fix with targeted script
- Create test script that forces a top-deck reserve
- Run simulation and verify no errors
- Check database has correct card IDs for top-deck reserves
- Verify CSV export works correctly

### Step 6: Validate with full simulation
- Run data collection with multiple games
- Monitor for any "int object has no attribute cost" errors
- Verify all games save successfully
- Check database integrity

## Validation Commands

Execute every command to validate the bug is fixed with zero regressions.

```bash
# Navigate to project root
cd /home/apprentyr/projects/splendor_ia

# Test 1: Create a script to force top-deck reserves
cat > scripts/test_topdeck_reserve.py << 'EOF'
"""Test script to verify top-deck reserve handling"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splendor.board import Board
from splendor.move import Move
from splendor.constants import RESERVE, TOP_DECK, TAKEONEGOLD
from copy import deepcopy

# Create a simple 2-player game
state = Board(2, ["ISMCTS_PARA", "ISMCTS_PARA"], debug=False)

# Manually create a top-deck reserve move
reserve_move = Move(RESERVE, 0, [0,0,0,0,0,0], None)  # Reserve from deck level 0

print(f"Move action type: {reserve_move.action}, type: {type(reserve_move.action)}")
print(f"Is action an int? {isinstance(reserve_move.action, int)}")
print(f"Is action in TOP_DECK? {reserve_move.action in TOP_DECK}")

# Get the card that would be reserved
card = state.decks[0][0] if len(state.decks[0]) > 0 else None
if card:
    print(f"Card to be reserved: VP={card.vp}, level={card.lvl}, cost={card.cost}")

# Execute the move
print(f"\nExecuting reserve move...")
try:
    state.doMove(reserve_move)
    print(f"Success! Player now has {len(state.players[0].reserved)} reserved cards")
    if state.players[0].reserved:
        reserved_card = state.players[0].reserved[-1]
        print(f"Reserved card: VP={reserved_card.vp}, cost={reserved_card.cost}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
EOF

python3 scripts/test_topdeck_reserve.py

# Test 2: Run a small simulation to test database save
# Backup current database
cp data/games.db data/games.db.backup 2>/dev/null || true

# Create minimal test config
cat > data/test_config.txt << 'EOF'
2: 5
EOF

# Run 5 games to increase chance of hitting top-deck reserve
echo "Running 5 test games..."
python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'scripts'))

from data_collector import run_data_collection
run_data_collection('data/test_config.txt', 'data/games.db', 'data/test_log.txt')
" 2>&1 | tee test_output.txt

# Check for errors
if grep -q "'int' object has no attribute 'cost'" test_output.txt; then
    echo "✗ ERROR STILL OCCURS: Bug not fixed"
    exit 1
else
    echo "✓ No 'int' object has no attribute cost' errors"
fi

# Check that all 5 games were saved
echo ""
echo "Checking game count in database..."
python3 -c "
import sqlite3
conn = sqlite3.connect('data/games.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM Game WHERE NbPlayers = 2')
count = cursor.fetchone()[0]
print(f'Games saved: {count}')
if count >= 5:
    print('✓ All test games saved successfully')
else:
    print(f'✗ Only {count}/5 games saved')
conn.close()
"

# Check for top-deck reserve actions in database
echo ""
echo "Checking for top-deck reserve actions..."
python3 -c "
import sqlite3
from splendor.constants import RESERVE

conn = sqlite3.connect('data/games.db')
cursor = conn.cursor()

# Count reserve actions
cursor.execute('SELECT COUNT(*) FROM Action WHERE Type = ?', (RESERVE,))
reserve_count = cursor.fetchone()[0]
print(f'Total reserve actions: {reserve_count}')

# Check if any have NULL card IDs (which would indicate top-deck reserves with our fix)
cursor.execute('SELECT COUNT(*) FROM Action WHERE Type = ? AND IDCard IS NULL', (RESERVE,))
null_card_count = cursor.fetchone()[0]

if reserve_count > 0:
    print(f'Reserves with NULL card ID: {null_card_count}')
    if null_card_count > 0:
        print('⚠ Note: NULL card IDs for top-deck reserves (expected if we chose Option 1)')
    print('✓ Reserve actions found and saved')
else:
    print('⚠ No reserve actions in test games (may be normal)')

conn.close()
"

# Test 3: Run full validation
echo ""
echo "=== Running full validation ==="

# Restore original database
cp data/games.db.backup data/games.db 2>/dev/null || true

# Clean up test files
rm -f test_output.txt data/test_config.txt data/test_log.txt

echo "✓ Validation complete"
```

## Notes

### Solution Approach: Retrieve Card from Board State (Option 2)

We chose to retrieve the actual card from the board state rather than saving NULL for top-deck reserves because:

**Why this approach:**
- Preserves complete game information for ML training and analysis
- We already have `board_states` available (captured before each action)
- The card ID is critical for understanding player strategy
- Minimal additional complexity since board states are already captured

**Key implementation details:**
- `board_states[turn]` contains the state **BEFORE** the action at turn `turn`
- For top-deck reserves, `a.action` is an integer (0=level 1, 1=level 2, 2=level 3)
- We retrieve the card with: `card = board.decks[a.action][0]` (top card of that deck)
- This gives us the exact card that was reserved, matching what `board.reserve()` did during gameplay

**Card visibility in Splendor:**
- **Top-deck reserves**: Card becomes visible **only to the reserving player** (hidden from opponents)
  - When `board.reserve(move)` is called, the card is added to `player.reserved`
  - The card's `visible` property tracks this (cards from top deck are NOT marked as visible)
  - This is important for ISMCTS (Information Set MCTS) which handles hidden information
- **Visible card reserves**: Card is already visible to all players
- **Database storage**: We save the actual card ID for both types (visibility doesn't affect storage)
  - This is crucial: even though opponents can't see the card, we record it for complete game analysis

### Why the comment exists

The comment on line 111 `#will fail if someone reserve a top deck` suggests this was a known issue that was never fixed. This might be because:
1. Top-deck reserves are rare in MCTS gameplay
2. The error was deemed non-critical (game continues after failure)
3. It was left as a TODO item

### Impact on CSV Export

The CSV export was failing because it depends on successful database save. Once database save succeeds, CSV export should also succeed (it has its own error handling).

### Testing Strategy

Since top-deck reserves are probabilistic, we need to:
1. Run enough games to hit the scenario (5-10 games should be sufficient)
2. Check database for reserve actions
3. Verify no errors in output
4. Check CSV files generated successfully

### Database Schema

The Action table allows NULL for IDCard field (for actions like TOKENS), but our solution (Option 2) saves the actual card ID for all reserve actions, providing complete game history for analysis.

### Summary

This fix resolves a critical bug where top-deck reserve actions would crash the game save process. By retrieving the actual card from `board_states[turn].decks[a.action][0]`, we:
1. Fix the crash (no more AttributeError)
2. Preserve complete game data (card ID is saved correctly)
3. Maintain consistency with gameplay logic (same card that was reserved)
4. Enable full game reconstruction and ML training analysis
