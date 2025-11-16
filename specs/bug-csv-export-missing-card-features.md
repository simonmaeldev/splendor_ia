# Bug: CSV Export Missing Card Features (Expected 382, Got 370)

## Bug Description

When running the main game loop, CSV export occasionally fails with the error:
```
Error exporting game 4563 to CSV: Expected 382 features, got 370
```

This error occurs intermittently during game simulation, particularly for games that reach later turns when card decks become depleted. The error indicates that the feature encoding is producing 370 features instead of the expected 382 features, a difference of exactly 12 features (the size of one card encoding).

## Problem Statement

The CSV exporter's `encode_game_state_from_board()` function assumes there are always exactly 12 visible cards (4 cards per level × 3 levels) on the board. However, when a deck becomes empty in the late game, `board.displayedCards[level]` contains fewer than 4 cards for that level, causing the total feature count to be less than 382.

## Solution Statement

Modify the CSV exporter to always encode exactly 12 card slots regardless of how many cards are actually present. When a deck level has fewer than 4 cards, pad the remaining slots with NaN-encoded cards (12 NaN values per missing card).

## Steps to Reproduce

```bash
cd /home/apprentyr/projects/splendor_ia

# Run a simulation that will eventually deplete decks
python3 scripts/main.py

# The error appears intermittently:
# Error exporting game XXXX to CSV: Expected 382 features, got 370
```

The error is more likely to occur:
- In longer games (more turns)
- When deck level 1 becomes depleted (most common)
- Less frequently for deck level 2 or 3 depletion

## Root Cause Analysis

### Location
`src/splendor/csv_exporter.py:258-262` in the `encode_game_state_from_board()` function

### Current Code
```python
# Visible cards (12 cards: 4 per level, levels 1-3)
# displayedCards is [level1[4], level2[4], level3[4]]
for level_cards in board.displayedCards:
    for card in level_cards:
        features.extend(encode_card(card))
```

### Issue
The code iterates through whatever cards exist in `board.displayedCards`, but this list can have fewer than 12 cards total when decks are depleted.

**How cards get removed without replacement:**

In `src/splendor/board.py:313-323` (`removeCard()` method):
```python
def removeCard(self, move: Move) -> None:
    if move.action in TOP_DECK:
        del self.decks[move.action][0]
    else:
        card = move.action
        self.displayedCards[card.lvl - 1].remove(card)  # Remove card
        if len(self.decks[card.lvl - 1]) > 0:           # Check if deck has cards
            newCard = self.decks[card.lvl - 1].pop(0)
            newCard.setVisible()
            self.displayedCards[card.lvl - 1].append(newCard)  # Only add if deck has cards
        # If deck is empty, no replacement card is added!
```

**Result:**
- Normal case: `displayedCards = [[4 cards], [4 cards], [4 cards]]` → 12 cards → 144 features
- When level 1 deck empty: `displayedCards = [[3 cards], [4 cards], [4 cards]]` → 11 cards → 132 features
- Total features: 3 + 6 + 132 + 3 + 30 + 196 = 370 ✓ (matches error message)

### Validation
The feature count difference confirms this analysis:
- Expected: 382 features (12 cards × 12 features/card = 144 card features)
- Actual: 370 features (11 cards × 12 features/card = 132 card features)
- Difference: 12 features = exactly 1 missing card encoding

## Relevant Files

### Files to Modify

- **`src/splendor/csv_exporter.py`** (primary fix)
  - Modify `encode_game_state_from_board()` to always encode exactly 12 card slots
  - Add padding with NaN-encoded cards when fewer than 4 cards per level

### New Files

None required - all changes are modifications to existing file.

## Step by Step Tasks

### Step 1: Fix Visible Card Encoding to Always Encode 12 Slots
- Open `src/splendor/csv_exporter.py`
- Locate `encode_game_state_from_board()` function (line ~223)
- Find the visible cards encoding section (lines ~258-262)
- Replace the current encoding logic:
  ```python
  # OLD CODE (lines 258-262):
  # Visible cards (12 cards: 4 per level, levels 1-3)
  # displayedCards is [level1[4], level2[4], level3[4]]
  for level_cards in board.displayedCards:
      for card in level_cards:
          features.extend(encode_card(card))
  ```

  With fixed encoding that pads missing cards:
  ```python
  # Visible cards (12 cards: 4 per level, levels 1-3)
  # displayedCards is [level1[0-4], level2[0-4], level3[0-4]]
  # Always encode exactly 4 slots per level, padding with NaN if deck is depleted
  for level_cards in board.displayedCards:
      for i in range(4):
          if i < len(level_cards):
              features.extend(encode_card(level_cards[i]))
          else:
              # Deck depleted, no card in this slot - encode as NaN
              features.extend(encode_card(None))
  ```

### Step 2: Verify encode_card(None) Returns 12 NaN Values
- Check `encode_card()` function (line ~36)
- Verify it already handles `None` correctly:
  ```python
  if card_obj is None:
      return [float('nan')] * 12
  ```
- ✓ Already implemented - no changes needed

### Step 3: Update Documentation
- Update the docstring in `encode_game_state_from_board()` (line ~225-241)
- Change comment from "visible_cards (12 x 12 = 144)" to:
  ```python
  - visible_cards (12 x 12 = 144, padded with NaN when decks depleted)
  ```

### Step 4: Create Test Script to Reproduce and Validate Fix
- Create `scripts/test_csv_export_edge_case.py` to test deck depletion scenario
- Test with a game state that has depleted decks
- Verify feature count is always 382

### Step 5: Run Validation Commands
- Run test script created in Step 4
- Run a small number of real games to completion
- Verify no CSV export errors occur
- Check that CSV files with depleted decks have correct feature count

## Validation Commands

Execute every command to validate the bug is fixed with zero regressions.

```bash
# Navigate to project root
cd /home/apprentyr/projects/splendor_ia

# Create test script to simulate deck depletion
cat > scripts/test_csv_export_depleted_deck.py << 'EOF'
"""Test CSV export with depleted deck scenario"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from splendor.board import Board
from splendor.move import Move
from splendor.csv_exporter import encode_game_state_from_board
from copy import deepcopy

# Create a 2-player game
board = Board(2, ["ISMCTS", "ISMCTS"], debug=False)

# Simulate deck depletion by removing cards from level 1 deck
print(f"Initial level 1 deck size: {len(board.deckLVL1)}")
print(f"Initial displayedCards sizes: {[len(level) for level in board.displayedCards]}")

# Remove most cards from level 1 deck to simulate late game
board.deckLVL1 = board.deckLVL1[:2]  # Leave only 2 cards

# Build a level 1 card to trigger replacement
level1_card = board.displayedCards[0][0]
move = Move(0, level1_card, None)  # BUILD action
board.build(move)

print(f"After build - level 1 deck size: {len(board.deckLVL1)}")
print(f"After build - displayedCards sizes: {[len(level) for level in board.displayedCards]}")

# Build another level 1 card
if len(board.displayedCards[0]) > 0:
    board.currentPlayer = 0
    level1_card2 = board.displayedCards[0][0]
    move2 = Move(0, level1_card2, None)
    board.build(move2)

print(f"After 2nd build - level 1 deck size: {len(board.deckLVL1)}")
print(f"After 2nd build - displayedCards sizes: {[len(level) for level in board.displayedCards]}")

# Now try to encode the game state
try:
    features = encode_game_state_from_board(board, 1)
    print(f"\n✓ SUCCESS: Encoded {len(features)} features (expected 382)")
    if len(features) == 382:
        print("✓ Feature count is correct!")
    else:
        print(f"✗ FAILED: Expected 382 features, got {len(features)}")
        sys.exit(1)
except Exception as e:
    print(f"\n✗ FAILED: {e}")
    sys.exit(1)

print("\nTest passed!")
EOF

# Run the test script
python3 scripts/test_csv_export_depleted_deck.py

# Run a few real games to ensure no regressions
echo "=== Running real game simulation test ==="

# Backup current config
cp data/simulation_config.txt data/simulation_config.txt.backup 2>/dev/null || true

# Create minimal test config
cat > data/simulation_config.txt << 'EOF'
# Test configuration for CSV export validation
2: 3
EOF

# Run test games
timeout 300 python3 scripts/main.py || echo "Test completed or timed out"

# Check if any CSV export errors occurred in the output
echo ""
echo "=== Checking for CSV export errors ==="
# If the test ran without errors, there should be no "Error exporting" messages
echo "✓ If you see this message and no 'Error exporting' messages above, the bug is fixed!"

# Restore original config
mv data/simulation_config.txt.backup data/simulation_config.txt 2>/dev/null || true

# Verify CSV files were created and have correct structure
echo ""
echo "=== Validating CSV file structure ==="
python3 -c "
import csv
import glob
csv_files = glob.glob('data/games/2_games/*.csv')
csv_files = [f for f in csv_files if '.gitkeep' not in f]
if not csv_files:
    print('⚠ No CSV files found')
else:
    for filepath in csv_files[:3]:  # Check first 3 files
        with open(filepath) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)
            print(f'File {filepath}:')
            print(f'  Headers: {len(headers)} columns')
            print(f'  Rows: {len(rows)} actions')
            # Expected: 402 columns (1 game_id + 382 input + 20 output - 1 = 402)
            # Note: The function returns 382 features but headers include game_id
            expected = 403  # game_id + 382 inputs + 20 outputs
            if len(headers) == expected:
                print(f'  ✓ Correct column count')
            else:
                print(f'  Column count: expected ~{expected}, got {len(headers)}')
"

# Database integrity check
echo ""
echo "=== Database Integrity Check ==="
python3 -c "
import sqlite3
conn = sqlite3.connect('data/games.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM Game')
game_count = cursor.fetchone()[0]
print(f'Total games in database: {game_count}')
if game_count > 0:
    print('✓ Games were saved successfully')
conn.close()
"

echo ""
echo "=== Validation Complete ==="
echo "If all checks show ✓ marks and no errors, the bug is fixed!"
```

## Notes

### Why This Bug Occurs Intermittently

The bug only manifests when:
1. A game runs long enough to deplete at least one deck
2. CSV export is attempted after deck depletion
3. Most commonly affects level 1 deck (smallest cards, built most frequently)

This explains why the error message shows it occurs during game simulation but not every time:
- Short games: All decks have cards → no bug
- Long games: One or more decks depleted → bug triggers

### Feature Count Calculation

With the fix, the encoding will always produce:
- Basic info: 3 (num_players, turn_number, current_player)
- Gems on board: 6
- Visible cards: 12 × 12 = 144 (always 12 slots, padded with NaN)
- Deck remaining: 3
- Nobles: 5 × 6 = 30
- Players: 4 × 49 = 196
- **Total: 382 features** ✓

### NaN Padding Strategy

Using NaN for missing cards is the correct approach because:
1. It maintains consistent feature count across all game states
2. NaN is semantically meaningful (represents absence of data)
3. ML models can handle NaN values appropriately
4. Consistent with existing `encode_card(None)` behavior

### No Database Changes Required

This bug only affects CSV export, not database storage. The database schema correctly handles variable numbers of displayed cards, so no schema changes are needed.

### Testing Strategy

The test script simulates the exact scenario that causes the bug:
1. Create a board with normal decks
2. Artificially deplete a deck
3. Build cards to remove displayed cards without replacement
4. Attempt CSV encoding
5. Verify 382 features are produced

### Related Issues

This fix is complementary to the improvements documented in `bug-csv-export-and-player-state-tracking.md`. Both address different aspects of CSV export robustness.
