# Feature: Advanced Feature Engineering for Splendor Imitation Learning

## Feature Description
Expand the feature engineering pipeline to include strategic game-state features that capture:
- Token availability and taking opportunities (per color)
- Card purchasing feasibility and distances (per player, per card)
- Noble acquisition opportunities and distances (per player, per noble)
- Player competitive positioning (leaderboard, VP gaps, buying power)
- Game progression metrics (distance to end, deck depletion)

This enhancement will significantly increase the model's ability to learn strategic decision-making by providing rich, interpretable features that encode domain knowledge about Splendor gameplay.

## User Story
As a **machine learning researcher training a Splendor AI**
I want to **extract strategic game-state features from raw CSV data**
So that **the neural network can learn better policies by understanding card affordability, noble opportunities, competitive positioning, and game phase**

## Problem Statement
The current feature set (~474 features after one-hot encoding) primarily consists of raw game state (board tokens, card costs, player inventories). While comprehensive, it lacks derived strategic features that encode:
- Whether a player can afford specific cards (with/without gold)
- How close players are to buying cards or attracting nobles
- Competitive dynamics (who's winning, who has the strongest economy)
- Game progression signals (how close to game end, deck depletion)

Expert Splendor players consider these factors when making decisions. Adding them as engineered features will help the model learn strategic patterns more efficiently.

## Solution Statement
Implement a modular feature engineering system that:
1. Creates `src/imitation_learning/feature_engineering.py` with reusable helper functions
2. Integrates into the existing `data_preprocessing.py` pipeline
3. Uses `src/splendor` core classes (Board, Player, Card) to compute strategic features
4. Adds ~300-500 new features covering token opportunities, card affordability, noble distances, player rankings, and game progression
5. Maintains backward compatibility with existing preprocessing pipeline

## Relevant Files

### Existing Files
- **src/imitation_learning/data_preprocessing.py** (lines 372-434: `engineer_features`)
  Current feature engineering with one-hot encoding. Will be extended to call new feature engineering module.

- **src/imitation_learning/utils.py**
  Contains `reconstruct_board_from_row()` for rebuilding Board state from CSV. Will be used heavily for feature computation.

- **src/splendor/board.py**
  Core game logic: `getNbMaxTokens()` (line 45), `getPossibleTokens()` (line 142), `getPossibleBuild()` (line 182).

- **src/splendor/player.py**
  Player state and affordability checks: `canBuild()` (line 176), `realCost()` (line 162), `realGoldCost()` (line 172), `getTotalBonus()` (line 159).

- **src/splendor/cards.py**
  Card representation with `vp`, `bonus`, `cost`, `lvl` attributes.

- **src/splendor/constants.py**
  Game constants: `NB_TOKEN_2=4`, `NB_TOKEN_3=5`, `NB_TOKEN_4=7`, `VP_GOAL=15`, `MAX_NB_TOKENS=10`.

### New Files
- **src/imitation_learning/feature_engineering.py**
  New module containing all feature engineering helper functions organized by category.

## Implementation Plan

### Phase 1: Foundation
Set up the feature engineering module with core infrastructure:
- Create `feature_engineering.py` with proper imports and type hints
- Implement board state reconstruction helper (wrapper around existing utils)
- Add utility functions for player lookups and safe value extraction
- Create feature name generation helpers for consistent naming

### Phase 2: Core Feature Implementation
Implement feature extraction functions organized by category:
- **Token features**: availability, remaining after actions, pile sizes
- **Card features**: affordability, gold requirements, purchase distances
- **Noble features**: acquisition distances and opportunities
- **Player comparison features**: rankings, VP gaps, economy strength
- **Game progression features**: distance to end, deck depletion

### Phase 3: Integration
Integrate new features into the preprocessing pipeline:
- Modify `data_preprocessing.py` to call feature engineering module
- Update normalization masks to handle new continuous features
- Validate feature shapes and data types
- Update configuration files with new input dimensions

## Step by Step Tasks

### 1. Create Feature Engineering Module Structure
- Create `src/imitation_learning/feature_engineering.py` with module docstring
- Add imports: `numpy`, `pandas`, `typing`, `src.splendor` classes
- Define type aliases for clarity: `FeatureDict = Dict[str, float]`
- Implement `extract_all_features(row: pd.Series) -> Dict[str, float]` as main entry point

### 2. Implement Token-Related Features
- **Function**: `extract_token_features(row: pd.Series, board: Board) -> Dict[str, float]`
- Features per color (white, blue, green, red, black):
  - `can_take2_{color}`: Binary (1 if pile >= 4, else 0)
  - `tokens_left_if_take2_{color}`: Numeric (gems_board_{color} - 2 if can_take2, else gems_board_{color})
  - `tokens_left_if_take1_{color}`: Numeric (gems_board_{color} - 1)
  - `max_tokens_pile_{color}`: Constant based on num_players (4/5/7 for regular, 5 for gold)
  - `maximum_takeable_this_turn_{color}`: min(2, gems_board_{color}) for each color
- Total: 5 colors × 5 features = 25 features (+ 1 for max_tokens_pile_gold = 26 token features)

### 3. Implement Card-Related Features (Per Player)
- **Function**: `extract_card_features_for_player(player: Player, visible_cards: List[Card], reserved_cards: List[Card], nobles: List[Character]) -> Dict[str, float]`
- Iterate over all cards (12 visible + up to 3 reserved per player):
  - `player{i}_card{j}_can_build`: Binary (use `player.canBuild(card)`)
  - `player{i}_card{j}_must_use_gold`: Binary (1 if can_build=True but need gold, else 0)
  - `player{i}_card{j}_distance_{color}`: max(0, real_cost[color] - player.tokens[color]) for each color
  - `player{i}_card{j}_distance_total`: sum of distances across all 5 colors (excluding gold)
  - `player{i}_card{j}_vp_if_buy`: player.vp + card.vp + 3 (if buying this card enables acquiring a noble, else +0)
- Per card: 1 + 1 + 5 + 1 + 1 = 9 features
- Per player: (12 visible + 3 reserved) × 9 = 135 features
- All 4 players: 4 × 135 = 540 card features

### 4. Implement Noble-Related Features (Per Player)
- **Function**: `extract_noble_features_for_player(player: Player, nobles: List[Character]) -> Dict[str, float]`
- For each of 5 nobles:
  - `player{i}_noble{j}_distance_{color}`: max(0, noble.cost[color] - player.reductions[color])
  - `player{i}_noble{j}_distance_total`: sum of color distances
  - `player{i}_noble{j}_acquirable`: Binary (1 if all distances are 0, else 0)
- Aggregate features:
  - `player{i}_closest_noble_distance`: min of all noble total distances
  - `player{i}_nobles_acquirable_count`: sum of acquirable nobles
- Per noble: 5 colors + 1 total + 1 binary = 7 features
- Per player: 5 nobles × 7 + 2 aggregate = 37 noble features
- All 4 players: 4 × 37 = 148 noble features

### 5. Implement Card-Noble Synergy Features
- **Function**: `extract_card_noble_synergy(player: Player, cards: List[Card], nobles: List[Character]) -> Dict[str, float]`
- For each card the player could build:
  - `player{i}_card{j}_nobles_after_buy`: Count of nobles acquirable if we buy this card
  - `player{i}_card{j}_closest_noble_distance_after_buy`: Min noble distance after adding card's bonus
- Per card: 2 features
- Per player: 15 cards × 2 = 30 features
- All 4 players: 4 × 30 = 120 synergy features

### 6. Implement Player Comparison Features
- **Function**: `extract_player_comparison_features(players: List[Player], current_player_idx: int) -> Dict[str, float]`
- Per player:
  - `player{i}_vp`: Victory points (already in data but useful for rankings)
  - `player{i}_total_gems_reduction`: sum(player.reductions)
  - `player{i}_buying_capacity`: sum(player.tokens[:5]) + sum(player.reductions)
  - `player{i}_total_gems_possessed`: sum(player.tokens)
  - `player{i}_total_gem_colors_possessed`: count of non-zero token colors
  - `player{i}_num_reserved_cards`: len(player.reserved)
  - `player{i}_num_nobles_acquired`: len(player.characters)
  - `player{i}_num_cards_bought`: len(player.built)
- Relative features (computed once per game state):
  - `max_vp_among_players`: max of all player VPs
  - `player{i}_distance_to_max_vp`: max_vp - player{i}_vp
  - `player{i}_leaderboard_position`: Ordinal rank by VP (0=1st, 1=2nd, etc.)
  - `player{i}_vp_gap_to_leader`: VP difference from 1st place
  - `max_gems_reduction_among_players`: max total reduction
  - `player{i}_gems_reduction_leaderboard_position`: Ordinal rank by gems reduction
- Per player base: 8 features
- Per player relative: 4 features
- Global: 2 features (max_vp, max_gems_reduction)
- Total: 4 × (8 + 4) + 2 = 50 comparison features

### 7. Implement Game Progression Features
- **Function**: `extract_game_progression_features(row: pd.Series, players: List[Player]) -> Dict[str, float]`
- Global features:
  - `distance_to_end_game`: 15 - max(player VPs)
  - `distance_to_avg_end_turns`: Estimated based on num_players (hardcoded heuristic: 2p→20 turns, 3p→18, 4p→16. These are example numbers, but are not the real ones)
  - `deck_level1_remaining`: From CSV column
  - `deck_level2_remaining`: From CSV column
  - `deck_level3_remaining`: From CSV column
  - `total_cards_bought`: sum(len(player.built) for all players)
- Per player:
  - `player{i}_vp_to_win`: 15 - player.vp
- Total: 5 global + 4 per-player = 9 progression features

### 8. Integrate into Data Preprocessing Pipeline
- Modify `data_preprocessing.py::engineer_features()`:
  - After one-hot encoding, call `feature_engineering.extract_all_features(row)` for each row
  - Append extracted features to dataframe
  - Update `feature_cols` list with new feature names
- Update `create_normalization_mask()`:
  - Exclude binary features (can_build, must_use_gold, acquirable, etc.)
  - Normalize continuous features (distances, counts, VPs)
- Add progress bar for feature extraction (using tqdm)

### 9. Update Feature Column Tracking
- In `feature_engineering.py`, create `get_all_feature_names()` function:
  - Returns ordered list of all feature names that will be generated
  - Used for validation and debugging
- Update `save_preprocessed_data()` to log new feature count

### 10. Handle Edge Cases
- **Missing players** (2-3 player games): Features for player2/player3 will be all zeros (already handled by fillna)
- **Missing nobles** (fewer than 5): Features should be all zeros (same as missing players/cards)
- **Missing cards** (reserved slots empty): Features should be zeros with position sentinel (-1)
- **Division by zero**: Use safe max/min operations with fallbacks

### 11. Add Unit Tests
- Create `tests/test_feature_engineering.py`:
  - Test token feature extraction with known board states
  - Test card affordability calculations
  - Test noble distance calculations
  - Test ranking/leaderboard calculations
  - Test edge cases (missing players, empty reserves, etc.)
- Run tests: `pytest tests/test_feature_engineering.py -v`

### 12. Update Documentation
- Add docstrings to all functions in `feature_engineering.py`
- Update `data_preprocessing.py` docstring with new feature count
- Document feature naming conventions (e.g., `player{i}_card{j}_feature_name`)
- Add example usage in module docstring

### 13. Validate Feature Extraction
- Run preprocessing on a small subset (10 games):
  - `python -m src.imitation_learning.data_preprocessing --config config_small.yaml --max-games 10`
- Check output shapes and feature counts
- Inspect `feature_cols.json` to verify all features are present
- Verify no NaN/Inf values in continuous features

### 14. Update Model Configuration
- Modify `config_small.yaml`, `config_medium.yaml`, `config_large.yaml`:
  - Update `model.input_dim` to `null` (auto-detect from preprocessed data)
  - Document expected input dimension increase (from ~474 to ~1400+ features)
- Update `train.py` to log input dimension on startup

### 15. Run Full Preprocessing Pipeline
- Execute preprocessing on full dataset:
  - `python -m src.imitation_learning.data_preprocessing --config config_small.yaml`
- Monitor for:
  - Execution time (may be slower due to feature computation)
  - Memory usage
  - Validation report (ensure no failures)

### 16. Validation Commands
Execute every command to validate the feature works correctly with zero regressions:
- `python -m pytest tests/test_feature_engineering.py -v` - Run unit tests for feature engineering
- `python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml --max-games 10` - Test preprocessing on small subset
- `python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml --single-file data/games/2_games/2.csv` - Test single file processing
- `python -c "import json; f=open('data/processed/feature_cols.json'); cols=json.load(f); print(f'Total features: {len(cols)}'); print('Sample features:', cols[:10], '...', cols[-10:])"` - Verify feature count and names
- `python -c "import numpy as np; X=np.load('data/processed/X_train.npy'); print(f'X_train shape: {X.shape}'); print(f'Contains NaN: {np.isnan(X).any()}'); print(f'Contains Inf: {np.isinf(X).any()}')"` - Validate preprocessed data integrity
- `python -m src.imitation_learning.train --config src/imitation_learning/configs/config_small.yaml --epochs 1` - Smoke test training with new features

## Testing Strategy

### Unit Tests
- **Token feature extraction**:
  - Test `can_take2` with pile sizes 0, 3, 4, 7
  - Test `max_tokens_pile` for 2, 3, 4 player games
  - Test `tokens_left_if_take2` calculations

- **Card feature extraction**:
  - Test `can_build` with sufficient/insufficient tokens
  - Test `must_use_gold` edge cases
  - Test distance calculations with various token/bonus combinations

- **Noble feature extraction**:
  - Test distance calculations with 0, partial, and full requirements
  - Test `acquirable` binary flag
  - Test aggregate metrics (closest, count)

- **Player comparison**:
  - Test leaderboard ranking with ties
  - Test VP gap calculations
  - Test gems reduction rankings

- **Game progression**:
  - Test `distance_to_end_game` at various game stages
  - Test deck depletion calculations

### Integration Tests
- **Full pipeline test**:
  - Process 10 games end-to-end
  - Verify output shapes match expected dimensions
  - Check feature value ranges are reasonable

- **Compatibility test**:
  - Ensure existing preprocessing still works
  - Verify masks are still generated correctly
  - Check normalization doesn't break

### Edge Cases
- **2-player game**: Player 2 and 3 features should be zeros
- **3-player game**: Player 3 features should be zeros
- **Fewer than 5 nobles**: Handle missing nobles gracefully
- **Empty reserves**: All reserve features should be zeros
- **Game start**: All player inventories are empty
- **Game end**: Max VP = 15+, check distance calculations
- **No affordable cards**: All `can_build` should be 0
- **All nobles acquirable**: Test when player has massive economy
- **Missing nobles**: All features should be 0 (no large constants)

## Acceptance Criteria
1. ✅ Feature engineering module (`feature_engineering.py`) created with all helper functions
2. ✅ All feature categories implemented:
   - Token features: 26 features
   - Card features: 540 features (4 players × 15 cards × 9)
   - Noble features: 148 features (4 players × 37)
   - Card-noble synergy: 120 features (4 players × 15 cards × 2)
   - Player comparison: 50 features
   - Game progression: 9 features
   - **Total new features**: ~893 features
3. ✅ Features integrated into `data_preprocessing.py` pipeline
4. ✅ Normalization mask updated to handle continuous vs binary features
5. ✅ All unit tests passing
6. ✅ Full preprocessing completes without errors on entire dataset
7. ✅ Preprocessed data validates correctly (no NaN/Inf, correct shapes)
8. ✅ Training runs successfully with new input dimension
9. ✅ Documentation updated with feature descriptions and naming conventions
10. ✅ Backward compatibility maintained (can still process data without new features if needed)

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions:
- `python -m pytest tests/test_feature_engineering.py -v` - Run unit tests for feature engineering
- `python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml --max-games 10` - Test preprocessing on small subset
- `python -m src.imitation_learning.data_preprocessing --config src/imitation_learning/configs/config_small.yaml --single-file data/games/2_games/2.csv` - Test single file processing
- `python -c "import json; f=open('data/processed/feature_cols.json'); cols=json.load(f); print(f'Total features: {len(cols)}'); print('Sample new features:'); [print(c) for c in cols if 'can_build' in c or 'distance' in c or 'leaderboard' in c][:20]; f.close()"` - Verify new feature names
- `python -c "import numpy as np; X=np.load('data/processed/X_train.npy'); print(f'X_train shape: {X.shape}'); print(f'Input dimension: {X.shape[1]}'); print(f'Contains NaN: {np.isnan(X).any()}'); print(f'Contains Inf: {np.isinf(X).any()}'); print(f'Min: {X.min()}, Max: {X.max()}')"` - Validate preprocessed data
- `python -c "import json; stats=json.load(open('data/processed/preprocessing_stats.json')); print(f\"Input dimension: {stats['input_dim']}\"); print(f\"Expected: ~1370 features (474 old + 896 new)\")"` - Check feature count matches expectations
- `python -m src.imitation_learning.train --config src/imitation_learning/configs/config_small.yaml --epochs 1` - Smoke test training with new features

## Notes

### Feature Count Breakdown
- **Existing features**: ~474 (after one-hot encoding)
- **Token features**: 26
- **Card features**: 540 (4 players × 15 cards × 9 features)
- **Noble features**: 148 (4 players × 37 features)
- **Card-noble synergy**: 120 (4 players × 15 cards × 2 features)
- **Player comparison**: 50
- **Game progression**: 9
- **Total**: ~474 + 893 = **~1367 features**

### Performance Considerations
- Feature extraction will add computational overhead to preprocessing
- Estimated 2-5x slower than current preprocessing (due to board reconstruction per row)
- Consider parallelization if needed (batch processing)

### Future Enhancements (Not in this spec)
- Opponent action simulation: "If I buy card X, can opponent buy card Y next turn?"
- Advanced combo features: "Cards that synergize with my current bonuses"
- Historical features: "VP gain rate over last 3 turns"
- Attention-based features: "Which cards are most contested by opponents?"

### Normalization Strategy
- **Binary features** (can_build, must_use_gold, acquirable): Do NOT normalize
- **Distances** (distance_to_buy, distance_to_noble): Normalize (can be 0-10+)
- **Counts** (tokens, gems_reduction, cards_bought): Normalize
- **VPs**: Normalize (range 0-20+)
- **Positions/ranks**: Normalize (ordinal 0-3)

### Dependencies
- No new external libraries required
- Uses existing `src/splendor` core classes
- Leverages `utils.py::reconstruct_board_from_row()` for state reconstruction
