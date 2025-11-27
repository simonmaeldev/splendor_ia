# Feature: AI Player Implementation for Neural Network Model

## Feature Description
Implement a new player type in Splendor that loads a trained neural network model and uses it to play the game. This AI player will:
- Load a pre-trained multi-head model from disk (.pth file)
- Reconstruct the board state from the current game position
- Generate legal action masks to ensure only valid moves can be predicted
- Compute engineered features from the board state (without writing/reading CSV)
- Use the model to predict action probabilities with legal move masking applied
- Translate the predicted action heads into executable Move objects

The implementation must be efficient (no unnecessary file I/O) and integrate seamlessly with the existing game engine infrastructure.

## User Story
As a machine learning researcher
I want to create an AI player that uses my trained neural network model
So that I can evaluate model performance through actual gameplay against MCTS and other baselines

## Problem Statement
Currently, the trained neural network models (stored in `data/models/`) cannot be used to actually play the game. The models are trained using offline supervised learning on CSV data, but there's no way to:
1. Load a trained model and use it for inference during live gameplay
2. Convert live game states into the feature representation expected by the model
3. Translate model predictions (multi-head outputs) back into valid game moves
4. Ensure the model only selects legal actions by applying masks before prediction

This prevents evaluation of the trained models in actual gameplay scenarios.

## Solution Statement
Create a new "MODEL" IA type that integrates with the existing Player class infrastructure. This player type will:
1. Accept a model path via the IA string (e.g., `"MODEL:data/models/baseline/best_model.pth"`)
2. Load the model once during initialization
3. On each turn, reconstruct the board state, compute features, generate masks, and use the model for masked prediction
4. Translate multi-head predictions into Move objects that the game engine understands
5. Handle all intermediate computation in-memory (no CSV file I/O)

The implementation will reuse existing modules (`feature_engineering.py`, `utils.py`, `state_reconstruction.py`, `model.py`) to avoid code duplication.

## Relevant Files
Use these files to implement the feature:

- `src/splendor/player.py` - Player class that currently handles MCTS AI; need to add MODEL IA type support
- `src/splendor/board.py` - Board class with getMoves() method for generating legal moves
- `src/splendor/move.py` - Move class that represents game actions
- `src/splendor/constants.py` - Game constants (BUILD, RESERVE, TOKENS, etc.)
- `src/imitation_learning/model.py` - Multi-head neural network architecture (MultiHeadSplendorNet)
- `src/imitation_learning/feature_engineering.py` - Feature extraction from board states
- `src/imitation_learning/utils.py` - Mask generation (generate_all_masks_from_row) and prediction utilities
- `src/imitation_learning/constants.py` - Pre-computed gem class mappings
- `src/utils/state_reconstruction.py` - Board reconstruction from CSV format (reference for feature extraction)

### New Files

- `src/splendor/ai_player.py` - Core AI player implementation with model inference logic
  - Handles model loading and caching
  - Converts board state to model input features
  - Applies masks to logits before prediction
  - Translates multi-head predictions to Move objects
  - Provides clean interface for Player class to call

- `src/splendor/action_decoder.py` - Translates model predictions to Move objects
  - Takes multi-head predictions (action_type, card_selection, etc.)
  - Decodes gem take3/take2 class indices back to token lists
  - Decodes gem removal class indices back to token lists
  - Maps card/noble indices to actual Card/Character objects from board
  - Constructs complete Move objects

- `tests/test_ai_player.py` - Unit tests for AI player functionality
  - Test model loading
  - Test feature extraction from board
  - Test mask generation
  - Test action decoding
  - Test end-to-end prediction flow

## Implementation Plan

### Phase 1: Foundation
Create the action decoder module that translates model predictions into Move objects. This is foundational because the AI player will depend on it to convert neural network outputs into game actions.

Set up the testing infrastructure to enable TDD (test-driven development) for the AI player components.

### Phase 2: Core Implementation
Implement the AI player module that handles model loading, feature extraction, masked inference, and action selection. This is the core logic that ties together feature engineering, masking, model inference, and action decoding.

Integrate the AI player with the existing Player class by adding MODEL IA type support.

### Phase 3: Integration
Test the AI player in actual gameplay scenarios to ensure it works correctly with the game engine. Create example scripts showing how to use the AI player in games.

## Step by Step Tasks

### 1. Create action decoder module
- Create `src/splendor/action_decoder.py`
- Implement function to decode gem_take3 class index to token list using COMBO_TO_CLASS_TAKE3
- Implement function to decode gem_take2 class index to token list (single color)
- Implement function to decode gems_removed class index to token list using CLASS_TO_REMOVAL
- Implement function to map card_selection index to Card object from board.displayedCards and player.reserved
- Implement function to map card_reservation index to Card or deck level (int)
- Implement function to map noble index to Character object from board.characters
- Implement main function `decode_predictions_to_move()` that takes all head predictions and board state, returns Move object
- Add comprehensive docstrings with examples

### 2. Create AI player module
- Create `src/splendor/ai_player.py`
- Implement `load_model()` function to load .pth checkpoint and create MultiHeadSplendorNet instance
- Implement `board_to_csv_row()` function to convert Board object to CSV row dictionary (reverse of state_reconstruction)
- Implement `get_model_input_features()` function that:
  - Converts board to CSV row dict
  - Calls `extract_all_features()` from feature_engineering.py
  - Converts feature dict to ordered tensor matching model's expected input
- Implement `apply_legal_move_masking()` function that:
  - Generates masks using `generate_all_masks_from_row()` from utils.py
  - Applies masks to model logits before argmax (strong negative penalty for illegal actions)
- Implement `ModelPlayer` class with:
  - `__init__(model_path)` - loads model and stores reference
  - `get_action(board)` - main inference method that returns Move object
  - Caches model config and feature column order for efficiency
- Add error handling for model loading failures, invalid paths, etc.

### 3. Integrate AI player with Player class
- Modify `src/splendor/player.py` askAction() method to detect "MODEL:" prefix in IA string
- Extract model path from IA string (e.g., "MODEL:data/models/baseline/best_model.pth")
- Lazy-load ModelPlayer instance on first action (cache in player.model_player attribute)
- Call model_player.get_action(board) to get Move object
- Store Move components in self.action as expected by existing code
- Ensure compatibility with existing getFinalAction() and getComplementaryAction() methods

### 4. Create unit tests
- Create `tests/test_ai_player.py`
- Test action decoder functions with various prediction combinations
- Test board_to_csv_row conversion produces correct format
- Test feature extraction produces correct number of features
- Test mask generation produces valid masks
- Create mock model for testing without actual .pth file
- Test end-to-end flow from board → features → masked predictions → move
- Test error handling (invalid model path, corrupted checkpoint, etc.)

### 5. Create integration test with actual model
- Create test script `scripts/test_model_player.py`
- Load a real trained model from `data/models/`
- Create a test game with MODEL player vs random/MCTS player
- Play several turns and verify MODEL player makes legal moves
- Log predictions and selected actions for debugging
- Verify no crashes and game completes successfully

### 6. Create example usage script
- Create `scripts/play_with_model.py`
- Show how to create a game with MODEL player: `Player("AI", "MODEL:data/models/baseline/best_model.pth")`
- Demonstrate 2-player game: MODEL vs MCTS
- Demonstrate 4-player game: multiple MODEL players with different models
- Add command-line arguments for model path, opponent type, number of games
- Print game results and statistics

### 7. Run validation commands
- Execute all validation commands listed below to ensure feature works correctly with zero regressions
- Fix any issues discovered during validation
- Verify MODEL player can complete full games without errors

## Testing Strategy

### Unit Tests
- **Action Decoder Tests**:
  - Test gem_take3 decoding for all 26 class indices
  - Test gem_take2 decoding for all 5 color indices
  - Test gems_removed decoding for representative removal patterns
  - Test card_selection mapping for visible cards (0-11) and reserved cards (12-14)
  - Test card_reservation mapping for visible cards and top-deck (12-14)
  - Test noble mapping for all 5 noble positions
  - Test decode_predictions_to_move() with various action type combinations

- **Feature Extraction Tests**:
  - Test board_to_csv_row() produces all required columns
  - Test CSV row has correct dtypes (floats, no strings except where expected)
  - Test feature extraction produces expected number of features (~893)
  - Test feature ordering matches model's expected input

- **Masking Tests**:
  - Test mask generation produces correct shapes (action_type: 4, card_selection: 15, etc.)
  - Test masks correctly identify legal vs illegal actions
  - Test mask application strongly penalizes illegal actions (logits → -1e10)
  - Test masked predictions always select legal actions

- **Model Loading Tests**:
  - Test model loads successfully from valid .pth file
  - Test error handling for missing files
  - Test error handling for corrupted checkpoints
  - Test model is set to eval() mode
  - Test model can be loaded to CPU (torch.device('cpu'))

### Integration Tests
- **End-to-End Gameplay**:
  - Create game with MODEL player and play multiple turns
  - Verify all selected actions are legal
  - Verify game completes without crashes
  - Compare MODEL player performance against random baseline

- **Multi-Player Games**:
  - Test 2-player game: MODEL vs MCTS
  - Test 3-player game: MODEL vs 2 MCTS
  - Test 4-player game: 2 MODEL vs 2 MCTS
  - Verify all players can take turns correctly

- **Model Performance**:
  - Play 10 games with MODEL player
  - Record win rate, average score, average game length
  - Compare against MCTS baseline
  - Verify MODEL player makes reasonable strategic decisions

### Edge Cases
- **Empty Action Spaces**:
  - Test when no nobles are available (nobles mask all zeros)
  - Test when player has no reserved cards
  - Test when player has max tokens (must remove)
  - Test when deck is empty (can't reserve from top deck)

- **Action Type Masking**:
  - Test when only BUILD is legal
  - Test when only TAKE_TOKENS is legal
  - Test when BUILD and RESERVE are both illegal

- **Model Prediction Edge Cases**:
  - Test when model predicts illegal action (should fall back due to masking)
  - Test when model outputs are very close (tie-breaking)
  - Test when all legal actions have very low probability

- **Resource Constraints**:
  - Test model loading on CPU-only systems
  - Test memory usage doesn't grow over many turns
  - Test model inference latency is reasonable (< 1 second per turn)

## Acceptance Criteria
- [ ] AI player can load a trained model from a .pth file specified in IA string
- [ ] AI player generates legal moves 100% of the time (no illegal action errors)
- [ ] AI player completes full games without crashes or exceptions
- [ ] Feature extraction from live board state produces identical features to CSV-based training data
- [ ] Legal action masking correctly filters out impossible moves before prediction
- [ ] Multi-head predictions are correctly decoded into Move objects with all components (action, card, tokens, noble)
- [ ] AI player can play in 2, 3, and 4 player games
- [ ] AI player can compete against MCTS players in mixed games
- [ ] All unit tests pass with 100% coverage of core logic
- [ ] Integration tests demonstrate MODEL player completes at least 10 games successfully
- [ ] Code follows existing project conventions and style
- [ ] No modifications to core splendor game engine (board.py, move.py, cards.py, etc.) except player.py integration
- [ ] Documentation includes clear examples of how to use MODEL player

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

- `cd /home/apprentyr/projects/splendor_ia && uv run pytest tests/test_ai_player.py -v` - Run AI player unit tests
- `cd /home/apprentyr/projects/splendor_ia && uv run python scripts/test_model_player.py` - Run integration test with actual model
- `cd /home/apprentyr/projects/splendor_ia && uv run python scripts/play_with_model.py --games 5` - Play 5 test games with MODEL player
- `cd /home/apprentyr/projects/splendor_ia && uv run pytest tests/ -v` - Run all existing tests to ensure no regressions

## Notes

### Model Configuration
The model expects a configuration dict with these keys when loading from checkpoint:
- `input_dim`: Number of input features (382 CSV features + ~893 engineered features = ~1275 total, or whatever the actual trained model used)
- `trunk_dims`: List of hidden layer dimensions for shared trunk
- `head_dims`: List of hidden layer dimensions for prediction heads
- `dropout`: Dropout probability
- `num_classes`: Dict mapping head names to output class counts

The checkpoint should contain `model_state_dict` and optionally `config` for easy loading.

### Feature Extraction Optimization
When processing each turn:
1. Reconstruct board once from current game state
2. Pass board to both mask generation AND feature engineering (avoid redundant reconstruction)
3. This 2x speedup is critical for real-time gameplay

### Action Decoding Details
- **action_type**: 0=BUILD, 1=RESERVE, 2=TAKE2, 3=TAKE3
- **card_selection**: 0-11 visible cards (level 1: 0-3, level 2: 4-7, level 3: 8-11), 12-14 reserved cards
- **card_reservation**: 0-11 visible cards, 12-14 top deck (level 1/2/3)
- **gem_take3**: 26 classes (0=empty, 1-5=single, 6-15=pairs, 16-25=triples)
- **gem_take2**: 5 classes (0=white, 1=blue, 2=green, 3=red, 4=black)
- **noble**: 5 classes (0-4 corresponding to board.characters[0-4])
- **gems_removed**: 84 classes (all combinations of removing 0-3 gems from 6 types)

### Performance Considerations
- Model inference should take < 1 second per turn on CPU
- Memory usage should be constant (no leaks from repeated inference)
- Consider caching model in Player instance rather than reloading each turn
- Use `torch.no_grad()` during inference to save memory

### Future Enhancements (Not in Scope)
- GPU acceleration for model inference
- Batch inference for parallel game simulations
- Model ensembling (combine predictions from multiple models)
- MCTS + neural network hybrid (use model as evaluation function)
- Self-play training loop using MODEL player
