"""Unit tests for AI player functionality.

This module tests:
- Action decoder functions
- Board to CSV conversion
- Feature extraction
- Mask generation
- Model inference (with mock model)
- End-to-end prediction flow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from splendor.action_decoder import (
    decode_gem_take3,
    decode_gem_take2,
    decode_gems_removed,
    decode_card_selection,
    decode_card_reservation,
    decode_noble_selection,
    decode_predictions_to_move
)
from splendor.ai_player import board_to_csv_row, apply_legal_move_masking
from splendor.board import Board
from splendor.constants import BUILD, RESERVE, WHITE, BLUE, GREEN, RED, BLACK
from splendor.move import Move
from imitation_learning.utils import generate_all_masks_from_row


class TestActionDecoder:
    """Test action decoding functions."""

    def test_decode_gem_take3_empty(self):
        """Test decoding empty gem selection (class 0)."""
        tokens = decode_gem_take3(0)
        assert tokens == [0, 0, 0, 0, 0, 0], "Class 0 should be no gems"

    def test_decode_gem_take3_single(self):
        """Test decoding single gem selections (classes 1-5)."""
        # Class 1: white
        assert decode_gem_take3(1) == [1, 0, 0, 0, 0, 0]
        # Class 2: blue
        assert decode_gem_take3(2) == [0, 1, 0, 0, 0, 0]
        # Class 5: black
        assert decode_gem_take3(5) == [0, 0, 0, 0, 1, 0]

    def test_decode_gem_take3_pairs(self):
        """Test decoding two-gem combinations (classes 6-15)."""
        # Class 6: white + blue
        assert decode_gem_take3(6) == [1, 1, 0, 0, 0, 0]

    def test_decode_gem_take3_triples(self):
        """Test decoding three-gem combinations (classes 16-25)."""
        # Class 16: white + blue + green
        assert decode_gem_take3(16) == [1, 1, 1, 0, 0, 0]

    def test_decode_gem_take2(self):
        """Test decoding gem_take2 selections."""
        assert decode_gem_take2(0) == [2, 0, 0, 0, 0, 0]  # White
        assert decode_gem_take2(1) == [0, 2, 0, 0, 0, 0]  # Blue
        assert decode_gem_take2(4) == [0, 0, 0, 0, 2, 0]  # Black

    def test_decode_gems_removed_no_removal(self):
        """Test decoding no gem removal (class 0)."""
        assert decode_gems_removed(0) == [0, 0, 0, 0, 0, 0]

    def test_decode_gems_removed_single(self):
        """Test decoding single gem removals."""
        # Class 1: remove 1 white
        assert decode_gems_removed(1) == [1, 0, 0, 0, 0, 0]

    def test_decode_card_selection_visible(self):
        """Test decoding visible card selections."""
        # Create test board
        board = Board(2, [None, None])

        # Test level 1, position 0 (index 0)
        card = decode_card_selection(0, board)
        assert card is not None
        assert card.lvl == 1

        # Test level 2, position 0 (index 4)
        card = decode_card_selection(4, board)
        assert card is not None
        assert card.lvl == 2

        # Test level 3, position 0 (index 8)
        card = decode_card_selection(8, board)
        assert card is not None
        assert card.lvl == 3

    def test_decode_card_selection_reserved(self):
        """Test decoding reserved card selections."""
        board = Board(2, [None, None])

        # Reserve a card for current player
        if len(board.displayedCards[0]) > 0:
            card_to_reserve = board.displayedCards[0][0]
            board.players[board.currentPlayer].reserved.append(card_to_reserve)

            # Test reserved card index (12)
            decoded = decode_card_selection(12, board)
            assert decoded is card_to_reserve

    def test_decode_card_reservation_visible(self):
        """Test decoding visible card reservations."""
        board = Board(2, [None, None])

        # Test visible card
        result = decode_card_reservation(0, board)
        assert result is not None

    def test_decode_card_reservation_top_deck(self):
        """Test decoding top deck reservations."""
        board = Board(2, [None, None])

        # Test top deck level 1 (index 12)
        result = decode_card_reservation(12, board)
        if len(board.decks[0]) > 0:
            assert result == 1  # Should return deck level
        else:
            assert result is None

    def test_decode_noble_selection(self):
        """Test decoding noble selections."""
        board = Board(2, [None, None])

        # Test valid noble index
        if len(board.characters) > 0:
            noble = decode_noble_selection(0, board)
            assert noble is not None
            assert noble == board.characters[0]

        # Test invalid index
        noble = decode_noble_selection(99, board)
        assert noble is None

    def test_decode_predictions_to_move_build(self):
        """Test decoding BUILD action predictions."""
        board = Board(2, [None, None])

        predictions = {
            'action_type': 0,  # BUILD
            'card_selection': 0,
            'card_reservation': 0,
            'gem_take3': 0,
            'gem_take2': 0,
            'noble': 0,
            'gems_removed': 0
        }

        move = decode_predictions_to_move(predictions, board)
        assert move.actionType == BUILD
        assert move.action is not None  # Should have selected a card

    def test_decode_predictions_to_move_reserve(self):
        """Test decoding RESERVE action predictions."""
        board = Board(2, [None, None])

        predictions = {
            'action_type': 1,  # RESERVE
            'card_selection': 0,
            'card_reservation': 0,
            'gem_take3': 0,
            'gem_take2': 0,
            'noble': 0,
            'gems_removed': 0
        }

        move = decode_predictions_to_move(predictions, board)
        assert move.actionType == RESERVE
        assert move.action is not None

    def test_decode_predictions_to_move_take2(self):
        """Test decoding TAKE2 action predictions."""
        board = Board(2, [None, None])

        predictions = {
            'action_type': 2,  # TAKE2
            'card_selection': 0,
            'card_reservation': 0,
            'gem_take3': 0,
            'gem_take2': 0,  # White
            'noble': 0,
            'gems_removed': 0
        }

        move = decode_predictions_to_move(predictions, board)
        assert move.actionType == 2
        assert move.action == [2, 0, 0, 0, 0, 0]

    def test_decode_predictions_to_move_take3(self):
        """Test decoding TAKE3 action predictions."""
        board = Board(2, [None, None])

        predictions = {
            'action_type': 3,  # TAKE3
            'card_selection': 0,
            'card_reservation': 0,
            'gem_take3': 16,  # White + Blue + Green
            'gem_take2': 0,
            'noble': 0,
            'gems_removed': 0
        }

        move = decode_predictions_to_move(predictions, board)
        assert move.actionType == 3
        assert move.action == [1, 1, 1, 0, 0, 0]


class TestBoardToCSV:
    """Test board to CSV conversion."""

    def test_board_to_csv_row_basic(self):
        """Test basic board to CSV conversion."""
        board = Board(2, [None, None])
        row = board_to_csv_row(board)

        # Check basic fields
        assert row['num_players'] == 2.0
        assert row['current_player'] == 0.0
        assert isinstance(row['turn_number'], float)

        # Check board tokens
        assert 'gems_board_white' in row
        assert 'gems_board_gold' in row

        # Check deck counts
        assert 'deck_level1_remaining' in row
        assert row['deck_level1_remaining'] == len(board.decks[0])

    def test_board_to_csv_row_nobles(self):
        """Test noble encoding in CSV."""
        board = Board(3, [None, None, None])
        row = board_to_csv_row(board)

        # Check nobles are encoded
        for i in range(len(board.characters)):
            assert f'noble{i}_vp' in row
            assert row[f'noble{i}_vp'] == board.characters[i].vp

        # Check padding for missing nobles
        for i in range(len(board.characters), 5):
            assert f'noble{i}_vp' in row
            assert np.isnan(row[f'noble{i}_vp'])

    def test_board_to_csv_row_cards(self):
        """Test card encoding in CSV."""
        board = Board(2, [None, None])
        row = board_to_csv_row(board)

        # Check first card is encoded
        assert 'card0_vp' in row
        assert 'card0_level' in row
        assert 'card0_cost_white' in row

    def test_board_to_csv_row_players(self):
        """Test player encoding with rotation."""
        board = Board(3, [None, None, None])
        board.currentPlayer = 1  # Make player 1 current

        row = board_to_csv_row(board)

        # Current player should be rotated to index 0
        assert row['current_player'] == 0.0

        # Player 0 in CSV should correspond to player 1 in board
        assert 'player0_vp' in row


class TestMasking:
    """Test legal move masking."""

    def test_apply_legal_move_masking(self):
        """Test masking application to logits."""
        # Create dummy logits
        logits = {
            'action_type': torch.tensor([0.5, 1.2, 0.8, 0.3])
        }

        # Create mask (only actions 0 and 2 are legal)
        masks = {
            'action_type': np.array([1, 0, 1, 0])
        }

        masked = apply_legal_move_masking(logits, masks)

        # Check that illegal actions have very negative logits
        assert masked['action_type'][1] < -1e9
        assert masked['action_type'][3] < -1e9

        # Check that legal actions are unchanged
        assert masked['action_type'][0] == 0.5
        assert masked['action_type'][2] == 0.8

    def test_masked_prediction_selects_legal(self):
        """Test that masked predictions select legal actions."""
        logits = {
            'action_type': torch.tensor([0.5, 10.0, 0.8, 0.3])  # Class 1 has highest score
        }

        masks = {
            'action_type': np.array([1, 0, 1, 0])  # But class 1 is illegal
        }

        masked = apply_legal_move_masking(logits, masks)
        predicted = masked['action_type'].argmax()

        # Should select class 2 (legal and highest among legal)
        assert predicted == 2


class TestMaskGeneration:
    """Test mask generation from board state."""

    def test_generate_masks_from_board(self):
        """Test generating masks from actual board state."""
        board = Board(2, [None, None])
        row = board_to_csv_row(board)

        masks = generate_all_masks_from_row(row, board)

        # Check all required masks are present
        assert 'action_type' in masks
        assert 'card_selection' in masks
        assert 'card_reservation' in masks
        assert 'gem_take3' in masks
        assert 'gem_take2' in masks
        assert 'noble' in masks
        assert 'gems_removed' in masks

        # Check mask shapes
        assert masks['action_type'].shape == (4,)
        assert masks['card_selection'].shape == (15,)
        assert masks['gem_take3'].shape == (26,)
        assert masks['gem_take2'].shape == (5,)

        # Check that at least some actions are legal
        assert np.sum(masks['action_type']) > 0, "At least one action type should be legal"

    def test_masks_match_legal_moves(self):
        """Test that masks correctly represent legal moves."""
        board = Board(2, [None, None])
        row = board_to_csv_row(board)

        # Get legal moves from board
        legal_moves = board.getMoves()

        # Get masks
        masks = generate_all_masks_from_row(row, board)

        # Check that if there are BUILD moves, BUILD is in action_type mask
        has_build = any(m.actionType == BUILD for m in legal_moves)
        if has_build:
            assert masks['action_type'][0] == 1, "BUILD should be legal in mask"


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_board_round_trip(self):
        """Test converting board to CSV and back maintains state."""
        from utils.state_reconstruction import reconstruct_board_from_csv_row

        board = Board(2, [None, None])
        row = board_to_csv_row(board)

        # Reconstruct board
        reconstructed = reconstruct_board_from_csv_row(row)

        # Check basic properties match
        assert len(reconstructed.players) == len(board.players)
        assert reconstructed.currentPlayer == 0  # Always 0 in rotated view
        assert len(reconstructed.characters) == len(board.characters)

    @pytest.mark.skipif(
        not Path("data/models/baseline/best_model.pth").exists(),
        reason="No trained model available"
    )
    def test_model_inference_with_real_model(self):
        """Test model inference with a real trained model."""
        from splendor.ai_player import ModelPlayer

        model_path = "data/models/baseline/best_model.pth"
        player = ModelPlayer(model_path)

        board = Board(2, [None, None])
        move = player.get_action(board)

        # Check move is valid
        assert move is not None
        assert isinstance(move, Move)
        assert move.actionType in [BUILD, RESERVE, 2, 3]

        # Check move is legal
        legal_moves = board.getMoves()
        # Note: Can't directly check if move is in legal_moves due to object identity
        # but we can check it doesn't crash when executed
        board_copy = board.clone()
        board_copy.doMove(move)  # Should not raise exception


def test_action_decoder_module():
    """Test that action decoder module can be imported and run."""
    import splendor.action_decoder as decoder
    # Module should have all required functions
    assert hasattr(decoder, 'decode_gem_take3')
    assert hasattr(decoder, 'decode_gem_take2')
    assert hasattr(decoder, 'decode_gems_removed')
    assert hasattr(decoder, 'decode_predictions_to_move')


def test_ai_player_module():
    """Test that AI player module can be imported."""
    import splendor.ai_player as ai_player
    assert hasattr(ai_player, 'ModelPlayer')
    assert hasattr(ai_player, 'load_model')
    assert hasattr(ai_player, 'board_to_csv_row')
    assert hasattr(ai_player, 'get_model_action')


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
