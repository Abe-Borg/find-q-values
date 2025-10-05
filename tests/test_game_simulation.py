# tests/test_game_simulation.py

import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.game_simulation import play_one_game
from agents.Agent import Agent
from environment.Environ import Environ


class TestGameSimulation:
    """Test suite for game simulation and validation."""
    
    @pytest.fixture
    def agents_and_environ(self):
        """Create fresh agents and environment for each test."""
        w_agent = Agent('W')
        b_agent = Agent('B')
        environ = Environ()
        return w_agent, b_agent, environ
    
    @pytest.fixture
    def valid_short_game(self):
        """
        A valid 4-ply game: 1.e4 e5 2.Nf3 Nc6
        """
        return {
            'game_id': 'TestGame1',
            'ply_count': 4,
            'moves': {
                'W1': 'e4',
                'B1': 'e5',
                'W2': 'Nf3',
                'B2': 'Nc6'
            }
        }
    
    @pytest.fixture
    def valid_scholars_mate(self):
        """
        Scholar's mate: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6 4.Qxf7#
        """
        return {
            'game_id': 'TestGame2',
            'ply_count': 7,
            'moves': {
                'W1': 'e4',
                'B1': 'e5',
                'W2': 'Bc4',
                'B2': 'Nc6',
                'W3': 'Qh5',
                'B3': 'Nf6',
                'W4': 'Qxf7#',
            }
        } 
    
# ============= TESTS FOR VALID GAMES =============
    
    def test_valid_short_game(self, agents_and_environ, valid_short_game):
        """Test that a valid short game processes without errors."""
        w_agent, b_agent, environ = agents_and_environ
        
        result = play_one_game(
            game_id=valid_short_game['game_id'],
            ply_count=valid_short_game['ply_count'],
            moves=valid_short_game['moves'],
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        # Should return None for valid games
        assert result is None
        
        # Verify environment state after game
        assert environ.turn_index == valid_short_game['ply_count']
    
    def test_valid_checkmate_game(self, agents_and_environ, valid_scholars_mate):
        """Test that a game ending in checkmate is valid."""
        w_agent, b_agent, environ = agents_and_environ
        
        result = play_one_game(
            game_id=valid_scholars_mate['game_id'],
            ply_count=valid_scholars_mate['ply_count'],
            moves=valid_scholars_mate['moves'],
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        assert result is None
        assert environ.turn_index == 7
        assert environ.board.is_checkmate()
        assert environ.turn_index == valid_scholars_mate['ply_count']
    
    def test_game_shorter_than_plycount(self, agents_and_environ):
        """Test a game that ends before reaching PlyCount (legitimate early termination)."""
        w_agent, b_agent, environ = agents_and_environ
        
        # Game ends at move 8 (checkmate) but PlyCount says 200
        # This should be valid - game naturally ended early
        moves = {
            'W1': 'e4',
            'B1': 'e5',
            'W2': 'Bc4',
            'B2': 'Nc6',
            'W3': 'Qh5',
            'B3': 'Nf6',
            'W4': 'Qxf7#',
        }
        
        result = play_one_game(
            game_id='EarlyEnd',
            ply_count=200,  # Claims 200 plies but ends at 8
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        assert result is None
        assert environ.board.is_checkmate() 

# ============= TESTS FOR CORRUPTED GAMES =============
    
    def test_illegal_move(self, agents_and_environ):
        """Test that an illegal move flags the game as corrupted."""
        w_agent, b_agent, environ = agents_and_environ
        
        # e5 is illegal as white's first move (pawn can't move there)
        moves = {
            'W1': 'e5',  # Illegal - pawn can't jump two squares to e5 from e2
            'B1': 'e5',
        }
        
        result = play_one_game(
            game_id='IllegalMove',
            ply_count=2,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        assert result == 'IllegalMove'
    
    def test_invalid_san_notation(self, agents_and_environ):
        """Test that invalid SAN notation flags the game as corrupted."""
        w_agent, b_agent, environ = agents_and_environ
        
        moves = {
            'W1': 'e4',
            'B1': 'XYZ123',  # Invalid SAN
        }
        
        result = play_one_game(
            game_id='InvalidSAN',
            ply_count=2,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        assert result == 'InvalidSAN'
    
    def test_move_for_wrong_piece(self, agents_and_environ):
        """Test that moving a piece that doesn't exist/can't make that move is caught."""
        w_agent, b_agent, environ = agents_and_environ
        
        moves = {
            'W1': 'e4',
            'B1': 'e5',
            'W2': 'Qh5',
            'B2': 'Nc6',
            'W3': 'Bd5',  # Bishop on f1 cannot reach d5 in this position
        }
        
        result = play_one_game(
            game_id='WrongPiece',
            ply_count=6,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        assert result == 'WrongPiece'
    
    def test_empty_move_in_middle(self, agents_and_environ):
        """Test that empty moves in the middle of a game flag it as corrupted."""
        w_agent, b_agent, environ = agents_and_environ
        
        moves = {
            'W1': 'e4',
            'B1': 'e5',
            'W2': '',  # Empty move in the middle
            'B2': 'Nf6',
        }
        
        result = play_one_game(
            game_id='EmptyMiddle',
            ply_count=4,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        assert result == 'EmptyMiddle'
    
    def test_nan_move_in_middle(self, agents_and_environ):
        """Test that NaN moves in the middle of a game flag it as corrupted."""
        w_agent, b_agent, environ = agents_and_environ
        
        moves = {
            'W1': 'e4',
            'B1': pd.NA,  # NaN move in the middle
            'W2': 'Nf3',
        }
        
        result = play_one_game(
            game_id='NaNMiddle',
            ply_count=3,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        assert result == 'NaNMiddle' 

# ============= EDGE CASE TESTS =============
    
    def test_zero_ply_game(self, agents_and_environ):
        """Test handling of a game with PlyCount=0."""
        w_agent, b_agent, environ = agents_and_environ
        
        moves = {}
        
        result = play_one_game(
            game_id='ZeroPly',
            ply_count=0,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        # Should exit immediately without errors
        assert result is None
        assert environ.turn_index == 0
    
    def test_environment_reset_between_games(self, agents_and_environ):
        """Test that environment properly resets between games."""
        w_agent, b_agent, environ = agents_and_environ
        
        # Play first game
        moves1 = {
            'W1': 'e4',
            'B1': 'e5',
        }
        
        play_one_game('Game1', 2, moves1, w_agent, b_agent, environ)
        
        # Reset environment
        environ.reset_environ()
        
        # Verify reset worked
        assert environ.turn_index == 0
        assert environ.board.fen() == 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        
        # Play second game - should work fine
        moves2 = {
            'W1': 'd4',
            'B1': 'd5',
        }
        
        result = play_one_game('Game2', 2, moves2, w_agent, b_agent, environ)
        assert result is None 

# ============= INTEGRATION TESTS FOR play_games() =============
    
    def test_play_games_empty_dataframe(self):
        """Test that empty DataFrame is handled gracefully."""
        from training.game_simulation import play_games
        
        empty_df = pd.DataFrame()
        corrupted = play_games(empty_df, pool=None)
        
        assert corrupted == []
    
    def test_play_games_all_valid_games(self):
        """Test processing a DataFrame with only valid games."""
        from training.game_simulation import play_games
        
        # Create DataFrame with 3 valid games
        data = {
            'PlyCount': [4, 4, 6],
            'W1': ['e4', 'd4', 'Nf3'],
            'B1': ['e5', 'd5', 'Nf6'],
            'W2': ['Nf3', 'Nf3', 'd4'],
            'B2': ['Nc6', 'Nf6', 'e6'],
            'W3': ['', '', 'c4'],
            'B3': ['', '', 'Bb4+'],
        }
        
        df = pd.DataFrame(data, index=['Game1', 'Game2', 'Game3'])
        corrupted = play_games(df, pool=None)
        
        assert len(corrupted) == 0
    
    def test_play_games_all_corrupted_games(self):
        """Test processing a DataFrame where all games are corrupted."""
        from training.game_simulation import play_games
        
        # Create DataFrame with 3 corrupted games
        data = {
            'PlyCount': [2, 2, 2],
            'W1': ['e5', 'XYZ', 'e4'],  # First two are invalid
            'B1': ['e5', 'e5', ''],     # Third has empty move
        }
        
        df = pd.DataFrame(data, index=['Bad1', 'Bad2', 'Bad3'])
        corrupted = play_games(df, pool=None)
        
        assert len(corrupted) == 3
        assert set(corrupted) == {'Bad1', 'Bad2', 'Bad3'}
    
    def test_play_games_mixed_valid_and_corrupted(self):
        """Test processing a DataFrame with mix of valid and corrupted games."""
        from training.game_simulation import play_games
        
        data = {
            'PlyCount': [4, 2, 4, 2],
            'W1': ['e4', 'e5', 'e4', 'd4'],      # Game2 has illegal first move
            'B1': ['e5', 'e5', '', 'd5'],        # Game3 has empty move
            'W2': ['Nf3', '', 'Nf3', 'Nf3'],
            'B2': ['Nc6', '', 'Nc6', 'Nf6'],
        }
        
        df = pd.DataFrame(data, index=['Good1', 'Bad1', 'Bad2', 'Good2'])
        corrupted = play_games(df, pool=None)
        
        assert len(corrupted) == 2
        assert set(corrupted) == {'Bad1', 'Bad2'}
    
    def test_play_games_missing_plycount_column(self):
        """Test that missing PlyCount column returns all games as corrupted."""
        from training.game_simulation import play_games
        
        # DataFrame without PlyCount column
        data = {
            'W1': ['e4', 'd4'],
            'B1': ['e5', 'd5'],
        }
        
        df = pd.DataFrame(data, index=['Game1', 'Game2'])
        corrupted = play_games(df, pool=None)
        
        # Should return all game IDs as corrupted
        assert len(corrupted) == 2
        assert set(corrupted) == {'Game1', 'Game2'} 

# ============= REAL-WORLD SCENARIO TESTS =============
    
    def test_long_valid_game(self, agents_and_environ):
        """Test a longer game (50+ plies) to ensure performance."""
        w_agent, b_agent, environ = agents_and_environ
        
        # Create a reasonable 50-ply game
        moves = {}
        # Simple back-and-forth moves that are all legal
        moves['W1'] = 'e4'
        moves['B1'] = 'e5'
        moves['W2'] = 'Nf3'
        moves['B2'] = 'Nc6'
        moves['W3'] = 'Bb5'
        moves['B3'] = 'a6'
        moves['W4'] = 'Ba4'
        moves['B4'] = 'Nf6'
        moves['W5'] = 'O-O'
        moves['B5'] = 'Be7'
        moves['W6'] = 'Re1'
        moves['B6'] = 'b5'
        moves['W7'] = 'Bb3'
        moves['B7'] = 'd6'
        moves['W8'] = 'c3'
        moves['B8'] = 'O-O'
        moves['W9'] = 'h3'
        moves['B9'] = 'Na5'
        moves['W10'] = 'Bc2'
        moves['B10'] = 'c5'
        moves['W11'] = 'd4'
        moves['B11'] = 'Qc7'
        moves['W12'] = 'Nbd2'
        moves['B12'] = 'cxd4'
        moves['W13'] = 'cxd4'
        moves['B13'] = 'Nc6'
        moves['W14'] = 'd5'
        moves['B14'] = 'Nb8'
        moves['W15'] = 'Nf1'
        moves['B15'] = 'Nbd7'
        moves['W16'] = 'N3h2'
        moves['B16'] = 'Bb7'
        moves['W17'] = 'Ng4'
        moves['B17'] = 'Rac8'
        moves['W18'] = 'Nfe3'
        moves['B18'] = 'Rfe8'
        moves['W19'] = 'Bd2'
        moves['B19'] = 'Nf8'
        moves['W20'] = 'Qf3'
        moves['B20'] = 'N6d7'
        moves['W21'] = 'Bb3'
        moves['B21'] = 'Nc5'
        moves['W22'] = 'Bc2'
        moves['B22'] = 'Nfd7'
        moves['W23'] = 'Nf5'
        moves['B23'] = 'Bf8'
        moves['W24'] = 'Rac1'
        moves['B24'] = 'g6'
        moves['W25'] = 'Nfh6+'
        
        result = play_one_game(
            game_id='LongGame',
            ply_count=49,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        assert result is None
        assert environ.turn_index == 49
    
    def test_game_with_captures_and_checks(self, agents_and_environ):
        """Test game with captures and check notation."""
        w_agent, b_agent, environ = agents_and_environ
        
        moves = {
            'W1': 'e4',
            'B1': 'd5',
            'W2': 'exd5',  # Capture
            'B2': 'Qxd5',  # Capture
            'W3': 'Nc3',
            'B3': 'Qd8',
            'W4': 'd4',
            'B4': 'Nf6',
            'W5': 'Nf3',
            'B5': 'Bg4',
            'W6': 'Bb5+',  # Check
        }
        
        result = play_one_game(
            game_id='CapturesChecks',
            ply_count=11,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        assert result is None
    
    def test_game_with_castling(self, agents_and_environ):
        """Test game with both kingside and queenside castling."""
        w_agent, b_agent, environ = agents_and_environ
        
        moves = {
            'W1': 'e4',
            'B1': 'e5',
            'W2': 'Nf3',
            'B2': 'Nc6',
            'W3': 'Bc4',
            'B3': 'Nf6',
            'W4': 'O-O',  # Kingside castling
            'B4': 'Bc5',
            'W5': 'd3',
            'B5': 'd6',
            'W6': 'Bg5',
            'B6': 'O-O',  # Kingside castling
        }
        
        result = play_one_game(
            game_id='Castling',
            ply_count=12,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        assert result is None
    

# ============= PLYCOUNT EDGE CASES =============
    
    def test_plycount_mismatch_game_ends_early(self, agents_and_environ):
        """Test when game naturally ends before reaching PlyCount."""
        w_agent, b_agent, environ = agents_and_environ
        
        # Checkmate at ply 8, but PlyCount says 100
        moves = {
            'W1': 'e4',
            'B1': 'e5',
            'W2': 'Bc4',
            'B2': 'Nc6',
            'W3': 'Qh5',
            'B3': 'Nf6',
            'W4': 'Qxf7#',
        }
        
        result = play_one_game(
            game_id='EarlyCheckmate',
            ply_count=100,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        # Should be valid - game ended naturally
        assert result is None
        assert environ.board.is_checkmate()
    
    def test_plycount_exceeds_actual_moves(self, agents_and_environ):
        """Test when PlyCount is larger than actual moves provided."""
        w_agent, b_agent, environ = agents_and_environ
        
        # Only 4 moves provided but PlyCount says 10
        moves = {
            'W1': 'e4',
            'B1': 'e5',
            'W2': 'Nf3',
            'B2': 'Nc6',
            # No more moves after this
        }
        
        result = play_one_game(
            game_id='TooFewMoves',
            ply_count=10,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        # Should be flagged as corrupted - empty moves detected
        assert result == 'TooFewMoves'
    
    def test_very_long_game_200_plies(self, agents_and_environ):
        """Test maximum length game (200 plies per side = 400 total)."""
        w_agent, b_agent, environ = agents_and_environ
        
        # Create a long repetitive game
        moves = {}
        # Knights moving back and forth
        for i in range(1, 101):  # 100 moves per side
            moves[f'W{i}'] = 'Nf3' if i == 1 else ('Ng1' if i % 2 == 0 else 'Nf3')
            moves[f'B{i}'] = 'Nf6' if i == 1 else ('Ng8' if i % 2 == 0 else 'Nf6')
        
        result = play_one_game(
            game_id='MaxLength',
            ply_count=200,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        assert result is None
    
    def test_inconsistent_move_keys(self, agents_and_environ):
        """Test game with missing move keys (gaps in the sequence)."""
        w_agent, b_agent, environ = agents_and_environ
        
        # Missing B2 in the sequence
        moves = {
            'W1': 'e4',
            'B1': 'e5',
            'W2': 'Nf3',
            # B2 is missing
            'W3': 'Bc4',
            'B3': 'Nc6',
        }
        
        result = play_one_game(
            game_id='MissingKeys',
            ply_count=6,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )
        
        # Missing move returns empty string, which should flag as corrupted
        assert result == 'MissingKeys'

    # ============== 

    def test_promotion_by_capture_and_check(self, agents_and_environ):
        """
        White promotes by capture (bxa8=Q), then Black promotes with check (gxf1=Q+).
        Sequence is legal from the initial position.
        """
        from training.game_simulation import play_one_game
        w_agent, b_agent, environ = agents_and_environ

        moves = {
            'W1': 'a4',
            'B1': 'h5',
            'W2': 'a5',
            'B2': 'h4',
            'W3': 'a6',
            'B3': 'h3',
            'W4': 'axb7',     # a6xb7
            'B4': 'hxg2',     # h3xg2
            'W5': 'bxa8=Q',   # b7xa8=Q
            'B5': 'gxf1=Q+',  # g2xf1=Q+
        }

        result = play_one_game(
            game_id='PromotionBothSides',
            ply_count=10,        # exactly the number of plies we supply
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )

        assert result is None
        assert environ.turn_index == 10  # all provided plies consumed

    def test_en_passant(self, agents_and_environ):
        """
        Classic en passant: 1.e4 d5 2.exd5 c5 3.dxc6 (e.p.)
        Note: SAN usually omits the 'e.p.' suffix; python-chess accepts 'dxc6'.
        """
        from training.game_simulation import play_one_game
        w_agent, b_agent, environ = agents_and_environ

        moves = {
            'W1': 'e4',
            'B1': 'd5',
            'W2': 'exd5',
            'B2': 'c5',       # two-square pawn push enabling en passant
            'W3': 'dxc6',     # en passant capture
        }

        result = play_one_game(
            game_id='EnPassant',
            ply_count=5,
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )

        assert result is None
        assert environ.turn_index == 5

    def test_queenside_castling(self, agents_and_environ):
        """
        White long castles after clearing b1/c1/d1 and avoiding attacked transit squares.
        Line: 1.d4 d5 2.Nc3 Nc6 3.Bf4 Bf5 4.Qd2 Qd7 5.O-O-O
        """
        from training.game_simulation import play_one_game
        w_agent, b_agent, environ = agents_and_environ

        moves = {
            'W1': 'd4',
            'B1': 'd5',
            'W2': 'Nc3',   # clears b1
            'B2': 'Nc6',
            'W3': 'Bf4',   # clears c1
            'B3': 'Bf5',
            'W4': 'Qd2',   # clears d1
            'B4': 'Qd7',
            'W5': 'O-O-O', # queenside castle
        }

        result = play_one_game(
            game_id='CastleLong',
            ply_count=9,  # up to W5
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )

        assert result is None
        assert environ.turn_index == 9

    def test_extra_moves_beyond_plycount_are_ignored(self, agents_and_environ):
        """
        Case B: Data includes legal moves beyond PlyCount.
        Our loop stops at ply_count and treats the game as valid.
        """
        from training.game_simulation import play_one_game
        w_agent, b_agent, environ = agents_and_environ

        # Provide 4 legal plies but set PlyCount=2
        moves = {
            'W1': 'e4',
            'B1': 'e5',
            'W2': 'Nf3',
            'B2': 'Nc6',
        }

        result = play_one_game(
            game_id='ExtraMovesBeyondPlyCount',
            ply_count=2,   # stop after W1,B1
            moves=moves,
            w_agent=w_agent,
            b_agent=b_agent,
            environ=environ
        )

        # Valid: we simply don't consume the extra moves
        assert result is None
        assert environ.turn_index == 2
