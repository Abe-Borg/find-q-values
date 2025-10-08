# project name: find-q-values
# file: environment/Environ.py

import chess
from utils import constants
from typing import Union, Dict, List, Tuple

class Environ:
    """Chess environment that manages board state and legal moves."""
    
    # Class-level turn list to avoid recreating for each instance
    _turn_list = None

    @classmethod
    def _get_turn_list(cls):
        """Initialize the turn list once and reuse it."""
        if cls._turn_list is None:
            max_turns = constants.max_num_turns_per_player * constants.num_players
            cls._turn_list = [f'{"W" if i % constants.num_players == 0 else "B"}{i // constants.num_players + 1}' 
                             for i in range(max_turns)]
        return cls._turn_list

    def __init__(self):
        """Initialize the environment with a chess board."""
        self.board = chess.Board()
        self.turn_list = self._get_turn_list()
        self.turn_index = 0

    def get_curr_state_and_legal_moves(self) -> Tuple[Dict[str, Union[int, str]], List[chess.Move]]:
        """
        Returns:
            Tuple of (state_dict, legal_moves_list)
        """
        curr_turn = self.turn_list[self.turn_index]
        legal_moves = list(self.board.legal_moves)
        
        # Return structured state without string conversions
        state = {
            'turn_index': self.turn_index, 
            'curr_turn': curr_turn,
            'white_to_move': self.board.turn == chess.WHITE,
        }
        
        return state, legal_moves

    def get_curr_turn(self) -> str:
        """Get the current turn identifier (e.g., 'W1', 'B1', etc.)."""
        return self.turn_list[self.turn_index]

    def update_curr_state(self) -> None:
        """Update the turn index after a move."""
        self.turn_index += 1

    def reset_environ(self) -> None:
        """Reset the environment to initial state."""
        self.board.reset()
        self.turn_index = 0

    def push_move_object(self, move: chess.Move) -> None:
        """
        Push a move object directly (fastest method).
        
        Args:
            move: chess.Move object to apply
        """
        self.board.push(move)
        self.update_curr_state()

    def convert_san_to_move_object(self, san_move: str) -> chess.Move:
        """
        Convert SAN string to Move object for the current position.
        
        Args:
            san_move: Move in Standard Algebraic Notation
            
        Returns:
            chess.Move object
            
        Raises:
            ValueError: If the move is invalid for the current position
        """
        return self.board.parse_san(san_move)

    def load_chessboard_for_q_est(self, analysis_results) -> None:
            move = analysis_results['anticipated_next_move']
            self.board.push(move)