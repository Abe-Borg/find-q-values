# file: agents/Agent.py

from typing import Dict

class Agent:
    """Chess agent that selects moves based on game data."""
    
    def __init__(self, color: str):
        """
        Initialize agent with a color.
        
        Args:
            color: 'W' for white or 'B' for black
        """
        self.color = color

    def choose_action(self, game_moves: Dict[str, str], curr_turn: str) -> str:
        """
        Choose an action based on the current game data.
        
        This is a simple replay agent that returns the move from the game data.
        
        Args:
            game_moves: Dictionary containing moves for the current game
            curr_turn: Current turn identifier (e.g., 'W1', 'B1')
            
        Returns:
            The move in SAN format from the game data, or empty string if not found
        """
        return game_moves.get(curr_turn, '')