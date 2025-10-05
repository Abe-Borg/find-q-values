# file: training/game_simulation.py

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import time
from agents.Agent import Agent
from utils import game_settings
from environment.Environ import Environ
from multiprocessing import Pool, cpu_count, shared_memory
import logging
import platform

# Determine if we're running on Windows
IS_WINDOWS = platform.system() == 'Windows'

# Logger setup
logger = logging.getLogger("game_simulation")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(str(game_settings.game_simulation_logger_filepath))
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class AdaptiveChunker:
    """Creates balanced chunks of games for parallel processing."""
    
    def __init__(self):
        self.calibrated_moves_per_second = 65000  # Based on your measured performance
    
    def create_balanced_chunks(self, 
                             game_indices: List[str], 
                             chess_data: pd.DataFrame, 
                             num_processes: int) -> List[List[str]]:
        """
        Create balanced chunks based on estimated processing times.
        
        Args:
            game_indices: List of game IDs to process
            chess_data: DataFrame containing game data
            num_processes: Number of worker processes
            
        Returns:
            List of chunks, where each chunk is a list of game IDs
        """
        if not game_indices:
            return []
        
        total_games = len(game_indices)
        
        # Auto-determine optimal chunk size based on total games
        if total_games <= 100:
            min_games_per_chunk = max(5, total_games // (num_processes * 2))
        elif total_games <= 1000:
            min_games_per_chunk = max(20, total_games // (num_processes * 3))
        else:
            min_games_per_chunk = max(100, total_games // (num_processes * 4))
        
        # Calculate optimal number of chunks
        max_chunks_by_workers = num_processes
        max_chunks_by_min_size = max(1, total_games // min_games_per_chunk)
        optimal_chunks = min(max_chunks_by_workers, max_chunks_by_min_size, total_games)
        
        print(f"Chunking {total_games} games into {optimal_chunks} chunks "
              f"(min {min_games_per_chunk} games/chunk)")
        
        # If very few games, one per chunk
        if total_games <= optimal_chunks:
            return [[game] for game in game_indices]
        
        # Estimate processing time for each game
        time_estimates = {}
        for game_id in game_indices:
            try:
                ply_count = int(chess_data.loc[game_id, 'PlyCount'])  # type: ignore
                time_estimates[game_id] = ply_count / self.calibrated_moves_per_second
            except (KeyError, TypeError, ValueError):
                time_estimates[game_id] = 0.001  # Default estimate
        
        # Sort games by estimated time (longest first for better balancing)
        sorted_games = sorted(game_indices, 
                            key=lambda g: time_estimates.get(g, 0), 
                            reverse=True)
        
        # Create balanced chunks using greedy algorithm
        chunks = [[] for _ in range(optimal_chunks)]
        chunk_times = [0.0] * optimal_chunks
        
        for game in sorted_games:
            # Assign to chunk with lowest total time
            min_idx = chunk_times.index(min(chunk_times))
            chunks[min_idx].append(game)
            chunk_times[min_idx] += time_estimates.get(game, 0)
        
        # Remove empty chunks
        return [chunk for chunk in chunks if chunk]

# Global chunker instance
chunker = AdaptiveChunker()

def init_worker(i):
    """Initialize worker process."""
    return f"Worker {i} initialized"

def warm_up_workers(pool, num_workers):
    """Initialize worker processes with warm-up tasks."""
    print("Warming up worker processes...")
    results = pool.map(init_worker, range(num_workers))
    for result in results:
        print(f"  {result}")

def create_shared_data(chess_data: pd.DataFrame) -> Dict:
    """
    Create shared memory representation of DataFrame.
    Uses different approach for Windows vs Unix systems.
    """
    if IS_WINDOWS:
        # On Windows, pass the DataFrame directly
        return {
            'windows_mode': True,
            'data': chess_data,
        }
    else:
        # Unix-based systems use shared memory for efficiency
        game_indices = list(chess_data.index)
        
        # Convert to numpy array for shared memory
        indices_arr = np.array(game_indices, dtype=object)
        indices_shm = shared_memory.SharedMemory(create=True, size=indices_arr.nbytes)
        indices_shared = np.ndarray(indices_arr.shape, dtype=indices_arr.dtype, 
                                   buffer=indices_shm.buf)
        indices_shared[:] = indices_arr[:]
        
        # Create shared memory for PlyCount
        ply_counts = chess_data['PlyCount'].values
        ply_counts = np.asarray(ply_counts, dtype=np.int64)  # Ensure standard numpy dtype
        ply_shm = shared_memory.SharedMemory(create=True, size=ply_counts.nbytes)
        ply_shared = np.ndarray(ply_counts.shape, dtype=ply_counts.dtype, buffer=ply_shm.buf)
        ply_shared[:] = ply_counts[:]
        
        # Create shared memory for move columns
        move_columns = {}
        for col in chess_data.columns:
            if col != 'PlyCount':
                col_data = chess_data[col].values
                col_data = np.asarray(col_data, dtype=object)  # Ensure standard numpy dtype (strings are object type)
                col_shm = shared_memory.SharedMemory(create=True, size=col_data.nbytes)
                col_shared = np.ndarray(col_data.shape, dtype=col_data.dtype, buffer=col_shm.buf)
                col_shared[:] = col_data[:]
        
                move_columns[col] = {
                    'shm_name': col_shm.name,
                    'shape': col_data.shape,
                    'dtype': str(col_data.dtype)
                }
        
        return {
            'windows_mode': False,
            'indices': {
                'shm_name': indices_shm.name,
                'shape': indices_arr.shape,
                'dtype': str(indices_arr.dtype)
            },
            'ply_counts': {
                'shm_name': ply_shm.name,
                'shape': ply_counts.shape,
                'dtype': str(ply_counts.dtype)
            },
            'move_columns': move_columns
        }

def cleanup_shared_data(shared_data: Dict) -> None:
    """Clean up shared memory resources."""
    if shared_data.get('windows_mode', False):
        return
    
    try:
        # Clean up indices shared memory
        indices_shm = shared_memory.SharedMemory(name=shared_data['indices']['shm_name'])
        indices_shm.close()
        indices_shm.unlink()
        
        # Clean up ply_counts shared memory
        ply_shm = shared_memory.SharedMemory(name=shared_data['ply_counts']['shm_name'])
        ply_shm.close()
        ply_shm.unlink()
        
        # Clean up all move column shared memory
        for col_info in shared_data['move_columns'].values():
            col_shm = shared_memory.SharedMemory(name=col_info['shm_name'])
            col_shm.close()
            col_shm.unlink()
    except FileNotFoundError:
        pass  # Already cleaned up

def play_games(chess_data: pd.DataFrame, pool=None) -> List[str]:
    """
    Process all games in the provided DataFrame.
    
    Args:
        chess_data: DataFrame containing chess games
        pool: Optional persistent process pool to use
        
    Returns:
        List of corrupted game IDs
    """
    if chess_data.empty:
        return []
    
    # Validate required columns
    if 'PlyCount' not in chess_data.columns:
        logger.critical("Missing required column: PlyCount")
        return list(chess_data.index)
    
    start_time = time.time()
    
    # Create shared memory representation
    shared_data = create_shared_data(chess_data)
    
    try:
        # Process games in parallel
        game_indices = list(chess_data.index)
        num_processes = max(1, min(cpu_count() - 1, len(game_indices)))
        
        # Create chunks
        chunks = chunker.create_balanced_chunks(game_indices, chess_data, num_processes)
        
        if not chunks:
            return []
        
        # Process chunks
        if pool is not None:
            # Use provided persistent pool
            results = pool.starmap(worker_process_games, 
                                 [(chunk, shared_data) for chunk in chunks])
        else:
            # Create new pool
            with Pool(processes=num_processes) as new_pool:
                warm_up_workers(new_pool, num_processes)
                results = new_pool.starmap(worker_process_games, 
                                         [(chunk, shared_data) for chunk in chunks])
        
        # Flatten results
        corrupted_games = [game for sublist in results for game in sublist]
        
        # Print performance metrics
        elapsed = time.time() - start_time
        games_per_second = len(chess_data) / elapsed
        moves_per_second = chess_data['PlyCount'].sum() / elapsed
        
        print(f"Processed {len(chess_data)} games in {elapsed:.2f}s "
              f"({games_per_second:.0f} games/s, {moves_per_second:.0f} moves/s)")
        print(f"Found {len(corrupted_games)} corrupted games "
              f"({len(corrupted_games)/len(chess_data)*100:.1f}%)")
        
        return corrupted_games
        
    finally:
        cleanup_shared_data(shared_data)

def worker_process_games(game_indices_chunk: List[str], shared_data: Dict) -> List[str]:
    """
    Worker function that processes a chunk of games.
    
    Args:
        game_indices_chunk: List of game IDs to process
        shared_data: Shared memory data structure
        
    Returns:
        List of corrupted game IDs from this chunk
    """
    corrupted_games = []
    
    # Create reusable agents and environment
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    
    # Handle Windows vs Unix data access
    if shared_data.get('windows_mode', False):
        # Windows: Direct DataFrame access
        chess_data = shared_data['data']
        
        for game_id in game_indices_chunk:
            try:
                row = chess_data.loc[game_id]
                ply_count = int(row['PlyCount'])
                
                # Extract moves
                moves = {}
                for col in chess_data.columns:
                    if col != 'PlyCount':
                        moves[col] = row[col]
                
                # Process game
                result = play_one_game(game_id, ply_count, moves, 
                                     w_agent, b_agent, environ)
                if result is not None:
                    corrupted_games.append(result)
                
                # Reset for next game
                environ.reset_environ()
                
            except Exception as e:
                logger.critical(f"Error processing game {game_id}: {e}")
                corrupted_games.append(game_id)
    else:
        # Unix: Shared memory access
        # Access shared memory arrays
        indices_shm = shared_memory.SharedMemory(name=shared_data['indices']['shm_name'])
        game_indices_arr = np.ndarray(tuple(shared_data['indices']['shape']), 
                                     dtype=np.dtype(shared_data['indices']['dtype']), 
                                     buffer=indices_shm.buf)
        game_indices_list = game_indices_arr.tolist()
        
        ply_shm = shared_memory.SharedMemory(name=shared_data['ply_counts']['shm_name'])
        ply_counts = np.ndarray(tuple(shared_data['ply_counts']['shape']), 
                               dtype=np.dtype(shared_data['ply_counts']['dtype']), 
                               buffer=ply_shm.buf)
        
        # Access move columns
        move_columns = {}
        move_shms = {}
        for col, col_info in shared_data['move_columns'].items():
            shm = shared_memory.SharedMemory(name=col_info['shm_name'])
            move_shms[col] = shm
            move_columns[col] = np.ndarray(tuple(col_info['shape']), 
                                         dtype=np.dtype(col_info['dtype']), 
                                         buffer=shm.buf)
        
        # Process games
        for game_id in game_indices_chunk:
            try:
                idx = game_indices_list.index(game_id)
                ply_count = int(ply_counts[idx])
                moves = {col: move_columns[col][idx] for col in move_columns}
                
                # Process game
                result = play_one_game(game_id, ply_count, moves, 
                                     w_agent, b_agent, environ)
                if result is not None:
                    corrupted_games.append(result)
                
                # Reset for next game
                environ.reset_environ()
                
            except ValueError:
                logger.critical(f"Game {game_id} not found in indices")
                corrupted_games.append(game_id)
            except Exception as e:
                logger.critical(f"Error processing game {game_id}: {e}")
                corrupted_games.append(game_id)
        
        # Clean up shared memory access
        indices_shm.close()
        ply_shm.close()
        for shm in move_shms.values():
            shm.close()
    
    return corrupted_games

def play_one_game(game_id: str, 
                  ply_count: int, 
                  moves: Dict[str, str], 
                  w_agent: Agent, 
                  b_agent: Agent, 
                  environ: Environ) -> Optional[str]:
    """
    Process a single game.
    
    Args:
        game_id: Game identifier
        ply_count: Total number of plies (half-moves) in the game
        moves: Dictionary of moves by turn
        w_agent: White player agent
        b_agent: Black player agent
        environ: Chess environment
        
    Returns:
        game_id if corrupted, None if successful
    """
    while True:
        # Get current state
        curr_state, legal_moves = environ.get_curr_state_and_legal_moves()
        
        # Check termination conditions
        turn_index: int = curr_state['turn_index'] # type: ignore
        if turn_index >= ply_count:
            break
        if environ.board.is_game_over():
            break
        
        # Determine which agent's turn
        curr_turn: str = curr_state['curr_turn'] # type: ignore
        white_to_move: bool = curr_state['white_to_move'] # type: ignore
        agent = w_agent if white_to_move else b_agent
        
        # Get move from game data
        chess_move_san = agent.choose_action(moves, curr_turn)
        
        # Handle empty moves - these indicate corruption
        if pd.isna(chess_move_san) or chess_move_san == '':
            logger.critical(f"Empty move found for game {game_id}, turn {curr_turn}")
            return game_id
        
        # Convert and validate move
        try:
            move_obj = environ.convert_san_to_move_object(chess_move_san)
        except Exception as e:
            logger.critical(f"Invalid move format '{chess_move_san}' "
                          f"for game {game_id}, turn {curr_turn}: {e}")
            return game_id
        
        # Check if move is legal
        if move_obj not in legal_moves:
            logger.critical(f"Illegal move '{chess_move_san}' "
                          f"for game {game_id}, turn {curr_turn}")
            return game_id
        
        # Apply move
        environ.push_move_object(move_obj)
    
    return None  # Game processed successfully

