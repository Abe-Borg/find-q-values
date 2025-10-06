# file: game_simulation/game_simulation.py

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import time
from agents.Agent import Agent
from utils import game_settings
from environment.Environ import Environ
from multiprocessing import Pool, cpu_count, shared_memory
import logging

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
        min_games_per_chunk = max(100, total_games // (num_processes * 4))
        
        # Calculate optimal number of chunks
        max_chunks_by_workers = num_processes
        max_chunks_by_min_size = max(1, total_games // min_games_per_chunk)
        optimal_chunks = min(max_chunks_by_workers, max_chunks_by_min_size, total_games)
        
        print(f"Chunking {total_games} games into {optimal_chunks} chunks "
              f"(min {min_games_per_chunk} games/chunk)")
        
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

def cleanup_shared_data(shared_data: Dict) -> None:
    """Clean up shared memory resources."""
    
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
        
    """
    start_time = time.time()
    
    # Create shared memory representation
    shared_data = chess_data
    
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
        
        
        # Print performance metrics
        elapsed = time.time() - start_time
        games_per_second = len(chess_data) / elapsed
        moves_per_second = chess_data['PlyCount'].sum() / elapsed
        
        print(f"Processed {len(chess_data)} games in {elapsed:.2f}s "
              f"({games_per_second:.0f} games/s, {moves_per_second:.0f} moves/s)")
        
        # not sure what to return, if anything.
        
    finally:
        cleanup_shared_data(shared_data)

def worker_process_games(game_indices_chunk: List[str], shared_data: Dict) -> List[str]:
    """
    Worker function that processes a chunk of games.
    
    Args:
        game_indices_chunk: List of game IDs to process
        shared_data: Shared memory data structure
        
    Returns:
        
    """
    # Create reusable agents and environment
    w_agent = Agent('W')
    b_agent = Agent('B')
    environ = Environ()
    
    chess_data = shared_data['data']
    
    for game_id in game_indices_chunk:
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
        
        # Reset for next game
        environ.reset_environ()
    
    # not sure what to return.

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
        
        # Convert and validate move
        move_obj = environ.convert_san_to_move_object(chess_move_san)

        # Apply move
        environ.push_move_object(move_obj)
    
    # not sure what to return, if anything.


def generate_q_est_df(chess_data: pd.DataFrame) -> pd.DataFrame:
    # Ensure all indices are strings and properly formatted
    game_indices = [str(idx) for idx in chess_data.index]
    
    # Create master DataFrame with exact structure
    master_df = pd.DataFrame(
        index=chess_data.index,  # Use original index
        columns=chess_data.columns,
        dtype=object
    )

    master_df['PlyCount'] = chess_data['PlyCount']
    move_cols = [col for col in chess_data.columns if col.startswith(('W', 'B'))]
    master_df[move_cols] = 0.0
    
    num_processes = min(cpu_count(), len(game_indices))
    chunks = chunkify(game_indices, num_processes)   

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            worker_generate_q_est, 
            [(chunk, chess_data.loc[chunk]) for chunk in chunks]
        )  

    for chunk_df in results:
        if isinstance(chunk_df, pd.DataFrame):
            for game_idx in chunk_df.index:
                if game_idx in master_df.index:
                    master_df.loc[game_idx, move_cols] = chunk_df.loc[game_idx, move_cols]
    
    return master_df

def generate_q_est_df_one_game(chess_data, game_number, environ, engine) -> pd.DataFrame:
    game_df = pd.DataFrame(
        index=[game_number],
        columns=chess_data.columns,
        dtype=float
    )
    game_df.fillna(0.0, inplace=True)
    
    num_moves = chess_data.at[game_number, 'PlyCount']
    curr_state = environ.get_curr_state()
    moves_processed = 0
    
    while moves_processed < num_moves:
        curr_turn = curr_state['curr_turn']
        try:
            chess_move = chess_data.at[game_number, curr_turn]
                
            if chess_move.endswith('#'):
                game_df.at[game_number, curr_turn] = constants.CHESS_MOVE_VALUES['mate_score']
                break
            
            try:
                apply_move_and_update_state(chess_move, environ)
                est_qval = find_estimated_q_value(environ, engine)
                game_df.at[game_number, curr_turn] = est_qval
                moves_processed += 1
                
                curr_state = environ.get_curr_state()
                if environ.board.is_game_over() or not curr_state['legal_moves']:
                    break
                    
            except chess.IllegalMoveError as e:
                logger.critical(f"Invalid move '{chess_move}' for game {game_number}, turn {curr_turn}")
                game_df.at[game_number, curr_turn] = 0
                
        except Exception as e:
            logger.critical(f"Error processing game {game_number}, turn {curr_turn}: {str(e)}")
            break
            
    return game_df

def find_estimated_q_value(environ, engine) -> int:
    # create temp board to avoid sending null move to chess engine
    temp_board = chess.Board(environ.board.fen())
    anticipated_next_move = analyze_board_state(temp_board, engine)
    environ.load_chessboard_for_q_est(anticipated_next_move)

    if environ.board.is_game_over() or not environ.get_legal_moves():
        environ.board.pop()

    temp_board = chess.Board(environ.board.fen())
    est_qval_analysis = analyze_board_state(temp_board, engine)

    if est_qval_analysis['mate_score'] is None:
        est_qval = est_qval_analysis['centipawn_score']
    else:
        est_qval = constants.CHESS_MOVE_VALUES['mate_score']

    environ.board.pop()
    return est_qval 

def analyze_board_state(board, engine) -> dict:
    analysis_result = engine.analyse(board, game_settings.search_limit, multipv = constants.chess_engine_num_moves_to_return)
    mate_score = None
    centipawn_score = None
    anticipated_next_move = None
    pov_score = analysis_result[0]['score'].white() if board.turn == chess.WHITE else analysis_result[0]['score'].black()
    
    if pov_score.is_mate():
        mate_score = pov_score.mate()
    else:
        centipawn_score = pov_score.score()

    anticipated_next_move = analysis_result[0]['pv'][0]
    return {
        'mate_score': mate_score,
        'centipawn_score': centipawn_score,
        'anticipated_next_move': anticipated_next_move
    }

def start_chess_engine(): 
    chess_engine = chess.engine.SimpleEngine.popen_uci(game_settings.stockfish_filepath)
    return chess_engine 

def worker_generate_q_est(game_indices_chunk, chunk_data):
    chunk_df = pd.DataFrame(
        index=game_indices_chunk,
        columns=chunk_data.columns,
        dtype=object
    )

    chunk_df['PlyCount'] = chunk_data['PlyCount']
    move_cols = [col for col in chunk_data.columns if col.startswith(('W', 'B'))]
    chunk_df[move_cols] = 0.0
    
    environ = Environ()
    engine = start_chess_engine()
    
    try:
        for game_number in game_indices_chunk:
            try:
                # Process one game
                ply_count = chunk_data.at[game_number, 'PlyCount']
                curr_state = environ.get_curr_state()
                move_count = 0
                
                while move_count < ply_count:
                    curr_turn = curr_state['curr_turn']
                    chess_move = chunk_data.at[game_number, curr_turn]
                    
                    if isinstance(chess_move, str) and chess_move.endswith('#'):
                        chunk_df.at[game_number, curr_turn] = constants.CHESS_MOVE_VALUES['mate_score']
                        break
                        
                    try:
                        environ.board.push_san(chess_move)
                        environ.update_curr_state()
                        est_qval = find_estimated_q_value(environ, engine)
                        chunk_df.at[game_number, curr_turn] = est_qval
                        move_count += 1
                        curr_state = environ.get_curr_state()
                        
                        if environ.board.is_game_over():
                            break
                            
                    except chess.IllegalMoveError:
                        logger.critical(f"Invalid move '{chess_move}' in game {game_number}, turn {curr_turn}")
                        chunk_df.at[game_number, curr_turn] = 0
                        break
                        
            except Exception as e:
                logger.critical(f"Error processing game {game_number} in worker_generate_q_est: {str(e)}")
                continue
            finally:
                environ.reset_environ()
                
    finally:
        engine.quit()
    return chunk_df
