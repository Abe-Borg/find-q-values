# file: game_simulation/generate_q_estimates.py

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
logger = logging.getLogger("generate_q_estimates")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(str(game_settings.generate_q_estimates_logger_filepath))
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)





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