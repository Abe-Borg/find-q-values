# file: main/play_games.py

import pandas as pd
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import game_settings
from tqdm import tqdm

def main():    
    est_q_values_dir = 'est_q_values'
    
    # Get all files in the est_q_values directory
    files = os.listdir(est_q_values_dir)
    
    for file_name in tqdm(files):
        # Only process files with 'chess_games' in the name
        if 'chess_games' not in file_name:
            continue
            
        # Create new file name by replacing 'chess_games' with 'est_q_values'
        new_file_name = file_name.replace('chess_games', 'est_q_values')
        
        # Rename the file
        old_path = os.path.join(est_q_values_dir, file_name)
        new_path = os.path.join(est_q_values_dir, new_file_name)
        os.rename(old_path, new_path)

if __name__ == '__main__':
    main()