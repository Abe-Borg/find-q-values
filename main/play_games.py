# file: main/play_games.py

import pandas as pd
import time
import sys
import os
from multiprocessing import Pool, cpu_count
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback
from utils import game_settings
import logging
from tqdm import tqdm
import argparse

# Import the refactored game simulation module
from training.game_simulation import play_games, warm_up_workers

# Logger setup
logger = logging.getLogger("play_games")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(game_settings.play_games_logger_filepath)
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def create_persistent_pool(num_workers=None):
    """
    Create a persistent process pool with optimal worker count.
    
    Args:
        num_workers: Number of worker processes (default: CPU count - 1)
        
    Returns:
        Pool object
    """
    if num_workers is None:
        # Leave one CPU for the main process
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Creating persistent pool with {num_workers} workers...")
    pool = Pool(processes=num_workers)
    
    # Warm up the workers
    warm_up_workers(pool, num_workers)
    print("Worker pool ready!")
    
    return pool


def parse_arguments():
    """Parse command line arguments for dataframe range."""
    parser = argparse.ArgumentParser(
        description='Process chess game dataframes for corruption detection.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python play_games.py                    # Process all 100 dataframes
  python play_games.py --start 1 --end 10 # Process dataframes 1-10
  python play_games.py --start 5 --end 5  # Process only dataframe 5
  python play_games.py --single 42        # Process only dataframe 42
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--single',
        type=int,
        metavar='N',
        help='Process only dataframe N (1-100)'
    )
    group.add_argument(
        '--start',
        type=int,
        metavar='N',
        help='Start dataframe number (1-100, use with --end)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        metavar='N',
        help='End dataframe number (1-100, use with --start)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.single is not None:
        if not 1 <= args.single <= 100:
            parser.error("--single must be between 1 and 100")
        return args.single, args.single
    
    if args.start is not None and args.end is None:
        parser.error("--start requires --end")
    
    if args.end is not None and args.start is None:
        parser.error("--end requires --start")
    
    if args.start is not None and args.end is not None:
        if not 1 <= args.start <= 100:
            parser.error("--start must be between 1 and 100")
        if not 1 <= args.end <= 100:
            parser.error("--end must be between 1 and 100")
        if args.start > args.end:
            parser.error("--start must be less than or equal to --end")
        return args.start, args.end
    
    # Default: process all 100 dataframes
    return 1, 100


def main():
    """Main function to process chess game files."""
    start_time = time.time()
    
    # Parse command line arguments
    start_df, end_df = parse_arguments()
    num_dataframes_to_process = end_df - start_df + 1
    
    print(f"Processing dataframes {start_df} to {end_df} ({num_dataframes_to_process} total)")
    
    total_games_processed = 0
    total_corrupted_games = 0
    
    # Create persistent process pool
    persistent_pool = create_persistent_pool()
    
    try:
        # Process each data file
        for part in tqdm(range(start_df, end_df + 1), 
                        desc="Processing parts", unit="part"):
            
            # Get file path
            file_path = getattr(game_settings, f'chess_games_filepath_part_{part}')
            
            try:
                # Load data
                part_start_time = time.time()
                chess_data = pd.read_pickle(file_path, compression='zip')
                
                print(f"\nPart {part}: Loaded {len(chess_data)} games")
                
                # Process games
                corrupted_games = play_games(chess_data, persistent_pool)
                
                # Update statistics
                total_games_processed += len(chess_data)
                total_corrupted_games += len(corrupted_games)
                
                # Report results
                part_elapsed = time.time() - part_start_time
                print(f"Part {part}: {len(corrupted_games)} corrupted games "
                      f"({len(corrupted_games)/len(chess_data)*100:.1f}%)")
                print(f"Part {part}: Completed in {part_elapsed:.2f}s")
                
                # Remove corrupted games and save
                if corrupted_games:
                    chess_data = chess_data.drop(corrupted_games)
                    chess_data.to_pickle(file_path, compression='zip')
                    print(f"Part {part}: Removed corrupted games and saved")
                
            except FileNotFoundError:
                print(f"\nPart {part}: File not found - {file_path}")
                logger.critical(f"File not found: {file_path}")
                continue
                
            except Exception as e:
                print(f"\nPart {part}: Error processing - {e}")
                logger.critical(f"Error processing {file_path}: {e}")
                logger.critical(traceback.format_exc())
                continue
    
    finally:
        # Clean up pool
        print("\nCleaning up worker pool...")
        persistent_pool.close()
        persistent_pool.join()
        print("Worker pool cleaned up")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Dataframe range: {start_df} to {end_df}")
    print(f"Total files processed: {num_dataframes_to_process}")
    print(f"Total games processed: {total_games_processed:,}")
    print(f"Total corrupted games: {total_corrupted_games:,} "
          f"({total_corrupted_games/total_games_processed*100:.2f}%)" if total_games_processed > 0 else "(N/A)")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Average time per file: {total_time/num_dataframes_to_process:.2f}s")
    
    if total_games_processed > 0:
        overall_games_per_sec = total_games_processed / total_time
        print(f"Overall performance: {overall_games_per_sec:.0f} games/s")


if __name__ == '__main__':
    main()