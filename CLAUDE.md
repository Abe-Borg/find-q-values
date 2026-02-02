# CLAUDE.md

## Project Overview

**find-q-values** is a Python application that processes large chess game databases and estimates Q-values (state-action values) for each move using the Stockfish chess engine. The estimated Q-values are intended for use in machine learning training pipelines.

The project reads chess games stored as Pandas DataFrames in pickle files, replays each game move-by-move on a chess board, uses Stockfish at depth 1 to evaluate positions, and outputs DataFrames of estimated Q-values.

> **Note:** The README describes this as "Clean Chess DB" (corruption detection), but the actual codebase implements Q-value estimation. The README is outdated.

## Project Structure

```
find-q-values/
├── agents/                    # Chess agent (move selection from game data)
│   └── Agent.py               # Agent class - retrieves stored moves by turn key
├── environment/               # Chess board state management
│   └── Environ.py             # Environ class - wraps python-chess Board
├── game_simulation/           # Core game processing and Q-value estimation
│   └── game_simulation.py     # Main logic: chunking, multiprocessing, Q-value calc
├── main/                      # Entry point scripts
│   └── play_games.py          # CLI entry point with argparse
├── utils/                     # Configuration and constants
│   ├── constants.py           # Chess move values, turn limits, engine settings
│   └── game_settings.py       # File paths (100 input/output pairs), engine config
├── tests/                     # Test directory (empty - no tests written yet)
├── debug/                     # Log output files
├── requirements.txt           # Python dependencies (UTF-16 encoded)
└── README.md                  # Project README (outdated - describes different project)
```

### Data directories (gitignored)

- `chess_data/` - Input pickle files (`chess_games_part_1.pkl` through `chess_games_part_100.pkl`)
- `est_q_values/` - Output pickle files (`est_q_values_part_1.pkl` through `est_q_values_part_100.pkl`)

## Key Modules

### `agents/Agent.py`
Simple agent that looks up moves from a dictionary of game data by turn key (e.g., `W1`, `B1`, `W2`, `B2`). One agent is created per color (White/Black).

### `environment/Environ.py`
Wraps `chess.Board` from python-chess. Manages:
- Board state and legal moves
- Turn tracking via a pre-computed turn list (`W1`, `B1`, `W2`, `B2`, ... up to 400 plies)
- Move conversion (SAN to move objects)
- Board reset between games

Uses a class-level `_turn_list` cache shared across instances.

### `game_simulation/game_simulation.py`
The core module (~430 lines). Contains two parallel processing workflows:

1. **Game replay** (`play_games` -> `worker_process_games` -> `play_one_game`): Replays games through the board. Currently incomplete (multiple TODO comments about return values and Q-value integration).

2. **Q-value generation** (`generate_q_est_df` -> `worker_generate_q_est` -> `find_estimated_q_value`): Processes games, evaluates positions with Stockfish, and produces Q-value DataFrames. This path is more complete.

Key class: `AdaptiveChunker` - balances game chunks across workers using a greedy algorithm based on estimated ply counts.

### `main/play_games.py`
CLI entry point. Supports:
```bash
python main/play_games.py                      # Process all 100 files
python main/play_games.py --single 42          # Process only file 42
python main/play_games.py --start 1 --end 10   # Process files 1-10
```

Creates a persistent multiprocessing pool (CPU count - 1 workers) and iterates through pickle files.

### `utils/constants.py`
Chess move value constants used for scoring:
- `mate_score`: 1000
- `promotion_queen`: 900
- `promotion`: 500
- `check`: 300
- Game limits: 200 turns per player (400 plies max)
- Engine: depth 1, single PV line

### `utils/game_settings.py`
All file paths (100 input files, 100 output files, 6 log files) and Stockfish engine search limits. Paths are constructed relative to the `utils/` directory using `pathlib.Path`.

**Missing config:** `stockfish_filepath` is referenced in `start_chess_engine()` but not defined in `game_settings.py`.

## Data Format

**Input DataFrame columns:** `PlyCount, W1, B1, W2, B2, W3, B3, ...`
- `PlyCount`: integer, total half-moves in the game
- `W1`, `B1`, etc.: SAN-format move strings (e.g., `e4`, `Nf3`, `Bxe5+`, `Qh7#`)

**Output DataFrame columns:** Same structure, but move columns contain float Q-values instead of SAN strings.

Files are stored as zip-compressed pickle: `pd.read_pickle(path, compression='zip')`

## Build and Run

### Prerequisites
- Python 3.11+
- Stockfish chess engine binary (path must be configured in `game_settings.py`)

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running
```bash
python main/play_games.py [--single N | --start N --end M]
```

### Testing
```bash
python -m pytest
```
No tests exist yet. The `tests/` directory contains only `__init__.py`.

## Dependencies

Core:
- `chess` (python-chess 1.10.0) - Board representation, move parsing, UCI engine interface
- `pandas` (2.1.4) - DataFrame operations for game data
- `numpy` (1.26.2) - Numerical operations
- `psutil` (5.9.6) - CPU count for worker pool sizing
- `tqdm` (4.66.4) - Progress bars

Dev/testing:
- `pytest`, `pytest-benchmark`, `pytest-cov`, `pytest-html`
- `memory-profiler`

## Code Conventions

### Naming
- **Classes:** PascalCase (`Agent`, `Environ`, `AdaptiveChunker`)
- **Functions/methods:** snake_case (`play_one_game`, `get_curr_state_and_legal_moves`)
- **Constants:** UPPER_SNAKE_CASE for dicts (`CHESS_MOVE_VALUES`), lower_snake_case for scalars (`max_num_turns_per_player`)
- **Files:** PascalCase for class files (`Agent.py`, `Environ.py`), snake_case for module files (`game_simulation.py`, `play_games.py`)

### Patterns
- **File headers:** Each file starts with `# project name: find-q-values` and `# file: <path>`
- **Logging:** File-based logging at CRITICAL level; logger per module via `logging.getLogger("<module_name>")`
- **Type hints:** Used on function signatures throughout
- **Multiprocessing:** `Pool.starmap()` for distributing chunks to workers
- **Resource cleanup:** try/finally for pool and engine lifecycle
- **Docstrings:** Google-style with Args/Returns sections

### Architecture decisions
- One `Environ` and one `Agent` per color are created per worker process and reused across games in a chunk
- `Environ` resets between games via `reset_environ()` rather than creating new instances
- Turn list is cached at class level to avoid recomputation
- Chunks are balanced by estimated processing time (ply count / calibrated moves-per-second)

## Known Issues and Incomplete Areas

1. **Incomplete game replay workflow:** `play_one_game()`, `play_games()`, and `worker_process_games()` all have TODO/uncertain return values. The Q-value estimation step inside `play_one_game` is marked as TODO.
2. **Missing Stockfish path:** `game_settings.stockfish_filepath` is used in `start_chess_engine()` but never defined.
3. **Missing `chess` import in game_simulation.py:** Functions like `find_estimated_q_value` and `analyze_board_state` use `chess.Board`, `chess.WHITE`, `chess.IllegalMoveError`, and `chess.engine` but `chess` is not imported at the top of the file.
4. **Missing `constants` import in game_simulation.py:** `constants` is referenced in Q-value functions but not imported.
5. **Missing `chunkify` function:** `generate_q_est_df` calls `chunkify()` which is not defined anywhere in the codebase.
6. **`cleanup_shared_data` called on wrong type:** In `play_games()`, `shared_data` is assigned the raw DataFrame but `cleanup_shared_data` expects a dict with shared memory names.
7. **No tests:** Test directory is empty.
8. **requirements.txt encoding:** File is UTF-16 encoded rather than standard UTF-8.
9. **README is outdated:** Describes "Clean Chess DB" project, not Q-value estimation.
