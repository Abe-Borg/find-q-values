# Clean Chess DB

A high-performance Python application for processing, validating, and cleaning large chess game databases.

## Overview

Clean Chess DB is designed to efficiently process large datasets of chess games, validate their move sequences, and identify corrupted or invalid games. It leverages multiprocessing to achieve high throughput and incorporates various performance optimizations to handle datasets with millions of games.

## Features

- **Fast Chess Game Processing**: Efficiently process and validate chess games from PGN/pickle databases
- **Multiprocessing Support**: Parallel game processing for optimal performance on multi-core systems
- **Corrupted Game Detection**: Identify and flag games with invalid or illegal moves
- **Performance Monitoring**: Built-in benchmarking and profiling tools
- **Memory Optimization**: Careful memory management for handling large datasets
- **Extensive Test Suite**: Comprehensive testing for correctness and performance

## Requirements

- Python 3.12+
- Dependencies:
  - pandas
  - python-chess
  - numpy
  - psutil
  - tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/clean-chess-db.git
cd clean-chess-db
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To process a chess database file:

```python
from training.game_simulation import play_games
import pandas as pd

# Load chess data
chess_data = pd.read_pickle("path/to/chess_data.pkl", compression="zip")

# Process games and identify corrupted ones
corrupted_games = play_games(chess_data)

# Remove corrupted games
clean_data = chess_data.drop(corrupted_games)

# Save cleaned data
clean_data.to_pickle("path/to/clean_chess_data.pkl", compression="zip")
```

### Processing Large Datasets

For large datasets, use the main script which processes files in parts:

```bash
python main/play_games.py
```

This script processes all parts specified in `utils/game_settings.py`.

## Project Structure

```
clean-chess-db/
├── agents/               # Chess-playing agent classes
├── debug/                # Log files
├── environment/          # Chess environment classes
├── main/                 # Main execution scripts
├── optimization_testing/ # Performance benchmarking tools
├── tests/                # Test suite
├── training/             # Game simulation logic
└── utils/                # Helper functions and constants
```

## Configuration

Modify `utils/game_settings.py` to configure:
- File paths for input/output data
- Logging settings
- Other application parameters

## Performance Optimization

This project implements several optimizations for performance:

1. **Multiprocessing**: Distributes workload across CPU cores
2. **Data Preloading**: Minimizes DataFrame lookups during processing
3. **Move Caching**: Caches legal moves to avoid recalculation
4. **Memory Management**: Optimized data structures to reduce memory footprint
5. **Chunked Processing**: Processes data in optimally sized chunks

## Benchmarking

The project includes benchmarking tools to measure performance:

```bash
# Run basic benchmarks
python -m pytest tests/test_performance_profiling.py -v

# Run detailed profiling
python optimization_testing/profiling_script.py
```

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest
```

## License

[Your license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Python Chess (python-chess) library
- [Any other acknowledgements]
