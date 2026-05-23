from pathlib import Path

DATA_DIR = Path().parent.resolve() / 'assets' / 'dataset' / 'wave_buoy_data_halifax.csv'
print(DATA_DIR)

INPUT_SIZE = 6
HORIZON = 24
TARGET = 'VCAR'
THRESHOLD_PERCENTILE = 0.99
