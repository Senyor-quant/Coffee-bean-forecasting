from pathlib import Path
from models.lstm_train import run_training
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    data_dir = Path(os.getenv("COFFEE_DATA_DIR", "./data"))
    input_file = os.getenv("LSTM_INPUT_FILE", "combined_data_arimax_05052025_1301.csv")
    target = os.getenv("LSTM_TARGET_COLUMN", "KC_1")
    window = int(os.getenv("LSTM_WINDOW", 30))
    run_training(data_dir=data_dir, input_file=input_file, target_column=target, n_steps=window)

if __name__ == "__main__":
    main()
